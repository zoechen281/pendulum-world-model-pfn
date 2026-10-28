"""Day9实验B：固定模型与验证起点，比较不同预测步的误差，不重新训练。"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import platform
import time

import torch

from src.day04_world_model import (
    DynamicsMLP,
    episode_observations,
    episode_windows,
    rollout_predictions,
    trajectory_step_mse,
)


def main():
    # 导师提供的恢复与数据准备辅助。使用实验A的full组及其原有统计量。
    root = Path(__file__).resolve().parents[1]
    record_path = (
        root / "results/day09_data_amount_20261017T130753_993032/comparison.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record_hash = hashlib.sha256(record_path.read_bytes()).hexdigest()
    dataset_path = root / "results/day03_pendulum_dataset.json"
    expected_hash = record["source_sha256"]["results\\day03_pendulum_dataset.json"]
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == expected_hash
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    model = DynamicsMLP()
    model.load_state_dict(
        {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in record["groups"]["full"]["final_state_dict"].items()
        }
    )
    model.eval()
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    input_mean = torch.tensor(record["input_mean"], dtype=torch.float32)
    input_std = torch.tensor(record["input_std"], dtype=torch.float32)
    max_horizon = 20
    horizons = (1, 5, 10, 20)
    # 所有比较都用同一批362个起点，短预测长度不额外加入轨迹尾部起点。
    # 一次递推20步后取相应步数；这是预测长度评估，不是MPC规划长度比较。
    windows = [
        window
        for episode in dataset["splits"]["val"]
        for window in episode_windows(episode, max_horizon)
    ]
    assert len(windows) == 362
    error_rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for window in windows:
            true_trajectory = episode_observations(window)
            initial_observation = true_trajectory[:1]
            actions = [
                torch.tensor(t["action"], dtype=torch.float32).reshape(1, 1)
                for t in window
            ]
            # 当前任务：调用rollout_predictions，保存为trajectory（Tensor列表）。
            trajectory = rollout_predictions(
                model, initial_observation, actions, input_mean, input_std
            )
            # 参数依次为model、initial_observation、actions、input_mean、input_std。

            # 后续：torch.cat沿dim0拼成predicted_trajectory (21,3)
            predicted_trajectory = torch.cat(trajectory, dim=0)
            # trajectory_step_mse得到step_mse (21,)，append到error_rows。
            step_mse = trajectory_step_mse(predicted_trajectory, true_trajectory)
            error_rows.append(step_mse)
    error_matrix = torch.stack(error_rows, dim=0)
    mean_step_mse = error_matrix.mean(dim=0)
    evaluation_seconds = time.perf_counter() - started
    for horizon in horizons:
        print(f"step {horizon}: mean MSE={mean_step_mse[horizon].item():.6f}")
    # 后续：stack成(362,21)，沿样本维求均值；报告horizons指定步的末步MSE。
    # 区分“第h步误差”与“前h步平均误差”；不把重叠窗口视为362次独立实验。
    # 记录分episode均值、完整误差矩阵和来源哈希，保存到新结果目录。
    # 不更新参数、不读test、不修改原权重；本实验不扩展控制评价或重训。
    # 以下为导师提供的协议检查与独立归档辅助。
    assert error_matrix.shape == (362, 21) and mean_step_mse.shape == (21,)
    assert torch.isfinite(error_matrix).all() and not error_matrix.requires_grad
    assert torch.count_nonzero(error_matrix[:, 0]) == 0
    assert not model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value, state_before[name]) for name, value in model.state_dict().items())
    assert hashlib.sha256(record_path.read_bytes()).hexdigest() == record_hash
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == expected_hash
    counts = [len(episode_windows(episode, max_horizon)) for episode in dataset["splits"]["val"]]
    episode_means = []
    offset = 0
    for count in counts:
        episode_means.append(error_matrix[offset:offset + count].mean(dim=0).tolist())
        offset += count
    stamp = datetime.now(timezone(timedelta(hours=8)))
    output = root / "results" / ("day09_prediction_horizons_" + stamp.strftime("%Y%m%dT%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    source_paths = [Path(__file__).resolve(), root / "src/day04_world_model.py"]
    for path in source_paths:
        (output / path.name).write_bytes(path.read_bytes())
    result = dict(
        recorded_at=stamp.isoformat(), checkpoint=str(record_path.relative_to(root)),
        checkpoint_sha256=record_hash, model_group="full", dataset_sha256=expected_hash,
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        versions=dict(python=platform.python_version(), torch=torch.__version__),
        input_mean=input_mean.tolist(), input_std=input_std.tolist(),
        protocol="Same 362 starting states, recorded action sequences, recursive predictions; report endpoint MSE at h.",
        horizons=list(horizons), max_horizon=max_horizon,
        validation_episode_indices=[6, 7], windows_per_episode=counts,
        error_matrix=error_matrix.tolist(), mean_step_mse=mean_step_mse.tolist(),
        per_episode_mean_step_mse=episode_means,
        endpoint_mse={str(h): mean_step_mse[h].item() for h in horizons},
        evaluation_seconds=evaluation_seconds, checkpoint_unchanged=True,
        model_parameters_unchanged=True, training_performed=False, test_evaluated=False,
        limitations=["Overlapping windows, only two validation episodes; not 362 independent repeats.",
                     "One fixed trained model; no universal monotonic-error claim.",
                     "Endpoint MSE differs from average MSE across first h steps.",
                     "Fixed start set differs from the 400-transition single-step validation set."],
    )
    report_path = output / "comparison.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ARCHIVE: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
