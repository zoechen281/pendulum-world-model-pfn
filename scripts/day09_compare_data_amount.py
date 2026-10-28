"""Day9数据量对照：普通残差模型，200条与1200条训练转移。"""

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

from src.day04_world_model import (
    DynamicsMLP,
    episodes_to_tensors,
    fit_input_mean,
    fit_input_std,
    predict_next_observation,
)


def main():
    root = Path(__file__).resolve().parents[1]
    # 导师提供的归档辅助：独立目录、来源哈希，不覆盖旧实验。
    source_paths = [Path(__file__).resolve(), root / "src/day04_world_model.py",
                    root / "results/day03_pendulum_dataset.json",
                    root / "results/day04_delta_normalized_run.json"]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_paths}
    dataset = json.loads(
        (root / "results/day03_pendulum_dataset.json").read_text(encoding="utf-8")
    )
    train_episodes = dataset["splits"]["train"]
    val_inputs, val_targets = episodes_to_tensors(dataset["splits"]["val"])

    # 当前任务：把首条完整训练轨迹转换成小数据组。
    # episodes_to_tensors要求“轨迹列表”，因此使用train_episodes[:1]，
    # 不用去掉外层列表的train_episodes[0]。
    # 返回结果依次保存为small_inputs (200,4)、small_targets (200,3)。
    small_inputs, small_targets = episodes_to_tensors(train_episodes[:1])
    full_inputs, full_targets = episodes_to_tensors(train_episodes)
    input_mean = fit_input_mean(small_inputs)
    input_std = fit_input_std(small_inputs)

    # 两组嵌套且来自固定轨迹，不是多个独立样本量重复实验。
    # 本次固定使用共同200条训练子集拟合的mean/std，两组复用，
    # 将标准化变换保持一致，且不让200条组额外读取其他1000条的统计量。
    # 这是本次控制变量协议，区别于“各训练子集分别拟合”的另一种合法协议。

    # 导师提供的工程辅助：相同初值、独立优化器、同一更新次数。
    assert small_inputs.shape == (200, 4) and small_targets.shape == (200, 3)
    assert full_inputs.shape == (1200, 4) and full_targets.shape == (1200, 3)
    assert torch.equal(small_inputs, full_inputs[:200])
    assert torch.equal(small_targets, full_targets[:200])
    torch.set_num_threads(1)
    torch.manual_seed(0)
    initial_model = DynamicsMLP()
    models = {name: copy.deepcopy(initial_model) for name in ("small", "full")}
    assert all(
        torch.equal(models["small"].state_dict()[name], value)
        for name, value in models["full"].state_dict().items()
    )
    groups = {
        "small": (small_inputs, small_targets),
        "full": (full_inputs, full_targets),
    }
    epochs = 200
    results = {}
    for group_name, (train_inputs, train_targets) in groups.items():
        model = models[group_name]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        observations = train_inputs[:, :3]
        actions = train_inputs[:, 3:4]
        started = time.perf_counter()
        model.train()
        for epoch in range(epochs):
            # 当前任务：完成一轮训练的五步。
            # 清梯度；调用predict_next_observation得到predictions (N,3)；
            # 和train_targets计算最终下一观测MSE为loss；backward；step。
            # 预测函数参数顺序：model、observations、actions、input_mean、input_std。
            predictions = predict_next_observation(
                model, observations, actions, input_mean, input_std
            )
            loss = torch.mean((predictions - train_targets) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_seconds = time.perf_counter() - started
        model.eval()
        with torch.no_grad():
            predictions = predict_next_observation(
                model, observations, actions, input_mean, input_std
            )
            train_mse = torch.mean((predictions - train_targets) ** 2).item()
            val_predictions = predict_next_observation(
                model, val_inputs[:, :3], val_inputs[:, 3:4], input_mean, input_std
            )
            val_mse = torch.mean((val_predictions - val_targets) ** 2).item()
        print(
            f"{group_name}: samples={len(train_inputs)}, "
            f"train MSE={train_mse:.6f}, val MSE={val_mse:.6f}"
        )
        # 导师提供的数值和协议检查，以及结果记录。
        assert predictions.shape == train_targets.shape
        assert val_predictions.shape == val_targets.shape == (400, 3)
        assert not predictions.requires_grad and not val_predictions.requires_grad
        assert torch.isfinite(val_predictions).all() and torch.isfinite(predictions).all()
        assert all(torch.isfinite(p).all() and p.grad is not None
                   and torch.isfinite(p.grad).all() for p in model.parameters())
        results[group_name] = dict(
            samples=len(train_inputs), train_mse=train_mse, validation_mse=val_mse,
            training_seconds=training_seconds,
            sample_presentations=epochs * len(train_inputs),
            parameters=sum(p.numel() for p in model.parameters()),
            component_validation_mse=((val_predictions - val_targets) ** 2).mean(dim=0).tolist(),
            final_state_dict={name: value.detach().tolist() for name, value in model.state_dict().items()},
        )
        print(f"{group_name}: training seconds={training_seconds:.6f}")

    # 更新次数相同不等于计算预算相同；包含数据覆盖差异，不能宣称纯数量因果。
    # 不读test评价，不覆盖旧结果；本次不加入结构模型或控制实验。
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(root))]
               for p in source_paths)
    recorded_at = datetime.now(timezone(timedelta(hours=8)))
    output_dir = root / "results" / ("day09_data_amount_" + recorded_at.strftime("%Y%m%dT%H%M%S_%f"))
    output_dir.mkdir(exist_ok=False)
    for source_path in source_paths[:2]:
        (output_dir / source_path.name).write_bytes(source_path.read_bytes())
    report = dict(
        recorded_at=recorded_at.isoformat(), source_sha256=hashes,
        seed=0, optimizer="SGD", learning_rate=0.01, updates_per_group=epochs,
        device="cpu", torch_threads=1, initial_weights_equal=True,
        initial_state_dict={name: value.detach().tolist() for name, value in initial_model.state_dict().items()},
        input_mean=input_mean.tolist(), input_std=input_std.tolist(),
        normalization_source="train episode0, same 200 transitions for both groups",
        train_episode_indices={"small": [0], "full": list(range(6))},
        validation_episode_indices=[6, 7], validation_samples=len(val_inputs),
        test_evaluated=False, historical_files_unchanged=True,
        versions=dict(python=platform.python_version(), torch=torch.__version__),
        groups=results,
        limitations=["Single initialization seed and fixed nested episode subsets.",
                     "Additional data changes coverage as well as quantity.",
                     "Equal update count, different sample presentations and compute budget.",
                     "Wall time is one short CPU measurement, not a stable benchmark."],
    )
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ARCHIVE: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
