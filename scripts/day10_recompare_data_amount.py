"""Day9新配置数据量复核：200条与1200条；默认不训练。"""

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from contextlib import redirect_stdout
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


def main(output_dir):
    root = Path(__file__).resolve().parents[1]
    # 导师提供的归档辅助：独立目录、来源哈希，不覆盖旧实验。
    source_paths = [
        Path(__file__).resolve(),
        root / "src/day04_world_model.py",
        root / "results/day03_pendulum_dataset.json",
        root / "results/day04_delta_normalized_run.json",
        root / "results/day09_data_amount_20261017T130753_993032/comparison.json",
        root / "results/day10_convergence_20261017T221844_116308/metrics.json",
    ]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
    }
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
    old = json.loads(source_paths[-2].read_text(encoding="utf-8"))
    convergence = json.loads(source_paths[-1].read_text(encoding="utf-8"))
    assert convergence["meaningful_improvement"] and convergence["best_step"] == 1950
    assert torch.equal(input_mean, torch.tensor(old["input_mean"], dtype=input_mean.dtype))
    assert torch.equal(input_std, torch.tensor(old["input_std"], dtype=input_std.dtype))

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
    for name, value in initial_model.state_dict().items():
        assert torch.equal(value, torch.tensor(old["initial_state_dict"][name], dtype=value.dtype))
    # 训练开始前留存来源与初值；失败时也有可追溯记录。
    for source_path in source_paths[:2]:
        (output_dir / source_path.name).write_bytes(source_path.read_bytes())
    with (output_dir / "initial_weights.json").open("x", encoding="utf-8") as f:
        json.dump({name: value.tolist() for name, value in initial_model.state_dict().items()}, f)
    models = {name: copy.deepcopy(initial_model) for name in ("small", "full")}
    assert all(
        torch.equal(models["small"].state_dict()[name], value)
        for name, value in models["full"].state_dict().items()
    )
    groups = {
        "small": (small_inputs, small_targets),
        "full": (full_inputs, full_targets),
    }
    # 学习者填写的新配置：两组使用相同固定预算，不分别早停。
    epochs = 1950
    learning_rate = 0.001
    results = {}
    for group_name, (train_inputs, train_targets) in groups.items():
        model = models[group_name]
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{group_name}: non-finite loss at {epoch + 1}")
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
        assert math.isfinite(train_mse) and math.isfinite(val_mse)
        assert any(not torch.equal(value, initial_model.state_dict()[name])
                   for name, value in model.state_dict().items())
        print(
            f"{group_name}: samples={len(train_inputs)}, "
            f"train MSE={train_mse:.6f}, val MSE={val_mse:.6f}"
        )
        # 导师提供的数值和协议检查，以及结果记录。
        assert predictions.shape == train_targets.shape
        assert val_predictions.shape == val_targets.shape == (400, 3)
        assert not predictions.requires_grad and not val_predictions.requires_grad
        assert (
            torch.isfinite(val_predictions).all() and torch.isfinite(predictions).all()
        )
        assert all(
            torch.isfinite(p).all()
            and p.grad is not None
            and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        results[group_name] = dict(
            samples=len(train_inputs),
            train_mse=train_mse,
            validation_mse=val_mse,
            training_seconds=training_seconds,
            sample_presentations=epochs * len(train_inputs),
            parameters=sum(p.numel() for p in model.parameters()),
            component_validation_mse=((val_predictions - val_targets) ** 2)
            .mean(dim=0)
            .tolist(),
            final_state_dict={
                name: value.detach().tolist()
                for name, value in model.state_dict().items()
            },
        )
        print(f"{group_name}: training seconds={training_seconds:.6f}")

    # 更新次数相同不等于计算预算相同；包含数据覆盖差异，不能宣称纯数量因果。
    # 不读test评价，不覆盖旧结果；本次不加入结构模型或控制实验。
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(root))]
        for p in source_paths
    )
    recorded_at = datetime.now(timezone(timedelta(hours=8)))
    report = dict(
        recorded_at=recorded_at.isoformat(),
        source_sha256=hashes,
        seed=0,
        optimizer=type(optimizer).__name__,
        learning_rate=learning_rate,
        updates_per_group=epochs,
        device="cpu",
        torch_threads=1,
        initial_weights_equal=True,
        initial_state_dict={
            name: value.detach().tolist()
            for name, value in initial_model.state_dict().items()
        },
        input_mean=input_mean.tolist(),
        input_std=input_std.tolist(),
        normalization_source="train episode0, same 200 transitions for both groups",
        train_episode_indices={"small": [0], "full": list(range(6))},
        validation_episode_indices=[6, 7],
        validation_samples=len(val_inputs),
        test_evaluated=False,
        historical_files_unchanged=True,
        versions=dict(python=platform.python_version(), torch=torch.__version__),
        groups=results,
        old_config=dict(optimizer=old["optimizer"], learning_rate=old["learning_rate"],
                        updates_per_group=old["updates_per_group"]),
        old_groups={mode: {k: v for k, v in g.items()
                           if k in ("samples", "train_mse", "validation_mse")}
                    for mode, g in old["groups"].items()},
        limitations=[
            "Single initialization seed and fixed nested episode subsets.",
            "Additional data changes coverage as well as quantity.",
            "Equal update count, different sample presentations and compute budget.",
            "Wall time is one short CPU measurement, not a stable benchmark.",
        ],
    )
    report_path = output_dir / "comparison.json"
    with report_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    rows = ["# Day9数据量：旧配置/新配置", "",
            "相同seed0初值、嵌套200/1200条train、共同200条统计量、相同400条val；不评估test。",
            "", "|配置|训练条数|训练MSE|验证MSE|", "|---|---:|---:|---:|"]
    for label, groups in (("旧SGD200", report["old_groups"]), ("新Adam1950", results)):
        for mode, g in groups.items():
            rows.append(f"|{label}|{g['samples']}|{g['train_mse']:.10g}|{g['validation_mse']:.10g}|")
    rows += ["", "原始旧结果保留。单seed与固定轨迹；数据覆盖随数量改变，同更新次数不等计算量。",
             "K来自P0-1验证集选择，本次仍属开发验证；不据此证明普遍数据量规律。",
             "本次只训练两组一次，未执行多步/控制；预测长度评估后续复用新full权重，不再训练。",
             "配置、统计量、初末权重、耗时与来源哈希见comparison.json；stdout.txt保存运行日志。"]
    with (output_dir / "SUMMARY.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"ARCHIVE: {report_path}")
    return report_path


def archive_run():
    """导师辅助：先建新目录，失败也保存日志；不改学习者训练循环。"""
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / ("day10_day09_adam_" + datetime.now(
        timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    capture = io.StringIO()
    started = time.perf_counter()
    try:
        with redirect_stdout(capture):
            path = main(output)
    except (Exception, KeyboardInterrupt) as error:
        import traceback
        with (output / "failed.json").open("x", encoding="utf-8") as f:
            json.dump(dict(error=repr(error), elapsed_seconds=time.perf_counter()-started,
                           test_evaluated=False), f, ensure_ascii=False, indent=2)
        capture.write(traceback.format_exc())
        raise
    finally:
        with (output / "stdout.txt").open("x", encoding="utf-8") as f:
            f.write(capture.getvalue())
        print(capture.getvalue(), end="")
        print(f"归档目录：{output}")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Day9新配置练习，训练前需审查和单独同意。"
    )
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()
    if args.run_training:
        archive_run()
    else:
        print("NOT RUN：Adam lr0.001，seed0，200/1200条组各1950更新；未训练、未创建实验目录。")
