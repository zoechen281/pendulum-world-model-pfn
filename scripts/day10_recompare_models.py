"""Day8新配置复核；默认不训练，显式授权后独立归档。"""

import copy
import hashlib
import io
import json
import math
import platform
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch

from src.day04_world_model import (
    DynamicsMLP,
    episodes_to_tensors,
    fit_input_mean,
    fit_input_std,
    predict_next_observation,
)
from src.day08_physics_model import PhysicsStructuredMLP


def predict_for_comparison(model, mode, observations, actions, input_mean, input_std):
    """两组统一返回原尺度下一观测 (N,3)，调用方控制梯度。"""
    if mode == "baseline":
        # 复用学习者已有的残差预测函数；它也支持当前按行独立的批量计算。
        return predict_next_observation(
            model, observations, actions, input_mean, input_std
        )
    if mode == "structured":
        # 当前任务：调用结构模型，并return其最终下一观测预测。
        # 参数依次为 observations、actions、input_mean、input_std。
        return model(observations, actions, input_mean, input_std)
    raise ValueError(f"未知组别：{mode}")


def main():
    # 导师提供的工程准备；不读取test集合做评估，不更新验证数据上的参数。
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads(
        (root / "results/day03_pendulum_dataset.json").read_text(encoding="utf-8")
    )
    train_inputs, train_targets = episodes_to_tensors(dataset["splits"]["train"])
    val_inputs, val_targets = episodes_to_tensors(dataset["splits"]["val"])
    input_mean = fit_input_mean(train_inputs)
    input_std = fit_input_std(train_inputs)
    observations = train_inputs[:, :3]
    actions = train_inputs[:, 3:4]

    torch.set_num_threads(1)
    torch.manual_seed(0)
    baseline = DynamicsMLP()
    torch.manual_seed(0)
    structured = PhysicsStructuredMLP()
    # 同架构隐藏层共享相同初值；输出头形状不同，不能逐参数全相同。
    for layer_index in (0, 2):
        for name, parameter in baseline.network[layer_index].named_parameters():
            assert torch.equal(
                parameter,
                dict(structured.network[layer_index].named_parameters())[name],
            )
    models = {"baseline": baseline, "structured": structured}
    initial = {mode: copy.deepcopy(model.state_dict()) for mode, model in models.items()}
    groups = {}
    # 学习者填写的新配置；固定预算，不分别用验证集早停。
    epochs = 1950
    learning_rate = 0.001
    for mode, model in models.items():
        started = time.perf_counter()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print(f"{mode}: parameters={sum(p.numel() for p in model.parameters())}")
        model.train()
        for epoch in range(epochs):
            # 后续训练任务：清梯度；调用predict_for_comparison得到predictions；
            # 和train_targets计算最终观测MSE；backward；step。
            # 两组使用相同固定更新次数；不把更新前loss误称最终指标。
            optimizer.zero_grad()
            predictions = predict_for_comparison(
                model, mode, observations, actions, input_mean, input_std
            )
            loss = torch.mean((predictions - train_targets) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{mode}: non-finite loss at {epoch + 1}")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = predict_for_comparison(
                model, mode, observations, actions, input_mean, input_std
            )
            mse = torch.mean((predictions - train_targets) ** 2).item()
            print(f"{mode}: final training MSE={mse:.6f}")
            predictions_val = predict_for_comparison(
                model,
                mode,
                val_inputs[:, :3],
                val_inputs[:, 3:4],
                input_mean,
                input_std,
            )
            mse_val = torch.mean((predictions_val - val_targets) ** 2).item()
            print(f"{mode}: final validation MSE={mse_val:.6f}")
            component_mse = ((predictions_val - val_targets) ** 2).mean(dim=0)
            print(f"{mode}: component MSE={component_mse.tolist()}")
            circle_errors = torch.abs(
                predictions_val[:, 0] ** 2 + predictions_val[:, 1] ** 2 - 1
            )
            print(f"{mode}: mean circle error={circle_errors.mean().item():.6f}")
            assert all(math.isfinite(v) for v in (mse, mse_val))
            assert predictions_val.shape == val_targets.shape
            assert torch.isfinite(predictions_val).all()
            final_state = copy.deepcopy(model.state_dict())
            assert all(torch.isfinite(p).all() for p in final_state.values())
            assert any(not torch.equal(initial[mode][k], v) for k, v in final_state.items())
            groups[mode] = dict(
                train_mse=mse, validation_mse=mse_val,
                component_mse=component_mse.tolist(),
                mean_circle_error=circle_errors.mean().item(),
                max_circle_error=circle_errors.max().item(),
                parameters=sum(p.numel() for p in model.parameters()),
                elapsed_seconds=time.perf_counter() - started,
                initial_state_dict=initial[mode], final_state_dict=final_state,
            )
        # 后续评估任务：eval + no_grad；重新计算最终训练/验证下一观测MSE，
        # 及验证各分量MSE、单位圆误差；两组使用相同统计量和数据。
        # 单训练种子、小预算，不以必胜为验收标准。多步/控制不在本次必做比较内。
    return dict(groups=groups, input_mean=input_mean, input_std=input_std,
                config=dict(seed=0, optimizer="Adam", learning_rate=learning_rate,
                            updates=epochs, device="cpu", threads=1,
                            train_count=len(train_inputs), val_count=len(val_inputs)),
                test_evaluated=False)


def archive_run():
    """导师工程辅助：新目录、源文件指纹、日志和旧/新比较；不改训练算法。"""
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root / "src/day04_world_model.py",
               root / "src/day08_physics_model.py",
               root / "results/day03_pendulum_dataset.json",
               root / "results/day04_delta_normalized_run.json",
               root / "results/day08_model_comparison_20261017T125516_537274/comparison.json",
               root / "results/day10_convergence_20261017T221844_116308/metrics.json"]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p.relative_to(root)): digest(p) for p in sources}
    prior = json.loads(sources[-1].read_text(encoding="utf-8"))
    assert prior["meaningful_improvement"] and prior["best_step"] == 1950
    old = json.loads(sources[-2].read_text(encoding="utf-8"))
    output = root / "results" / ("day10_day08_adam_" + datetime.now(
        timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    for p in sources[:3]:
        (output / p.name).write_bytes(p.read_bytes())
    capture = io.StringIO()
    def write_json(name, value):
        with (output / name).open("x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False,
                      default=lambda x: x.detach().cpu().tolist() if isinstance(x, torch.Tensor) else str(x))
    try:
        with redirect_stdout(capture):
            result = main()
        assert all(digest(p) == before[str(p.relative_to(root))] for p in sources)
        result.update(source_sha256=before, historical_files_unchanged=True,
                      versions=dict(python=platform.python_version(), torch=torch.__version__),
                      old_config=dict(optimizer="SGD", learning_rate=0.01, updates=200),
                      old_groups={mode: {k: v for k, v in group.items()
                                        if k in ("train_mse", "validation_mse", "mean_circle_error", "parameters")}
                                  for mode, group in old["groups"].items()})
        write_json("comparison.json", result)
        rows = ["# Day8旧配置/新配置比较", "",
                "同数据、seed0、全train统计量；新模型从初值训练，固定Adam lr0.001、1950步。",
                "", "|配置|模型|训练MSE|验证MSE|单位圆误差|", "|---|---|---:|---:|---:|"]
        for label, groups in (("旧SGD200", result["old_groups"]), ("新Adam1950", result["groups"])):
            for mode, g in groups.items():
                rows.append(f"|{label}|{mode}|{g['train_mse']:.10g}|{g['validation_mse']:.10g}|{g['mean_circle_error']:.10g}|")
        rows += ["", "旧结果保留；未用test，未评估多步或控制。单seed，参数量不同，不能由此证明普遍优势。",
                 "K由P0-1验证集选择，再用同一验证集比较，属于开发结果。统计量、初末权重、版本、耗时与来源指纹见comparison.json。"]
        (output / "SUMMARY.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    except (Exception, KeyboardInterrupt) as error:
        import traceback
        write_json("failed.json", dict(error=repr(error), source_sha256=before, test_evaluated=False))
        capture.write(traceback.format_exc())
        raise
    finally:
        (output / "stdout.txt").write_text(capture.getvalue(), encoding="utf-8")
        print(capture.getvalue(), end="")
        print(f"归档：{output}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Day8新配置复核：训练前需单独取得用户同意。"
    )
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()
    if args.run_training:
        archive_run()
    else:
        print("NOT RUN：Adam lr0.001，seed0，两模型各1950次全批量更新；未训练、未创建结果目录。")
