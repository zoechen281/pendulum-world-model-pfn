"""P0-1 收敛检查；核心循环由学习者填写，工程辅助由导师提供。

默认仅核对数据/初始化，不训练、不创建结果目录。
明确使用--run-training才执行已获同意的训练。旧SGD结果只读复用。
"""

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch

from src.day04_world_model import (
    DynamicsMLP, episodes_to_tensors, fit_input_mean, fit_input_std,
    make_delta_targets, normalize_inputs, predict_next_observation,
)


@dataclass(frozen=True)
class ConvergenceConfig:
    seed: int = 0
    learning_rate: float = 1e-3
    max_updates: int = 5000
    min_updates: int = 1000
    eval_every: int = 50
    patience_checks: int = 10
    min_delta: float = 1e-5
    meaningful_relative_reduction: float = 0.20
    meaningful_absolute_reduction: float = 0.001


OLD_SGD_RECORD = (
    Path(__file__).resolve().parents[1] / "results/day04_delta_normalized_run.json"
)
OUTPUT_PREFIX = "results/day10_convergence_"


def train_adam_with_early_stopping(
    model: torch.nn.Module,
    train_inputs: torch.Tensor,
    train_next_observations: torch.Tensor,
    val_inputs: torch.Tensor,
    val_next_observations: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    config: ConvergenceConfig,
) -> dict:
    """由学习者实现；输入原尺度(N,4)/(N,3)，统计量(1,4)，CPU float32。

    返回history、best_step、stop_step、best_state_dict、final_state_dict、
    best_train_mse、best_val_mse、stop_reason；调用方记录timing。
    Adam从seed0初始权重开始，不从旧SGD最终权重续训。
    """
    # 以下训练/评估/快照/早停核心由学习者完成；有限值检查由导师补充。
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    normalized_train_inputs = normalize_inputs(train_inputs, input_mean, input_std)
    delta_targets = make_delta_targets(train_inputs, train_next_observations)
    history = []
    best_val_mse = float("inf")
    best_train_mse = None
    best_step = 0
    best_state_dict = None
    patience_reference = float("inf")
    no_improvement_checks = 0
    stop_reason = "max_updates"
    for step in range(1, config.max_updates + 1):
        # 切换训练模式、清梯度
        model.train()
        optimizer.zero_grad()
        # 用 normalized_train_inputs 预测 predicted_deltas
        predicted_deltas = model(normalized_train_inputs)
        # 与 delta_targets 计算 MSE，保存为 loss
        loss = torch.mean((predicted_deltas - delta_targets) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}")
        # 反向传播、更新参数
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Invalid parameter gradient at step {step}")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError(f"Non-finite model parameter at step {step}")
        if step % config.eval_every == 0:
            model.eval()
            with torch.no_grad():
                normalized_val_inputs = normalize_inputs(
                    val_inputs, input_mean, input_std
                )
                predicted_val_deltas = model(normalized_val_inputs)
                val_predictions = val_inputs[:, :3] + predicted_val_deltas
                val_mse = torch.mean(
                    (val_predictions - val_next_observations) ** 2
                ).item()
                evaluated_train_deltas = model(normalized_train_inputs)
                train_predictions = train_inputs[:, :3] + evaluated_train_deltas
                train_mse = torch.mean(
                    (train_predictions - train_next_observations) ** 2
                ).item()
            if not math.isfinite(train_mse) or not math.isfinite(val_mse):
                raise FloatingPointError(f"Non-finite evaluation at step {step}")
            history.append(
                {
                    "step": step,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                }
            )
            print(f"step={step}: train MSE={train_mse:.10f}, val MSE={val_mse:.10f}")
            if val_mse < best_val_mse:
                # 更新 best_val_mse、best_train_mse、best_step
                best_val_mse = val_mse
                best_train_mse = train_mse
                best_step = step
                # 保存 best_state_dict
                best_state_dict = copy.deepcopy(model.state_dict())
            if val_mse < patience_reference - config.min_delta:
                # 更新参考误差，并把连续无改善次数清零
                patience_reference = val_mse
                no_improvement_checks = 0
            else:
                # 连续无改善次数加 1
                no_improvement_checks += 1
            if (
                no_improvement_checks >= config.patience_checks
                and step >= config.min_updates
            ):
                stop_reason = "early_stopping"
                break
    stop_step = step
    final_state_dict = copy.deepcopy(model.state_dict())
    return {
        "history": history,
        "best_step": best_step,
        "stop_step": stop_step,
        "best_state_dict": best_state_dict,
        "final_state_dict": final_state_dict,
        "best_train_mse": best_train_mse,
        "best_val_mse": best_val_mse,
        "stop_reason": stop_reason,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_hash(state: dict) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(state.items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    def tensor_to_list(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        raise TypeError(type(item).__name__)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=tensor_to_list, allow_nan=False)


def prepare_experiment(config: ConvergenceConfig) -> dict:
    """导师辅助：只读数据、核对统计量和初值；绝不调用优化器或训练函数。"""
    assert config.max_updates >= config.min_updates >= 1
    assert config.eval_every >= 1 and config.max_updates % config.eval_every == 0
    assert config.patience_checks >= 1 and config.min_delta >= 0
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "results/day03_pendulum_dataset.json"
    old = json.loads(OLD_SGD_RECORD.read_text(encoding="utf-8"))
    assert old["seed"] == config.seed == 0
    assert old["optimizer"] == "SGD" and old["learning_rate"] == 0.01 and old["updates"] == 200
    assert _sha256(dataset_path) == old["dataset_sha256"]
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    train_inputs, train_next = episodes_to_tensors(dataset["splits"]["train"])
    val_inputs, val_next = episodes_to_tensors(dataset["splits"]["val"])
    assert train_inputs.shape == (1200, 4) and train_next.shape == (1200, 3)
    assert val_inputs.shape == (400, 4) and val_next.shape == (400, 3)
    mean, std = fit_input_mean(train_inputs), fit_input_std(train_inputs)
    assert torch.equal(mean, torch.tensor(old["input_mean"], dtype=torch.float32))
    assert torch.equal(std, torch.tensor(old["input_std"], dtype=torch.float32))
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = DynamicsMLP()
    # Day4未保存初值，不声称可直接核对其初值快照。
    # 用独立保存的Day9同架构seed0初值作字节级交叉核对，Day4依据其元数据/源配置。
    reference_path = root / "results/day09_data_amount_20261017T130753_993032/comparison.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    assert reference["seed"] == config.seed
    assert all(torch.equal(value, torch.tensor(reference["initial_state_dict"][name], dtype=value.dtype))
               for name, value in model.state_dict().items())
    sources = [Path(__file__).resolve(), root / "src/day04_world_model.py", dataset_path,
               OLD_SGD_RECORD, reference_path]
    hashes = {str(p.relative_to(root)): _sha256(p) for p in sources}
    return dict(root=root, model=model, train_inputs=train_inputs, train_next=train_next,
                val_inputs=val_inputs, val_next=val_next, input_mean=mean, input_std=std,
                old=old, sources=sources, hashes=hashes,
                initial_hash=_state_hash(model.state_dict()))


@torch.no_grad()
def evaluate_checkpoint(model, state, prepared):
    """导师辅助：在模型副本上核对best/final指标，不改变训练后模型。"""
    probe = copy.deepcopy(model)
    probe.load_state_dict(state)
    probe.eval()
    metrics = {}
    for split in ("train", "val"):
        inputs = prepared[f"{split}_inputs"]
        target = prepared[f"{split}_next"]
        predicted = predict_next_observation(probe, inputs[:, :3], inputs[:, 3:4],
                                             prepared["input_mean"], prepared["input_std"])
        assert predicted.shape == target.shape and not predicted.requires_grad
        assert torch.isfinite(predicted).all()
        metrics[f"{split}_mse"] = ((predicted - target) ** 2).mean().item()
    return metrics


def main(run_training: bool = False):
    """默认只读准备；显式执行开关须在用户同意本次训练后使用。"""
    config = ConvergenceConfig()
    prepared = prepare_experiment(config)
    print("准备检查通过：train=1200 / val=400；统计量匹配Day4；seed0初值匹配Day9存档。")
    print("初值核对限制：Day4没有保存initial_state_dict，不能直接验证其历史初值字节。")
    print(f"旧SGD200：train={prepared['old']['train_mse']:.10f}, val={prepared['old']['val_mse']:.10f}")
    print(f"拟运行：Adam lr={config.learning_rate}, max={config.max_updates}, min={config.min_updates}, "
          f"eval_every={config.eval_every}, patience={config.patience_checks}; CPU/1 thread")
    if not run_training:
        print("NOT RUN：未调用训练函数，未创建结果目录。训练前必须取得用户同意。")
        return None

    # 以下仅在用户明确同意后执行；不自动启动Day8/Day9条件重跑。
    from contextlib import redirect_stdout
    import io
    model = prepared["model"]
    initial_state = copy.deepcopy(model.state_dict())
    timestamp = datetime.now(timezone(timedelta(hours=8)))
    output = prepared["root"] / (OUTPUT_PREFIX + timestamp.strftime("%Y%m%dT%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    for path in prepared["sources"][:2]:
        (output / path.name).write_bytes(path.read_bytes())
    capture = io.StringIO()
    started = time.perf_counter()
    try:
        with redirect_stdout(capture):
            result = train_adam_with_early_stopping(
                model, prepared["train_inputs"], prepared["train_next"], prepared["val_inputs"],
                prepared["val_next"], prepared["input_mean"], prepared["input_std"], config,
            )
    except (Exception, KeyboardInterrupt) as error:
        import traceback
        _write_json(output / "failed.json", dict(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__, error=str(error), config=asdict(config),
            elapsed_seconds=time.perf_counter() - started, source_sha256=prepared["hashes"],
            initial_state_sha256=prepared["initial_hash"], test_evaluated=False,
        ))
        _write_json(output / "initial_weights.json", initial_state)
        (output / "stdout.txt").write_text(capture.getvalue() + traceback.format_exc(), encoding="utf-8")
        print(f"失败/中断记录已保存：{output}")
        raise
    seconds = time.perf_counter() - started
    assert result["best_state_dict"] is not None and result["history"]
    best = evaluate_checkpoint(model, result["best_state_dict"], prepared)
    final = evaluate_checkpoint(model, result["final_state_dict"], prepared)
    assert math.isclose(best["val_mse"], result["best_val_mse"], abs_tol=1e-8, rel_tol=1e-6)
    assert math.isclose(best["train_mse"], result["best_train_mse"], abs_tol=1e-8, rel_tol=1e-6)
    assert _state_hash(initial_state) == prepared["initial_hash"]
    assert _state_hash(result["final_state_dict"]) != prepared["initial_hash"]
    assert all(_sha256(p) == prepared["hashes"][str(p.relative_to(prepared["root"]))]
               for p in prepared["sources"])
    reduction = prepared["old"]["val_mse"] - best["val_mse"]
    relative_reduction = reduction / prepared["old"]["val_mse"]
    meaningful = (reduction >= config.meaningful_absolute_reduction
                  and relative_reduction >= config.meaningful_relative_reduction)
    _write_json(output / "config.json", dict(
        adam=asdict(config), optimizer="Adam", device="cpu", threads=1,
        data_counts=dict(train=1200, val=400), test_evaluated=False,
        input_mean=prepared["input_mean"], input_std=prepared["input_std"],
        source_sha256=prepared["hashes"], initial_state_sha256=prepared["initial_hash"],
        initialization_evidence="seed0 architecture/config; exact match to saved Day9 initial state; Day4 initial state unavailable",
        versions=dict(python=platform.python_version(), torch=torch.__version__),
    ))
    _write_json(output / "initial_weights.json", initial_state)
    _write_json(output / "best_weights.json", result["best_state_dict"])
    _write_json(output / "final_weights.json", result["final_state_dict"])
    metrics = {k: v for k, v in result.items() if k not in ("best_state_dict", "final_state_dict")}
    metrics.update(best=best, final=final, elapsed_seconds=seconds,
                   old_sgd=dict(path=str(OLD_SGD_RECORD.relative_to(prepared["root"])),
                                train_mse=prepared["old"]["train_mse"], val_mse=prepared["old"]["val_mse"],
                                updates=200, retrained=False),
                   absolute_val_reduction=reduction, relative_val_reduction=relative_reduction,
                   meaningful_improvement=meaningful, historical_files_unchanged=True,
                   test_evaluated=False, conditional_reruns_performed=False,
                   evidence_scope="Optimizer, rate and budget change together; single-seed development validation, not proof of global convergence.")
    _write_json(output / "metrics.json", metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    for key in ("train_mse", "val_mse"):
        axis.plot([r["step"] for r in result["history"]], [r[key] for r in result["history"]], label=key)
    axis.axhline(prepared["old"]["val_mse"], linestyle="--", label="old SGD200 validation")
    axis.set(xlabel="Parameter updates", ylabel="Next-observation MSE", title="Seed0 Adam convergence check")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png")
    plt.close(figure)
    summary = (f"best step={result['best_step']}, stop step={result['stop_step']}, "
               f"reason={result['stop_reason']}, best val={best['val_mse']:.10f}, "
               f"meaningful improvement={meaningful}\nARCHIVE: {output}\n")
    (output / "stdout.txt").write_text(capture.getvalue() + summary, encoding="utf-8")
    print(capture.getvalue(), end="")
    print(summary, end="")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0-1: default read-only preparation; training needs prior consent")
    parser.add_argument("--run-training", action="store_true", help="Only after explicit user consent for this Adam run")
    args = parser.parse_args()
    main(run_training=args.run_training)
