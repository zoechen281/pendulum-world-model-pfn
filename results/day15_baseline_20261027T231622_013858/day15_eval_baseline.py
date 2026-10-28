"""Day15 保底评估（导师辅助）：固定单任务 MLP vs 精确贝叶斯上下文推断，不训练、不用 test。

固定 MLP 预测分布的冻结规则（评估前确定，不在留出标签上调）：N(MLP 的 Δω, noise_std²)。
留出任务用固定种子从先验采样，K 取同一批上下文的嵌套前缀，查询点对所有 K 相同。
"""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.day04_world_model import DynamicsMLP, normalize_inputs
from src.day15_bayes_oracle import posterior_predictive
from src.day15_pfn_prior import pendulum_delta_omega, sample_tasks

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/day10_convergence_20261017T221844_116308"
DATASET = ROOT / "results/day03_pendulum_dataset.json"
CONFIG = {
    "eval_seed": 20261028,
    "num_tasks": 1000,
    "max_context": 32,
    "num_query": 20,
    "context_sizes": [0, 1, 2, 4, 8, 16, 32],
    "noise_std": 0.02,
    "dt": 0.05,
    "fixed_mlp_predictive": "N(mlp_delta_omega, noise_std^2)",
    "coverage_level": 0.9,
}
Z90 = 1.6448536269514722


def load_fixed_mlp():
    cfg = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    weights = json.loads((CHECKPOINT / "best_weights.json").read_text(encoding="utf-8"))
    model = DynamicsMLP()
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()})
    model.eval()
    return model, torch.tensor(cfg["input_mean"]), torch.tensor(cfg["input_std"])


@torch.no_grad()
def mlp_delta_omega(model, mean, std, obs, torque):
    inputs = torch.cat([obs, torque], dim=-1)
    shape = inputs.shape[:-1]
    out = model(normalize_inputs(inputs.reshape(-1, 4), mean, std))
    return out[:, 2:3].reshape(*shape, 1)


@torch.no_grad()
def original_val_check(model, mean, std):
    """加载自检：原 val 协议下的全观测 MSE 应与 P0-1 记录（约 3.1e-4）一致。"""
    data = json.loads(DATASET.read_text(encoding="utf-8"))["splits"]["val"]
    obs = torch.tensor([t["observation"] for ep in data for t in ep])
    act = torch.tensor([t["action"] for ep in data for t in ep])
    nxt = torch.tensor([t["next_observation"] for ep in data for t in ep])
    pred = model(normalize_inputs(torch.cat([obs, act], 1), mean, std))
    return {
        "val_full_obs_mse": ((pred - (nxt - obs)) ** 2).mean().item(),
        "val_delta_omega_mse": ((pred[:, 2] - (nxt - obs)[:, 2]) ** 2).mean().item(),
        "val_omega_abs_max": obs[:, 2].abs().max().item(),
    }


def gaussian_metrics(mean, var, y):
    per_point_sq = (mean - y) ** 2
    nll = 0.5 * torch.log(2 * math.pi * var) + per_point_sq / (2 * var)
    covered = ((mean - y).abs() <= Z90 * var.sqrt()).float()
    per_task_mse = per_point_sq.mean(dim=(1, 2))
    n = per_task_mse.numel()
    return {
        "mse": per_point_sq.mean().item(),
        "mse_task_se": (per_task_mse.std() / math.sqrt(n)).item(),
        "nll": nll.mean().item(),
        "coverage90": covered.mean().item(),
    }


@torch.no_grad()
def main():
    out_dir = ROOT / f"results/day15_baseline_{datetime.now():%Y%m%dT%H%M%S_%f}"
    out_dir.mkdir(parents=True)
    model, mean, std = load_fixed_mlp()
    load_check = original_val_check(model, mean, std)

    c = CONFIG
    task = sample_tasks(c["num_tasks"], c["max_context"], c["num_query"],
                        torch.Generator().manual_seed(c["eval_seed"]),
                        noise_std=c["noise_std"], dt=c["dt"])
    x_q, y_q, obs_q = task["x_qry"], task["y_qry"], task["obs_qry"]
    a = task["params"][:, 0].reshape(-1, 1, 1)
    b = task["params"][:, 1].reshape(-1, 1, 1)
    clean = pendulum_delta_omega(x_q[..., :1], x_q[..., 1:], a, b, c["dt"])

    mlp_mean = mlp_delta_omega(model, mean, std, obs_q, x_q[..., 1:])
    mlp_var = torch.full_like(mlp_mean, c["noise_std"] ** 2)
    mlp_metrics = gaussian_metrics(mlp_mean, mlp_var, y_q)
    default_clean = pendulum_delta_omega(x_q[..., :1], x_q[..., 1:], torch.tensor(15.0),
                                         torch.tensor(3.0), c["dt"])
    diagnostics = {
        "mlp_mse_vs_default_params_clean": ((mlp_mean - default_clean) ** 2).mean().item(),
        "query_clip_rate": ((obs_q[..., 2:3] + clean).abs() > 8.0).float().mean().item(),
        "fraction_tasks_nonpositive_param": (task["params"] <= 0).any(1).float().mean().item(),
        "noise_floor_mse": c["noise_std"] ** 2,
    }

    rows = []
    for k in c["context_sizes"]:
        m, v = posterior_predictive(task["x_ctx"][:, :k], task["y_ctx"][:, :k], x_q,
                                    noise_std=c["noise_std"], dt=c["dt"])
        rows.append({"K": k, "bayes": gaussian_metrics(m, v, y_q), "fixed_mlp": mlp_metrics})

    ks = [r["K"] for r in rows]
    xs = [max(k, 0.5) for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bayes_mse = [r["bayes"]["mse"] for r in rows]
    bayes_se = [r["bayes"]["mse_task_se"] for r in rows]
    axes[0].errorbar(xs, bayes_mse, yerr=bayes_se, marker="o", label="Bayes posterior (in-context)")
    axes[0].axhline(mlp_metrics["mse"], color="tab:red", ls="--", label="Fixed single-task MLP")
    axes[0].axhline(c["noise_std"] ** 2, color="gray", ls=":", label="Noise floor σ²")
    axes[0].set(xscale="symlog", yscale="log", xlabel="Context size K (K=0 plotted at 0.5)",
                ylabel="Query MSE", title="Prediction error vs context size")
    axes[0].set_xticks(xs, [str(k) for k in ks])
    axes[0].legend(fontsize=8)
    axes[1].plot(xs, [r["bayes"]["coverage90"] for r in rows], marker="o", label="Bayes posterior")
    axes[1].axhline(mlp_metrics["coverage90"], color="tab:red", ls="--", label="Fixed MLP (+σ²)")
    axes[1].axhline(0.9, color="gray", ls=":", label="Nominal 90%")
    axes[1].set(xscale="symlog", ylim=(0, 1.02), xlabel="Context size K",
                ylabel="Empirical coverage", title="90% predictive interval coverage")
    axes[1].set_xticks(xs, [str(k) for k in ks])
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_vs_context.png", dpi=150)

    sources = ["scripts/day15_eval_baseline.py", "src/day15_pfn_prior.py",
               "src/day15_bayes_oracle.py", "src/day04_world_model.py"]
    for s in sources:
        shutil.copy2(ROOT / s, out_dir / Path(s).name)
    hashes = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}
    for p in (CHECKPOINT / "best_weights.json", CHECKPOINT / "config.json", DATASET):
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    result = {"config": c, "load_check": load_check, "diagnostics": diagnostics, "rows": rows,
              "source_sha256": hashes, "trained": False, "test_evaluated": False}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir.relative_to(ROOT)), "load_check": load_check,
                      "diagnostics": diagnostics}, indent=1))
    for r in rows:
        print(f"K={r['K']:>2}  bayes mse={r['bayes']['mse']:.3e}±{r['bayes']['mse_task_se']:.1e}"
              f"  nll={r['bayes']['nll']:.3f}  cov90={r['bayes']['coverage90']:.3f}"
              f" | mlp mse={r['fixed_mlp']['mse']:.3e} nll={r['fixed_mlp']['nll']:.3f}"
              f" cov90={r['fixed_mlp']['coverage90']:.3f}")


if __name__ == "__main__":
    main()
