"""Day15 三方法对比评估（导师辅助）：固定MLP vs 贝叶斯oracle vs PFN，不训练、不用test。

评估任务由固定种子从先验采样，与训练采样使用不同的 generator；K 为嵌套前缀。
"""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.day04_world_model import DynamicsMLP, normalize_inputs
from src.day15_bayes_oracle import posterior_predictive
from src.day15_pfn_model import ContextPFN
from src.day15_pfn_prior import pendulum_delta_omega, sample_tasks

ROOT = Path(__file__).resolve().parents[1]
FIXED = ROOT / "results/day10_convergence_20261017T221844_116308"
CONFIG = {
    "eval_seed": 20261028,
    "num_tasks": 1000,
    "max_context": 32,
    "num_query": 20,
    "context_sizes": [0, 1, 2, 4, 8, 16, 32],
    "noise_std": 0.02,
    "dt": 0.05,
    "coverage_level": 0.9,
}
Z90 = 1.6448536269514722


def load_fixed_mlp():
    cfg = json.loads((FIXED / "config.json").read_text(encoding="utf-8"))
    weights = json.loads((FIXED / "best_weights.json").read_text(encoding="utf-8"))
    model = DynamicsMLP()
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()})
    model.eval()
    return model, torch.tensor(cfg["input_mean"]), torch.tensor(cfg["input_std"])


def load_pfn(weights_dir: Path):
    cfg = json.loads((weights_dir / "metrics.json").read_text(encoding="utf-8"))["config"]
    model = ContextPFN(**cfg["model"])
    weights = json.loads((weights_dir / "weights.json").read_text(encoding="utf-8"))
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()})
    model.eval()
    return model


@torch.no_grad()
def mlp_delta_omega(model, mean, std, obs, torque):
    inputs = torch.cat([obs, torque], dim=-1)
    shape = inputs.shape[:-1]
    out = model(normalize_inputs(inputs.reshape(-1, 4), mean, std))
    return out[:, 2:3].reshape(*shape, 1)


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
def main(weights_dir: str):
    run = Path(weights_dir)
    if not run.is_absolute():
        run = ROOT / run
    out_dir = ROOT / f"results/day15_eval_{datetime.now():%Y%m%dT%H%M%S_%f}"
    out_dir.mkdir(parents=True)

    mlp, mean, std = load_fixed_mlp()
    pfn = load_pfn(run)

    c = CONFIG
    task = sample_tasks(c["num_tasks"], c["max_context"], c["num_query"],
                        torch.Generator().manual_seed(c["eval_seed"]),
                        noise_std=c["noise_std"], dt=c["dt"])
    x_q, y_q, obs_q = task["x_qry"], task["y_qry"], task["obs_qry"]

    mlp_mean = mlp_delta_omega(mlp, mean, std, obs_q, x_q[..., 1:])
    mlp_var = torch.full_like(mlp_mean, c["noise_std"] ** 2)
    mlp_metrics = gaussian_metrics(mlp_mean, mlp_var, y_q)

    rows = []
    for k in c["context_sizes"]:
        xc, yc = task["x_ctx"][:, :k], task["y_ctx"][:, :k]
        mb, vb = posterior_predictive(xc, yc, x_q, noise_std=c["noise_std"], dt=c["dt"])
        mp, lvp = pfn(xc, yc, x_q)
        rows.append({
            "K": k,
            "bayes": gaussian_metrics(mb, vb, y_q),
            "pfn": gaussian_metrics(mp, lvp.exp(), y_q),
            "fixed_mlp": mlp_metrics,
        })

    ks = [r["K"] for r in rows]
    xs = [max(k, 0.5) for k in ks]
    labels = [str(k) for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for name, color in (("bayes", "tab:blue"), ("pfn", "tab:green")):
        axes[0].errorbar(
            xs, [r[name]["mse"] for r in rows],
            yerr=[r[name]["mse_task_se"] for r in rows],
            marker="o", label=name.upper() if name == "pfn" else "Bayes posterior",
            color=color,
        )
    axes[0].axhline(mlp_metrics["mse"], color="tab:red", ls="--", label="Fixed single-task MLP")
    axes[0].axhline(c["noise_std"] ** 2, color="gray", ls=":", label="Noise floor σ²")
    axes[0].set(xscale="symlog", yscale="log",
                xlabel="Context size K (K=0 plotted at 0.5)", ylabel="Query MSE",
                title="Prediction error vs context size")
    axes[0].set_xticks(xs, labels)
    axes[0].legend(fontsize=8)
    axes[1].plot(xs, [r["bayes"]["coverage90"] for r in rows], marker="o",
                 color="tab:blue", label="Bayes posterior")
    axes[1].plot(xs, [r["pfn"]["coverage90"] for r in rows], marker="s",
                 color="tab:green", label="PFN")
    axes[1].axhline(mlp_metrics["coverage90"], color="tab:red", ls="--", label="Fixed MLP (+σ²)")
    axes[1].axhline(0.9, color="gray", ls=":", label="Nominal 90%")
    axes[1].set(xscale="symlog", ylim=(0, 1.02), xlabel="Context size K",
                ylabel="Empirical coverage", title="90% predictive interval coverage")
    axes[1].set_xticks(xs, labels)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pfn_vs_context.png", dpi=150)

    sources = ["scripts/day15_eval_pfn.py", "src/day15_pfn_prior.py",
               "src/day15_bayes_oracle.py", "src/day15_pfn_model.py",
               "src/day04_world_model.py"]
    for s in sources:
        shutil.copy2(ROOT / s, out_dir / Path(s).name)
    hashes = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}
    for p in (FIXED / "best_weights.json", FIXED / "config.json",
              run / "weights.json", run / "metrics.json"):
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    result = {"config": c, "pfn_checkpoint": str(run.relative_to(ROOT)), "rows": rows,
              "source_sha256": hashes, "trained": False, "test_evaluated": False}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir.relative_to(ROOT)),
                      "mlp": mlp_metrics}, indent=1))
    for r in rows:
        print(f"K={r['K']:>2}  pfn mse={r['pfn']['mse']:.3e} nll={r['pfn']['nll']:.3f}"
              f" cov={r['pfn']['coverage90']:.3f} | bayes mse={r['bayes']['mse']:.3e}"
              f" cov={r['bayes']['coverage90']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python -m scripts.day15_eval_pfn results/day15_pfn_<时间戳>")
        sys.exit(1)
    main(sys.argv[1])
