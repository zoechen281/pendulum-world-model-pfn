"""固定协议比较掩码PFN、旧联合查询PFN、贝叶斯和固定MLP；不训练、不用test。"""

from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from scripts.day15_eval_pfn import (
    CONFIG, FIXED, gaussian_metrics, load_fixed_mlp, load_pfn, mlp_delta_omega,
)
from src.day15_bayes_oracle import posterior_predictive
from src.day15_pfn_model_masked import MaskedContextPFN
from src.day15_pfn_prior import sample_tasks

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/day15_pfn_20261028T162639_073060"
OLD_EVAL = ROOT / "results/day15_eval_20261028T162911_018421/metrics.json"


@torch.no_grad()
def main(weights_dir):
    torch.set_num_threads(1)
    run = Path(weights_dir)
    run = run if run.is_absolute() else ROOT / run
    cfg = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["config"]
    masked = MaskedContextPFN(**cfg["model"])
    weights = json.loads((run / "weights.json").read_text(encoding="utf-8"))
    masked.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()})
    masked.eval()
    old = load_pfn(OLD)
    mlp, mean, std = load_fixed_mlp()
    c = dict(CONFIG)
    task = sample_tasks(c["num_tasks"], c["max_context"], c["num_query"],
                        torch.Generator().manual_seed(c["eval_seed"]),
                        noise_std=c["noise_std"], dt=c["dt"])
    xq, yq = task["x_qry"], task["y_qry"]
    mm = mlp_delta_omega(mlp, mean, std, task["obs_qry"], xq[..., 1:])
    mlp_metrics = gaussian_metrics(mm, torch.full_like(mm, c["noise_std"] ** 2), yq)
    rows = []
    for k in c["context_sizes"]:
        xc, yc = task["x_ctx"][:, :k], task["y_ctx"][:, :k]
        mb, vb = posterior_predictive(xc, yc, xq, noise_std=c["noise_std"], dt=c["dt"])
        mo, lo = old(xc, yc, xq)
        mn, ln = masked(xc, yc, xq)
        rows.append({"K": k, "bayes": gaussian_metrics(mb, vb, yq),
                     "pfn_unmasked": gaussian_metrics(mo, lo.exp(), yq),
                     "pfn_masked": gaussian_metrics(mn, ln.exp(), yq),
                     "fixed_mlp": mlp_metrics})

    previous = json.loads(OLD_EVAL.read_text(encoding="utf-8"))
    if previous["config"] != c:
        raise ValueError("Evaluation config differs from archived protocol")
    max_old_difference = max(abs(row[new][metric] - oldrow[oldkey][metric])
                             for row, oldrow in zip(rows, previous["rows"])
                             for new, oldkey in [("bayes", "bayes"), ("fixed_mlp", "fixed_mlp"),
                                                 ("pfn_unmasked", "pfn")]
                             for metric in ["mse", "nll", "coverage90"])
    if max_old_difference > 1e-5:
        raise ValueError(f"Old metrics changed beyond numerical tolerance: {max_old_difference}")

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S_%f")
    out = ROOT / f"results/day15_eval_masked_{stamp}"
    out.mkdir()
    xs = [max(row["K"], 0.5) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for key, label, color in [("bayes", "Bayes posterior", "tab:blue"),
                              ("pfn_unmasked", "PFN joint-query (old)", "tab:green"),
                              ("pfn_masked", "PFN masked", "tab:purple"),
                              ("fixed_mlp", "Fixed single-task MLP", "tab:red")]:
        axes[0].errorbar(xs, [r[key]["mse"] for r in rows],
                         yerr=[r[key]["mse_task_se"] for r in rows],
                         marker="o", color=color, label=label)
        axes[1].plot(xs, [r[key]["coverage90"] for r in rows],
                     marker="o", color=color, label=label)
    axes[0].axhline(c["noise_std"] ** 2, color="gray", ls=":", label="Noise variance")
    axes[1].axhline(0.9, color="gray", ls=":", label="Nominal 90%")
    axes[0].set(yscale="log", ylabel="Query MSE", title="Prediction error vs context size")
    axes[1].set(ylim=(0, 1.02), ylabel="Empirical coverage", title="90% predictive interval coverage")
    for ax in axes:
        ax.set_xscale("symlog")
        ax.set_xlabel("Context size K (K=0 plotted at 0.5)")
        ax.set_xticks(xs, [str(r["K"]) for r in rows])
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "pfn_vs_context.png", dpi=150)
    plt.close(fig)
    sources = ["scripts/day15_eval_pfn_masked.py", "scripts/day15_eval_pfn.py",
               "src/day15_pfn_model_masked.py", "src/day15_pfn_model.py",
               "src/day15_pfn_prior.py", "src/day15_bayes_oracle.py", "src/day04_world_model.py"]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {s: digest(ROOT / s) for s in sources}
    for s in sources:
        shutil.copy2(ROOT / s, out / Path(s).name)
    for p in [run / "weights.json", run / "metrics.json", OLD / "weights.json",
              OLD / "metrics.json", OLD_EVAL, FIXED / "best_weights.json", FIXED / "config.json"]:
        hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    result = {"config": c, "masked_checkpoint": run.relative_to(ROOT).as_posix(),
              "unmasked_checkpoint": OLD.relative_to(ROOT).as_posix(), "rows": rows,
              "task_bank_sha256": {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()
                                   for k, v in task.items()},
              "max_archived_metric_difference": max_old_difference,
              "source_sha256": hashes, "trained": False, "test_evaluated": False}
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": out.relative_to(ROOT).as_posix(),
                      "max_archived_metric_difference": max_old_difference}, indent=2))
    for r in rows:
        print(f"K={r['K']:2}: masked MSE={r['pfn_masked']['mse']:.8g}, "
              f"NLL={r['pfn_masked']['nll']:.6f}, cov90={r['pfn_masked']['coverage90']:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.day15_eval_pfn_masked results/day15_pfn_masked_<timestamp>")
    main(sys.argv[1])
