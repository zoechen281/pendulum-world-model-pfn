"""Day15 PFN训练驱动（导师辅助）：先说明配置再运行；核心训练循环由学习者写。"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.day15_pfn_model import ContextPFN, train_pfn

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "seed": 0,
    "device": "cpu",
    "threads": 1,
    "model": {"d_model": 64, "n_heads": 4, "n_layers": 2},
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "num_steps": 3000,
    "batch_size": 64,
    "num_query": 16,
    "noise_std": 0.02,
    "train_context_sizes": [0, 1, 2, 4, 8, 16],
    "note": "K=32 留给评估，训练采样不含32",
}


def main():
    if "--run" not in sys.argv:
        print("训练NOT RUN。已说明配置；加 --run 执行。")
        return
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(CONFIG["threads"])
    out_dir = ROOT / f"results/day15_pfn_{datetime.now():%Y%m%dT%H%M%S_%f}"
    out_dir.mkdir(parents=True)

    model = ContextPFN(**CONFIG["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    generator = torch.Generator().manual_seed(CONFIG["seed"])

    start = time.time()
    losses = train_pfn(
        model, optimizer, CONFIG["num_steps"], CONFIG["batch_size"], generator,
        num_query=CONFIG["num_query"], noise_std=CONFIG["noise_std"],
    )
    elapsed = time.time() - start

    weights_path = out_dir / "weights.json"
    weights_path.write_text(
        json.dumps({k: v.tolist() for k, v in model.state_dict().items()}),
        encoding="utf-8",
    )
    steps = [s for s, _ in losses]
    values = [v for _, v in losses]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, values, lw=0.6)
    ax.set(xlabel="step", ylabel="Gaussian NLL", title="PFN training loss")
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=150)

    sources = ["scripts/day15_train_pfn.py", "src/day15_pfn_model.py",
               "src/day15_pfn_prior.py"]
    for s in sources:
        shutil.copy2(ROOT / s, out_dir / Path(s).name)
    result = {
        "config": CONFIG,
        "elapsed_seconds": elapsed,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "first_losses": values[:10],
        "last_losses": values[-10:],
        "final_loss": values[-1],
        "min_loss": min(values),
        "source_sha256": {
            s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources
        },
        "test_evaluated": False,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "losses.json").write_text(json.dumps(losses), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir.relative_to(ROOT)),
                      "elapsed_seconds": round(elapsed, 1),
                      "final_loss": values[-1]}, indent=1))


if __name__ == "__main__":
    main()
