"""只读归档 JSON，输出公开文档所用数字；不加载模型、不训练、不评估。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "masked_training": "results/day15_pfn_masked_20261028T183404_691538/metrics.json",
    "masked_eval": "results/day15_eval_masked_20261028T183604_691086/metrics.json",
    "single_step": "results/day10_day08_adam_20261018T102005_973952/comparison.json",
    "horizons": "results/day10_day09_horizons_20261018T103223_004434/comparison.json",
    "control": "results/day10_oracle_20261018T110145_857569/comparison.json",
    "pfn_training": "results/day15_pfn_20261028T162639_073060/metrics.json",
    "pfn_eval": "results/day15_eval_20261028T162911_018421/metrics.json",
    "data_amount": "results/day10_day09_adam_20261018T103015_801539/comparison.json",
}


def read_metrics():
    data = {k: json.loads((ROOT / p).read_text(encoding="utf-8")) for k, p in PATHS.items()}
    groups = data["single_step"]["groups"]
    episodes = data["control"]["episodes"]
    control = []
    for h in (5, 15, 30):
        row = {"H": h}
        for mode in ("learned", "oracle"):
            eps = [e for e in episodes if e["horizon"] == h and e["mode"] == mode]
            row[mode] = {"successes": sum(e["success"] for e in eps), "episodes": len(eps)}
        control.append(row)
    random = [e for e in episodes if e["mode"] == "random"]
    return {
        "evidence_paths": PATHS,
        "single_step": {k: {f: v[f] for f in ("validation_mse", "parameters", "mean_circle_error")}
                        for k, v in groups.items()},
        "horizons": data["horizons"]["endpoint_mse"],
        "windows_per_episode": data["horizons"]["windows_per_episode"],
        "control": control,
        "random": {"successes": sum(e["success"] for e in random), "episodes": len(random)},
        "data_amount": {k: v["validation_mse"] for k, v in data["data_amount"]["groups"].items()},
        "pfn_training": {k: data["pfn_training"][k]
                         for k in ("config", "elapsed_seconds", "num_parameters", "final_loss")},
        "pfn_eval_config": data["pfn_eval"]["config"],
        "pfn_eval": data["pfn_eval"]["rows"],
        "masked_training": {k: data["masked_training"][k] for k in ("config", "elapsed_seconds", "num_parameters", "final_loss")},
        "masked_eval": data["masked_eval"]["rows"],
        "masked_eval_config": data["masked_eval"]["config"],
    }


if __name__ == "__main__":
    print(json.dumps(read_metrics(), ensure_ascii=False, indent=2))
