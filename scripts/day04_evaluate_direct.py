"""恢复已保存的直接预测模型，只评估验证集，不重新训练。"""

import hashlib
import json
from pathlib import Path
from src.day01_regression import mse_loss

import torch

from src.day01_regression import evaluate_model
from src.day04_world_model import DynamicsMLP, episodes_to_tensors

root = Path(__file__).resolve().parents[1]
record = json.loads(
    (root / "results/day04_direct_run.json").read_text(encoding="utf-8")
)
dataset_path = root / "results/day03_pendulum_dataset.json"
assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == record["dataset_sha256"]
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
model = DynamicsMLP()
model.load_state_dict(
    {
        name: torch.tensor(values, dtype=torch.float32)
        for name, values in record["final_state_dict"].items()
    }
)

# TODO 1：将 dataset["splits"]["val"] 转换为 val_inputs、val_targets。
val_inputs, val_targets = episodes_to_tensors(dataset["splits"]["val"])
# TODO 2：调用 evaluate_model，保存结果为 val_mse。
val_mse = evaluate_model(model, val_inputs, val_targets)
# TODO 3：打印 val_mse。
print(val_mse)

baseline_predictions = val_inputs[:, :3]
baseline_mse = mse_loss(baseline_predictions, val_targets)
print(baseline_mse.item())
