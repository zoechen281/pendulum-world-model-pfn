"""Day 1：恢复已有模型参数，由学习者完成四个新输入的评估。"""

import json
from pathlib import Path

import torch

from src.day01_regression import (
    build_mlp_model,
    evaluate_model,
    make_inputs,
    make_targets,
)

# 工程准备：读取学习率 0.01 实验中实际保存的最终参数，不重新训练。
project_dir = Path(__file__).resolve().parents[1]
record_path = project_dir / "results" / "day01_mlp_lr001_run.json"
record = json.loads(record_path.read_text(encoding="utf-8"))
model = build_mlp_model()
saved_parameters = {}
for name, values in record["final_state_dict"].items():
    saved_parameters[name] = torch.tensor(values, dtype=torch.float32)
model.load_state_dict(saved_parameters)

# state_dict 保存各层权重和偏置。上面创建模型结构，再加载已训练的数值。
# 当前练习由你完成：
# 1. 用 make_inputs 创建 [-1.5, -0.5, 0.5, 1.5]，保存为 val_inputs。
# 2. 用 make_targets(val_inputs) 生成 val_targets。
# 3. 调用 evaluate_model(model, val_inputs, val_targets)，保存为 val_mse。
# 4. 打印 val_mse。
# 这些点未参与训练，但都位于训练输入范围 [-2,2] 内；本次评估区间内插值。
# 不重新创建模型、优化器，不求导或训练；完成后删除占位异常。
val_inputs = make_inputs([-1.5, -0.5, 0.5, 1.5])
val_targets = make_targets(val_inputs)
val_mse = evaluate_model(model, val_inputs, val_targets)
print(f"Validation MSE: {val_mse}")
