"""直接预测基线：加载固定数据，训练循环由学习者填写。"""

import json
from pathlib import Path

import torch

from src.day01_regression import mse_loss
from src.day04_world_model import (
    DynamicsMLP,
    episodes_to_tensors,
    make_delta_targets,
    fit_input_mean,
    fit_input_std,
    normalize_inputs,
)

root = Path(__file__).resolve().parents[1]
dataset = json.loads(
    (root / "results/day03_pendulum_dataset.json").read_text(encoding="utf-8")
)
train_inputs, train_targets = episodes_to_tensors(dataset["splits"]["train"])

# 小型 CPU 实验固定线程数和种子；不读取验证/测试目标用于参数更新。
torch.set_num_threads(1)
torch.manual_seed(0)
model = DynamicsMLP()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_history = []

model.train()
delta_targets = make_delta_targets(train_inputs, train_targets)
inputs_mean = fit_input_mean(train_inputs)
inputs_std = fit_input_std(train_inputs)
normalized_train_inputs = normalize_inputs(train_inputs, inputs_mean, inputs_std)
for epoch in range(200):
    # TODO：亲手填写清零、预测、训练 MSE、反向传播、参数更新五步。
    # 模型输入用 train_inputs，预测目标用 train_targets，损失变量命名 loss。
    optimizer.zero_grad()
    predictions = model(normalized_train_inputs)
    loss = mse_loss(predictions, delta_targets)
    loss.backward()
    optimizer.step()

    # 这里记录的是本次参数更新前的 loss，而非更新后的最终模型误差。
    loss_history.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, training MSE before update: {loss.item():.6f}")


model.eval()
val_inputs, val_targets = episodes_to_tensors(dataset["splits"]["val"])
normalized_val_inputs = normalize_inputs(val_inputs, inputs_mean, inputs_std)
with torch.no_grad():
    # 1. 用 model(val_inputs) 得到 predicted_deltas。
    predicted_deltas = model(normalized_val_inputs)
    # 2. 当前观测加上 predicted_deltas，得到 next_predictions。
    current_observations = val_inputs[:, :3]
    next_predictions = current_observations + predicted_deltas
    # 3. 计算 next_predictions 与 val_targets 的 MSE，保存为 val_mse。
    val_mse = mse_loss(next_predictions, val_targets)

    predicted_deltas = model(normalized_val_inputs)
    next_predictions = val_inputs[:, :3] + predicted_deltas
    val_mse = mse_loss(next_predictions, val_targets)

    predicted_cos = next_predictions[:, 0]
    predicted_sin = next_predictions[:, 1]
    circle_errors = torch.abs(predicted_cos**2 + predicted_sin**2 - 1)
    mean_circle_error = circle_errors.mean()

print(val_mse.item())
print("Validation MSE:", val_mse.item())
print("Mean circle error:", mean_circle_error.item())
