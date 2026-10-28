"""Day 1：复用学习者已写的训练循环，仅替换为带 ReLU 的多层模型。"""

import torch

from src.day01_regression import (
    build_mlp_model,
    make_inputs,
    make_targets,
    mse_loss,
)

# 学习率对照：保持种子 0，输入 [-2,-1,0,1,2]，无噪声目标，
# SGD 学习率 0.01，200 次全批量更新。
torch.manual_seed(0)
inputs = make_inputs([-2.0, -1.0, 0.0, 1.0, 2.0])
targets = make_targets(inputs)
model = build_mlp_model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
for i in range(200):
    predictions = model(inputs)
    loss = mse_loss(predictions, targets)
    loss.backward()
    print(f"Iteration {i}, Loss: {loss.item()}")
    optimizer.step()
    optimizer.zero_grad()
# 请亲手复用 day01_train_linear.py 中自己写过的准备与训练代码。
# 用 build_mlp_model() 创建新 model；优化器仍接收新模型的 model.parameters()。
# 每轮打印 i 和更新前的 loss.item()，保留线性基线文件及其结果。
# 完成后删除占位异常。
