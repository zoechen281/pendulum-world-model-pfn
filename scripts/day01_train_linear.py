"""Day 1：线性回归基线，先由学习者准备数据、模型和优化器。"""

import torch

from src.day01_regression import (
    build_linear_model,
    make_inputs,
    make_targets,
    mse_loss,
)

# 固定模型初始化使用的随机数，便于重复比较同一个实验。
torch.manual_seed(0)

# 当前只完成准备阶段：
# 1. 用 make_inputs 创建 [-2.0, -1.0, 0.0, 1.0, 2.0]，保存为 inputs。
# 2. 用 make_targets(inputs) 生成 targets。
# 3. 用 build_linear_model() 创建 model。
# 4. 创建 SGD 优化器 optimizer，传入 model.parameters()，学习率为 0.05。
# 参数与优化器只创建一次；训练循环在本阶段验收后再编写。
# 完成准备阶段后删除占位异常。

inputs = make_inputs([-2.0, -1.0, 0.0, 1.0, 2.0])
targets = make_targets(inputs)
model = build_linear_model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 训练循环由你补充，重复 200 次：
# 1. 清理旧梯度。
# 2. predictions = model(inputs)，根据当前参数重新预测。
# 3. 调用你写的 mse_loss(predictions, targets)，得到本轮 loss。
# 4. 求导，再更新参数。
# 5. 打印轮次 i 和本轮更新前计算的 loss.item()。
# 不在循环内重建模型或优化器，不用目标公式直接代替模型预测。
optimizer.zero_grad()
for i in range(200):
    predictions = model(inputs)
    loss = mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Iteration {i}, Loss: {loss.item()}")
