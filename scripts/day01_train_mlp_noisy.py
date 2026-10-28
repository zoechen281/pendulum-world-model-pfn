"""Day 1：复用学习者已写的训练循环，仅替换为带 ReLU 的多层模型。"""

import torch

from src.day01_regression import (
    build_mlp_model,
    make_inputs,
    make_noisy_targets,
    make_targets,
    mse_loss,
    evaluate_model,
)

# 带噪声训练：保持种子 0，输入 [-2,-1,0,1,2]，目标加入标准差为 0.2 的高斯噪声。
# SGD 学习率 0.01，200 次全批量更新。
torch.manual_seed(0)
inputs = make_inputs([-2.0, -1.0, 0.0, 1.0, 2.0])
targets = make_noisy_targets(inputs, noise_std=0.2)
torch.manual_seed(0)
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

val_inputs = make_inputs([-1.5, -0.5, 0.5, 1.5])
val_targets = make_targets(val_inputs)
val_mse = evaluate_model(model, val_inputs, val_targets)
print(val_mse)


# 请亲手复用 day01_train_linear.py 中自己写过的准备与训练代码。
# 用 build_mlp_model() 创建新 model；优化器仍接收新模型的 model.parameters()。
# 每轮打印 i 和更新前的 loss.item()，保留线性基线文件及其结果。
# 完成后删除占位异常。
class RegressionMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义 self.fc1：1 → 16
        self.fc1 = torch.nn.Linear(1, 16)
        # 定义 self.relu
        self.relu = torch.nn.ReLU()
        # 定义 self.fc2：16 → 1
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. x 经过 self.fc1，结果保存为 hidden
        hidden = self.fc1(x)
        # 2. hidden 经过 self.relu，结果仍保存为 hidden
        hidden = self.relu(hidden)
        # 3. hidden 经过 self.fc2，结果保存为 predictions
        predictions = self.fc2(hidden)
        # 4. 返回 predictions
        return predictions
