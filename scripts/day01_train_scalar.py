"""Day 1：用循环重复 10 次标量参数更新，由学习者填写。"""

import torch

# 循环外：创建初值为 1.0、需要求导的 w，再创建学习率 0.1 的 SGD。
# 循环内：重复 10 次清理梯度、重算 loss=(w-3)^2、求导、更新。
# 每轮打印轮次 i 和本轮算出的 loss（它是该轮更新前的损失）。
# 循环结束后打印最终 w。
# 参数和优化器只创建一次，不在循环内重建。
# 完成后删除占位异常。
w = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)
for i in range(10):
    optimizer.zero_grad()
    loss = (w - 3) ** 2
    loss.backward()
    print(f"Iteration {i}, Loss: {loss.item()}")
    optimizer.step()
print(f"Final w: {w.item()}")
