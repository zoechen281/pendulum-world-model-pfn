"""Day 1：先观察求导结果，再由学习者补充一次参数更新。"""

import torch

# 请亲手完成以下步骤：
# 1. 创建数值为 1.0 的标量 Tensor w，并设置 requires_grad=True。
# 2. 计算 loss = (w - 3) 的平方。
# 3. 对 loss 调用 backward。
# 4. 打印 w.grad 和 w，观察梯度与参数值的区别。
# 完成后删除下面的占位异常。
w = torch.tensor(1.0, requires_grad=True)
loss = (w - 3) ** 2
loss.backward()
print(w.grad)
print(w)

# 下一步由你补充：
# 1. 用 torch.optim.SGD 创建优化器，负责参数 w，学习率为 0.1。
# 2. 调用优化器的 step 方法，使用前面算出的梯度执行一次更新。
# 3. 再打印 w，比较更新前后的参数值。
# 当前只更新一次，不写循环。
optimizer = torch.optim.SGD([w], lr=0.1)
optimizer.step()
print(w)

# 第二次更新由你补充，复用上面的 w 和 optimizer：
# 1. 调用 optimizer.zero_grad() 清理旧梯度。
# 2. 根据当前 w 重新计算平方损失，保存到 new_loss。
# 3. 对 new_loss 调用 backward，并打印 w.grad。
# 4. 再调用 optimizer.step()，然后打印 w。
# 不重新创建 w 或 optimizer，不写循环。
optimizer.zero_grad()
new_loss = (w - 3) ** 2
new_loss.backward()
print(w.grad)
optimizer.step()
print(w)
