"""在一个训练窗口上更新一次，检查两步梯度流程；不代表效果对照实验。"""

import hashlib
import json
from pathlib import Path

import torch
from src.day04_world_model import (
    DynamicsMLP,
    episode_observations,
    rollout_predictions,
    multistep_loss,
)

root = Path(__file__).resolve().parents[1]
record = json.loads(
    (root / "results/day04_delta_normalized_run.json").read_text(encoding="utf-8")
)
dataset_path = root / "results/day03_pendulum_dataset.json"
assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == record["dataset_sha256"]
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
torch.set_num_threads(1)
model = DynamicsMLP()
model.load_state_dict(
    {
        name: torch.tensor(values, dtype=torch.float32)
        for name, values in record["final_state_dict"].items()
    }
)
input_mean = torch.tensor(record["input_mean"], dtype=torch.float32)
input_std = torch.tensor(record["input_std"], dtype=torch.float32)

# 使用训练集首轨迹的前两条转移；验证/测试数据不参与此次更新。
window = dataset["splits"]["train"][0][:2]
true_trajectory = episode_observations(window)
initial_observation = true_trajectory[:1]
actions = [torch.tensor(t["action"], dtype=torch.float32).reshape(1, 1) for t in window]
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
parameters_before = {
    name: value.detach().clone() for name, value in model.named_parameters()
}
model.train()

# TODO 1：清零梯度。
optimizer.zero_grad()
# TODO 2：调用 rollout_predictions，保存为 trajectory。
trajectory = rollout_predictions(
    model, initial_observation, actions, input_mean, input_std
)
# TODO 3：用 torch.cat(..., dim=0) 得到 predicted_trajectory。
predicted_trajectory = torch.cat(trajectory, dim=0)
# TODO 4：调用 multistep_loss，保存为 loss。
loss = multistep_loss(predicted_trajectory, true_trajectory)
# TODO 5：反向传播。
loss.backward()
# TODO 6：更新参数。
optimizer.step()

# 以下为导师提供的流程检查；这里不保存或覆盖模型权重。
assert predicted_trajectory.shape == true_trajectory.shape == (3, 3)
assert loss.requires_grad and torch.isfinite(loss)
assert all(
    p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
)
assert any(
    not torch.equal(value, parameters_before[name])
    for name, value in model.named_parameters()
)
print(f"Two-step loss before update: {loss.item():.6f}")
print("PASS: finite gradients and parameters updated")
