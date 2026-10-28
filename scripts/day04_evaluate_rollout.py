"""用固定验证轨迹前20个动作评估连续预测，不重新训练。"""

import hashlib
import json
from pathlib import Path

import torch
from src.day04_world_model import (
    DynamicsMLP,
    episode_observations,
    rollout_predictions,
    trajectory_step_mse,
    episode_windows,
)

root = Path(__file__).resolve().parents[1]
record = json.loads(
    (root / "results/day04_delta_normalized_run.json").read_text(encoding="utf-8")
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
input_mean = torch.tensor(record["input_mean"], dtype=torch.float32)
input_std = torch.tensor(record["input_std"], dtype=torch.float32)

# 先观察第一条验证轨迹的前20步；不能据此代表全部轨迹。
episode = dataset["splits"]["val"][0][:20]
true_trajectory = episode_observations(episode)
initial_observation = true_trajectory[:1]
actions = [
    torch.tensor(t["action"], dtype=torch.float32).reshape(1, 1) for t in episode
]

model.eval()
with torch.no_grad():
    # TODO 1：调用 rollout_predictions，结果保存为 trajectory。
    # 参数依次为 model、initial_observation、actions、input_mean、input_std。
    trajectory = rollout_predictions(
        model, initial_observation, actions, input_mean, input_std
    )
    # TODO 2：将 trajectory 沿 dim=0 用 torch.cat 拼接为 predicted_trajectory。
    predicted_trajectory = torch.cat(trajectory, dim=0)
    # TODO 3：调用 trajectory_step_mse，比较预测与真实轨迹，保存为 step_mse。
    step_mse = trajectory_step_mse(predicted_trajectory, true_trajectory)

for step in [0, 1, 5, 10, 20]:
    print(f"step={step}, MSE={step_mse[step].item():.6f}")

horizon = 20
all_step_errors = []

with torch.no_grad():
    # 验证集中的所有轨迹
    for episode in dataset["splits"]["val"]:

        # 验证集中的所有完整轨迹
        for window in episode_windows(episode, horizon):
            # 当前 episode 的所有合法窗口
            true_trajectory = episode_observations(window)
            initial_observation = true_trajectory[:1]
            actions = [
                torch.tensor(t["action"], dtype=torch.float32).reshape(1, 1)
                for t in window
            ]
            trajectory = rollout_predictions(
                model, initial_observation, actions, input_mean, input_std
            )
            predicted_trajectory = torch.cat(trajectory, dim=0)
            step_mse = trajectory_step_mse(predicted_trajectory, true_trajectory)
            all_step_errors.append(step_mse)

error_matrix = torch.stack(all_step_errors, dim=0)
mean_step_mse = error_matrix.mean(dim=0)  # 形状 (21,)
