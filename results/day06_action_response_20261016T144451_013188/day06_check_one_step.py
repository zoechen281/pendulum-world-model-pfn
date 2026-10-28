"""同一状态、同一动作的单步诊断；一个转移不能解释整个控制回合。"""

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.day04_world_model import DynamicsMLP, predict_next_observation


def main(torque: float = 1.0, env_seed: int = 100):
    # 导师提供的恢复与环境辅助；不重新训练、不覆盖权重。
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = root / "results/day04_delta_normalized_run.json"
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    record = json.loads(checkpoint_bytes.decode("utf-8"))
    torch.set_num_threads(1)
    model = DynamicsMLP()
    model.load_state_dict(
        {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in record["final_state_dict"].items()
        }
    )
    model.eval()
    input_mean = torch.tensor(record["input_mean"], dtype=torch.float32)
    input_std = torch.tensor(record["input_std"], dtype=torch.float32)
    env = gym.make("Pendulum-v1")
    try:
        observation, info = env.reset(seed=env_seed)
        observation_tensor = torch.tensor(observation, dtype=torch.float32).reshape(
            1, 3
        )
        # 固定合法动作便于比较；此处不选择或优化动作。
        action = np.array([torque], dtype=np.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32).reshape(1, 1)
        assert env.action_space.contains(action)
        with torch.no_grad():
            # TODO 1（当前步骤）：调用 predict_next_observation，保存为 predicted_next。
            # 参数依次为 model、observation_tensor、action_tensor、input_mean、input_std。
            # 输出是原尺度下一观测 (1,3)，函数内已完成标准化及变化量加回。
            predicted_next = predict_next_observation(
                model, observation_tensor, action_tensor, input_mean, input_std
            )

        # TODO 2（随后填写）：调用 env.step(action)，得到真实 next_observation 等五项。
        # 模型和环境都从上述真实 observation 出发，期间不得先执行其他动作。
        next_observation, reward, terminated, truncated, info = env.step(action)
        # TODO 3：将真实 next_observation 转为 float32 Tensor，形状 (1,3)。
        next_observation_tensor = torch.tensor(
            next_observation, dtype=torch.float32
        ).reshape(1, 3)
        # TODO 4：打印预测/真实下一观测、三个分量的平方误差及整体 MSE。
        # 分量顺序为 cos、sin、角速度；混合分量 MSE 不是统一物理单位误差。
        # 此单点只能用于流程核对，不能证明它就是 MPC 效果不佳的原因。
        print("Predicted next observation:", predicted_next)
        print("True next observation:", next_observation_tensor)
        squared_errors = (predicted_next - next_observation_tensor) ** 2
        mse = squared_errors.mean()
        print("Squared errors:", squared_errors)
        print("Mean squared error:", mse)

        assert (
            hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
