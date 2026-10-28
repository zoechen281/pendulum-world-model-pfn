"""Day8：网络学习角速度变化，再由运动学结构重建下一观测。"""

import torch

from src.day04_world_model import normalize_inputs


class PhysicsStructuredMLP(torch.nn.Module):
    """内部网络 4->64->64->1；最终输出原尺度下一观测 (N,3)。

    仅固定运动学与观测重建，不把完整重力/力矩动力学公式写进网络。
    两个隐藏层宽度与基线相同，输出头不同，参数量不完全相同，比较时需报告。
    """

    def __init__(self):
        super().__init__()
        # 当前任务：用 torch.nn.Sequential 定义 self.network。
        # 依次为 Linear(4,64)、ReLU、Linear(64,64)、ReLU、Linear(64,1)。
        # 最后一层不加ReLU：角速度变化量既可能为正，也可能为负。
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
    ) -> torch.Tensor:
        """原尺度 observations (N,3)、actions (N,1)，统计量 (1,4)。

        观测排列为 [cos(theta), sin(theta), angular_velocity]；
        输入与参数同设备/浮点类型、有限值，cos/sin不同时为零，动作合法。
        输入只用训练统计量标准化，不原地修改；返回原尺度下一观测 (N,3)。
        调用方决定训练/评估模式和梯度开关。
        dt=0.05、速度裁剪[-8,8]对应本项目默认Pendulum-v1。
        """
        # 后续任务（本次先不写）：
        # 1. 拼接观测和动作，标准化后送入self.network，得到 delta_velocity (N,1)。
        # 2. 用atan2从原始cos/sin恢复theta (N,1)，取原始角速度 (N,1)。
        #    切片要保留列维度，避免 (N,) 与 (N,1) 相加广播为 (N,N)。
        # 3. 当前角速度加变化量，裁剪到[-8,8]，得到next_velocity。
        # 4. 用更新后的next_velocity乘dt更新theta，得到next_theta。
        # 5. 用cos/sin重建角度分量，与next_velocity按列拼接为 (N,3)，返回。
        # 不无条件加入机械能守恒损失；有外力矩时能量可以改变。
        inputs = torch.cat([observations, actions], dim=1)
        normalized_inputs = normalize_inputs(inputs, input_mean, input_std)
        delta_velocity = self.network(normalized_inputs)
        angular_velocity = observations[:, 2:3]
        next_velocity = torch.clamp(angular_velocity + delta_velocity, -8.0, 8.0)
        theta = torch.atan2(observations[:, 1:2], observations[:, 0:1])
        next_theta = theta + next_velocity * 0.05
        next_cos = torch.cos(next_theta)
        next_sin = torch.sin(next_theta)
        next_observation = torch.cat([next_cos, next_sin, next_velocity], dim=1)
        return next_observation
