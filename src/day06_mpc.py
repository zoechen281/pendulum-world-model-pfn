"""Day 6：用已学动力学模型进行短期规划；核心函数由学习者填写。"""

import torch

from src.day04_world_model import normalize_inputs


def pendulum_rewards(
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """计算执行动作前的 Pendulum 即时奖励。

    observations: (N,3)，原尺度 [cos(theta), sin(theta), angular_velocity]。
    actions: (N,1)，原尺度力矩，调用方保证在 [-2,2] 内。
    输入约定为同设备、同浮点类型、有限值，cos/sin 不同时为零。
    返回 (N,) 奖励，不求平均，不修改输入；梯度开关由调用方控制。
    模型预测的 cos/sin 可能不在单位圆上，这里用 atan2 提取方向，
    不因此认为预测状态已经满足物理约束。
    """
    # TODO 1：分别取 observations 的第 1、2 列，得到 cos_theta、sin_theta。
    cos_theta = observations[:, 0]
    sin_theta = observations[:, 1]
    # TODO 2：使用 torch.atan2(sin_theta, cos_theta) 得到 theta。
    theta = torch.atan2(sin_theta, cos_theta)
    # TODO 3：取 observations 第 3 列为 angular_velocity；取 actions 唯一列为 torque。
    angular_velocity = observations[:, 2]
    torque = actions[:, 0]
    # 上述四个取列结果以及 theta 均应为 (N,)。
    # TODO 4：按 -(theta平方 + 0.1*角速度平方 + 0.001*力矩平方) 计算 rewards。
    rewards = -(theta**2 + 0.1 * angular_velocity**2 + 0.001 * torque**2)
    # TODO 5：返回 rewards，完成后删除下面的占位异常。
    return rewards


def score_action_sequences(
    model: torch.nn.Module,
    initial_observation: torch.Tensor,
    candidate_actions: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
) -> torch.Tensor:
    """用模型模拟候选动作序列，返回各自的预测累计奖励。

    initial_observation: (1,3)，原尺度真实初始观测。
    candidate_actions: (N,H,1)，N>=1、H>=1，力矩在 [-2,2] 内。
    mean/std: (1,4)，沿用训练统计量，std 为正。
    输入及模型使用同设备、同浮点类型；返回 scores (N,)，不取平均。
    先计算当前状态及动作的奖励，再预测下一观测；无折扣、无终端奖励。
    不修改传入的观测、候选动作或模型参数。
    调用方负责 model.eval() 和 torch.no_grad()；本函数不训练模型。
    """
    # 当前小步骤：初始化候选数量、各候选观测和累计分数。
    # TODO 1：从 candidate_actions 第 0 维读取 num_candidates。
    num_candidates = candidate_actions.shape[0]
    # TODO 2：用 repeat 将 initial_observation 复制为 observations (N,3)。
    observations = initial_observation.repeat(num_candidates, 1)
    # TODO 3：创建 scores (N,) 全零 Tensor。
    # 工程提示：initial_observation.new_zeros(num_candidates)
    # 与 torch.zeros 一样生成零，但自动沿用观测的 dtype 和 device。
    scores = initial_observation.new_zeros(num_candidates)

    # 后续再填写循环：沿 candidate_actions 第 1 维遍历规划步。
    for step in range(candidate_actions.shape[1]):
        # 每一步依次完成：
        # - 取所有候选在当前步的 actions (N,1)。
        actions = candidate_actions[:, step]
        # - 调用 pendulum_rewards 得到 rewards (N,)，累加到 scores。
        rewards = pendulum_rewards(observations, actions)
        scores += rewards
        # - 拼接 observations、actions 为 inputs (N,4)，再标准化。
        inputs = torch.cat([observations, actions], dim=1)
        normalized_inputs = normalize_inputs(inputs, input_mean, input_std)
        # - model 输出 predicted_deltas (N,3)，加回原尺度 observations。
        predicted_deltas = model(normalized_inputs)
        observations = observations + predicted_deltas
    #   使用非原地相加，避免修改初始观测或破坏梯度图。

    # 循环结束后返回 scores。
    return scores


@torch.no_grad()
def plan_random_shooting(
    model: torch.nn.Module,
    initial_observation: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    num_candidates: int,
    horizon: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """随机采样候选、用模型评分，返回最佳序列的第一个动作 (1,)。

    initial_observation: (1,3)，mean/std: (1,4)，均沿用原有契约。
    num_candidates 和 horizon 为正整数；动作每维范围 [-2,2)。
    调用方先设置 model.eval()；装饰器关闭本函数内的梯度追踪。
    generator 可由调用方设置种子以复现采样，其设备需匹配观测设备。
    不在函数内重置种子，避免闭环每一步重复相同随机候选。
    返回动作与观测同 dtype/device，不训练或修改模型参数。
    只在抽到的候选中选最好，不保证全局最优或真实环境控制效果。
    """
    # TODO 1（当前步骤）：生成 candidate_actions，形状 (num_candidates,horizon,1)。
    # 使用 torch.rand，指定 dtype=initial_observation.dtype、
    # device=initial_observation.device、generator=generator。
    # 再用之前推导的乘法与减法，把 [0,1) 映射为 [-2,2)。
    candidate_actions = (
        torch.rand(
            (num_candidates, horizon, 1),
            dtype=initial_observation.dtype,
            device=initial_observation.device,
            generator=generator,
        )
        * 4
        - 2
    )  # 映射到 [-2, 2)
    # TODO 2（随后填写）：调用 score_action_sequences 得到 scores (N,)。
    # 参数顺序：model、initial_observation、candidate_actions、input_mean、input_std。
    scores = score_action_sequences(
        model, initial_observation, candidate_actions, input_mean, input_std
    )
    # TODO 3：用 torch.argmax 找到 best_index。
    best_index = torch.argmax(scores)
    # TODO 4：取最佳候选的第一个动作 best_action (1,)，并返回。
    best_action = candidate_actions[best_index, 0]
    return best_action
