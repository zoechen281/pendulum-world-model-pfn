"""Day 4：将转移整理为监督学习数据，再学习单步动力学。"""

import torch


def transition_to_tensors(transition: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """将一条从 JSON 读取的转移转换为 (模型输入, 直接预测目标)。

    当前约定 observation、action、next_observation 均为数值列表，
    长度依次为 3、1、3，数值有限；暂不扩展异常输入处理。
    输入按 [cos(theta), sin(theta), angular_velocity, torque] 排列，
    float32，形状 (4,)；目标为下一观测，float32，形状 (3,)。
    不包含 reward 或结束标记，不修改原始转移。
    本函数暂时直接预测下一观测，不计算变化量、不做标准化。
    """
    # TODO 1：将 observation 列表与 action 列表拼接，保存为 input_values。
    input_values = transition["observation"] + transition["action"]
    # TODO 2：将 input_values 转为 float32 Tensor，保存为 inputs。
    inputs = torch.tensor(input_values, dtype=torch.float32)
    # TODO 3：将 next_observation 转为 float32 Tensor，保存为 targets。
    targets = torch.tensor(transition["next_observation"], dtype=torch.float32)
    # TODO 4：返回 inputs、targets。
    return inputs, targets


class DynamicsMLP(torch.nn.Module):
    """直接预测下一观测：4 -> 64 -> 64 -> 3，两个隐藏层后使用 ReLU。"""

    def __init__(self):
        super().__init__()
        # TODO：用 Sequential 定义 self.network，按上述维度连接层。
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """输入 (N,4)，返回下一观测预测 (N,3)，不更新参数。"""
        # TODO：将 inputs 送入 self.network，返回计算结果。
        return self.network(inputs)


def episodes_to_tensors(
    episodes: list[list[dict]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """将一个已划分集合的全部轨迹转换为两张二维数据表。

    当前约定 episodes 非空，每条轨迹非空，记录格式同 transition_to_tensors。
    按 episode 顺序及轨迹内部时间顺序遍历，复用单条转换函数。
    返回 float32 inputs (N,4)、targets (N,3)，N 为转移总数。
    只在集合内部展开，不混合训练/验证/测试，不修改原始记录。
    """
    # TODO 1：创建两个空列表 input_rows、target_rows。
    input_rows, target_rows = [], []
    # TODO 2：外层循环遍历 episode，内层循环遍历 transition。
    for episode in episodes:
        for transition in episode:
            inputs, targets = transition_to_tensors(transition)
            input_rows.append(inputs)
            target_rows.append(targets)
    # TODO 3：调用 transition_to_tensors，将结果分别 append 到两个列表。
    # TODO 4：分别用 torch.stack(..., dim=0) 堆叠为 inputs、targets。
    inputs = torch.stack(input_rows, dim=0)
    targets = torch.stack(target_rows, dim=0)
    # TODO 5：返回 inputs、targets。
    return inputs, targets


def make_delta_targets(
    inputs: torch.Tensor,
    next_observations: torch.Tensor,
) -> torch.Tensor:
    """inputs 形状 (N,4)，next_observations 形状 (N,3)。
    返回下一观测减去当前观测，形状 (N,3)。
    """
    # 从 inputs 取出所有行的前三列，再计算差值并返回。
    current_observations = inputs[:, :3]
    return next_observations - current_observations


def fit_input_mean(train_inputs: torch.Tensor) -> torch.Tensor:
    """输入 (N,4)，返回各特征均值，形状 (1,4)。"""
    # 沿样本维求均值，保留维度，并返回。
    return train_inputs.mean(dim=0, keepdim=True)


def fit_input_std(train_inputs: torch.Tensor) -> torch.Tensor:
    """返回各列标准差，形状 (1,4)，最小值为 1e-6。"""
    # 1. 沿样本维计算标准差，保留维度，使用 correction=0。
    std = train_inputs.std(dim=0, keepdim=True, correction=0)
    # 2. 设置下限并返回。
    std = torch.clamp(std, min=1e-6)
    return std


def normalize_inputs(
    inputs: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
) -> torch.Tensor:
    """inputs 为 (N,4)，mean/std 为 (1,4)，返回 (N,4)。"""
    # 按 (输入 - 均值) / 标准差计算并返回，不修改原输入。
    return (inputs - input_mean) / input_std


def predict_next_observation(
    model: torch.nn.Module,
    observation: torch.Tensor,
    action: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
) -> torch.Tensor:
    """observation: (1,3)，action: (1,1)。
    mean/std: (1,4)，返回原尺度的下一观测 (1,3)。
    调用方负责 model.eval() 和 torch.no_grad()。
    """
    # 1. 沿 dim=1 用 torch.cat 拼接 observation、action，得到 inputs。
    inputs = torch.cat([observation, action], dim=1)
    # 2. 用 normalize_inputs 标准化 inputs。
    normalized_inputs = normalize_inputs(inputs, input_mean, input_std)
    # 3. 将标准化输入送入 model，得到 predicted_deltas。
    predicted_deltas = model(normalized_inputs)
    # 4. 返回 observation + predicted_deltas。
    return observation + predicted_deltas


def rollout_predictions(
    model: torch.nn.Module,
    initial_observation: torch.Tensor,
    actions: list[torch.Tensor],
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
) -> list[torch.Tensor]:
    """初始观测 (1,3)，每个动作 (1,1)。
    返回包含初始观测的预测轨迹，长度为 len(actions)+1。
    调用方负责 model.eval() 和 torch.no_grad()。
    """
    predicted_observation = initial_observation
    trajectory = [predicted_observation]

    for action in actions:
        # 1. 调用 predict_next_observation，更新 predicted_observation。
        predicted_observation = predict_next_observation(
            model, predicted_observation, action, input_mean, input_std
        )
        # 2. 将更新后的 predicted_observation append 到 trajectory。
        trajectory.append(predicted_observation)

    return trajectory


def episode_observations(episode: list[dict]) -> torch.Tensor:
    """非空 episode，观测为 JSON 数值列表。
    返回包含初始观测的真实轨迹，float32，形状 (T+1,3)。
    """
    observations = [episode[0]["observation"]]

    for transition in episode:
        # 将本条转移的 next_observation append 到 observations。
        observations.append(transition["next_observation"])

    # 将 observations 转为 float32 Tensor 并返回。
    return torch.tensor(observations, dtype=torch.float32)


def trajectory_step_mse(
    predicted_trajectory: torch.Tensor,
    true_trajectory: torch.Tensor,
) -> torch.Tensor:
    """两者形状均为 (T+1,3)，返回逐时刻 MSE，形状 (T+1,)。
    包含初始时刻，不修改输入。
    """
    # 1. 计算逐元素平方误差。
    squared_errors = (predicted_trajectory - true_trajectory) ** 2
    # 2. 对三个观测分量求平均，保留时间维，并返回。
    mse_per_step = squared_errors.mean(dim=1)
    return mse_per_step


def episode_windows(
    episode: list[dict],
    horizon: int,
) -> list[list[dict]]:
    """horizon 为正整数，返回该 episode 内所有连续转移窗口。
    每个窗口恰有 horizon 条转移；长度不足时返回空列表。
    不跨 episode，不修改输入。
    """
    windows = []
    # 1. 遍历所有合法起点。
    for start in range(len(episode) - horizon + 1):
        window = episode[start : start + horizon]
        windows.append(window)
    # 2. 切出 horizon 条转移，append 到 windows。
    return windows


def multistep_loss(
    predicted_trajectory: torch.Tensor,
    true_trajectory: torch.Tensor,
) -> torch.Tensor:
    """输入均为 (H+1,3)，H>=1，包含初始观测。
    返回排除初始时刻后的平均 MSE，保留梯度。
    """
    # 1. 调用 trajectory_step_mse，得到 step_mse。
    step_mse = trajectory_step_mse(predicted_trajectory, true_trajectory)
    # 2. 跳过初始项，对其余项求平均并返回。
    return step_mse[1:].mean()
