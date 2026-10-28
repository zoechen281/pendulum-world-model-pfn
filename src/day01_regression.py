"""Day 1：一次只实现一个小功能。"""

import torch


def make_inputs(values: list[float]) -> torch.Tensor:
    """把非空数值列表转换为 float32 Tensor，形状为 (样本数, 1)。

    例：[-2.0, -1.0, 0.0, 1.0] 对应 4 行、1 列。
    当前只处理非空、有限数值列表，不要求扩展异常输入处理。
    """
    # TODO：由你创建 Tensor、调整为列向量，并返回结果。

    x = torch.tensor(values, dtype=torch.float32)
    x = x.reshape(len(values), 1)
    return x


def make_targets(inputs: torch.Tensor) -> torch.Tensor:
    """按 y = 2*x^2 + 3*x + 1 为每条输入样本计算目标值，不加噪声。

    输入：float32 Tensor，形状 (样本数, 1)。
    输出：相同形状、相同类型的目标 Tensor；不修改输入。
    """
    # TODO：由你对 Tensor 逐元素计算公式，并返回结果。
    y = 2 * inputs**2 + 3 * inputs + 1
    return y


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算预测与目标的均方误差，返回只含一个数的标量 Tensor。

    当前约定：两个输入都是非空 float32 Tensor，形状相同且为 (样本数, 1)。
    保留 Tensor 运算，以便后续对损失调用 backward；不修改输入。
    """
    # TODO：依次计算差值、平方、平均，并返回结果。
    squared_errors = (predictions - targets) ** 2
    MSE = squared_errors.mean()
    return MSE


def build_linear_model() -> torch.nn.Linear:
    """创建并返回一个每条样本 1 个输入、1 个输出的线性层。

    模型执行 y_hat=w*x+b，权重和偏置均可训练。
    本函数只创建模型；输入数据在调用处送入模型。
    """
    # TODO：创建 torch.nn.Linear 模型，并返回模型对象。
    model = torch.nn.Linear(in_features=1, out_features=1)
    return model


def make_noisy_targets(inputs: torch.Tensor, noise_std: float = 0.2) -> torch.Tensor:
    """计算二次函数目标，并添加均值为 0、标准差为 noise_std 的高斯噪声。

    输入为非空 float32 Tensor，形状 (样本数, 1)。
    当前约定 noise_std 为有限非负数，输出形状和类型与输入相同。
    不修改输入；随机种子由调用者设置，本函数不重置种子。
    """
    # TODO 1：调用已有函数，得到无噪声目标 clean_targets。
    # TODO 2：生成同形状的标准正态随机数，再乘 noise_std。
    # TODO 3：返回无噪声目标与噪声之和。
    clean_targets = make_targets(inputs)
    noise = torch.randn_like(clean_targets) * noise_std
    return clean_targets + noise


def evaluate_model(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> float:
    """保持参数不变，返回给定数据上的 MSE 数值。

    输入与目标为同形状的非空 float32 Tensor，形状 (样本数, 1)。
    设置评估模式，在 no_grad 范围内预测并计算 MSE，再返回 Python 数值。
    本函数结束后模型保持评估模式；恢复训练时由调用者执行 model.train()。
    """
    # TODO：评估模式 → 不追踪梯度的预测与损失 → 返回数值。
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        loss = mse_loss(predictions, targets)
    return loss.item()


def build_mlp_model() -> torch.nn.Sequential:
    """创建并返回 Linear(1,16) → ReLU → Linear(16,1) 的小型网络。

    输入形状：(样本数, 1)。隐藏特征形状：(样本数, 16)。
    输出形状：(样本数, 1)。本函数只创建模型，不训练。
    """
    # TODO：使用 Sequential 按顺序组合三项，再返回模型对象。
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=1, out_features=16),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=16, out_features=1),
    )
    return model


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
