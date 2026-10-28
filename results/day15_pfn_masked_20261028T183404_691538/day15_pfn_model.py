"""Day15：PFN 式上下文推断网络（概念验证规模）。结构由导师提供，统计核心由学习者实现。"""

import torch
import math
from src.day15_pfn_prior import sample_tasks


class ContextPFN(torch.nn.Module):
    """上下文 + 查询的 Transformer；输出每个查询点的高斯预测分布。

    记号约定（与 src/day15_pfn_prior.sample_tasks 一致）：
    - 每个“任务”固定一组物理参数 (a, b)，网络只能看到上下文样本，看不到 (a, b)。
    - 上下文 token 编码 (x=[sinθ,u], y, 标记=1)；查询 token 编码 (x, 0, 标记=0)。
    - 查询之间未屏蔽（联合自注意力），预测可依赖其他查询的特征。
    - K=0 时使用可学习的“先验 token”，避免注意力作用于空集合。
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.token_projection = torch.nn.Linear(4, d_model)
        self.prior_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 2),
        )

    def forward(
        self,
        x_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        x_qry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x_ctx (B,K,2)、y_ctx (B,K,1)、x_qry (B,Q,2) -> mean (B,Q,1)、log_var (B,Q,1)。

        token 内容（最后一维 4 个特征）：上下文 [sinθ, u, y, 1]，查询 [sinθ, u, 0, 0]；
        第 4 维标记 1=上下文、0=查询。
        """
        B, K, _ = x_ctx.shape
        ctx_tokens = torch.cat(
            [x_ctx, y_ctx, torch.ones(B, K, 1, dtype=x_ctx.dtype)], dim=-1
        )
        qry_tokens = torch.cat(
            [x_qry, torch.zeros(B, x_qry.shape[1], 2, dtype=x_ctx.dtype)], dim=-1
        )
        tokens = torch.cat([ctx_tokens, qry_tokens], dim=1)
        if K == 0:
            tokens = torch.cat(
                [torch.zeros(B, 1, 4, dtype=x_ctx.dtype) + self.prior_token, tokens],
                dim=1,
            )
        encoded = self.encoder(self.token_projection(tokens))
        mean_logvar = self.head(encoded[:, -x_qry.shape[1] :, :])
        return mean_logvar[..., :1], mean_logvar[..., 1:]


def gaussian_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """返回逐样本 N( mean, exp(log_var) ) 下 y 的负对数似然均值。

    mean/log_var/y 形状相同 (...,1)，返回标量（对全部元素取 mean）。
    公式：0.5 * ( log(2π) + log_var + (y-mean)²/exp(log_var) )。
    """
    # TODO（学习者）：用一行或多行实现上述公式。提示：torch.log(torch.tensor(2*math.pi))。
    log_2pi = torch.log(torch.tensor(2 * math.pi))
    nll = 0.5 * (log_2pi + log_var + (y - mean) ** 2 / torch.exp(log_var))
    return nll.mean()


def train_pfn(
    model: ContextPFN,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    batch_size: int,
    generator: torch.Generator,
    num_query: int = 16,
    noise_std: float = 0.02,
) -> list[tuple[int, float]]:
    """每一步从先验现采一批任务训练；返回 [(步数, 当步NLL)]。"""
    context_sizes = [0, 1, 2, 4, 8, 16]
    losses = []
    model.train()
    for step in range(num_steps):
        index = torch.randint(0, len(context_sizes), (1,), generator=generator).item()
        k = context_sizes[index]
        task = sample_tasks(batch_size, k, num_query, generator, noise_std=noise_std)
        mean, log_var = model(task["x_ctx"], task["y_ctx"], task["x_qry"])
        optimizer.zero_grad()
        loss = gaussian_nll(mean, log_var, task["y_qry"])
        loss.backward()
        optimizer.step()
        losses.append((step, loss.item()))
    return losses
