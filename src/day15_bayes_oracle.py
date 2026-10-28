"""Day15：已知特征形式时的精确贝叶斯后验预测（贝叶斯线性回归）；核心由学习者实现。"""

import torch

from src.day15_pfn_prior import PRIOR_MEAN, PRIOR_STD


def posterior_predictive(
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_qry: torch.Tensor,
    prior_mean: tuple[float, float] = PRIOR_MEAN,
    prior_std: tuple[float, float] = PRIOR_STD,
    noise_std: float = 0.02,
    dt: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回查询点标签 y 的后验预测均值和方差（含观测噪声）。

    模型：y = phi @ [a, b] + eps，phi = dt * x = dt * [sin(theta), u]，eps ~ N(0, noise_std²)。
    先验：[a, b] ~ N(prior_mean, diag(prior_std²))，各任务独立。

    输入：x_ctx (B,K,2)、y_ctx (B,K,1)、x_qry (B,Q,2)，K 可以为 0。
    输出：mean (B,Q,1)、var (B,Q,1)，与输入同 dtype。不原地修改输入，不使用任务真值 params。

    公式（矩阵形式）：
      Lambda = Sigma0^-1 + Phi^T Phi / sigma²          (B,2,2)
      Sigma_n = Lambda^-1
      mu_n = Sigma_n (Sigma0^-1 mu0 + Phi^T y / sigma²) (B,2,1)
      mean = phi* @ mu_n，var = phi* Sigma_n phi*^T 的对角 + sigma²
    """
    # TODO（学习者），建议步骤：
    # 1. phi_ctx = dt * x_ctx (B,K,2)，phi_qry = dt * x_qry (B,Q,2)。
    phi_ctx = dt * x_ctx
    phi_qry = dt * x_qry
    # 2. 先验：mu0 = torch.tensor(prior_mean).reshape(1,2,1)；
    #    prior_precision = torch.diag(1 / torch.tensor(prior_std) ** 2)，形状 (2,2)，可广播到 (B,2,2)。
    #    注意 dtype 与 x_ctx 一致（可用 dtype=x_ctx.dtype）。
    mu0 = torch.tensor(prior_mean, dtype=x_ctx.dtype).reshape(1, 2, 1)
    prior_precision = torch.diag(
        1 / torch.tensor(prior_std, dtype=x_ctx.dtype) ** 2
    ).reshape(1, 2, 2)
    # 3. Lambda = prior_precision + phi_ctx.transpose(1, 2) @ phi_ctx / noise_std**2   (B,2,2)
    Lambda = (
        prior_precision + torch.matmul(phi_ctx.transpose(1, 2), phi_ctx) / noise_std**2
    )
    # 4. Sigma_n = torch.linalg.inv(Lambda)
    Sigma_n = torch.linalg.inv(Lambda)
    # 5. mu_n = Sigma_n @ (prior_precision @ mu0 + phi_ctx.transpose(1, 2) @ y_ctx / noise_std**2)  (B,2,1)
    mu_n = torch.matmul(
        Sigma_n,
        prior_precision @ mu0
        + torch.matmul(phi_ctx.transpose(1, 2), y_ctx) / noise_std**2,
    )
    # 6. mean = phi_qry @ mu_n   (B,Q,1)
    mean = torch.matmul(phi_qry, mu_n)
    # 7. var：对每个查询点算 phi* Sigma_n phi*^T。提示：((phi_qry @ Sigma_n) * phi_qry).sum(-1, keepdim=True)
    #    再加 noise_std**2。
    var = ((phi_qry @ Sigma_n) * phi_qry).sum(-1, keepdim=True) + noise_std**2
    return mean, var
