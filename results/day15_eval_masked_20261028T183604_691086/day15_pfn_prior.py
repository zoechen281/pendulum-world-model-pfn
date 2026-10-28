"""Day15：未裁剪单摆增量的合成任务；核心函数由学习者实现。"""

import torch


def pendulum_delta_omega(
    sin_theta: torch.Tensor,
    torque: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt: float = 0.05,
) -> torch.Tensor:
    """返回无噪声、未裁剪的角速度变化量。

    sin_theta 已是角度的正弦值，不要再次取 sin。
    输入为可广播的 CPU 浮点 Tensor，不原地修改任何输入。
    标量输入返回标量 Tensor。
    批量约定：sin_theta/torque 为 (B,N,1)，a/b 为 (B,1,1)，
    输出为 (B,N,1)。
    本函数不生成噪声、不采样参数、不裁剪下一角速度。
    关系式见指定计划：Delta omega = dt * (a * sin(theta) + b * torque)。
    """
    # TODO（学习者）：用一条 return 实现上述关系式，替换下面的占位异常。
    return dt * (a * sin_theta + b * torque)


PRIOR_MEAN = (15.0, 3.0)
PRIOR_STD = (3.0, 0.75)


def sample_tasks(
    batch_size: int,
    num_context: int,
    num_query: int,
    generator: torch.Generator,
    noise_std: float = 0.02,
    dt: float = 0.05,
) -> dict[str, torch.Tensor]:
    """从先验采样 batch_size 个单摆任务，每个任务含 K 个上下文点和 Q 个查询点。

    记 B=batch_size、K=num_context、Q=num_query、N=K+Q。所有随机数都传 generator=generator。
    先验（未截断、相互独立）：a ~ N(15, 3²)，b ~ N(3, 0.75²)；同一任务内所有点共用同一组 (a, b)。
    每个点独立采样：theta ~ U[-pi, pi]，omega ~ U[-6, 6]，u ~ U[-2, 2]。
    标签 y = pendulum_delta_omega(sin(theta), u, a, b, dt) + noise_std * 标准正态噪声（不裁剪）。

    返回字典（全部 float32）：
      x_ctx (B,K,2) = [sin(theta), u]，y_ctx (B,K,1)
      x_qry (B,Q,2)，                  y_qry (B,Q,1)
      params (B,2) = [a, b]            仅诊断用，不能给 PFN 或贝叶斯推断器当输入
      obs_qry (B,Q,3) = [cos(theta), sin(theta), omega]  仅供固定 MLP 评估使用
    K 可以为 0（此时 x_ctx/y_ctx 的第二维为 0）。
    """
    # TODO（学习者），建议步骤：
    # 1. 采样 a、b，各为 (B,1,1)：均值 + 标准差 * torch.randn(..., generator=generator)。
    a = PRIOR_MEAN[0] + PRIOR_STD[0] * torch.randn(
        (batch_size, 1, 1), generator=generator
    )
    b = PRIOR_MEAN[1] + PRIOR_STD[1] * torch.randn(
        (batch_size, 1, 1), generator=generator
    )
    # 2. 一次性为 N 个点采样 theta、omega、u，各为 (B,N,1)：
    #    U[low, high] = low + (high - low) * torch.rand(..., generator=generator)。
    N = num_context + num_query
    theta = -torch.pi + 2 * torch.pi * torch.rand(
        (batch_size, N, 1), generator=generator
    )
    omega = -6 + 12 * torch.rand((batch_size, N, 1), generator=generator)
    u = -2 + 4 * torch.rand((batch_size, N, 1), generator=generator)
    # 3. 计算 sin_theta，用 pendulum_delta_omega 得到无噪声增量，再加噪声得到 y (B,N,1)。
    sin_theta = torch.sin(theta)
    delta_omega = pendulum_delta_omega(sin_theta, u, a, b, dt)
    noise = noise_std * torch.randn((batch_size, N, 1), generator=generator)
    y = delta_omega + noise
    # 4. 拼出 x = [sin_theta, u] (B,N,2) 和 obs = [cos_theta, sin_theta, omega] (B,N,3)。
    cos_theta = torch.cos(theta)
    x = torch.cat([sin_theta, u], dim=-1)
    obs = torch.cat([cos_theta, sin_theta, omega], dim=-1)
    # 5. 沿第二维切片：前 K 个为上下文，后 Q 个为查询；params 由 a、b 拼成 (B,2)。
    x_ctx = x[:, :num_context, :]
    y_ctx = y[:, :num_context, :]
    x_qry = x[:, num_context:, :]
    y_qry = y[:, num_context:, :]
    params = torch.stack([a.squeeze(-1).squeeze(-1), b.squeeze(-1).squeeze(-1)], dim=-1)
    obs_qry = obs[:, num_context:, :]
    return {
        "x_ctx": x_ctx,
        "y_ctx": y_ctx,
        "x_qry": x_qry,
        "y_qry": y_qry,
        "params": params,
        "obs_qry": obs_qry,
    }
