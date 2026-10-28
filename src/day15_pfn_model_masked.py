"""Day15 掩码 PFN：结构接线由助手提供，build_attention_mask 由学习者填写。

保留旧 ContextPFN 和旧权重。K>0 时前缀为 K 个上下文 token；
K=0 时保持原逻辑，前缀为一个可学习先验 token。没有查询标签输入。
"""

import torch


def build_attention_mask(num_prefix: int, num_query: int) -> torch.Tensor:
    """返回 CPU bool Tensor，shape=(L,L)，L=num_prefix+num_query。

    有效输入：num_prefix>=1，num_query>=1。
    行表示正在读取信息的 token，列表示被读取的 token。
    True=禁止关注，False=允许关注。
    所有行只能读取前 num_prefix 列；其余列全部禁止。
    先验 token 也属于前缀，因此 K=0 时仍有允许关注的位置。
    """
    # TODO（学习者）：
    # 1. 计算 L，使用 torch.ones 创建 (L,L) 且 dtype=torch.bool 的全 True 张量。
    L = num_prefix + num_query
    mask = torch.ones(L, L, dtype=torch.bool)
    # 2. 通过二维切片，将所有行的前 num_prefix 列设为 False。
    mask[:, :num_prefix] = False
    # 3. 返回该张量。
    return mask


class MaskedContextPFN(torch.nn.Module):
    """上下文 + 查询的 Transformer；输出每个查询点的高斯预测分布。

    记号约定（与 src/day15_pfn_prior.sample_tasks 一致）：
    - 每个“任务”固定一组物理参数 (a, b)，网络只能看到上下文样本，看不到 (a, b)。
    - 上下文 token 编码 (x=[sinθ,u], y, 标记=1)；查询 token 编码 (x, 0, 标记=0)。
    - 前缀和查询的注意力都只能读取前缀，前缀不能读取查询。
    - 查询自身特征通过残差连接保留；因此各查询输出不依赖其他查询。
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
        num_prefix = K if K > 0 else 1
        mask = build_attention_mask(num_prefix, x_qry.shape[1]).to(tokens.device)
        encoded = self.encoder(self.token_projection(tokens), mask=mask)
        mean_logvar = self.head(encoded[:, -x_qry.shape[1] :, :])
        return mean_logvar[..., :1], mean_logvar[..., 1:]
