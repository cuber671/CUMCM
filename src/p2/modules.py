"""P2 M1-E 模型掩码原语（masked pooling / 缺失感知嵌入 / 可用性约束注意力）。

契约依据：docs/问题二实施步骤.md §1.3/§3 + 建模方案 §6.1/§6.2。
- masked_pool：keep 位均值；全 keep=0 → 稳定零向量（由门控/δ 表达缺失）。
- MissingEmbedding：z = p_i + x_eff + (1−δ)·e^m，其中 x_eff = where(δ=1, x, 0)。
  这是 δ·x 的数值安全实现：IEEE 里 0×NaN=NaN，乘法门控挡不住 NaN 占位，
  where 保证 a=0 位的 x（含 NaN）物理上不进入计算图——§1.3"缺失位不得读取"
  的强化版。special/padding 位 δ=0，同样只得到 p+e，但它们另被结构 mask 排除。
- AvailabilityAttention：证据 mask KVM_j = s_j·δ_j；K 侧 masked_fill(-inf)、
  V 侧 where 置零（双侧防 NaN）；Σ KVM=0 的行 softmax 退化为 NaN，
  nan_to_num 置 0 权重 → 输出严格零（等价"跳过 softmax"，契约 B3 单测）。
  缺失位不能作证据，但允许作 query 接受上下文补偿（方案 §6.2）。
- 组合不变性（§1.3 核心单测，见 tests/test_p2_m1e_modules.py）：
  x 在 a=0 位（content 缺失 / special / padding）任意扰动——含 NaN 注入——
  组合层输出逐位不变；a=1 位扰动必须改变输出（防恒等假阳性）。
"""
from __future__ import annotations

import math

import torch
from torch import nn


def count_params(module: nn.Module) -> int:
    """参数量报告（契约 §3"报告各级参数量"）。"""
    return sum(p.numel() for p in module.parameters())


def masked_pool(h: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """(N,T,D)+(N,T)→(N,D)：keep 位均值；excluded 位不读取（where 实现，免疫 NaN）；
    整段 keep=0 → 零向量。"""
    keep_b = keep > 0.5
    h_eff = torch.where(keep_b[..., None], h,
                        torch.zeros((), dtype=h.dtype, device=h.device))
    denom = keep_b.to(h.dtype).sum(dim=1).clamp(min=1.0)
    return h_eff.sum(dim=1) / denom[..., None]


class MissingEmbedding(nn.Module):
    """z = p_i + x_eff + (1−δ)·e^(m)；p 为 50 位可学习位置嵌入，e^m 按模态独立。"""

    def __init__(self, d: int = 64, t_grid: int = 50, n_mod: int = 3):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(t_grid, d))
        self.miss = nn.Parameter(torch.zeros(n_mod, d))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.miss, std=0.02)

    def forward(self, x: torch.Tensor, a: torch.Tensor, mod_idx: int) -> torch.Tensor:
        a_b = a > 0.5
        x_eff = torch.where(a_b[..., None], x,
                            torch.zeros((), dtype=x.dtype, device=x.device))
        e = self.miss[mod_idx]
        return self.pos[None, :, :] + x_eff + (~a_b).to(x.dtype)[..., None] * e


class AvailabilityAttention(nn.Module):
    """单向跨模态注意力：Q←源模态序列，K/V←证据模态序列，d=64/h=2（契约默认）。"""

    def __init__(self, d: int = 64, h: int = 2):
        super().__init__()
        assert d % h == 0
        self.d, self.h, self.dh = d, h, d // h
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)

    def forward(self, q_seq: torch.Tensor, kv_seq: torch.Tensor,
                q_mask: torch.Tensor, kv_avail: torch.Tensor) -> torch.Tensor:
        """q_mask：(N,Tq) 结构 mask（padding 行输出无意义、下游池化排除）；
        kv_avail：(N,Tkv) = s·δ，证据可用性。返回 (N,Tq,D)，全屏蔽样本行严格零。"""
        n, tq = q_seq.shape[:2]
        nkv = kv_seq.shape[1]
        q = self.wq(q_seq).view(n, tq, self.h, self.dh).transpose(1, 2)
        k = self.wk(kv_seq).view(n, nkv, self.h, self.dh).transpose(1, 2)
        v = self.wv(kv_seq).view(n, nkv, self.h, self.dh).transpose(1, 2)
        avail = kv_avail > 0.5
        v = torch.where(avail[:, None, :, None], v,
                        torch.zeros((), dtype=v.dtype, device=v.device))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)      # (N,h,Tq,Tkv)
        scores = scores.masked_fill(~avail[:, None, None, :], float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out = (attn @ v).transpose(1, 2).reshape(n, tq, self.d)
        out = self.wo(out)
        # Σ KVM=0 的样本：attn 已为 0，但 wo 的 bias 仍会加非零值——契约要求
        # "该方向输出直接置零"，故在此显式置零（B3 单测覆盖）
        empty = ~avail.any(dim=1)                                  # (N,)
        return torch.where(empty[:, None, None],
                           torch.zeros((), dtype=out.dtype, device=out.device), out)


class ModalityEncoder(nn.Module):
    """单模态入口：raw 投影 → 缺失感知嵌入（B0 只用 proj+masked_pool，
    B1+ 复用本件的嵌入输出；参数量随阶梯报告）。"""

    def __init__(self, d_in: int, d: int = 64, t_grid: int = 50, mod_idx: int = 0):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.emb = MissingEmbedding(d, t_grid)
        self.mod_idx = mod_idx

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.emb(self.proj(x), a, self.mod_idx)
