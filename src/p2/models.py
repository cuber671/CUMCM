"""P2 模型阶梯（M2+）：B0 掩码池化地板基线；B1–B3 与 MRFN 在后续里程碑扩展。

契约 §3/§5：B0 = 逐模态投影 → masked_pool(content) → concat → 双头 MLP；
无缺失状态输入（地板基线，B3-naive 对照在 §8.5）；回归头 3·tanh ∈ [-3,3]。
"""
from __future__ import annotations

import torch
from torch import nn

from src.p2.modules import AvailabilityAttention, count_params, masked_pool

MODALITIES = ("text", "audio", "vision")


class B0(nn.Module):
    def __init__(self, d_text: int = 768, d_audio: int = 74, d_vision: int = 35,
                 d: int = 64, hidden: int = 64, dropout: float = 0.3, n_cls: int = 3):
        super().__init__()
        self.enc = nn.ModuleList([nn.Linear(d_text, d), nn.Linear(d_audio, d),
                                  nn.Linear(d_vision, d)])
        self.drop = nn.Dropout(dropout)

        def head(out_dim):
            return nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, out_dim))

        self.cls_head = head(n_cls)
        self.reg_head = head(1)

    def forward(self, feats: dict, content: torch.Tensor) -> dict:
        """feats: {name: (N,T,D_in)} 已标准化特征；content: (N,T) bool（结构掩码）。
        返回 {"logits": (N,3), "reg": (N,)∈[-3,3]}。"""
        pooled = [masked_pool(enc(self.drop(feats[name])), content)
                  for name, enc in zip(MODALITIES, self.enc)]
        h = self.drop(torch.cat(pooled, dim=-1))
        return {"logits": self.cls_head(h),
                "reg": 3.0 * torch.tanh(self.reg_head(h).squeeze(-1))}


MODEL_REGISTRY = {"B0": B0}


class _LadderHeads(nn.Module):
    """阶梯共用双头：Linear(in→hidden)-ReLU-Dropout-Linear(hidden→out)。"""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3, n_cls: int = 3):
        super().__init__()
        self.cls_head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, n_cls))
        self.reg_head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> dict:
        return {"logits": self.cls_head(h),
                "reg": 3.0 * torch.tanh(self.reg_head(h).squeeze(-1))}


class B1(nn.Module):
    """各模态独立编码（Linear-ReLU-Dropout-Linear）+掩码池化 → concat → 双头。"""

    def __init__(self, d_text: int = 768, d_audio: int = 74, d_vision: int = 35,
                 d: int = 64, hidden: int = 64, dropout: float = 0.3, n_cls: int = 3):
        super().__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Linear(din, d), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(d, d))
            for din in (d_text, d_audio, d_vision)])
        self.drop = nn.Dropout(dropout)
        self.heads = _LadderHeads(3 * d, hidden, dropout, n_cls)

    def forward(self, feats: dict, content: torch.Tensor, avail: dict | None = None) -> dict:
        pooled = [masked_pool(enc(self.drop(feats[name])), content)
                  for name, enc in zip(MODALITIES, self.enc)]
        return self.heads(self.drop(torch.cat(pooled, dim=-1)))


class _ProjBiGRU(nn.Module):
    """proj(d_in→d) → 双向 GRU(d, hidden=d/向) → 输出 2d。"""

    def __init__(self, d_in: int, d: int = 64, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.proj(x))
        return out                                # (N,T,2d)


class B2(nn.Module):
    """B1 模态内换双向 GRU（hidden 64/向）→ 掩码池化 → concat(384) → 双头。
    输入按约定零填充后，padding/特殊位全零、不携带样本信息（无 pack 需求）。"""

    def __init__(self, d_text: int = 768, d_audio: int = 74, d_vision: int = 35,
                 d: int = 64, hidden: int = 64, dropout: float = 0.3, n_cls: int = 3):
        super().__init__()
        self.enc = nn.ModuleList([_ProjBiGRU(din, d, dropout)
                                  for din in (d_text, d_audio, d_vision)])
        self.drop = nn.Dropout(dropout)
        self.heads = _LadderHeads(3 * 2 * d, hidden, dropout, n_cls)

    def forward(self, feats: dict, content: torch.Tensor, avail: dict | None = None) -> dict:
        pooled = [masked_pool(enc(self.drop(feats[name])), content)
                  for name, enc in zip(MODALITIES, self.enc)]
        return self.heads(self.drop(torch.cat(pooled, dim=-1)))


class B3(nn.Module):
    """B2 + 6 方向跨模态注意力（KVM_j = s_j·δ_j 屏蔽 K/V，Q 侧 content mask；
    方向 t←a、t←v、a←t、a←v、v←t、v←a）。
    每模态表示 = [BiGRU 输出(128) ｜ 两个方向注意力输出(64×2)] → 池化 → concat(768) → 双头。
    forward 必须传 avail = {模态: (N,T) bool 可用性}（clean = o；掩码评测 = o·(1−b)）。"""

    DIRS = (("text", "audio"), ("text", "vision"), ("audio", "text"),
            ("audio", "vision"), ("vision", "text"), ("vision", "audio"))

    def __init__(self, d_text: int = 768, d_audio: int = 74, d_vision: int = 35,
                 d: int = 64, hidden: int = 64, dropout: float = 0.3, n_cls: int = 3,
                 n_heads: int = 2):
        super().__init__()
        self.enc = nn.ModuleList([_ProjBiGRU(din, d, dropout)
                                  for din in (d_text, d_audio, d_vision)])
        self.pre_attn = nn.ModuleList([nn.Linear(2 * d, d) for _ in MODALITIES])
        self.attn = nn.ModuleList([AvailabilityAttention(d, n_heads) for _ in self.DIRS])
        self.drop = nn.Dropout(dropout)
        self.heads = _LadderHeads(3 * (2 * d + 2 * d), hidden, dropout, n_cls)

    def forward(self, feats: dict, content: torch.Tensor, avail: dict | None = None) -> dict:
        if avail is None:                     # 仅调试用；主实验必传真实可用性
            avail = {m: content for m in MODALITIES}
        E = {name: enc(self.drop(feats[name]))
             for name, enc in zip(MODALITIES, self.enc)}
        A = {name: self.pre_attn[i](E[name])          # 128 → 64（注意力工作空间）
             for i, name in enumerate(MODALITIES)}
        pooled = []
        for m in MODALITIES:
            parts = [E[m]]
            for i, (qm, kvm) in enumerate(self.DIRS):
                if qm == m:
                    parts.append(self.attn[i](A[qm], A[kvm], content, avail[kvm]))
            pooled.append(masked_pool(self.drop(torch.cat(parts, dim=-1)), content))
        return self.heads(self.drop(torch.cat(pooled, dim=-1)))


MODEL_REGISTRY.update({"B1": B1, "B2": B2, "B3": B3})


def build_model(name: str, **kw) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 {name}；可用 {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kw)
