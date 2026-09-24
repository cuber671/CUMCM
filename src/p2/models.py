"""P2 模型阶梯（M2+）：B0 掩码池化地板基线；B1–B3 与 MRFN 在后续里程碑扩展。

契约 §3/§5：B0 = 逐模态投影 → masked_pool(content) → concat → 双头 MLP；
无缺失状态输入（地板基线，B3-naive 对照在 §8.5）；回归头 3·tanh ∈ [-3,3]。
"""
from __future__ import annotations

import torch
from torch import nn

from src.p2.modules import count_params, masked_pool

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


def build_model(name: str, **kw) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 {name}；可用 {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kw)
