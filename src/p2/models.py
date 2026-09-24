"""P2 模型阶梯（M2+）：B0 掩码池化地板基线；B1–B3 与 MRFN 在后续里程碑扩展。

契约 §3/§5：B0 = 逐模态投影 → masked_pool(content) → concat → 双头 MLP；
无缺失状态输入（地板基线，B3-naive 对照在 §8.5）；回归头 3·tanh ∈ [-3,3]。
"""
from __future__ import annotations

import torch
from torch import nn

from src.p2.modules import (AvailabilityAttention, MissingEmbedding, content_window_gru,
                            count_params, masked_pool)
from src.p2.pipeline import zero_fill

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

    def forward(self, x: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return content_window_gru(self.gru, h, content)   # (N,T,2d)


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
        pooled = [masked_pool(enc(self.drop(feats[name]), content), content)
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
        E = {name: enc(self.drop(feats[name]), content)
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


class MRFN(nn.Module):
    """缺失感知可靠融合网络（方案 §6）= B3 + 显式缺失状态编码 + 可靠性门控。

    §6.1 输入端：z = p_i + x_eff + (1−δ)·e^m（MissingEmbedding 原语，内部门控，
    a=0 位的 x 不进计算图——取代阶梯的 zero_fill 约定，这是与 B3 的受控差异点 1）。
    §6.2 交互：与 B3 相同的 6 方向可用性约束注意力（差异点 2 之前先有 §6.1）。
    §6.3 融合：c_m = Σ s·δ / Σ s（覆盖率高证据）；附件2 无 quality 字段，
    q_m 退化并入特征（接口保留）；g = softmax(MLP([h̃_t;h̃_a;h̃_v;c_t;c_a;c_v]))；
    融合 h = Σ g_m·h̃_m，双头作用于 h。forward 必传 avail = {模态: (N,T) bool}。
    返回附加 "gates" (N,3) 与 "coverage" (N,3) 供 §8.4 门控验证。

    消融开关（§9，默认全开 = MRFN 完整版）：
      use_missing_state=False → 退回零填充输入（B3 语义），无缺失嵌入；
      use_masked_attn=False   → 注意力不屏蔽不可用证据（K/V 仅按 content mask）；
      use_gate=False          → 无门控融合，三模态 h̃ concat → 双头（此时无 gates 输出）。
    """

    def __init__(self, d_text: int = 768, d_audio: int = 74, d_vision: int = 35,
                 d: int = 64, hidden: int = 64, dropout: float = 0.3, n_cls: int = 3,
                 n_heads: int = 2, t_grid: int = 50,
                 use_missing_state: bool = True, use_masked_attn: bool = True,
                 use_gate: bool = True, gate_ln: bool = False,
                 evidence_pool: bool = False):
        super().__init__()
        self.use_missing_state = use_missing_state
        self.use_masked_attn = use_masked_attn
        self.use_gate = use_gate
        self.gate_ln = gate_ln
        self.evidence_pool = evidence_pool
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(din, d), nn.ReLU(),
                                                 nn.Dropout(dropout))
                                   for din in (d_text, d_audio, d_vision)])
        if use_missing_state:
            self.miss = MissingEmbedding(d, t_grid, n_mod=3)
        self.gru = nn.ModuleList([nn.GRU(d, d, batch_first=True, bidirectional=True)
                                  for _ in MODALITIES])
        self.pre_attn = nn.ModuleList([nn.Linear(2 * d, d) for _ in MODALITIES])
        self.attn = nn.ModuleList([AvailabilityAttention(d, n_heads) for _ in B3.DIRS])
        self.drop = nn.Dropout(dropout)
        branch_dim = 4 * d                      # BiGRU(2d) + 两个注意力方向(2×d)
        if use_gate:
            gate_in = 3 * branch_dim + 3        # [h̃_t;h̃_a;h̃_v] + [c_t;c_a;c_v]
            self.gate = nn.Sequential(nn.Linear(gate_in, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 3))
            if gate_ln:                         # Round 4：门控/融合共用归一化表示
                self.ln = nn.ModuleList([nn.LayerNorm(branch_dim) for _ in MODALITIES])
        self.heads = _LadderHeads(branch_dim if use_gate else 3 * branch_dim,
                                  hidden, dropout, n_cls)

    def forward(self, feats: dict, content: torch.Tensor, avail: dict) -> dict:
        z = {}
        for i, m in enumerate(MODALITIES):
            if self.use_missing_state:
                z[m] = self.miss(self.proj[i](feats[m]), avail[m], i)   # §6.1
            else:
                z[m] = self.proj[i](zero_fill(feats[m], avail[m]))      # B3 语义
        E, A = {}, {}
        for i, m in enumerate(MODALITIES):
            e = content_window_gru(self.gru[i], z[m], content)
            E[m] = e
            A[m] = self.pre_attn[i](e)
        branch = {}
        for m in MODALITIES:
            parts = [E[m]]
            for i, (qm, kvm) in enumerate(B3.DIRS):
                if qm == m:
                    kv = avail[kvm] if self.use_masked_attn else content
                    parts.append(self.attn[i](A[qm], A[kvm], content, kv))
            branch[m] = self.drop(torch.cat(parts, dim=-1))      # (N,T,256)
        h_tilde = {}
        for m in MODALITIES:
            # Round 5 证据池化：只在可用位上池化，缺失位不再稀释证据强度
            keep = (content & avail[m]) if self.evidence_pool else content
            h_tilde[m] = masked_pool(branch[m], keep)
        ssum = content.sum(dim=1).clamp(min=1.0)
        cov = torch.stack([(content & avail[m]).sum(dim=1) / ssum for m in MODALITIES], dim=1)
        if self.use_gate:
            if self.gate_ln:                    # Round 4：门控与融合共用归一化表示
                h_tilde = {m: self.ln[i](h_tilde[m]) for i, m in enumerate(MODALITIES)}
            g = torch.softmax(self.gate(torch.cat(list(h_tilde.values()) + [cov], dim=-1)),
                              dim=-1)
            h = sum(g[:, i][:, None] * h_tilde[m] for i, m in enumerate(MODALITIES))
            out = self.heads(h)
            out["gates"] = g
        else:
            out = self.heads(self.drop(torch.cat(list(h_tilde.values()), dim=-1)))
        out["coverage"] = cov
        return out


MODEL_REGISTRY.update({
    "MRFN": MRFN,
    "MRFN_gLN": lambda **kw: MRFN(gate_ln=True, **kw),
    "MRFN_ePool": lambda **kw: MRFN(evidence_pool=True, **kw),
    "MRFN_noState": lambda **kw: MRFN(use_missing_state=False, **kw),
    "MRFN_noMaskAttn": lambda **kw: MRFN(use_masked_attn=False, **kw),
    "MRFN_noGate": lambda **kw: MRFN(use_gate=False, **kw),
})


def build_model(name: str, **kw) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 {name}；可用 {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kw)
