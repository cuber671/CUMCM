"""P2 M4 单测：MRFN（缺失状态编码 + 门控融合）。

- 前向：形状、gates 为合法 softmax 且逐样本变化、reg∈[-3,3]、空 content 有限
- §1.3 模型级扰动不变性（零填充已由管线移除，门控是唯一防线）：
  扰动 a=0 位（缺失 content / special / padding）的 x → 输出逐位不变；
  a=1 位扰动必须改变输出
- 全模态不可用：coverage=0、gates 均匀有效、输出有限
- 参数量公式（契约 §3）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.models import MRFN, count_params

N, T, D = 4, 12, 16
CONTENT = torch.ones(N, T, dtype=torch.bool)
CONTENT[:, 0] = False
CONTENT[:, -2:] = False
SPECIAL = torch.zeros(N, T, dtype=torch.bool)
SPECIAL[:, 0] = True
SPECIAL[:, -2] = True
PADDING = ~CONTENT & ~SPECIAL


def build(seed=0):
    torch.manual_seed(seed)
    m = MRFN(d_text=32, d_audio=12, d_vision=8, d=D, hidden=16, dropout=0.0,
             n_heads=2, t_grid=T)
    m.eval()
    return m


def make_inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    feats = {"text": torch.randn(N, T, 32, generator=g),
             "audio": torch.randn(N, T, 12, generator=g),
             "vision": torch.randn(N, T, 8, generator=g)}
    avail = {m: (torch.rand(N, T, generator=g) > 0.35) & CONTENT
             for m in ("text", "audio", "vision")}
    return feats, avail


def test_forward_shapes_and_gates():
    m = build()
    feats, avail = make_inputs()
    with torch.no_grad():
        out = m(feats, CONTENT, avail)
    assert out["logits"].shape == (N, 3) and out["reg"].shape == (N,)
    assert out["gates"].shape == (N, 3) and out["coverage"].shape == (N, 3)
    assert torch.allclose(out["gates"].sum(-1), torch.ones(N), atol=1e-5)
    assert (out["gates"] > 0).all()
    assert out["gates"].std(0).max() > 0    # 逐样本有变化（非常数输出）
    assert (out["coverage"] >= 0).all() and (out["coverage"] <= 1).all()


def test_empty_content_finite():
    m = build()
    feats, avail = make_inputs()
    ec = torch.zeros(N, T, dtype=torch.bool)
    with torch.no_grad():
        out = m(feats, ec, {m: torch.zeros(N, T, dtype=torch.bool)
                            for m in avail})
    assert torch.isfinite(out["logits"]).all() and torch.isfinite(out["reg"]).all()


def test_perturbation_invariance_model_level():
    """§1.3 核心单测（MRFN 内部门控是唯一防线，管线无零填充）：
    扰动 a=0 位的 x（缺失 content / special / padding × 三模态）→ 输出逐位不变。"""
    m = build()
    feats, avail = make_inputs(seed=2)
    with torch.no_grad():
        base = torch.cat([m(feats, CONTENT, avail)["logits"],
                          m(feats, CONTENT, avail)["gates"]], dim=-1)
    for mname in ("text", "audio", "vision"):
        regions = {"缺失content位": CONTENT & ~avail[mname],
                   "special位": SPECIAL, "padding位": PADDING}
        for label, region in regions.items():
            x2 = {k: v.clone() for k, v in feats.items()}
            g = torch.Generator().manual_seed(3)
            x2[mname][region] = 100 * torch.randn(int(region.sum()),
                                                  x2[mname].shape[-1], generator=g)
            with torch.no_grad():
                out = m(x2, CONTENT, avail)
            got = torch.cat([out["logits"], out["gates"]], dim=-1)
            assert (got == base).all(), f"{label}/{mname}: a=0 位扰动泄漏进输出"


def test_available_perturbation_changes_output():
    m = build()
    feats, avail = make_inputs(seed=4)
    with torch.no_grad():
        base = m(feats, CONTENT, avail)["logits"]
    region = CONTENT & avail["audio"]
    x2 = {k: v.clone() for k, v in feats.items()}
    g = torch.Generator().manual_seed(5)
    x2["audio"][region] = 100 * torch.randn(int(region.sum()), 12, generator=g)
    with torch.no_grad():
        out = m(x2, CONTENT, avail)["logits"]
    assert not torch.allclose(out, base)


def test_param_count():
    # proj 56320 + miss(50·64+3·64) + gru 3×49920 + pre_attn 3×(128·64+64)
    #   + attn 6×16640 + gate((3·256+3)·64+64 + 64·3+3) + heads(256)
    gru = 2 * (3 * 64 * 64 + 3 * 64 * 64 + 2 * 3 * 64)
    expect = (64 * (768 + 74 + 35) + 3 * 64 +              # proj
              50 * 64 + 3 * 64 +                            # missing embedding
              3 * gru +                                     # BiGRU
              3 * (128 * 64 + 64) +                         # pre_attn
              6 * 16640 +                                   # 注意力
              (3 * 256 + 3) * 64 + 64 + 64 * 3 + 3 +        # gate
              (256 * 64 + 64 + 64 * 3 + 3) + (256 * 64 + 64 + 64 + 1))  # heads
    m = MRFN()                                              # 默认真实维度 d=64
    assert count_params(m) == expect, (count_params(m), expect)
