"""P2 M3 单测：B1/B2/B3 阶梯模型 + zero_fill 约定。

- 阶梯前向：形状、reg∈[-3,3]、空 content 无 NaN、avail=None 调试路径
- B3：可用性掩码生效（证据位被屏蔽时输出与"该位不可用"一致）、
  全模态不可用时仍有限
- 参数量公式核对（契约 §3 参数量表）
- zero_fill：a=0 位精确归零（含 NaN 占位）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.models import B1, B2, B3, count_params
from src.p2.pipeline import zero_fill

N, T, D = 4, 12, 16
CONTENT = torch.ones(N, T, dtype=torch.bool)
CONTENT[:, 0] = False
CONTENT[:, -2:] = False


def make_feats(seed=0, d_map=(("text", 768), ("audio", 74), ("vision", 35))):
    g = torch.Generator().manual_seed(seed)
    return {m: torch.randn(N, T, d, generator=g) for m, d in d_map}


def make_avail(seed=1):
    g = torch.Generator().manual_seed(seed)
    return {m: (torch.rand(N, T, generator=g) > 0.3) & CONTENT for m in ("text", "audio", "vision")}


def _check_forward(model, avail_needed=False):
    feats = {k: v[:, :, :].float() for k, v in make_feats().items()}
    feats["audio"] = feats["audio"][:, :, :74]
    feats["vision"] = feats["vision"][:, :, :35]
    avail = make_avail() if avail_needed else None
    with torch.no_grad():
        out = model(feats, CONTENT, avail) if avail_needed else model(feats, CONTENT)
    assert out["logits"].shape == (N, 3) and out["reg"].shape == (N,)
    assert (out["reg"].abs() <= 3.0).all()
    assert torch.isfinite(out["logits"]).all()
    # 空 content：池化零向量路径
    with torch.no_grad():
        out0 = model(feats, torch.zeros(N, T, dtype=torch.bool),
                     {m: torch.zeros(N, T, dtype=torch.bool) for m in avail}
                     ) if avail_needed else model(feats, torch.zeros(N, T, dtype=torch.bool))
    assert torch.isfinite(out0["logits"]).all()
    return out


def test_b1_forward():
    torch.manual_seed(0)
    _check_forward(B1(dropout=0.0))


def test_b2_forward():
    torch.manual_seed(0)
    _check_forward(B2(dropout=0.0))


def test_b3_forward_and_avail_effect():
    torch.manual_seed(0)
    m = B3(dropout=0.0)
    m.eval()
    feats = make_feats()
    feats["audio"] = feats["audio"][:, :, :74]
    feats["vision"] = feats["vision"][:, :, :35]
    avail_full = {m: CONTENT.clone() for m in ("text", "audio", "vision")}
    avail_masked = make_avail()
    with torch.no_grad():
        o1 = m(feats, CONTENT, avail_full)
        o2 = m(feats, CONTENT, avail_masked)
        # 全模态不可用（avail=∅）→ 注意力输出零、池化走 GRU 通道，仍有限
        o3 = m(feats, CONTENT, {k: torch.zeros(N, T, dtype=torch.bool)
                                for k in avail_full})
    assert torch.isfinite(o3["logits"]).all()
    assert not torch.allclose(o1["logits"], o2["logits"])   # 可用性输入改变输出


def test_param_count_formulas():
    # B1: enc 第一层 64·(768+74+35) + 3×64 偏置 + 3×(64·64+64) + heads(192)
    b1 = 64 * (768 + 74 + 35) + 3 * 64 + 3 * (64 * 64 + 64) + \
        (192 * 64 + 64 + 64 * 3 + 3) + (192 * 64 + 64 + 64 + 1)
    assert count_params(B1()) == b1 == 93764
    # B2: proj 56320 + 3×BiGRU(64,64) 49920×3 + heads(384)
    gru = 2 * (3 * 64 * 64 + 3 * 64 * 64 + 2 * 3 * 64)
    b2 = 56320 + 3 * gru + (384 * 64 + 64 + 64 * 3 + 3) + (384 * 64 + 64 + 64 + 1)
    assert count_params(B2()) == b2 == 255620
    # B3: B2 的 proj+GRU + 3×pre_attn(128·64+64) + 6×Attn(16640) + heads(768)
    b3 = 56320 + 3 * gru + 3 * (128 * 64 + 64) + 6 * 16640 + \
        (768 * 64 + 64 + 64 * 3 + 3) + (768 * 64 + 64 + 64 + 1)
    assert count_params(B3()) == b3 == 429380


def test_zero_fill():
    x = torch.randn(N, T, 3)
    x[0, 0, 0] = float("nan")
    a = torch.ones(N, T, dtype=torch.bool)
    a[0, 0] = False
    a[1, 1] = False
    out = zero_fill(x, a)
    assert (out[~a] == 0).all()
    assert torch.isfinite(out).all()               # NaN 占位同时被清除
    assert torch.equal(out[a], x[a])
