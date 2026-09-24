"""P2 M5 单测：消融变体 + 前向填充。

- forward_fill：段内复制最后可用值；前无可用 → 0；全不可用 → 全零
- 消融开关语义：
    noState   → 输入按 B3 语义零填充（可用性变化仍影响输出，经由零填充）
    noMaskAttn(+state) → 注意力不屏蔽证据，但嵌入仍用 δ → 输出随 avail 变化
    noState+noMaskAttn → avail 对输出完全无影响（结构不变性）
    noGate    → 无 gates 输出、heads 走 concat
- 注册表变体可解析；参数量：noState/noGate 严格小于 full，noMaskAttn == full
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.models import MRFN, MODEL_REGISTRY, count_params
from src.p2.pipeline import forward_fill

N, T, D = 3, 10, 16
CONTENT = torch.ones(N, T, dtype=torch.bool)
CONTENT[:, 0] = False
CONTENT[:, -2:] = False


def build(name=None, seed=0, **kw):
    torch.manual_seed(seed)
    m = (MODEL_REGISTRY[name] if name else MRFN)(
        d_text=32, d_audio=12, d_vision=8, d=D, hidden=16, dropout=0.0,
        n_heads=2, t_grid=T, **kw)
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


def test_forward_fill():
    x = torch.tensor([[[10.0], [11.0], [12.0], [13.0], [14.0]]])
    a = torch.tensor([[True, False, False, True, False]])
    out = forward_fill(x, a)
    assert out[0, 0, 0] == 10 and out[0, 1, 0] == 10 and out[0, 2, 0] == 10
    assert out[0, 3, 0] == 13 and out[0, 4, 0] == 13
    a2 = torch.zeros(1, 5, dtype=torch.bool)
    assert (forward_fill(x, a2) == 0).all()          # 前无可用 → 0
    assert torch.equal(forward_fill(x, torch.ones(1, 5, dtype=torch.bool)), x)


def test_variants_forward():
    feats, avail = make_inputs()
    for name in ("MRFN_noState", "MRFN_noMaskAttn", "MRFN_noGate"):
        m = build(name)
        with torch.no_grad():
            out = m(feats, CONTENT, avail)
        assert out["logits"].shape == (N, 3)
        assert torch.isfinite(out["logits"]).all()
        if name == "MRFN_noGate":
            assert "gates" not in out
        else:
            assert out["gates"].shape == (N, 3)
            assert torch.allclose(out["gates"].sum(-1), torch.ones(N), atol=1e-5)


def test_noState_noMaskAttn_noGate_avail_invariant():
    """三开关全关 + 管线预填充：avail 完全不进计算图 → 输出逐位不变。
    （只关 State/MaskAttn 时，门控覆盖率 c_m 仍合法消费 avail——那是设计行为。）"""
    from src.p2.pipeline import zero_fill
    m = build(use_missing_state=False, use_masked_attn=False, use_gate=False)
    feats, avail = make_inputs()
    feats = {k: zero_fill(v, avail[k]) for k, v in feats.items()}   # 管线预填充
    with torch.no_grad():                    # warm-up：避开 cudnn 首调 autotune 浮点差
        m(feats, CONTENT, avail)
    with torch.no_grad():
        o1 = m(feats, CONTENT, avail)
        o2 = m(feats, CONTENT, {k: CONTENT.clone() for k in avail})
    assert "gates" not in o1
    assert torch.equal(o1["logits"], o2["logits"])


def test_gate_coverage_consumes_avail():
    """门控开启时 avail 经覆盖率 c_m 合法进入门控：改变 avail → gates 改变。"""
    m = build(use_missing_state=False, use_masked_attn=False, use_gate=True)
    feats, avail = make_inputs()
    with torch.no_grad():
        g1 = m(feats, CONTENT, avail)["gates"]
        g2 = m(feats, CONTENT, {k: CONTENT.clone() for k in avail})["gates"]
    assert not torch.allclose(g1, g2)


def test_full_mrfn_avail_matters():
    m = build()
    feats, avail = make_inputs()
    with torch.no_grad():
        o1 = m(feats, CONTENT, avail)
        o2 = m(feats, CONTENT, {k: CONTENT.clone() for k in avail})
    assert not torch.allclose(o1["logits"], o2["logits"])


def test_registry_and_param_ordering():
    p_full = count_params(build())
    p_nostate = count_params(build(name="MRFN_noState"))
    p_nomask = count_params(build(name="MRFN_noMaskAttn"))
    p_nogate = count_params(build(name="MRFN_noGate"))
    assert p_nomask == p_full                       # 行为开关不改架构
    assert p_nostate < p_full                       # 少 MissingEmbedding
    # noGate：去 gate(−49.6k) 但 heads 输入 256→768(+65.5k) → 净增，参数量必不等
    assert p_nogate != p_full and p_nogate > 0
    for name in ("MRFN_noState", "MRFN_noMaskAttn", "MRFN_noGate"):
        assert name in MODEL_REGISTRY


def test_s_select_formula():
    """Round 1 鲁棒早停公式（契约 §8 预注册）。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_p2_mrfn import s_select_compute
    assert s_select_compute(1.0, 1.0, 1.0, 1.0) == 1.0
    expect = 0.5 * 0.8 + 0.25 * 0.6 + 0.15 * 0.4 + 0.10 * 0.2
    assert s_select_compute(0.8, 0.6, 0.4, 0.2) == pytest.approx(expect)
