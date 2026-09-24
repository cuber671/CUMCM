"""P2 M1-E 单测：模型掩码原语与 §1.3 扰动不变性（M1 收口核心测试）。

覆盖审阅清单：
- masked_pool：排除 special/padding；全不可用 → 稳定零向量；excluded 位 NaN 不读取
- AvailabilityAttention：KVM 证据屏蔽正确（对照手算 reference）；
  Σ KVM=0 行输出严格零且无 NaN/Inf（契约 B3 单测）
- MissingEmbedding：公式逐位；模态独立 e^m
- 组合扰动不变性：扰动 a=0 位的 x（content 缺失/special/padding × 两模态），
  输出逐位不变；NaN 注入同样不变；a=1 位扰动必须改变输出（防恒等假阳性）
- 参数量公式核对（契约 §3）
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.modules import (AvailabilityAttention, MissingEmbedding,
                            ModalityEncoder, count_params, masked_pool)

torch.manual_seed(0)
N, T, D = 5, 12, 16
# 结构：0=CLS，1..9=content（sep_pos=10），10=SEP，11=padding
CONTENT = torch.zeros(N, T, dtype=torch.bool)
CONTENT[:, 1:10] = True
SPECIAL = torch.zeros(N, T, dtype=torch.bool)
SPECIAL[:, 0] = True
SPECIAL[:, 10] = True
PADDING = ~CONTENT & ~SPECIAL


def make_availability(seed=1, empty_rows=(0,)):
    """content 上的可用性 a；指定行整段不可用（附件2 有 110/15/28 条 vision 全零）。"""
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(N, T, generator=g) > 0.4).float() * CONTENT.float()
    for r in empty_rows:
        a[r, :] = 0.0
    return a


def make_x(d_in, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, T, d_in, generator=g)


# ---- masked_pool ----

def test_masked_pool_excludes_and_handles_empty():
    h = torch.randn(N, T, D)
    out = masked_pool(h, CONTENT)
    manual = h[:, 1:10, :].mean(1)
    assert torch.allclose(out, manual, atol=1e-6)
    empty = torch.zeros(N, T)
    out0 = masked_pool(h, empty)
    assert (out0 == 0).all()                       # 全不可用 → 稳定零向量


def test_masked_pool_nan_at_excluded_positions():
    h = torch.randn(N, T, D)
    h[:, 0, :] = float("nan")                      # CLS 位注 NaN
    h[:, 10:, :] = float("nan")                    # SEP+padding 注 NaN
    out = masked_pool(h, CONTENT)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, h[:, 1:10, :].mean(1), atol=1e-6)


# ---- MissingEmbedding ----

def test_missing_embedding_formula():
    emb = MissingEmbedding(d=D, t_grid=T, n_mod=3)
    x = torch.randn(N, T, D)
    a = make_availability()
    z = emb(x, a, mod_idx=2)
    x_eff = x * (a > 0.5).float()[..., None]       # 有限值下与 where 等价
    ref = emb.pos[None] + x_eff + (1 - (a > 0.5).float())[..., None] * emb.miss[2]
    assert torch.allclose(z, ref, atol=1e-6)
    z0 = emb(x, a, mod_idx=0)
    assert not torch.allclose(z, z0)               # e^m 模态独立


# ---- AvailabilityAttention ----

def test_attention_matches_reference():
    """W 全置单位阵 → 输出 = softmax(QKᵀ/√dh)·V；与独立 numpy 参考实现逐行比对。"""
    d_ref, h_ref = 8, 2
    att = AvailabilityAttention(d=d_ref, h=h_ref)
    with torch.no_grad():
        for lin in (att.wq, att.wk, att.wv, att.wo):
            lin.weight.copy_(torch.eye(d_ref))
            lin.bias.zero_()
    g = torch.Generator().manual_seed(4)
    q_seq, kv_seq = torch.randn(N, T, d_ref, generator=g), torch.randn(N, T, d_ref, generator=g)
    kv_avail = make_availability(seed=4)
    out = att(q_seq, kv_seq, CONTENT, kv_avail).detach().numpy()

    def softmax_rows(s):
        e = np.exp(s - s.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    q, k, v = q_seq.numpy(), kv_seq.numpy(), kv_seq.numpy()
    for i in range(N):
        avail = (kv_avail[i] > 0.5).numpy()
        vv = v[i].copy()
        vv[~avail] = 0.0                          # 与模块 V 侧 where 一致
        ref = np.zeros((T, d_ref))
        for h_i in range(h_ref):
            sl = slice(h_i * 4, (h_i + 1) * 4)
            s = q[i][:, sl] @ k[i][:, sl].T / math.sqrt(4)
            s[:, ~avail] = -1e30
            ref[:, sl] = softmax_rows(s) @ vv[:, sl]
        assert np.allclose(out[i], ref, atol=1e-5), i


def test_attention_fully_masked_kv_outputs_zero():
    torch.manual_seed(5)
    att = AvailabilityAttention(d=D, h=2)
    q_seq, kv_seq = torch.randn(N, T, D), torch.randn(N, T, D)
    kv_avail = make_availability(seed=6, empty_rows=(0, 3))
    out = att(q_seq, kv_seq, CONTENT, kv_avail)
    assert torch.isfinite(out).all()
    assert (out[[0, 3]] == 0).all()                # Σ KVM=0 行严格零（契约 B3 单测）
    assert not (out[[1, 2, 4]] == 0).all()         # 其余行非零


def test_attention_nan_at_masked_evidence():
    """被屏蔽证据位的 NaN 不得传播（V 侧 where + K 侧 masked_fill 双防）。
    注入位置动态选取，确保确实是 kv_avail=0 的位（可用位的 NaN 属合法进入）。"""
    torch.manual_seed(7)
    att = AvailabilityAttention(d=D, h=2)
    q_seq = torch.randn(N, T, D)
    kv_seq = torch.randn(N, T, D)
    kv_avail = make_availability(seed=8, empty_rows=())
    for i in (0, 1):
        masked_idx = torch.where(~(kv_avail[i] > 0.5))[0]
        assert len(masked_idx) > 0
        kv_seq[i, masked_idx[0], :] = float("nan")
    out = att(q_seq, kv_seq, CONTENT, kv_avail)
    assert torch.isfinite(out).all()


# ---- 组合扰动不变性（§1.3 核心）----

def build_composite(seed=9):
    torch.manual_seed(seed)
    enc_t = ModalityEncoder(d_in=8, d=D, t_grid=T, mod_idx=0)
    enc_a = ModalityEncoder(d_in=6, d=D, t_grid=T, mod_idx=1)
    att = AvailabilityAttention(d=D, h=2)
    for m in (enc_t, enc_a, att):
        m.eval()
    return enc_t, enc_a, att


def composite_forward(enc_t, enc_a, att, x_t, x_a, a_t, a_a):
    z_t = enc_t(x_t, a_t)
    z_a = enc_a(x_a, a_a)
    h_t = att(z_t, z_a, CONTENT, CONTENT & (a_a > 0.5))
    return torch.cat([masked_pool(h_t, CONTENT), masked_pool(z_t, CONTENT)], dim=-1)


def perturb(x, region_mask, seed=11, nan=False):
    xp = x.clone()
    g = torch.Generator().manual_seed(seed)
    if nan:
        xp[region_mask] = float("nan")
    else:
        xp[region_mask] = 100 * torch.randn(int(region_mask.sum()), x.shape[-1], generator=g)
    return xp


def test_perturbation_invariance_all_regions():
    """扰动 a=0 位的 x（缺失 content / special / padding × 两模态）→ 输出逐位不变。"""
    enc_t, enc_a, att = build_composite()
    a_t = make_availability(seed=12, empty_rows=())
    a_a = make_availability(seed=13, empty_rows=(0,))
    x_t, x_a = make_x(8, 14), make_x(6, 15)
    base = composite_forward(enc_t, enc_a, att, x_t, x_a, a_t, a_a)
    assert torch.isfinite(base).all()

    a_t_b, a_a_b = a_t > 0.5, a_a > 0.5
    regions = {
        "t缺失content位": CONTENT & ~a_t_b,
        "a缺失content位": CONTENT & ~a_a_b,
        "t特殊位": SPECIAL,
        "a特殊位": SPECIAL,
        "t padding位": PADDING,
        "a padding位": PADDING,
    }
    for name, region in regions.items():
        assert region.any(), name
        x2_t = perturb(x_t, region) if name.startswith("t") else x_t
        x2_a = perturb(x_a, region) if name.startswith("a") else x_a
        out = composite_forward(enc_t, enc_a, att, x2_t, x2_a, a_t, a_a)
        assert (out == base).all(), f"{name}: 扰动改变了输出（缺失位泄漏）"
        # NaN 注入同样不变
        x3_t = perturb(x_t, region, nan=True) if name.startswith("t") else x_t
        x3_a = perturb(x_a, region, nan=True) if name.startswith("a") else x_a
        out3 = composite_forward(enc_t, enc_a, att, x3_t, x3_a, a_t, a_a)
        assert (out3 == base).all(), f"{name}: NaN 注入改变了输出"
        assert torch.isfinite(out3).all()


def test_available_position_perturbation_changes_output():
    """a=1 位扰动必须改变输出（证明不变性不是恒等映射假阳性）。"""
    enc_t, enc_a, att = build_composite()
    a_t = make_availability(seed=16, empty_rows=())
    a_a = make_availability(seed=17, empty_rows=(0,))
    x_t, x_a = make_x(8, 18), make_x(6, 19)
    base = composite_forward(enc_t, enc_a, att, x_t, x_a, a_t, a_a)
    region = CONTENT & (a_t > 0.5)
    assert region.any()
    out = composite_forward(enc_t, enc_a, att, perturb(x_t, region), x_a, a_t, a_a)
    assert not torch.allclose(out, base)


# ---- 参数量（契约 §3）----

def test_param_counts():
    att = AvailabilityAttention(d=64, h=2)
    assert count_params(att) == 4 * (64 * 64 + 64)            # wq/wk/wv/wo
    emb = MissingEmbedding(d=64, t_grid=50, n_mod=3)
    assert count_params(emb) == 50 * 64 + 3 * 64              # p + e^m
    enc = ModalityEncoder(d_in=74, d=64, t_grid=50, mod_idx=1)
    assert count_params(enc) == 74 * 64 + 64 + count_params(emb)
