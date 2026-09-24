"""P2 M2 单测：B0 模型 + 输入管线（含 M1-B 遗留项闭合）。

- zscore_reset：o=0 位精确置 0；o=1 位 NaN/Inf 抛错（M1-B 闭合）；
  NaN 在 o=0 位被安全清除
- B0 前向：形状、reg∈[-3,3]、全 content=False 无 NaN（池化零向量路径）、参数量
- 训练性：小批量 CE+L1 若干 epoch 后 loss 必须下降（反向传播存在性）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.models import B0, count_params
from src.p2.pipeline import zscore_reset

N, T = 4, 10


def test_zscore_reset_semantics():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, T, 3))
    mean, std = np.zeros(3), np.ones(3) + 0.5
    o = rng.random((N, T)) > 0.3
    out = zscore_reset(x, mean, std, o)
    assert out.dtype == np.float32
    assert (out[~o] == 0).all()                          # o=0 位精确占位零
    assert np.allclose(out[o], (x[o] / std).astype(np.float32))  # o=1 位标准化保留
    assert np.isfinite(out).all()


def test_zscore_reset_rejects_nan_at_observed():
    x = np.zeros((N, T, 3))
    o = np.ones((N, T), dtype=bool)
    x[0, 0, 0] = np.nan
    try:
        zscore_reset(x, np.zeros(3), np.ones(3), o)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "o=1 位 NaN 必须抛错（M1-B 闭合）"
    o2 = np.zeros((N, T), dtype=bool)
    o2[0, 0] = True
    x2 = np.full((N, T, 3), np.nan)
    x2[0, 0] = 1.0
    out = zscore_reset(x2, np.zeros(3), np.ones(3), o2)  # NaN 只在 o=0 位 → 清除
    assert np.isfinite(out).all() and out[0, 0, 0] == 1.0


def test_b0_forward_shapes_and_range():
    torch.manual_seed(0)
    m = B0(dropout=0.0)
    m.eval()
    content = torch.ones(3, T, dtype=torch.bool)
    content[:, -2:] = False
    feats = {"text": torch.randn(3, T, 768), "audio": torch.randn(3, T, 74),
             "vision": torch.randn(3, T, 35)}
    out = m(feats, content)
    assert out["logits"].shape == (3, 3) and out["reg"].shape == (3,)
    assert (out["reg"].abs() <= 3.0).all()               # 3·tanh 值域
    assert torch.isfinite(out["logits"]).all()


def test_b0_all_content_false_no_nan():
    """整批无 content 位：池化走零向量路径，输出仍有限（由偏置驱动）。"""
    torch.manual_seed(1)
    m = B0(dropout=0.0)
    m.eval()
    content = torch.zeros(2, T, dtype=torch.bool)
    feats = {k: torch.randn(2, T, d) for k, d in
             (("text", 768), ("audio", 74), ("vision", 35))}
    out = m(feats, content)
    assert torch.isfinite(out["logits"]).all() and torch.isfinite(out["reg"]).all()


def test_b0_param_count_formula():
    m = B0(d=64, hidden=64, dropout=0.3)
    enc = 768 * 64 + 64 + 74 * 64 + 64 + 35 * 64 + 64
    head = (3 * 64) * 64 + 64 + 64 * 3 + 3 + (3 * 64) * 64 + 64 + 64 * 1 + 1
    assert count_params(m) == enc + head


def test_tiny_overfit_loss_decreases():
    """16 样本全批训练：loss 必须显著下降（反向传播/掩码池化梯度通路存在性）。"""
    torch.manual_seed(2)
    m = B0(dropout=0.0)
    content = torch.ones(16, T, dtype=torch.bool)
    feats = {k: torch.randn(16, T, d) for k, d in
             (("text", 768), ("audio", 74), ("vision", 35))}
    y = torch.randint(0, 3, (16,))
    yr = torch.randn(16).clamp(-3, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        out = m(feats, content)
        loss = ce(out["logits"], y) + (out["reg"] - yr).abs().mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])
