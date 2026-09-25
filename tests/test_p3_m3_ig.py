"""P3 M3 单测：IG 数学（线性完备性/积分误差）、架构级梯度阻断、基线构造、alpha 词聚合。

第二轮审阅修正补：三子词样本的 alpha 聚合语义（质量守恒求和）、avail≡0 退化实证、
中点求积。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.p3.ig import (N_STEPS, aggregate_words, av_baseline, ig_modality,
                       p1_word_grid)
from src.p3.shapley import MODS, SampleTensors


def _st(o_vision_zero: bool = False) -> SampleTensors:
    tb = np.zeros((1, 3, 50), dtype=np.int64)
    tb[0, 0, 0] = 101; tb[0, 1, 0] = 1
    tb[0, 0, 1:4] = 9999; tb[0, 1, 1:4] = 1
    tb[0, 0, 4] = 102; tb[0, 1, 4] = 1
    content = np.zeros((1, 50), bool); content[0, 1:4] = True
    o_v = content.copy()
    if o_vision_zero:
        o_v[:] = False
    return SampleTensors(
        n=1, text_bert=tb, content=content,
        o={"text": content.copy(), "audio": content.copy(), "vision": o_v},
        audio=np.ones((1, 50, 74), np.float32),
        vision=np.ones((1, 50, 35), np.float32), raw_text="hello bright world")


class _LinearFuse(nn.Module):
    """接口同 MRFN 的线性对照模型：logits/reg = 各模态特征线性组合（IG 对线性精确）。"""

    def __init__(self):
        super().__init__()
        self.wt = nn.Parameter(torch.ones(768))
        self.wa = nn.Parameter(torch.full((74,), 2.0))
        self.wv = nn.Parameter(torch.full((35,), -1.0))
        self.head = nn.Linear(3, 3, bias=False)

    def forward(self, feats, content, avail):
        pooled = {m: (feats[m] * content[..., None]).sum(1) for m in MODS}
        v3 = torch.stack([pooled["text"] @ self.wt,
                          pooled["audio"] @ self.wa,
                          pooled["vision"] @ self.wv], dim=1)
        return {"logits": self.head(v3), "reg": v3.sum(dim=1, keepdim=True)}


class TestIGMath:
    def test_completeness_linear_exact(self):
        """中点 64 步在线性模型上完备（积分对常值被积函数精确）。"""
        torch.manual_seed(0)
        st = _st()
        model = _LinearFuse()
        x = {"text": torch.rand(1, 50, 768), "audio": torch.from_numpy(st.audio),
             "vision": torch.from_numpy(st.vision)}
        base = {m: torch.zeros_like(x[m]) for m in MODS}
        avail = {m: torch.from_numpy(st.o[m]) for m in MODS}
        ig = ig_modality(model, st, "cpu", 0, base, x, avail)
        content_t = torch.from_numpy(st.content)
        with torch.no_grad():
            fx = model(x, content_t, avail); f0 = model(base, content_t, avail)
        for o in ("cls", "reg"):
            tot = sum(float(ig[m][o].sum()) for m in MODS)
            df = float(fx["logits"][0, 0] - f0["logits"][0, 0]) if o == "cls" \
                else float(fx["reg"][0] - f0["reg"][0])
            assert abs(tot - df) <= 1e-5 * max(abs(df), 1.0), (o, tot, df)

    def test_structural_positions_zero_ig(self):
        """线性对照：content 池化 → special/padding 位 IG 恒零（架构级）。"""
        torch.manual_seed(0)
        st = _st()
        model = _LinearFuse()
        x = {"text": torch.rand(1, 50, 768), "audio": torch.from_numpy(st.audio),
             "vision": torch.from_numpy(st.vision)}
        base = {m: torch.zeros_like(x[m]) for m in MODS}
        avail = {m: torch.from_numpy(st.o[m]) for m in MODS}
        ig = ig_modality(model, st, "cpu", 0, base, x, avail)
        for m in MODS:
            for o in ("cls", "reg"):
                assert float(np.abs(ig[m][o][~st.content[0]]).max()) == 0.0

    def test_empty_avail_degenerate_ig(self):
        """avail≡0（= v(∅) 同构路径）→ IG 全零：平凡博弈、全哑玩家——
        第二轮审阅修正的设计依据（故 IG 基线固定 avail=o）。"""
        torch.manual_seed(0)
        st = _st()
        model = _LinearFuse()          # 即便对无架构阻断的线性模型也成立：
        # LinearFuse 用 content 池化（等价 avail），avail 不进模型；改用真 MRFN 验证架构级
        from src.p2.models import MODEL_REGISTRY
        m2 = MODEL_REGISTRY["MRFN"](dropout=0.0).eval()
        x = {"text": torch.rand(1, 50, 768), "audio": torch.rand(1, 50, 74),
             "vision": torch.rand(1, 50, 35)}
        base = {m: torch.zeros_like(x[m]) for m in MODS}
        z0 = {m: torch.zeros_like(torch.from_numpy(st.o[m])) for m in MODS}
        ig = ig_modality(m2, st, "cpu", 0, base, x, z0, n_steps=4)
        assert all(float(np.abs(ig[m][o]).max()) == 0.0
                   for m in MODS for o in ("cls", "reg"))


class TestArchitecturalBlocking:
    def test_real_mrfn_zero_grad_when_vision_unavailable(self):
        """#13 同构：o_vision 全零 → ∂F/∂vision ≡ 0（架构级，随机权重即可验证）。"""
        from src.p2.models import MODEL_REGISTRY
        torch.manual_seed(0)
        st = _st(o_vision_zero=True)
        model = MODEL_REGISTRY["MRFN"](dropout=0.0).eval()
        feats = {"text": torch.rand(1, 50, 768), "audio": torch.rand(1, 50, 74),
                 "vision": torch.rand(1, 50, 35, requires_grad=True)}
        avail = {m: torch.from_numpy(st.o[m]) for m in MODS}
        out = model(feats, torch.from_numpy(st.content), avail)
        (out["logits"].sum() + out["reg"].sum()).backward()
        assert feats["vision"].grad is not None
        assert float(feats["vision"].grad.abs().max()) == 0.0


class TestBaselines:
    def test_av_baseline_values(self):
        stats = {"audio": {"mean": np.full(74, 2.0), "std": np.full(74, 0.5)}}
        z = av_baseline("zero", "audio", stats)
        assert z.shape == (50, 74) and np.allclose(z, -4.0)      # (0−2)/0.5
        assert np.allclose(av_baseline("missing", "audio", stats), 0.0)
        assert np.allclose(av_baseline("mean", "audio", stats), 0.0)  # 与 missing 重合留痕

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError):
            av_baseline("noise", "audio", {"audio": {"mean": np.zeros(74),
                                                     "std": np.ones(74)}})


class TestWordAggregation:
    def test_p1_word_grid_alignment(self):
        """P1 build_grid 映射：hello / world 两词，alpha 词内均匀。"""
        wg = p1_word_grid("hello world")
        wbp, alpha = wg["word_by_position"], wg["alpha"]
        assert wbp[0] is None and wbp[3] is None       # CLS 与其后 padding
        assert wbp[1] == 0 and wbp[2] == 1
        assert np.isclose(alpha[1], 1.0) and np.isclose(alpha[2], 1.0)

    def test_multi_subword_word_alpha_and_sum(self):
        """多子词样本（审阅要求）：alpha=1/n 逐片均匀；词级归因 = 片求和（质量守恒），
        不是均值——求和保证 Σ_w A_w = Σ_j a_j（词级完备性）。"""
        from src.p1.grid import build_grid
        wg = p1_word_grid("unbelievably")              # un ##bel ##ie ##va ##bly（5 片）
        n = wg["n_pieces"][0]
        assert n == 5
        assert np.allclose([wg["alpha"][j] for j in range(1, 1 + n)], 1 / n)
        pos = np.zeros(50)
        pos[1:1 + n] = [0.1, 0.2, 0.3, 0.4, 0.5]
        content = np.zeros((1, 50), bool); content[0, 1:1 + n] = True
        agg = aggregate_words(pos, wg, content)
        assert agg["words"] == {0: pytest.approx(1.5)}   # 求和（1.5），非均值（0.3）
        assert agg["structural_leak"] == 0.0 and agg["n_words"] == 1

    def test_aggregate_words_defensive(self):
        wg = p1_word_grid("hello world")
        pos = np.arange(50, dtype=float)
        content = np.zeros((1, 50), bool); content[0, 1:3] = True
        agg = aggregate_words(pos, wg, content)
        assert agg["words"] == {0: 1.0, 1: 2.0}
        assert agg["structural_leak"] == float(pos[~content[0]].sum())
        assert agg["n_words"] == 2
