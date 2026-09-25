"""P3 M1 单测：附件4 loader 维度归一、schema 硬校验、掩码推导、预注册口径（真实数据）。"""
from __future__ import annotations

import numpy as np
import pytest

from src.p3.data import (ALIGNED_DIR, FIELD_SHAPES, PREREG, Att4Sample,
                         _as_batch3, att4_masks, load_att4, prereg_summary)

HAVE_ATT4 = ALIGNED_DIR.exists()


def _synth_text_bert(n_content=3):
    """合成合法 text_bert：CLS + content + SEP + padding，token_type 全零。"""
    tb = np.zeros((3, 50), dtype=np.float32)
    tb[0, 0] = 101; tb[1, 0] = 1; tb[0, 1:n_content + 1] = 9999
    tb[1, 1:n_content + 1] = 1
    tb[0, n_content + 1] = 102; tb[1, n_content + 1] = 1
    return tb


class TestLoader:
    def test_batch_dim_normalized(self):
        x = np.zeros((50, 74), dtype=np.float32)
        assert _as_batch3(x, "audio").shape == (1, 50, 74)
        assert _as_batch3(x[None], "audio").shape == (1, 50, 74)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            _as_batch3(np.zeros((49, 74)), "audio")

    def test_field_set_violation_raises(self, monkeypatch, tmp_path):
        import pickle as pkl
        f = tmp_path / "05.pkl"
        with open(f, "wb") as h:
            pkl.dump({"raw_text": "x", "id": "05", "audio": np.zeros((50, 74)),
                      "vision": np.zeros((50, 35)), "text": np.zeros((50, 768)),
                      "text_bert": _synth_text_bert(), "extra": 1}, h)
        monkeypatch.setattr("src.p3.data.ALIGNED_DIR", tmp_path)
        with pytest.raises(ValueError, match="字段集"):
            load_att4(5)

    def test_id_mismatch_raises(self, monkeypatch, tmp_path):
        import pickle as pkl
        f = tmp_path / "05.pkl"
        with open(f, "wb") as h:
            pkl.dump({"raw_text": "x", "id": "06", "audio": np.zeros((50, 74)),
                      "vision": np.zeros((50, 35)), "text": np.zeros((50, 768)),
                      "text_bert": _synth_text_bert()}, h)
        monkeypatch.setattr("src.p3.data.ALIGNED_DIR", tmp_path)
        with pytest.raises(ValueError, match="id"):
            load_att4(5)


class TestMasks:
    def test_masks_on_synthetic(self):
        s = Att4Sample(1, "hello world", np.ones((1, 50, 74), np.float32),
                       np.ones((1, 50, 35), np.float32), np.ones((1, 50, 768)),
                       _synth_text_bert()[None])
        tm, obs = att4_masks(s)
        assert tm.content.sum() == 3 and tm.special.sum() == 2
        assert obs["o_text"].sum() == 3
        # 全 1 特征行 → a/v 全可用；整行置零 → 自然缺失
        assert obs["o_audio"].sum() == 3 and obs["o_vision"].sum() == 3
        s.audio[0, 1] = 0.0
        _, obs2 = att4_masks(s)
        assert not obs2["o_audio"][0, 1] and obs2["o_audio"].sum() == 2


@pytest.mark.skipif(not HAVE_ATT4, reason="附件4 不在本机")
class TestRealAtt4:
    def test_prereg(self):
        got = prereg_summary(load_att4())
        assert got == {k: PREREG[k] for k in got}

    def test_shapes_finite(self):
        for s in load_att4():
            for f, shape in FIELD_SHAPES.items():
                assert getattr(s, f).shape == (1,) + shape
                assert np.isfinite(getattr(s, f)).all()
