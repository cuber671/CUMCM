"""P3 M4 单测：删除算子语义、证据序、线性模型上删除曲线/LOO↔IG 精确一致性。"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.p3.fidelity import (available_positions, deletion_insertion,
                             evidence_ranking, removal_state)
from src.p3.shapley import MODS, SampleTensors


def _st() -> SampleTensors:
    tb = np.zeros((1, 3, 50), dtype=np.int64)
    tb[0, 0, 0] = 101; tb[0, 1, 0] = 1
    tb[0, 0, 1:4] = 9999; tb[0, 1, 1:4] = 1
    tb[0, 0, 4] = 102; tb[0, 1, 4] = 1
    content = np.zeros((1, 50), bool); content[0, 1:4] = True
    return SampleTensors(
        n=1, text_bert=tb, content=content,
        o={"text": content.copy(), "audio": content.copy(),
           "vision": content & np.array([False, True, True, False] + [False] * 46)},
        audio=np.ones((1, 50, 4), np.float32),
        vision=np.ones((1, 50, 3), np.float32), raw_text="hello bright world")


class TestRemovalState:
    def test_text_replacement(self):
        st = _st()
        ids, _ = removal_state(st, {("text", 2)})
        assert ids[0, 0, 2] == 103 and ids[0, 0, 1] == 9999 and ids[0, 0, 3] == 9999
        assert ids[0, 0, 0] == 101 and ids[0, 0, 4] == 102      # 结构不动
        assert ids[0, 1].tolist() == st.text_bert[0, 1].tolist()

    def test_av_avail_flip_only(self):
        st = _st()
        _, avail = removal_state(st, {("audio", 1)})
        assert not avail["audio"][0, 1] and avail["audio"][0, 2]
        assert np.array_equal(avail["vision"], st.o["vision"])

    def test_unavailable_raises(self):
        st = _st()
        with pytest.raises(ValueError, match="不可用"):
            removal_state(st, {("vision", 3)})            # o_v=0
        with pytest.raises(ValueError, match="不可用"):
            removal_state(st, {("text", 0)})              # CLS

    def test_all_removed_equals_empty_state(self):
        st = _st()
        ids, avail = removal_state(st, set(available_positions(st)))
        assert (ids[0, 0, 1:4] == 103).all()               # text 全 [MASK]
        assert all(not avail[m].any() for m in MODS)


class TestRanking:
    def test_only_available_desc(self):
        st = _st()
        ig = {"text": np.arange(50, dtype=float),
              "audio": np.zeros(50), "vision": np.zeros(50)}
        rank = evidence_ranking(ig, st)
        assert [r[0] for r in rank[:3]] == ["text"] * 3
        assert [r[1] for r in rank[:3]] == [3, 2, 1]        # 降序
        assert ("vision", 3) not in [(m, j) for m, j, _ in rank]


class _WeightedLinear(nn.Module):
    """logits = [Σz, −Σz, z_text]，z_m = Σ_j avail_j · w · feats_j（删除序解析可测）。"""

    def __init__(self, wt, wa, wv):
        super().__init__()
        for name, w in (("wt", wt), ("wa", wa), ("wv", wv)):
            self.register_buffer(name, torch.tensor(w, dtype=torch.float32))

    def forward(self, feats, content, avail):
        pooled = {}
        for m, w in (("text", self.wt), ("audio", self.wa), ("vision", self.wv)):
            a = avail[m][..., None].to(feats[m].dtype)
            pooled[m] = (feats[m] * a).sum(1) @ w    # 时间池化后加权 → (1,)
        z = torch.stack([pooled[m] for m in MODS], dim=1)
        return {"logits": torch.stack([z.sum(1), -z.sum(1), z[:, 0]], 1),
                "reg": z.sum(1, keepdim=True)}


class _EyeBert(nn.Module):
    """恒等编码：ids (1,3,50) → one-hot(128) 嵌入；content id 11/12/13 各占独立维。"""

    def __init__(self, d=128):
        super().__init__()
        self.d = d

    def forward(self, ids):
        return torch.nn.functional.one_hot(ids[:, 0], self.d).float()


class TestCurvesLinear:
    def _setup(self):
        # content ids 11/12/13（独立维）；CLS=101/SEP=102/MASK=103 维权重全 0
        tb = np.zeros((1, 3, 50), dtype=np.int64)
        tb[0, 0, 0] = 101; tb[0, 1, 0] = 1
        tb[0, 0, 1:4] = [11, 12, 13]; tb[0, 1, 1:4] = 1
        tb[0, 0, 4] = 102; tb[0, 1, 4] = 1
        content = np.zeros((1, 50), bool); content[0, 1:4] = True
        st = SampleTensors(
            n=1, text_bert=tb, content=content,
            o={"text": content.copy(), "audio": content.copy(),
               "vision": content & np.array([False, True, True, False] + [False] * 46)},
            audio=np.eye(4, dtype=np.float32)[np.arange(50) % 4][None],   # 位置型
            vision=np.eye(3, dtype=np.float32)[np.arange(50) % 3][None],
            raw_text="x")
        wt = np.zeros(128, np.float32); wt[[11, 12, 13]] = [0.9, 0.2, 0.4]
        wa = np.array([0.0, 0.1, 0.3, 0.0], dtype=np.float32)
        wv = np.array([0.0, 0.15, 0.2], dtype=np.float32)
        ens = [(_WeightedLinear(wt, wa, wv).eval(), _EyeBert())]

        def pad(w, d):
            return np.concatenate([np.asarray(w, float), np.zeros(50 - d)])
        # IG（线性精确）：ig[j] = w[id_j]（text）/ w[j%D]（a/v 位置型特征）
        ig = {"text": pad([0, .9, .2, .4], 4),
              "audio": pad(wa[np.arange(50) % 4], 50),
              "vision": pad(np.where(np.arange(50) % 3 < 3,
                                     wv[np.minimum(np.arange(50) % 3, 2)], 0), 50)}
        ig["vision"] = np.zeros(50)
        ig["vision"][[1, 2]] = [wv[1], wv[2]]   # 位置 1→e1(wv[1]), 2→e2(wv[2])
        return st, ens, ig

    def test_deletion_attr_beats_random(self):
        st, ens, ig = self._setup()
        res = deletion_insertion(ens, st, ig, "cpu", 0, 1,
                                 fractions=(0.25, 0.5), n_rand=5)
        assert res["aopc"] > 0                                   # 归因序删除更快
        assert res["del_auc_attr"] < res["del_auc_rand"]
        assert res["ins_auc_attr"] > res["ins_auc_rand"]
        assert res["compr_at_20"] > 0                            # 删 top20% 掉概率

    def test_loo_ig_exact_on_linear(self):
        """线性模型：单点删除 Δp = w[id_j] ∝ IG_j → Spearman = 1。"""
        from src.p3.fidelity import loo_consistency
        st, ens, ig = self._setup()
        out = loo_consistency(ens, st, ig, "cpu", 0)
        assert out["text"]["rho"] == pytest.approx(1.0)
        assert out["vision"]["rho"] == pytest.approx(1.0)
        assert out["audio"]["rho"] == pytest.approx(1.0)
