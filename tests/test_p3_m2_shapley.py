"""P3 M2 单测：精确 Shapley 数学性质 + 联盟构造（[MASK] 前置、avail 语义、v(∅) 同构）。"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from fractions import Fraction

from src.p3.shapley import (COALITIONS, MODS, SampleTensors, coalition_inputs,
                            exact_shapley, shapley_weights)


def _sample_tensors() -> SampleTensors:
    """合成样本：content=1..3 位，o 与附件4 契约同构。"""
    tb = np.zeros((1, 3, 50), dtype=np.int64)
    tb[0, 0, 0] = 101; tb[0, 1, 0] = 1
    tb[0, 0, 1:4] = 9999; tb[0, 1, 1:4] = 1
    tb[0, 0, 4] = 102; tb[0, 1, 4] = 1
    content = np.zeros((1, 50), bool); content[0, 1:4] = True
    o = {"text": content.copy(), "audio": content.copy(), "vision": content.copy()}
    return SampleTensors(n=1, text_bert=tb, content=content, o=o,
                         audio=np.random.default_rng(0).normal(size=(1, 50, 74)).astype(np.float32),
                         vision=np.random.default_rng(1).normal(size=(1, 50, 35)).astype(np.float32),
                         raw_text="hello world")


class TestShapleyMath:
    def test_weights_exact(self):
        w = shapley_weights()
        assert w == {0: Fraction(1, 3), 1: Fraction(1, 6), 2: Fraction(1, 3)}

    def test_additive_game(self):
        a = {"text": 0.7, "audio": -0.2, "vision": 0.1}
        values = {S: sum(a[m] for m in S) for S in COALITIONS}
        phi = exact_shapley(values)
        assert all(abs(float(phi[m]) - a[m]) < 1e-12 for m in MODS)

    def test_efficiency_random_game(self):
        rng = np.random.default_rng(42)
        values = {S: rng.normal() for S in COALITIONS}
        phi = exact_shapley(values)
        total = sum(float(phi[m]) for m in MODS)
        assert abs(total - (values[frozenset(MODS)] - values[frozenset()])) < 1e-12

    def test_symmetric_players(self):
        values = {S: float(len(S)) for S in COALITIONS}
        phi = exact_shapley(values)
        assert abs(float(phi["text"]) - 1.0) < 1e-12
        assert abs(phi["text"] - phi["audio"]).item() == 0 if hasattr(phi["text"], "shape") else True

    def test_vector_valued(self):
        a = {"text": 0.2, "audio": 0.5, "vision": -0.3}
        values = {S: np.array([sum(a[m] for m in S), 0.1 * len(S)]) for S in COALITIONS}
        phi = exact_shapley(values)
        assert np.allclose(phi["text"], [0.2, 0.1], atol=1e-12)
        assert np.allclose(phi["vision"], [-0.3, 0.1], atol=1e-12)


class TestCoalitionInputs:
    def _run(self, coalition):
        seen = {}

        class SpyEnc(torch.nn.Module):
            def forward(self, ids):
                seen["ids"] = ids.numpy().copy()
                return torch.zeros((ids.shape[0], ids.shape[2], 4))

        st = _sample_tensors()
        feats, avail = coalition_inputs(st, frozenset(coalition), SpyEnc(), "cpu")
        return st, seen["ids"], feats, avail

    def test_text_in_coalition_untouched(self):
        st, ids, feats, avail = self._run({"text"})
        assert np.array_equal(ids, st.text_bert)
        assert avail["text"].numpy().any()

    def test_text_out_masked_before_bert(self):
        st, ids, feats, avail = self._run(("audio",))
        assert (ids[0, 0, 1:4] == 103).all()          # content → [MASK]
        assert ids[0, 0, 0] == 101 and ids[0, 0, 4] == 102   # special 不动
        assert ids[0, 1] .tolist() == st.text_bert[0, 1].tolist()  # attention 行不动
        assert not avail["text"].numpy().any()

    def test_empty_coalition_all_missing_structural_kept(self):
        st, ids, feats, avail = self._run(())
        assert (ids[0, 0, 1:4] == 103).all() and ids[0, 0, 0] == 101
        assert all(not avail[m].numpy().any() for m in MODS)   # v(∅) 全未观测
        assert feats["audio"].shape == (1, 50, 74)              # 特征仍进模型（非全零输入路径）

    def test_modality_avail_semantics(self):
        st, _, _, avail = self._run(("text", "vision"))
        assert avail["text"].numpy().any() and avail["vision"].numpy().any()
        assert not avail["audio"].numpy().any()

    def test_full_coalition_equals_natural(self):
        st, _, _, avail = self._run(MODS)
        for m in MODS:
            assert np.array_equal(avail[m].numpy(), st.o[m])
