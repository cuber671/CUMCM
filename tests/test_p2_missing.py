"""P2 M1-C 单测：缺失生成器（synthetic 构造）。

覆盖 M1-C 审阅要点：
- b 只落在 o=1 的位置（special/padding 永不抹除，四种机制全覆盖）
- 同 seed 逐位复现、异 seed 产生不同 mask
- mcar 实际率≈请求率；block 连续且 head/mid/tail 落位正确
- joint 两模态 b 完全一致、段长服从 pmf、段不重叠
- full: b==o；全零样本无 crash
- 预注册网格：47 组合、目录名唯一、无三模态全抹
- build 级：K=8 实例、ids 记录、跨 split 文件隔离
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2 import missing as M

N, T = 6, 16


@pytest.fixture(scope="module")
def masks():
    rng = np.random.default_rng(11)
    content = np.zeros((N, T), dtype=bool)
    content[:, 1:13] = True            # content = 1..12，13=SEP，0=CLS，14-15=pad
    o_a = content & (rng.random((N, T)) > 0.15)
    o_v = content & (rng.random((N, T)) > 0.25)
    o_v[2, :] = False                  # 造一个整样本无观测（附件2 有 110/15/28 条）
    return content, {"t": content.copy(), "a": o_a, "v": o_v}


def _all_b(combo, masks, seed=1):
    content, o = masks
    return M.gen_combo_instances(combo, 0, content, o, seed=seed)


# ---- 不越界（最重要契约）----

def test_b_never_outside_o(masks):
    content, o = masks
    combos = [dict(mechanism="mcar", modalities="t", rate=40, position="random"),
              dict(mechanism="block", modalities="a", rate=60, position="random"),
              dict(mechanism="block", modalities="v", rate=40, position="head"),
              dict(mechanism="block", modalities="v", rate=40, position="tail"),
              dict(mechanism="joint", modalities="av", rate=40, position="random"),
              dict(mechanism="full", modalities="v", rate=100, position="random")]
    for combo in combos:
        for bm in _all_b(combo, masks):
            for m, b in bm.items():
                assert not int(((b == 1) & ~o[m]).sum()), (combo, m)


# ---- mcar ----

def test_mcar_rate_and_reproducibility(masks):
    content, o = masks
    combo = dict(mechanism="mcar", modalities="a", rate=40, position="random")
    b1 = _all_b(combo, masks, seed=7)
    b2 = _all_b(combo, masks, seed=7)
    b3 = _all_b(combo, masks, seed=8)
    for x, y in zip(b1, b2):
        for m in x:
            assert (x[m] == y[m]).all()          # 同 seed 逐位复现
    diff = any((x[m] != y[m]).any() for x, y in zip(b1, b3) for m in x)
    assert diff                                   # 异 seed 不同
    for bm in b1:
        act = bm["a"].sum() / o["a"].sum()
        assert abs(act - 0.4) < 0.02


# ---- block ----

def test_block_contiguous_and_positions(masks):
    content, o = masks
    dense = {"t": content.copy(), "a": content.copy(), "v": content.copy()}
    for pos in ("head", "mid", "tail", "random"):
        combo = dict(mechanism="block", modalities="t", rate=50, position=pos)
        bm = M.gen_combo_instances(combo, 3, content, dense, seed=5)[0]
        b = bm["t"]
        for i in range(N):
            runs = M.run_lengths(b[i])
            assert len(runs) <= 1, (pos, i, runs)          # 单块
            ell = int(round(0.5 * 12))
            if runs:
                assert runs[0] == ell
                s = int(np.where(b[i])[0][0])
                c0 = 1
                if pos == "head":
                    assert s == c0
                elif pos == "tail":
                    assert s + ell - 1 == 12
                elif pos == "mid":
                    assert s == c0 + (12 - ell) // 2
                else:
                    assert c0 <= s <= 12 - ell + 1


def test_block_actual_le_request_with_holes(masks):
    content, o = masks
    combo = dict(mechanism="block", modalities="v", rate=80, position="random")
    for bm in _all_b(combo, masks):
        act = bm["v"].sum() / max(o["v"].sum(), 1)
        assert act <= 0.8 + 1e-9


# ---- joint ----

def test_joint_shared_and_segments(masks):
    content, o = masks
    pmf = (0.7, 0.1, 0.1, 0.1)
    combo = dict(mechanism="joint", modalities="av", rate=40, position="random")
    for bm in _all_b(combo, masks):
        assert (bm["a"] == bm["v"]).all()                  # 两模态完全同位
        avail = content & o["a"] & o["v"]
        assert not int(((bm["a"] == 1) & ~avail).sum())    # 只抹双方都观测的位

    # 段长服从 pmf（大样本统计）+ 互不重叠
    rng = np.random.default_rng(3)
    apos = np.arange(300)
    segs = M.sample_joint_segments(apos, 200, pmf, rng)
    occ = np.zeros(400, dtype=bool)
    lens = []
    for s, ell in segs:
        seg = np.arange(s, s + ell)
        assert not occ[seg].any()
        occ[seg] = True
        lens.append(ell)
    hist = np.bincount(lens, minlength=5)[1:5] / len(lens)
    assert np.abs(hist - np.array(pmf)).max() < 0.15       # 宽松统计界


# ---- full / 退化样本 ----

def test_full_and_degenerate(masks):
    content, o = masks
    bm = M.gen_combo_instances(dict(mechanism="full", modalities="v", rate=100,
                                    position="random"), 1, content, o, seed=2)[0]
    assert (bm["v"] == o["v"]).all()
    assert bm["v"][2].sum() == 0                            # 整样本无观测 → 无抹除
    # 全零模态样本上各机制无 crash、不越界（sample2 只有 v 全零，a 仍有观测属正常）
    for mech, mods in (("mcar", "v"), ("block", "v"), ("joint", "av"), ("full", "v")):
        c = dict(mechanism=mech, modalities=mods, rate=40, position="random")
        for inst in _all_b(c, masks):
            for m, b in inst.items():
                assert b[2].sum() == 0                      # avail=content&a&v=0 或 o=0
                assert not int(((b == 1) & ~o[m]).sum())


# ---- 网格与 build 级 ----

def test_grid_preregistered():
    g = M.library_grid()
    assert len(g) == 43
    dirs = [M.combo_dir(Path("x"), "train", c) for c in g]
    assert len(set(dirs)) == 43                             # 目录名唯一
    assert not any(set(list(c["modalities"])) == {"t", "a", "v"} for c in g)
    tags = {t for c in g for t in c["tags"]}
    assert {"train_A", "train_B", "train_C", "eval_main", "eval_mech"} <= tags


def test_build_library_end_to_end(masks, tmp_path):
    content, o = masks
    ids = {"train": np.array([f"tr{i}" for i in range(N)]),
           "valid": np.array([f"va{i}" for i in range(N)]),
           "test": np.array([f"te{i}" for i in range(N)])}
    grid = M.library_grid()[:6]                             # 子集即可验证机制
    for split in ids:
        for ci, combo in enumerate(grid):
            inst = M.gen_combo_instances(combo, ci, content, o, seed=M.MASK_LIBRARY_SEED)
            assert len(inst) == M.K_INSTANCES
            keys = [frozenset(np.flatnonzero(b["a" if "a" in b else "t"]).tolist())
                    for b in inst]
            assert len(set(keys)) >= 2                      # 实例间确有差异
            arrs = {"ids": ids[split]}
            for k_i, bm in enumerate(inst):
                for m, b in bm.items():
                    arrs[f"b_{k_i}_{m}"] = b
            p = M.combo_dir(tmp_path, split, combo)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **arrs)
    # split 隔离：train 与 valid 同组合文件不同路径、各自 ids
    z_tr = np.load(M.combo_dir(tmp_path, "train", grid[0]))
    z_va = np.load(M.combo_dir(tmp_path, "valid", grid[0]))
    assert (z_tr["ids"] != z_va["ids"]).all()
    assert str(M.combo_dir(tmp_path, "train", grid[0])) != \
        str(M.combo_dir(tmp_path, "valid", grid[0]))


def test_att3_pmf_frozen_values():
    assert abs(sum(M.ATT3_SEGMENT_PMF) - 1.0) < 1e-9
    assert len(M.ATT3_SEGMENT_PMF) == 4
