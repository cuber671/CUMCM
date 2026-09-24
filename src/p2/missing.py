"""P2 M1-C 缺失生成器与固定掩码库。

契约依据：docs/问题二实施步骤.md §1.2/§2 + docs/问题二建模方案.md §3/§4/§8.2。
- b_m 语义：合成抹除指示，只允许落在 o_m=1 的位置（o 自带 content 限定），
  special/padding 永不抹除；最终可用 a = o·(1−b)。
- 请求缺失率分母 = o_m=1 的可用位（2026-09-24 拍板）；请求率与实际有效率双记录。
- 机制（方案 §4）：
    mcar   均匀随机位（MCAR 点缺失）
    block  单个模型索引连续块，ℓ=round(rate·w)（w=|o=1|，上限=content 长度），
           位置 head/mid/tail/random；块覆盖 o=0 的洞是 no-op，实际率如实记录
    joint  a+v 同位联合缺失（附件3式）：段只能在 o_a=o_v=1 的位置、模型索引连续、
           段长 ~ P{1,2,3,4}（附件3 实测经验分布）、段互不重叠、贪心至目标数
    full   整模态抹除（b = o）
- 附件3 实测 pmf（2026-09-24：30 文件/605 content 位/131 联合零行/95 段，无 >4 段）：
    {1: 0.7474, 2: 0.1684, 3: 0.0421, 4: 0.0421}，平均段长 1.379。
- 可复现：全部随机性来自 default_rng([mask_library_seed, combo_idx, k])，
  combo_idx 为预注册网格中的全局序号；同 seed 逐位复现，异 seed 产生不同 mask。
本模块只依赖 numpy。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

MASK_LIBRARY_SEED = 2026
K_INSTANCES = 8
MODALITY_DIRS = ("t", "a", "v", "ta", "tv", "av")
POSITIONS = ("head", "mid", "tail", "random")

# 附件3 联合缺失段长经验分布（冻结值；生成脚本会重测并核对）
ATT3_SEGMENT_PMF = (71 / 95, 16 / 95, 4 / 95, 4 / 95)
ATT3_JOINT_ZERO = 131   # 预注册核对数（方案 §1）
ATT3_CONTENT = 605


def instance_rng(seed: int, combo_idx: int, k: int) -> np.random.Generator:
    """实例级确定性随机源：同 (seed, combo_idx, k) 必然同序列。"""
    return np.random.default_rng([int(seed), int(combo_idx), int(k)])


def _targets(o: np.ndarray, rate: float) -> np.ndarray:
    """每样本目标抹除数 k=round(rate·w)，w=|o=1|（分母=自然可用位）。"""
    assert 0.0 <= rate <= 1.0
    return np.round(rate * o.sum(axis=1)).astype(np.int64)


def run_lengths(b_row: np.ndarray) -> list:
    """一行 b 中被抹位置的极长连续段长（模型索引空间）。"""
    idx = np.where(b_row)[0]
    if len(idx) == 0:
        return []
    return [len(s) for s in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)]


def gen_b_mcar(o: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """均匀随机位：从 o=1 的位置无放回均匀抽 k 个（keys 排序法，o=0 沉底）。"""
    b = np.zeros(o.shape, dtype=np.uint8)
    keys = np.where(o == 1, rng.random(o.shape), 2.0)
    order = np.argsort(keys, axis=1)
    k = _targets(o, rate)
    for i in range(o.shape[0]):
        if k[i] > 0:
            b[i, order[i, :k[i]]] = 1
    return b


def _block_start(c0: int, width: int, ell: int, position: str, rng) -> int:
    if ell >= width:
        return c0
    if position == "head":
        return c0
    if position == "tail":
        return c0 + width - ell
    if position == "mid":
        return c0 + (width - ell) // 2
    if position == "random":
        return c0 + int(rng.integers(0, width - ell + 1))
    raise ValueError(f"未知 position: {position}")


def gen_b_block(content: np.ndarray, o: np.ndarray, rate: float, position: str,
                rng: np.random.Generator) -> np.ndarray:
    """单个模型索引连续块；b 只落在块内 o=1 的位，实际率可能低于请求率（如实记录）。"""
    b = np.zeros(o.shape, dtype=np.uint8)
    for i in range(o.shape[0]):
        cidx = np.where(content[i])[0]
        if len(cidx) == 0:
            continue
        ell = min(int(round(rate * int(o[i].sum()))), len(cidx))
        if ell <= 0:
            continue
        start = _block_start(int(cidx[0]), len(cidx), ell, position, rng)
        b[i, start:start + ell] = 1
    return b & o.astype(np.uint8)


def sample_joint_segments(apos: np.ndarray, k: int, pmf, rng) -> list:
    """贪心不重叠段采样：随机置换起点，段长 ~pmf，段必须整体落在可用位上；
    达到 k 个抹除位或起点耗尽即停。返回段起点列表（可测试/可审计）。"""
    segs = []
    occupied = np.zeros(int(apos.max()) + 1 if len(apos) else 1, dtype=bool)
    total = 0
    for s in rng.permutation(apos):
        if total >= k:
            break
        ell = int(rng.choice(np.arange(1, 5), p=pmf))
        seg = np.arange(int(s), int(s) + ell)
        if seg[-1] >= len(occupied) or occupied[seg].any() or not np.isin(seg, apos).all():
            continue
        occupied[seg] = True
        segs.append((int(s), ell))
        total += ell
    return segs


def gen_b_joint(content: np.ndarray, o_a: np.ndarray, o_v: np.ndarray, rate: float,
                pmf, rng: np.random.Generator):
    """a+v 同位联合缺失：两模态 b 完全一致；段限 o_a=o_v=1 且模型索引连续。"""
    avail = content & o_a & o_v
    b = np.zeros(content.shape, dtype=np.uint8)
    k = _targets(avail, rate)
    for i in range(content.shape[0]):
        apos = np.where(avail[i])[0]
        if k[i] <= 0 or len(apos) == 0:
            continue
        for s, ell in sample_joint_segments(apos, k[i], pmf, rng):
            b[i, s:s + ell] = 1
    return b, b.copy()


def gen_b_full(o: np.ndarray) -> np.ndarray:
    """整模态抹除：b = o（可用位全抹）。"""
    return o.astype(np.uint8).copy()


def gen_combo_instances(combo: dict, combo_idx: int, content: np.ndarray,
                        o_by_mod: dict, pmf=ATT3_SEGMENT_PMF,
                        seed: int = MASK_LIBRARY_SEED, k_instances: int = K_INSTANCES) -> list:
    """一个组合的 K 个实例：返回 [ {mod: b(N,50)uint8} ] 及逐实例统计。"""
    mech, mods = combo["mechanism"], list(combo["modalities"])
    rate = combo["rate"] / 100.0
    instances = []
    for k_i in range(k_instances):
        rng = instance_rng(seed, combo_idx, k_i)
        if mech == "mcar":
            b_map = {m: gen_b_mcar(o_by_mod[m], rate, rng) for m in mods}
        elif mech == "block":
            b_map = {m: gen_b_block(content, o_by_mod[m], rate, combo["position"], rng)
                     for m in mods}
        elif mech == "joint":
            ba, bv = gen_b_joint(content, o_by_mod["a"], o_by_mod["v"], rate, pmf, rng)
            b_map = {"a": ba, "v": bv}
        elif mech == "full":
            b_map = {m: gen_b_full(o_by_mod[m]) for m in mods}
        else:
            raise ValueError(f"未知机制: {mech}")
        instances.append(b_map)
    return instances


def instance_stats(b_map: dict, o_by_mod: dict, content: np.ndarray) -> dict:
    """单实例统计：各模态实际有效缺失率（分母=o=1）+ 抹除段长直方 + 越界审计。"""
    per_mod, run_hist = {}, Counter()
    for m, b in b_map.items():
        w = int(o_by_mod[m].sum())
        per_mod[m] = round(float(b.sum()) / w, 6) if w else 0.0
        bad = int(((b == 1) & ~(o_by_mod[m] == 1)).sum())
        assert bad == 0, f"b 越界（o=0 处 b=1）{bad} 位"
        for i in range(b.shape[0]):
            run_hist.update(run_lengths(b[i]))
    return {"actual_rate": per_mod,
            "run_len_hist": {str(k): v for k, v in sorted(run_hist.items())}}


def library_grid() -> list:
    """预注册网格（顺序即 combo_idx，43 项；顺序冻结，插入即破坏复现）。

    注意：机制对比 @40 的 block/joint 复用 train_B/train_A 的 rate40 条目
    （加 eval_mech tag），目录名唯一，不产生重复组合。
    """
    g = []
    for m in ("t", "a", "v"):                                   # eval 主线+训练B 15
        for r in (10, 20, 40, 60, 80):
            g.append(dict(mechanism="mcar", modalities=m, rate=r, position="random",
                          tags=["eval_main", "train_B"]))
    for m in ("ta", "tv", "av"):                                # eval 组合 3
        g.append(dict(mechanism="mcar", modalities=m, rate=40, position="random",
                      tags=["eval_combo"]))
    for m in ("t", "a", "v"):                                   # eval 位置 9
        for p in ("head", "mid", "tail"):
            g.append(dict(mechanism="block", modalities=m, rate=40, position=p,
                          tags=["eval_position"]))
    for r in (10, 20, 40, 60):                                  # 训练 A 4（rate40 兼任机制对比）
        tags = ["train_A"] + (["eval_mech"] if r == 40 else [])
        g.append(dict(mechanism="joint", modalities="av", rate=r, position="random",
                      tags=tags))
    for m in ("t", "a", "v"):                                   # 训练 B-块 9
        for r in (20, 40, 60):
            tags = ["train_B"] + (["eval_mech"] if r == 40 else [])
            g.append(dict(mechanism="block", modalities=m, rate=r, position="random",
                          tags=tags))
    for m in ("t", "a", "v"):                                   # 训练 C 3
        g.append(dict(mechanism="full", modalities=m, rate=100, position="random",
                      tags=["train_C"]))
    return g


def combo_dir(root: Path, split: str, combo: dict, seed: int = MASK_LIBRARY_SEED) -> Path:
    return (Path(root) / split / combo["mechanism"] / combo["modalities"] /
            f"rate{combo['rate']}" / combo["position"] / f"seed{seed}.npz")
