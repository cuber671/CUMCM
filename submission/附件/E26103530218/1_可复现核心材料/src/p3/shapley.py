"""P3 M2：模态级精确 Shapley（8 联盟枚举）——MCEF 联盟层。

口径（docs/问题三建模方案.md §3 + M2 预注册 2026-09-25）：
- 联盟 S ⊆ {text,audio,vision}，8 个全枚举，固定顺序（combo 序冻结，缓存键）；
- 联盟构造 = 对 content 位执行合成缺失（P2 语义）：m∈S → avail_m = o_m；m∉S → avail_m = 0；
  special/padding 由结构 mask 保持不变，绝不触碰。text∉S 时必须在 BERT 前把 content 位
  替换为 [MASK] 重编码（mask_text_tokens 越界即拒），不允许对完整 embedding 事后置零；
- v(∅) = 全模态未观测状态（与 P2 Full 缺失同构）：text 全 [MASK] 重编码 + 三模态 avail=0，
  结构 mask 全保留，非全零输入；
- v 取双值：分类概率向量 p∈Δ³（softmax）与回归强度 ŷ；集成 = 3 seeds 均值（P2 终评同口径）；
- 精确 Shapley：φ_m = Σ_{S⊆N\{m}} |S|!(3-|S|-1)!/3!·[v(S∪{m}) − v(S)]；
- 完备性（效率公理）：Σ_m φ_m = v(N) − v(∅)，分类逐类、回归分别检验，容差 1e-5。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import torch

from src.p2.data import CONTRACT
from src.p2.pipeline import load_stats, zscore_reset
from src.p2.text_mask import mask_text_tokens

MODS = ("text", "audio", "vision")
# 8 联盟固定顺序（冻结：缓存与报告键序）
COALITIONS: tuple[frozenset[str], ...] = tuple(
    frozenset(c) for r in range(4) for c in combinations(MODS, r))
COMPLETENESS_TOL = 1e-5


@dataclass
class SampleTensors:
    """附件4 单条样本的模型输入张量（标准化后；o 掩码按 P2 契约推导）。"""
    n: int
    text_bert: np.ndarray            # (1,3,50) int64（原始，未抹除）
    content: np.ndarray              # (1,50) bool
    o: dict[str, np.ndarray]         # 各模态 (1,50) bool 自然观测
    audio: np.ndarray                # (1,50,74) float32 已 z-score（o=0 位已归零）
    vision: np.ndarray               # (1,50,35) float32 同上
    raw_text: str


def prepare_sample(s, stats: dict | None = None) -> SampleTensors:
    """Att4Sample → 模型输入（audio/vision 用 train 统计量标准化，o=0 位归零）。"""
    from src.p3.data import att4_masks
    tm, obs = att4_masks(s)
    stats = stats or load_stats()
    tb_int = s.text_bert.astype(np.int64)
    aud = zscore_reset(s.audio.astype(np.float32), stats["audio"]["mean"],
                       stats["audio"]["std"], obs["o_audio"], name=f"att4#{s.n:02d}.audio")
    vis = zscore_reset(s.vision.astype(np.float32), stats["vision"]["mean"],
                       stats["vision"]["std"], obs["o_vision"], name=f"att4#{s.n:02d}.vision")
    return SampleTensors(n=s.n, text_bert=tb_int, content=tm.content,
                         o={"text": obs["o_text"], "audio": obs["o_audio"],
                            "vision": obs["o_vision"]},
                         audio=aud, vision=vis, raw_text=s.raw_text)


def coalition_inputs(st: SampleTensors, coalition: frozenset[str], bert_enc,
                     device: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """构造联盟 S 的模型输入：feats + avail。text∉S 时 [MASK] 重编码（BERT 前）。"""
    if "text" in coalition:
        ids = torch.from_numpy(st.text_bert).to(device)
    else:
        b_text = st.content.copy()          # 全 content 位抹除（越界由 mask_text_tokens 拒）
        ids = torch.from_numpy(mask_text_tokens(st.text_bert, b_text, CONTRACT)).to(device)
    feats = {"text": bert_enc(ids), "audio": torch.from_numpy(st.audio).to(device),
             "vision": torch.from_numpy(st.vision).to(device)}
    avail = {m: torch.from_numpy(st.o[m] if m in coalition
                                 else np.zeros_like(st.o[m])).to(device)
             for m in MODS}
    return feats, avail


def ensemble_v(ensemble, st: SampleTensors, coalition: frozenset[str], device: str) -> dict:
    """联盟值：3 seeds 概率/回归均值（附带门控均值，供 g–φ 与解释卡用）。"""
    probs, regs, gates = [], [], []
    for model, bert_enc in ensemble:
        feats, avail = coalition_inputs(st, coalition, bert_enc, device)
        with torch.no_grad():
            o = model(feats, torch.from_numpy(st.content).to(device), avail)
        probs.append(torch.softmax(o["logits"], dim=-1)[0].cpu().numpy())
        regs.append(float(o["reg"][0]))
        gates.append(o["gates"][0].cpu().numpy())
    return {"v_cls": np.mean(probs, axis=0), "v_reg": float(np.mean(regs)),
            "gates": np.mean(gates, axis=0)}


def shapley_weights(n_players: int = 3) -> dict[int, Fraction]:
    """|S| → 权重 |S|!(n-|S|-1)!/n!（精确分数，杜绝浮点权重误差）。"""
    return {k: Fraction(factorial(k) * factorial(n_players - k - 1), factorial(n_players))
            for k in range(n_players)}


def exact_shapley(values: dict[frozenset, np.ndarray | float]) -> dict[str, object]:
    """通用精确 Shapley：v 值可为标量或向量（逐分量），输出同形 φ。"""
    w = shapley_weights()
    phi = {}
    for m in MODS:
        acc = None
        others = [x for x in MODS if x != m]
        for r in range(len(others) + 1):
            for c in combinations(others, r):
                S, Sm = frozenset(c), frozenset(c) | {m}
                term = (np.asarray(values[Sm], dtype=np.float64)
                        - np.asarray(values[S], dtype=np.float64)) * float(w[r])
                acc = term if acc is None else acc + term
        phi[m] = acc
    return phi


def phi_report(ensemble, st: SampleTensors, device: str) -> dict:
    """单样本完整 φ 表 + 完备性检验（分类逐类 + 回归）。"""
    vals = {S: ensemble_v(ensemble, st, S, device) for S in COALITIONS}
    phi_cls = exact_shapley({S: vals[S]["v_cls"] for S in COALITIONS})
    phi_reg = exact_shapley({S: vals[S]["v_reg"] for S in COALITIONS})
    p_full, p_empty = vals[frozenset(MODS)]["v_cls"], vals[frozenset()]["v_cls"]
    r_full, r_empty = vals[frozenset(MODS)]["v_reg"], vals[frozenset()]["v_reg"]
    sum_cls = sum(phi_cls[m] for m in MODS)
    sum_reg = sum(float(phi_reg[m]) for m in MODS)
    return {
        "sample_id": f"{st.n:02d}",
        "pred_class": int(p_full.argmax()), "pred_intensity": r_full,
        "confidence": float(p_full.max()),
        "coalition_values": {"|".join(sorted(S)) if S else "∅":
                             {"v_cls": np.round(v["v_cls"], 6).tolist(),
                              "v_reg": round(v["v_reg"], 6)} for S, v in vals.items()},
        "gates_full": np.round(vals[frozenset(MODS)]["gates"], 4).tolist(),
        "phi_cls": {m: np.round(phi_cls[m], 6).tolist() for m in MODS},
        "phi_reg": {m: round(float(phi_reg[m]), 6) for m in MODS},
        "completeness": {
            "cls_residual": float(np.abs(sum_cls - (p_full - p_empty)).max()),
            "reg_residual": abs(sum_reg - (r_full - r_empty)),
            "tol": COMPLETENESS_TOL,
            "pass": bool(np.abs(sum_cls - (p_full - p_empty)).max() < COMPLETENESS_TOL
                         and abs(sum_reg - (r_full - r_empty)) < COMPLETENESS_TOL)},
    }
