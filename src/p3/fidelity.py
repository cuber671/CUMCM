"""P3 M4：保真度验证层——删除/插入曲线（AOPC）、Compr/Suff、随机化、稳定性、IG↔LOO。

口径（docs/问题三建模方案.md §5 + M4 预注册 2026-09-25）：

- **抹除算子（复用 P2 语义，位置级）**：删除证据位 (m,j) =
  text → 该 WordPiece 编码前替换 [MASK]（`mask_text_tokens`，同联盟构造）；
  a/v → avail 位清 0（值保留、标记合成缺失，同 P2 b 掩码）。结构与 padding 永不触碰。
- **删除曲线**：全局证据序（三模态可用位按 |IG_cls|（M3 主基线）降序，跨模态合并），
  逐比例（10%..90%）删除 → 目标类概率 p；随机序对照 = 5 组种子化随机排列均值。
  AOPC = mean_f [p_rand(f) − p_attr(f)]（>0 即归因序删除更快压低目标概率）。
- **插入曲线**：起点 = 全删除态（text 全 [MASK]、a/v avail≡0），按归因序逐比例恢复，
  AUC 差（归因 − 随机）>0 即归因序更快重建预测。
- **Comprehensiveness / Sufficiency（ERASER）**：compr@f = p_full − p_del(top f)，
  suff@f = p_ins(top f) − p_full（f=0.2）。
- **权重随机化**：MRFN 头重初始化（固定种子，BERT 编码器不变），目标类取随机模型
  自身 argmax；期望归因排序与原模型解耦（Spearman 中位数 ≤ 0.3）。
- **扰动稳定性**：连续特征加噪 ε~N(0, 0.05σ_m)（text 嵌入与 a/v z 特征，
  按可用位数值 std 定标），重算 IG，Spearman(可用位排序)。
- **LOO 裁判**：逐可用位单点删除 Δp = p_full − p_del({(m,j)})，与 |IG| 排序比 Spearman。

预注册门槛（check_p3_m4.py 全量跑前冻结）：
  A 删除优越性 mean AOPC>0 且逐样本胜 ≥15/20；B 插入优越性 mean ΔAUC>0 且 ≥15/20；
  C compr@0.2 均值>0 且 |suff@0.2| 均值<compr 均值；D 随机化坍缩 中位 ρ≤0.3（各模态）；
  E 稳定性 text mean ρ≥0.6（主证据通道；a/v 报告不设门）；F IG↔LOO text mean ρ≥0.3
  （a/v 报告）。
"""
from __future__ import annotations

import numpy as np
import torch

from src.p2.data import CONTRACT
from src.p2.text_mask import mask_text_tokens
from src.p3.shapley import MODS, SampleTensors

FRACTIONS = tuple(round(0.1 * i, 1) for i in range(1, 10))
N_RANDOM = 5
COMPR_FRAC = 0.2


def available_positions(st: SampleTensors) -> list[tuple[str, int]]:
    """全部可用证据位 (m, j)：o_m=1（text=o_text=content）。"""
    out = []
    for m in MODS:
        out += [(m, int(j)) for j in np.where(st.o[m][0])[0]]
    return out


def evidence_ranking(ig_cls: dict[str, np.ndarray], st: SampleTensors) -> list:
    """全局证据序：三模态可用位按 |IG_cls| 降序（[(mod, pos, |ig|)]）。"""
    items = [(m, j, abs(float(ig_cls[m][j])))
             for (m, j) in available_positions(st)]
    return sorted(items, key=lambda t: -t[2])


def removal_state(st: SampleTensors, removed: set) -> tuple:
    """证据删除算子（P2 语义位置级）。removed ⊆ available_positions。

    返回 (ids_np, avail_np)：text 位 [MASK] 前置替换（BERT 前）；a/v 位 avail 清 0
    （特征值保留）。removed 越界（不可用位）直接拒绝。
    """
    avail_pos = set(available_positions(st))
    bad = removed - avail_pos
    if bad:
        raise ValueError(f"删除位不可用: {sorted(bad)[:3]}")
    b_text = np.zeros_like(st.content)
    avail = {m: st.o[m].copy() for m in MODS}
    for (m, j) in removed:
        avail[m][0, j] = False                  # P2 语义：b=1 → avail 清位（含 text）
        if m == "text":
            b_text[0, j] = True                 # text 同时 [MASK] 前置替换
    ids = mask_text_tokens(st.text_bert, b_text, CONTRACT) if b_text.any() \
        else st.text_bert.copy()
    return ids, avail


def ens_forward(ensemble, st: SampleTensors, ids: np.ndarray,
                avail: dict, device: str) -> tuple:
    """3-seed 均值前向：(p 向量, 强度)。text 用各自 seed 的 BERT 现场编码。"""
    ps, regs = [], []
    content_t = torch.from_numpy(st.content).to(device)
    for model, be in ensemble:
        with torch.no_grad():
            feats = {"text": be(torch.from_numpy(ids).to(device)),
                     "audio": torch.from_numpy(st.audio).to(device),
                     "vision": torch.from_numpy(st.vision).to(device)}
            o = model(feats, content_t,
                      {m: torch.from_numpy(avail[m]).to(device) for m in MODS})
        ps.append(torch.softmax(o["logits"], -1)[0].cpu().numpy())
        regs.append(float(o["reg"][0]))
    return np.mean(ps, axis=0), float(np.mean(regs))


def deletion_insertion(ensemble, st: SampleTensors, ig_cls: dict, device: str,
                       target_cls: int, sample_n: int,
                       fractions=FRACTIONS, n_rand=N_RANDOM) -> dict:
    """删除/插入曲线 + AOPC/AUC + Compr/Suff（一个样本的完整保真度实验）。"""
    rank = [(m, j) for m, j, _ in evidence_ranking(ig_cls, st)]
    n_avail = len(rank)
    all_set = set(rank)
    p_full, _ = ens_forward(ensemble, st, st.text_bert, st.o, device)
    p0 = float(p_full[target_cls])

    rows, ins_rows = [], []
    for f in fractions:
        k = max(1, int(round(f * n_avail)))
        # 删除：归因序 vs 随机序
        ids, av = removal_state(st, set(rank[:k]))
        p_attr = float(ens_forward(ensemble, st, ids, av, device)[0][target_cls])
        p_rands = []
        for rep in range(n_rand):
            rng = np.random.default_rng([2026, sample_n, rep, int(f * 100)])
            sub = set(rng.choice(n_avail, size=k, replace=False).tolist())
            ids_r, av_r = removal_state(st, {rank[i] for i in sub})
            p_rands.append(float(
                ens_forward(ensemble, st, ids_r, av_r, device)[0][target_cls]))
        rows.append({"frac": f, "p_attr": p_attr, "p_rand_mean": float(np.mean(p_rands)),
                     "p_rand_std": float(np.std(p_rands)), "p_full": p0})
        # 插入：全删除态起，按序恢复 k 位（= 删除 剩余位）
        ids_i, av_i = removal_state(st, set(rank[k:]))
        p_ins = float(ens_forward(ensemble, st, ids_i, av_i, device)[0][target_cls])
        p_ins_rands = []
        for rep in range(n_rand):
            rng = np.random.default_rng([2027, sample_n, rep, int(f * 100)])
            sub = rng.choice(n_avail, size=k, replace=False).tolist()
            keep = {rank[i] for i in sub}
            ids_ir, av_ir = removal_state(st, all_set - keep)
            p_ins_rands.append(float(
                ens_forward(ensemble, st, ids_ir, av_ir, device)[0][target_cls]))
        ins_rows.append({"frac": f, "p_attr": p_ins,
                         "p_rand_mean": float(np.mean(p_ins_rands)),
                         "p_rand_std": float(np.std(p_ins_rands))})

    aopc = float(np.mean([r["p_rand_mean"] - r["p_attr"] for r in rows]))
    del_auc_attr = float(np.trapz([r["p_attr"] for r in rows],
                                  [r["frac"] for r in rows]))
    del_auc_rand = float(np.trapz([r["p_rand_mean"] for r in rows],
                                  [r["frac"] for r in rows]))
    ins_auc_attr = float(np.trapz([r["p_attr"] for r in ins_rows],
                                  [r["frac"] for r in ins_rows]))
    ins_auc_rand = float(np.trapz([r["p_rand_mean"] for r in ins_rows],
                                  [r["frac"] for r in ins_rows]))
    # Compr/Suff @0.2
    k20 = max(1, int(round(COMPR_FRAC * n_avail)))
    ids_d, av_d = removal_state(st, set(rank[:k20]))
    compr = p0 - float(ens_forward(ensemble, st, ids_d, av_d, device)[0][target_cls])
    ids_s, av_s = removal_state(st, set(rank[k20:]))
    suff = float(ens_forward(ensemble, st, ids_s, av_s, device)[0][target_cls]) - p0
    return {"n_avail": n_avail, "p_full_target": p0,
            "deletion": rows, "insertion": ins_rows,
            "aopc": aopc,
            "del_auc_attr": del_auc_attr, "del_auc_rand": del_auc_rand,
            "ins_auc_attr": ins_auc_attr, "ins_auc_rand": ins_auc_rand,
            "compr_at_20": compr, "suff_at_20": suff}


def loo_consistency(ensemble, st: SampleTensors, ig_cls: dict, device: str,
                    target_cls: int) -> dict:
    """逐可用位单点删除 Δp 与 |IG| 的排序一致性（逐模态 Spearman）。"""
    from scipy.stats import spearmanr
    p_full, _ = ens_forward(ensemble, st, st.text_bert, st.o, device)
    p0 = float(p_full[target_cls])
    out = {}
    for m in MODS:
        js = [int(j) for j in np.where(st.o[m][0])[0]]
        if not js:                                  # 模态整缺失（#13 vision）
            out[m] = {"rho": float("nan"), "n": 0}
            continue
        loo = []
        for j in js:
            ids, av = removal_state(st, {(m, j)})
            loo.append(p0 - float(ens_forward(ensemble, st, ids, av, device)[0][target_cls]))
        ig_abs = [abs(float(ig_cls[m][j])) for j in js]
        if np.ptp(loo) == 0 or np.ptp(ig_abs) == 0:
            rho = float("nan")            # 全零模态（如 #13 vision）
        else:
            rho = float(spearmanr(loo, ig_abs).statistic)
        out[m] = {"rho": rho, "n": len(js)}
    return out


def stability_ig(ensemble, st: SampleTensors, stats: dict, device: str,
                 target_cls: int, mean_embs: dict, sample_n: int,
                 n_steps: int = 64) -> tuple:
    """扰动 x′ = x + ε（0.05σ_m，可用位定标）重算 IG；返回 (ig_noisy, 噪声 σ)。"""
    from src.p3.ig import ig_modality, text_baseline, av_baseline
    rng = np.random.default_rng([2028, sample_n])
    igs = {m: [] for m in MODS}
    sig = {}
    for (model, be), seed in zip(ensemble, (1, 2, 3)):
        with torch.no_grad():
            x = {"text": be(torch.from_numpy(st.text_bert).to(device)),
                 "audio": torch.from_numpy(st.audio).to(device),
                 "vision": torch.from_numpy(st.vision).to(device)}
            base = {"text": text_baseline("missing", st, be, device, mean_embs[seed]),
                    "audio": torch.from_numpy(
                        av_baseline("missing", "audio", stats)[None]).to(device),
                    "vision": torch.from_numpy(
                        av_baseline("missing", "vision", stats)[None]).to(device)}
        x_n, base_n = {}, {}
        for m in MODS:
            mask_t = torch.from_numpy(st.o[m]).to(device)[..., None]
            vals = x[m][mask_t.expand_as(x[m]).bool()]
            sig[m] = float(vals.std().cpu()) * 0.05 if vals.numel() else 0.0
            eps = torch.from_numpy(
                rng.normal(0.0, sig[m], size=tuple(x[m].shape)).astype(np.float32)
            ).to(device) * mask_t                       # 仅可用位加噪
            x_n[m] = x[m] + eps
            base_n[m] = base[m] + eps * 0.0             # 基线不扰（对照同一参照系）
        avail = {m: torch.from_numpy(st.o[m]).to(device) for m in MODS}
        ig = ig_modality(model, st, device, target_cls, base_n, x_n, avail,
                         n_steps=n_steps)
        for m in MODS:
            igs[m].append(ig[m]["cls"])
    return {m: np.mean(igs[m], axis=0) for m in MODS}, sig


def randomization_ig(st: SampleTensors, stats: dict, device: str, bert_enc,
                     seed: int = 999, n_steps: int = 64) -> tuple:
    """权重随机化（MRFN 头重初始化，BERT 不变）的 IG 与其自身 argmax 目标类。"""
    from src.p2.models import MODEL_REGISTRY
    from src.p3.ig import ig_modality, text_baseline, av_baseline
    torch.manual_seed(seed)
    model = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device).eval()
    with torch.no_grad():
        x = {"text": bert_enc(torch.from_numpy(st.text_bert).to(device)),
             "audio": torch.from_numpy(st.audio).to(device),
             "vision": torch.from_numpy(st.vision).to(device)}
        base = {"text": text_baseline("missing", st, bert_enc, device),
                "audio": torch.from_numpy(
                    av_baseline("missing", "audio", stats)[None]).to(device),
                "vision": torch.from_numpy(
                    av_baseline("missing", "vision", stats)[None]).to(device)}
        avail = {m: torch.from_numpy(st.o[m]).to(device) for m in MODS}
        content_t = torch.from_numpy(st.content).to(device)
        o = model(x, content_t, avail)
    target = int(o["logits"][0].argmax())
    ig = ig_modality(model, st, device, target, base, x, avail, n_steps=n_steps)
    return {m: ig[m]["cls"] for m in MODS}, target
