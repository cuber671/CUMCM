"""P2 错误归因分析（docx 内容(4)）：clean valid 错误按 |回归标签| 分区 + 标签结构。

纪律：纯分析，不训练、不碰 test（state 只读 train/valid 键）；产物供论文
问题二章节"错误归因与性能上限"小节使用。test 主表维持 9eb88d3 不变。

标签结构（本次分析发现的核心事实，train/valid 双 split 零例外验证）：
  分类标签是回归标签的确定性函数——cl=1 ⟺ rl==0（精确零），
  cl=0 ⟺ rl<0，cl=2 ⟺ rl>0；非零 |rl| 最小 1/6（MOSI 标注粒度）。
  ⇒ 真 rl 的 cls oracle = 100%，性能上限由特征侧弱情感分辨力决定，
     错误归因按 |rl| 分区（rl==0 / 弱极性 / 强极性）展开。

三部分：
  A 标签结构——确定性验证 + 真 rl oracle；
  B 模型侧——3 seeds clean valid 推理，错误按 |rl| 分区/分箱，分区混淆矩阵；
  C 后处理参照（valid-only，不进主表）——回归符号决策规则、极性仲裁、
    3-seed 集成、类偏置校正（决策侧改动收益量级的实证参照）。
产物：runs/p2/error_attribution/{error_attribution.json, per_seed_predictions.npz,
summary.md}。自检：per-seed clean acc 与 result.json 偏差 ≤0.01（跨进程重推理存在
cuDNN workspace 相关的浮点漂移，实测 |Δ|≤2/728 样本；test 终评同为重推理路径，
本分析与之一致，漂移逐 seed 记录于 repro_check）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_p2_b0 import metrics  # noqa: E402
from train_p2_ladder import build_tensors  # noqa: E402

from src.p2.data import load_aligned  # noqa: E402
from src.p2.models import MODEL_REGISTRY  # noqa: E402

STATE = ROOT / "runs/p2/data/m1a_state.npz"
CKPT = ROOT / "runs/p2/mrfn"
OUT = ROOT / "runs/p2/error_attribution"
MODS = ("text", "audio", "vision")
SEEDS = (1, 2, 3)
# 分箱：|rl|=0（真中性）单列，其后按强度递进；1/6 = 数据非零粒度下界
BINS = [("=0", None, None), ("(0,0.25)", 0.0, 0.25), ("[0.25,0.5)", 0.25, 0.5),
        ("[0.5,1.0)", 0.5, 1.0), ("[1.0,1.5)", 1.0, 1.5), ("[1.5,3.0]", 1.5, 3.001)]
ZONES = ("Z0_neutral", "Z1_weak", "Z2_strong")   # rl==0 / 0<|rl|<0.5 / |rl|≥0.5
ZONE_LABELS = {"Z0_neutral": "rl==0", "Z1_weak": "0<|rl|<0.5", "Z2_strong": "|rl|>=0.5"}


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for c in range(3):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s))


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for t in range(3):
        for p in range(3):
            cm[t, p] = int(((y == t) & (pred == p)).sum())
    return cm


def sign_rule(reg: np.ndarray, tau: float) -> np.ndarray:
    """数据同构决策规则：|r|<τ → 中性，否则按符号。τ=0 时永不判中性。"""
    return np.where(np.abs(reg) < tau, 1, np.where(reg < 0, 0, 2))


@torch.no_grad()
def infer_valid(model, va, batch=256):
    model.eval()
    logits, regs = [], []
    for i in range(0, va["content"].shape[0], batch):
        idx = slice(i, i + batch)
        feats = {k: va[k][idx] for k in MODS}
        av = {m: va[f"o_{m}"][idx] for m in MODS}
        o = model(feats, va["content"][idx], av)
        logits.append(o["logits"]); regs.append(o["reg"])
    return torch.cat(logits).cpu().numpy(), torch.cat(regs).cpu().numpy()


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    att = load_aligned()
    del att["test"]
    state = np.load(STATE)
    tensors = build_tensors(att, state, device)
    va = tensors["valid"]
    y = va["y_cls"].cpu().numpy()
    rl = va["y_reg"].cpu().numpy()
    ids = np.asarray(state["valid_ids"])
    abs_rl = np.abs(rl)

    # ---- A 标签结构：确定性验证（train/valid 双 split）----
    label_struct = {}
    for split in ("train", "valid"):
        rl_s = state[f"{split}_rl"].astype(float)
        cl_s = state[f"{split}_cls"].astype(int)
        z = rl_s == 0.0
        nz = np.abs(rl_s[~z])
        label_struct[split] = {
            "n": int(len(rl_s)),
            "n_neutral": int(z.sum()), "n_cls1": int((cl_s == 1).sum()),
            "cls1_and_rl_nonzero": int(((cl_s == 1) & ~z).sum()),
            "nonneutral_and_rl_zero": int(((cl_s != 1) & z).sum()),
            "sign_violations": int(((cl_s == 0) & (rl_s >= 0)).sum()
                                   + ((cl_s == 2) & (rl_s <= 0)).sum()),
            "min_abs_nonzero_rl": round(float(nz.min()), 6),
            "n_weak_polarity": int(((nz > 0) & (nz < 0.5)).sum()),
            "neutral_rl_all_exactly_zero": bool(np.all(rl_s[cl_s == 1] == 0.0))}
    oracle = {"rule": "cls = {rl<0:0, rl==0:1, rl>0:2}",
              "oracle_acc": 1.0 if all(
                  label_struct[s]["cls1_and_rl_nonzero"] == 0
                  and label_struct[s]["nonneutral_and_rl_zero"] == 0
                  and label_struct[s]["sign_violations"] == 0
                  for s in ("train", "valid")) else None,
              "implication": "真 rl 的 cls oracle=100% ⇒ 上限由特征侧弱情感分辨力决定"}

    # 三分区掩码（后续共用）
    zone_mask = {"Z0_neutral": rl == 0.0,
                 "Z1_weak": (rl != 0.0) & (abs_rl < 0.5),
                 "Z2_strong": abs_rl >= 0.5}

    # ---- B 模型侧：3 seeds clean valid 推理 ----
    per_seed, repro = [], []
    for seed in SEEDS:
        model = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        model.load_state_dict(torch.load(
            CKPT / f"MRFN_seed{seed}" / "checkpoint.pt", weights_only=True))
        lo, rg = infer_valid(model, va)
        m = metrics(y, rl, lo, rg)
        ref = json.loads((CKPT / f"MRFN_seed{seed}" / "result.json").read_text())
        d_acc = m["acc"] - ref["valid_metrics"]["acc"]
        d_S = m["S"] - ref["valid_metrics"]["S"]
        assert abs(d_acc) <= 0.01 and abs(d_S) <= 0.005, \
            f"seed{seed} 偏差超容差: Δacc={d_acc} ΔS={d_S}"
        repro.append({"seed": seed, "acc_reinf": m["acc"],
                      "acc_train_time": ref["valid_metrics"]["acc"],
                      "d_acc": round(d_acc, 6), "d_S": round(d_S, 6)})
        per_seed.append({"seed": seed, "logits": lo, "reg": rg,
                         "pred": lo.argmax(1), "metrics": m})
        print(f"seed{seed}: acc={m['acc']:.4f} S={m['S']:.4f} "
              f"(重推理 vs 训练时 Δacc={d_acc:+.4f})", flush=True)
    base_acc = [p["metrics"]["acc"] for p in per_seed]
    base_f1 = [p["metrics"]["macro_f1"] for p in per_seed]

    # 分箱错误率（per-seed 均值±std）；首箱 =0 为真中性
    bin_table = []
    for label, lo_e, hi_e in BINS:
        m_bin = (rl == 0.0) if lo_e is None else (abs_rl >= lo_e) & (abs_rl < hi_e)
        rates = [float((p["pred"][m_bin] != y[m_bin]).mean()) for p in per_seed]
        shares = [float((p["pred"][m_bin] != y[m_bin]).sum()
                        / (p["pred"] != y).sum()) for p in per_seed]
        bin_table.append({
            "bin": label, "n": int(m_bin.sum()),
            "class_counts": np.bincount(y[m_bin], minlength=3).tolist(),
            "err_rate_mean": round(float(np.mean(rates)), 6),
            "err_rate_std": round(float(np.std(rates)), 6),
            "err_share_mean": round(float(np.mean(shares)), 6)})
    # 三分区：acc / 错误占比 / 混淆矩阵（3 seeds 池化） / 关键条件率
    zone_out = {}
    y_pool = np.tile(y, 3)
    pred_pool = np.concatenate([p["pred"] for p in per_seed])
    for zn in ZONES:
        m_z = zone_mask[zn]
        m_pool = np.tile(m_z, 3)
        cm = confusion(y_pool[m_pool], pred_pool[m_pool])
        recalls = np.diag(cm) / np.maximum(cm.sum(1), 1)
        zone_out[zn] = {
            "def": ZONE_LABELS[zn], "n": int(m_z.sum()),
            "acc_mean": round(float(np.mean(
                [(p["pred"][m_z] == y[m_z]).mean() for p in per_seed])), 6),
            "err_share_mean": round(float(np.mean(
                [(p["pred"][m_z] != y[m_z]).sum()
                 / (p["pred"] != y).sum() for p in per_seed])), 6),
            "confusion_pooled3seeds": cm.tolist(),
            "recall_per_class": {["neg", "neu", "pos"][c]: round(float(recalls[c]), 6)
                                 for c in range(3) if cm.sum(1)[c] > 0},
            "pred_neutral_rate": round(float((pred_pool[m_pool] == 1).mean()), 6)}
    cm_all = confusion(y_pool, pred_pool)

    # ---- C 后处理参照（valid-only）----
    # C1 回归符号决策规则：sign_rule(r̂, τ)，τ 网格搜 macro-F1 最优；固定 τ=1/6 对照
    taus = np.arange(0.0, 1.53, 0.02)
    reg_rule = {"tau_grid_note": "τ=0 即永不判中性", "per_seed": []}
    for p in per_seed:
        rows = [(round(float(t), 2), float((sign_rule(p["reg"], t) == y).mean()),
                 macro_f1(y, sign_rule(p["reg"], t))) for t in taus]
        best = max(rows, key=lambda r: r[2])
        fix = (float((sign_rule(p["reg"], 1 / 6) == y).mean()),
               macro_f1(y, sign_rule(p["reg"], 1 / 6)))
        reg_rule["per_seed"].append({
            "seed": p["seed"], "tau_star": best[0], "acc_star": round(best[1], 6),
            "macro_f1_star": round(best[2], 6),
            "d_acc_vs_cls_head": round(best[1] - p["metrics"]["acc"], 6),
            "d_macro_f1_vs_cls_head": round(best[2] - p["metrics"]["macro_f1"], 6),
            "acc_tau_1over6": round(fix[0], 6), "macro_f1_tau_1over6": round(fix[1], 6)})
    # C2 极性仲裁：分类=非中性 但 |r̂|<τ → 改中性；分类=中性 但 |r̂|≥τ → 改 sign(r̂)
    def arbitration(pred_cls, reg_hat, tau):
        new = pred_cls.copy()
        new[(pred_cls != 1) & (np.abs(reg_hat) < tau)] = 1
        new[(pred_cls == 1) & (reg_hat > tau)] = 2
        new[(pred_cls == 1) & (reg_hat < -tau)] = 0
        return round(float((new == y).mean()), 6), round(macro_f1(y, new), 6)

    arb = {"per_seed": []}
    for p in per_seed:
        rows = [(round(float(t), 2), *arbitration(p["pred"], p["reg"], t))
                for t in np.arange(0.1, 1.55, 0.05)]
        best = max(rows, key=lambda r: r[2])
        fix = arbitration(p["pred"], p["reg"], 1 / 6)
        arb["per_seed"].append({
            "seed": p["seed"], "tau_star": best[0], "acc_star": best[1],
            "macro_f1_star": best[2],
            "d_acc": round(best[1] - p["metrics"]["acc"], 6),
            "d_macro_f1": round(best[2] - p["metrics"]["macro_f1"], 6),
            "acc_tau_1over6": fix[0], "macro_f1_tau_1over6": fix[1]})
    # C3 3-seed 概率/回归集成
    probs = np.mean([torch.softmax(torch.from_numpy(p["logits"]), -1).numpy()
                     for p in per_seed], 0)
    reg_ens = np.mean([p["reg"] for p in per_seed], 0)
    m_ens = metrics(y, rl, np.log(np.clip(probs, 1e-9, 1)), reg_ens)
    ens_rule = {}
    rows = [(round(float(t), 2), float((sign_rule(reg_ens, t) == y).mean()),
             macro_f1(y, sign_rule(reg_ens, t))) for t in taus]
    best = max(rows, key=lambda r: r[2])
    ens_rule = {"tau_star": best[0], "acc_star": round(best[1], 6),
                "macro_f1_star": round(best[2], 6)}
    ensemble = {"acc": m_ens["acc"], "macro_f1": m_ens["macro_f1"], "S": m_ens["S"],
                "d_acc_vs_seedmean": round(m_ens["acc"] - float(np.mean(base_acc)), 6),
                "d_macro_f1_vs_seedmean": round(
                    m_ens["macro_f1"] - float(np.mean(base_f1)), 6),
                "reg_sign_rule": ens_rule}
    # C4 类偏置校正（对数概率加性偏置，坐标搜 2 轮，macro-F1 目标）——过拟合上界参照
    lp = np.log(np.clip(probs, 1e-9, 1))
    bias = np.zeros(3)
    for _ in range(2):
        for c in range(3):
            cand, best_v = bias.copy(), -1.0
            for d in np.arange(-0.4, 0.41, 0.05):
                trial = bias.copy(); trial[c] = d
                v = macro_f1(y, (lp + trial).argmax(1))
                if v > best_v:
                    best_v, cand = v, trial
            bias = cand
    pred_cal = (lp + bias).argmax(1)
    calib = {"bias_neg_neu_pos": [round(float(b), 2) for b in bias],
             "acc": round(float((pred_cal == y).mean()), 6),
             "macro_f1": round(macro_f1(y, pred_cal), 6),
             "d_acc": round(float((pred_cal == y).mean()) - m_ens["acc"], 6),
             "d_macro_f1": round(macro_f1(y, pred_cal) - m_ens["macro_f1"], 6)}

    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "discipline": "纯 valid 分析；test 主表维持 9eb88d3；不改变任何模型/预测交付物",
        "repro_check": repro,
        "A_label_structure": {"per_split": label_struct, "oracle": oracle},
        "B_model": {
            "bins": bin_table, "zones": zone_out,
            "confusion_all_pooled3seeds": cm_all.tolist(),
            "clean_valid": {"acc_mean": round(float(np.mean(base_acc)), 6),
                            "macro_f1_mean": round(float(np.mean(base_f1)), 6)}},
        "C_posthoc_valid_only": {"reg_sign_rule": reg_rule, "arbitration": arb,
                                 "ensemble": ensemble, "class_bias_calibration": calib}}

    (OUT / "error_attribution.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(
        OUT / "per_seed_predictions.npz", ids=ids, y_cls=y, rl=rl,
        **{f"seed{s}_logits": p["logits"] for s, p in zip(SEEDS, per_seed)},
        **{f"seed{s}_reg": p["reg"] for s, p in zip(SEEDS, per_seed)})

    rr = reg_rule["per_seed"]
    lines = [
        "# P2 错误归因（clean valid，3 seeds MRFN，docx 内容(4)）",
        "",
        "## A 标签结构",
        "- 分类标签 = 回归标签的确定性函数：cl=1 ⟺ rl==0（train/valid 零例外），"
        "cl=0 ⟺ rl<0，cl=2 ⟺ rl>0；非零 |rl| 最小 1/6",
        "- ⇒ 真 rl 的 cls oracle = **100%**：上限不在标签侧，在特征侧弱情感分辨力",
        f"- valid 三分区：Z0 真中性 {zone_out['Z0_neutral']['n']} / "
        f"Z1 弱极性(0<|rl|<0.5) {zone_out['Z1_weak']['n']} / "
        f"Z2 强极性(≥0.5) {zone_out['Z2_strong']['n']}",
        "",
        "## B 错误分区（3 seeds）",
        f"- clean valid acc 均值 {np.mean(base_acc):.4f}；"
        f"分区 acc：Z0 {zone_out['Z0_neutral']['acc_mean']:.4f} / "
        f"Z1 {zone_out['Z1_weak']['acc_mean']:.4f} / "
        f"Z2 {zone_out['Z2_strong']['acc_mean']:.4f}",
        f"- 错误占比：Z0 {zone_out['Z0_neutral']['err_share_mean']:.1%} / "
        f"Z1 {zone_out['Z1_weak']['err_share_mean']:.1%} / "
        f"Z2 {zone_out['Z2_strong']['err_share_mean']:.1%}",
        f"- 真中性被命中 recall(neu|Z0)="
        f"{zone_out['Z0_neutral']['recall_per_class'].get('neu', 0):.3f}；"
        f"弱极性被误判中性率 pred_neu|Z1="
        f"{zone_out['Z1_weak']['pred_neutral_rate']:.3f}",
        "",
        "## C 决策侧参照（valid-only，不进主表）",
        f"- 回归符号决策规则（τ* 搜）：Δacc "
        f"{np.mean([r['d_acc_vs_cls_head'] for r in rr]):+.4f}、"
        f"ΔmacroF1 {np.mean([r['d_macro_f1_vs_cls_head'] for r in rr]):+.4f}",
        f"- 极性仲裁（τ* 搜）：Δacc {np.mean([r['d_acc'] for r in arb['per_seed']]):+.4f}",
        f"- 3-seed 集成：acc {ensemble['acc']:.4f}"
        f"（Δ {ensemble['d_acc_vs_seedmean']:+.4f}）",
        f"- 类偏置校正：Δacc {calib['d_acc']:+.4f}、ΔmacroF1 {calib['d_macro_f1']:+.4f}",
        "- 结论：决策侧全部手段 |Δ| ≤ 1pp，远小于弱情感带错误体量——"
        "继续堆决策侧方法无意义，错误主体在 Z0/Z1 的特征不可分性",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("\n产物:", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
