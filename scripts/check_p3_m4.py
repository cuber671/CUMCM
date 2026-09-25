"""P3 M4 验收脚本：附件4 全 20 条保真度验证（预注册门槛见 src/p3/fidelity.py 头注）。

检查项（门槛在跑前冻结）：
  A deletion_superiority：mean AOPC>0 且逐样本胜（AOPC>0）≥15/20；
  B insertion_superiority：mean ΔAUC(插入)>0 且逐样本胜 ≥15/20；
  C compr_suff：compr@0.2 均值>0 且 |suff@0.2| 均值<compr 均值；
  D randomization_collapse：随机权重模型 IG 与原 IG 排序 Spearman 中位数 ≤0.3（各模态）；
  E stability_text：text 扰动稳定性 mean ρ≥0.6（a/v 报告不设门）；
  F ig_loo_text：IG↔LOO text mean ρ≥0.3（a/v 报告）。
产物：runs/p3/m4/fidelity_att4.json + curves.csv（逐样本逐比例，供论文曲线图）。
退出码：全过 = 0。
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import load_aligned  # noqa: E402
from src.p2.models import MODEL_REGISTRY  # noqa: E402
from src.p2.pipeline import load_stats  # noqa: E402
from src.p2.text_mask import BertTextEncoder  # noqa: E402
from src.p3.data import load_att4  # noqa: E402
from src.p3.fidelity import (deletion_insertion, loo_consistency,  # noqa: E402
                             randomization_ig, stability_ig)
from src.p3.ig import train_mean_text_embedding  # noqa: E402
from src.p3.shapley import MODS, prepare_sample  # noqa: E402

CKPT = ROOT / "runs/p2/bft/MRFN_seed{seed}"
STATE = ROOT / "runs/p2/data/m1a_state.npz"
OUT = ROOT / "runs/p3/m4"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("✅" if ok else "❌"), name, "" if ok else json.dumps(detail, ensure_ascii=False)[:400])

    m2 = json.loads((ROOT / "runs/p3/m2/shapley_att4.json").read_text())
    m3 = json.loads((ROOT / "runs/p3/m3/ig_att4.json").read_text())
    target = {r["sample_id"]: r["pred_class"] for r in m2["reports"]}
    ig_cache = {r["sample_id"]: {m: np.array(r["baselines"]["missing"]["ig"][m]["cls"])
                                 for m in MODS} for r in m3["reports"]}

    att = load_aligned()
    state = np.load(STATE)
    stats = load_stats()
    ensemble, mean_embs, fps = [], {}, []
    for seed in (1, 2, 3):
        sd = Path(str(CKPT).format(seed=seed))
        be = BertTextEncoder(unfreeze_last=1).to(device)
        be.load_state_dict(torch.load(sd / "bert_checkpoint.pt", weights_only=True)); be.eval()
        m = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        m.load_state_dict(torch.load(sd / "checkpoint.pt", weights_only=True)); m.eval()
        ensemble.append((m, be))
        fps.append(hashlib.sha256(Path(sd / "checkpoint.pt").read_bytes()).hexdigest()[:12])
        mean_embs[seed] = train_mean_text_embedding(
            be, np.asarray(att["train"]["text_bert"]).astype(np.int64),
            state["train_content"].astype(bool), device)
    del att
    print("ensemble ready", flush=True)

    samples = load_att4()
    curves_rows, per_sample = [], []
    loo_all = {m: [] for m in MODS}
    stab_all = {m: [] for m in MODS}
    for s in samples:
        st = prepare_sample(s, stats)
        sid, tc = f"{s.n:02d}", target[f"{s.n:02d}"]
        ig_cls = ig_cache[sid]
        fid = deletion_insertion(ensemble, st, ig_cls, device, tc, s.n)
        for row in fid["deletion"]:
            curves_rows.append({"sample_id": sid, "curve": "deletion", **row})
        for row in fid["insertion"]:
            curves_rows.append({"sample_id": sid, "curve": "insertion", **row})
        loo = loo_consistency(ensemble, st, ig_cls, device, tc)
        for m in MODS:
            if loo[m]["rho"] == loo[m]["rho"]:
                loo_all[m].append(loo[m]["rho"])
        ig_noisy, sig = stability_ig(ensemble, st, stats, device, tc, mean_embs, s.n)
        stab = {}
        for m in MODS:
            js = np.where(st.o[m][0])[0]
            if len(js) == 0 or np.ptp(ig_cls[m][js]) == 0 or np.ptp(ig_noisy[m][js]) == 0:
                stab[m] = float("nan")
            else:
                stab[m] = float(spearmanr(ig_cls[m][js], ig_noisy[m][js]).statistic)
                stab_all[m].append(stab[m])
        per_sample.append({"sample_id": sid, "target_cls": tc, **fid,
                           "loo": loo, "stability": stab, "noise_sigma": sig})
        print(f"  #{sid} AOPC={fid['aopc']:+.3f} compr={fid['compr_at_20']:+.3f} "
              f"loo_t={loo['text']['rho']:.2f} stab_t={stab['text']:.2f}", flush=True)

    # ---- D 随机化坍缩（全 20 条，BERT 固定 seed1）----
    rnd_rho = {m: [] for m in MODS}
    for s in samples:
        st = prepare_sample(s, stats)
        sid = f"{s.n:02d}"
        ig_rnd, _ = randomization_ig(st, stats, device, ensemble[0][1])
        for m in MODS:
            js = np.where(st.o[m][0])[0]
            if len(js) == 0 or np.ptp(ig_cache[sid][m][js]) == 0 or np.ptp(ig_rnd[m][js]) == 0:
                continue
            rnd_rho[m].append(float(spearmanr(ig_cache[sid][m][js],
                                              ig_rnd[m][js]).statistic))
        print(f"  #{sid} randomized", flush=True)

    # ---- A/B/C/D/E/F ----
    aopcs = [r["aopc"] for r in per_sample]
    add("A_deletion_superiority",
        float(np.mean(aopcs)) > 0 and sum(a > 0 for a in aopcs) >= 15,
        {"mean_aopc": round(float(np.mean(aopcs)), 4),
         "wins": int(sum(a > 0 for a in aopcs)), "n": len(aopcs),
         "per_sample": [round(a, 4) for a in aopcs]})
    dauc = [r["ins_auc_attr"] - r["ins_auc_rand"] for r in per_sample]
    add("B_insertion_superiority",
        float(np.mean(dauc)) > 0 and sum(d > 0 for d in dauc) >= 15,
        {"mean_dauc": round(float(np.mean(dauc)), 4), "wins": int(sum(d > 0 for d in dauc))})
    mc, ms = (float(np.mean([r["compr_at_20"] for r in per_sample])),
              float(np.mean([abs(r["suff_at_20"]) for r in per_sample])))
    add("C_compr_suff", mc > 0 and ms < mc,
        {"mean_compr_at_20": round(mc, 4), "mean_abs_suff_at_20": round(ms, 4)})
    rnd_med = {m: round(float(np.median(v)), 4) for m, v in rnd_rho.items() if v}
    add("D_randomization_collapse", all(v <= 0.3 for v in rnd_med.values()),
        {"median_rho": rnd_med, "gate": "≤0.3 各模态"})
    add("E_stability_text",
        float(np.mean(stab_all["text"])) >= 0.6,
        {"text_mean": round(float(np.mean(stab_all["text"])), 4), "gate": "≥0.6",
         "av_report": {m: round(float(np.mean(stab_all[m])), 4) if stab_all[m] else None
                       for m in ("audio", "vision")}})
    add("F_ig_loo_text", float(np.mean(loo_all["text"])) >= 0.3,
        {"text_mean": round(float(np.mean(loo_all["text"])), 4), "gate": "≥0.3",
         "av_report": {m: round(float(np.mean(loo_all[m])), 4) if loo_all[m] else None
                       for m in ("audio", "vision")}})

    # ---- 曲线 CSV + JSON ----
    with open(OUT / "curves.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(curves_rows[0].keys()))
        w.writeheader(); w.writerows(curves_rows)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "MRFN+ (3-seed ensemble)", "checkpoint_sha12": fps,
        "gates_prereg": "A mean AOPC>0 & ≥15/20；B mean ΔAUC>0 & ≥15/20；C compr>0 & |suff|<compr；"
                        "D median ρ≤0.3；E text ρ≥0.6；F text ρ≥0.3（fidelity.py 头注冻结）",
        "evidence_source": "M3 主基线 IG_cls（runs/p3/m3/ig_att4.json）",
        "per_sample": per_sample, "randomization": {"median_rho": rnd_med},
        "checks": checks, "all_pass": all(c["ok"] for c in checks)}
    (OUT / "fidelity_att4.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ncache → {OUT / 'fidelity_att4.json'} + curves.csv")
    print("ALL PASS" if payload["all_pass"] else "FAILED")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
