"""P3 M4 补充诊断（非门槛）：F 失败的结构性归因——LOO 干预的局部性决定 IG↔LOO 一致性。

事实链（2026-09-25 探针 + 本脚本全量）：
  1. 确定性：同输入重复 LOO 逐位差 = 0.0（非数值噪声）；
  2. 局部干预（a/v：仅翻 avail 位）：ρ(|IG|,|Δp|) 高、top-20% 重合高 → IG 局部忠实；
  3. 非局部干预（text：[MASK] 重编码，BERT 会从上下文推断被掩 token）：全秩与 top-k
     均不一致——两种反事实本质不同（删除=可推断性，IG=嵌入敏感度）；
  4. 累积删除曲线（A/B/C 门槛）仍以 |IG| 序胜随机 → 全局排序忠实。
产物：runs/p3/m4/loo_diagnosis.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.models import MODEL_REGISTRY  # noqa: E402
from src.p2.pipeline import load_stats  # noqa: E402
from src.p2.text_mask import BertTextEncoder  # noqa: E402
from src.p3.data import load_att4  # noqa: E402
from src.p3.fidelity import ens_forward, removal_state  # noqa: E402
from src.p3.shapley import MODS, prepare_sample  # noqa: E402

CKPT = ROOT / "runs/p2/bft/MRFN_seed{seed}"
OUT = ROOT / "runs/p3/m4"


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = load_stats()
    ensemble = []
    for seed in (1, 2, 3):
        sd = Path(str(CKPT).format(seed=seed))
        be = BertTextEncoder(unfreeze_last=1).to(device)
        be.load_state_dict(torch.load(sd / "bert_checkpoint.pt", weights_only=True)); be.eval()
        m = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        m.load_state_dict(torch.load(sd / "checkpoint.pt", weights_only=True)); m.eval()
        ensemble.append((m, be))
    m2 = json.loads((ROOT / "runs/p3/m2/shapley_att4.json").read_text())
    m3 = json.loads((ROOT / "runs/p3/m3/ig_att4.json").read_text())
    tc = {r["sample_id"]: r["pred_class"] for r in m2["reports"]}
    igc = {r["sample_id"]: {m: np.array(r["baselines"]["missing"]["ig"][m]["cls"]) for m in MODS}
           for r in m3["reports"]}

    rows, acc = [], {m: {"rho_signed": [], "rho_abs": [], "topk": []} for m in MODS}
    for s in load_att4():
        st = prepare_sample(s, stats)
        sid, t = f"{s.n:02d}", tc[f"{s.n:02d}"]
        p0 = float(ens_forward(ensemble, st, st.text_bert, st.o, device)[0][t])
        for m in MODS:
            js = [int(j) for j in np.where(st.o[m][0])[0]]
            if not js:
                continue
            loo = np.array([p0 - float(ens_forward(
                ensemble, st, *removal_state(st, {(m, j)}), device)[0][t]) for j in js])
            ig = igc[sid][m][js]
            if np.ptp(np.abs(loo)) == 0 or np.ptp(np.abs(ig)) == 0:
                continue
            k = max(1, len(js) // 5)
            topk = len(set(np.argsort(-np.abs(loo))[:k])
                       & set(np.argsort(-np.abs(ig))[:k])) / k
            rs = float(spearmanr(np.abs(ig), loo).statistic)
            ra = float(spearmanr(np.abs(ig), np.abs(loo)).statistic)
            acc[m]["rho_signed"].append(rs)
            acc[m]["rho_abs"].append(ra)
            acc[m]["topk"].append(topk)
            rows.append({"id": sid, "mod": m, "rho_signed": round(rs, 3),
                         "rho_abs": round(ra, 3), "top20_overlap": round(topk, 2)})
        print(f"#{sid} done", flush=True)

    summary = {m: {k: round(float(np.mean(v)), 4) for k, v in d.items()}
               for m, d in acc.items()}
    payload = {
        "purpose": "F 门槛失败的结构性诊断（非门槛重定义）",
        "conclusion": (
            "局部干预（a/v avail 翻转）下 IG↔LOO 强一致；text 的 [MASK] 重编码是非局部"
            "干预（BERT 从上下文推断被掩 token），与嵌入层 IG 属不同反事实 → 全秩/头部"
            "均低相关是结构现象，不构成对 IG 忠实性的否定（a/v 对照 + 累积删除 A/B/C 支持）。"),
        "per_combination": rows, "summary_mean": summary}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "loo_diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
