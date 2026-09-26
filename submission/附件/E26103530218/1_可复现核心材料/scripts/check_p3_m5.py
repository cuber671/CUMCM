"""P3 M5 验收脚本：g–φ 一致性分析（valid 728 条全量，方案 §6）。

定位（预注册）：**分析**模型信任机制（门控 g，融合权重）与实际贡献（φ，Shapley）
的趋势一致性——二者含义不同，不验证等同。
  - g_m：MRFN 门控（3-seed 均值，自然输入满联盟）；
  - φ_m：M2 同款 8 联盟精确 Shapley（valid 全量 728 条 × 3 seeds 均值），
    分类取预测类分量（带符号），回归取标量；
  - 主分析：逐模态 Pearson 与 Spearman 双报告（g_m vs |φ_m|，cls 与 reg 两口径）；
  - 辅助：argmax|φ| 与 argmax g 的逐样本一致率。
趋势门槛（跑前冻结）：Spearman(g_text, |φ_text^cls,pred|) > 0（文本主通道趋势为正，
n=728）；其余模态/口径只报告不设门。
产物：runs/p3/m5/gphi_valid.json + scatter.csv（论文散点图数据）。
退出码：趋势门槛过 = 0。
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
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.p2.data import CONTRACT, load_aligned  # noqa: E402
from src.p2.models import MODEL_REGISTRY  # noqa: E402
from src.p2.pipeline import load_stats, zscore_reset  # noqa: E402
from src.p2.text_mask import BertTextEncoder, mask_text_tokens  # noqa: E402
from src.p3.shapley import COALITIONS, MODS, exact_shapley  # noqa: E402

CKPT = ROOT / "runs/p2/bft/MRFN_seed{seed}"
STATE = ROOT / "runs/p2/data/m1a_state.npz"
OUT = ROOT / "runs/p3/m5"
BATCH = 32
O_KEY = {"text": "o_text", "audio": "o_audio", "vision": "o_vision"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("✅" if ok else "❌"), name, "" if ok else json.dumps(detail, ensure_ascii=False))

    att = load_aligned()["valid"]
    state = np.load(STATE)
    stats = load_stats()
    n = att["audio"].shape[0]
    tb = np.asarray(att["text_bert"]).astype(np.int64)
    content = state["valid_content"].astype(bool)
    o = {m: state[f"valid_{O_KEY[m]}"].astype(bool) for m in MODS}
    aud = zscore_reset(att["audio"], stats["audio"]["mean"], stats["audio"]["std"],
                       o["audio"], name="valid.audio")
    vis = zscore_reset(att["vision"], stats["vision"]["mean"], stats["vision"]["std"],
                       o["vision"], name="valid.vision")

    ensemble, fps = [], []
    for seed in (1, 2, 3):
        sd = Path(str(CKPT).format(seed=seed))
        be = BertTextEncoder(unfreeze_last=1).to(device)
        be.load_state_dict(torch.load(sd / "bert_checkpoint.pt", weights_only=True)); be.eval()
        m = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        m.load_state_dict(torch.load(sd / "checkpoint.pt", weights_only=True)); m.eval()
        ensemble.append((m, be))
        fps.append(hashlib.sha256(Path(sd / "checkpoint.pt").read_bytes()).hexdigest()[:12])

    # ---- 批量联盟前向：v(S)（3-seed 均值）+ 满联盟门控 ----
    v_cls = {S: np.zeros((n, 3)) for S in COALITIONS}
    v_reg = {S: np.zeros(n) for S in COALITIONS}
    gates = np.zeros((n, 3))
    for i in range(0, n, BATCH):
        sl = slice(i, i + BATCH)
        c_t = torch.from_numpy(content[sl]).to(device)
        aud_t = torch.from_numpy(aud[sl]).to(device)
        vis_t = torch.from_numpy(vis[sl]).to(device)
        for S in COALITIONS:
            ids = tb[sl] if "text" in S else mask_text_tokens(
                tb[sl], content[sl], CONTRACT)
            ids_t = torch.from_numpy(ids).to(device)
            ps, rs = [], []
            for model, be in ensemble:
                avail = {m: (torch.from_numpy(o[m][sl]).to(device) if m in S
                             else torch.zeros_like(c_t)) for m in MODS}
                with torch.no_grad():
                    feats = {"text": be(ids_t), "audio": aud_t, "vision": vis_t}
                    out = model(feats, c_t, avail)
                ps.append(torch.softmax(out["logits"], -1).cpu().numpy())
                rs.append(out["reg"].reshape(-1).cpu().numpy())
                if S == frozenset(MODS):
                    gates[sl] += out["gates"].cpu().numpy() / len(ensemble)
            v_cls[S][sl] = np.mean(ps, axis=0)
            v_reg[S][sl] = np.mean(rs, axis=0)
        print(f"  batch {i // BATCH + 1}/{(n + BATCH - 1) // BATCH}", flush=True)
    gates /= 1.0                                    # 已在循环内按 seed 平均
    pred = v_cls[frozenset(MODS)].argmax(1)

    phi_cls = exact_shapley({S: v_cls[S] for S in COALITIONS})      # m → (N,3)
    phi_reg = exact_shapley({S: v_reg[S] for S in COALITIONS})      # m → (N,)

    # ---- 相关性（逐模态，双口径双系数）----
    mi = {m: i for i, m in enumerate(MODS)}
    rows, corr = [], {}
    for m in MODS:
        g = gates[:, mi[m]]
        pc = np.abs(phi_cls[m][np.arange(n), pred])
        pr = np.abs(phi_reg[m])
        corr[m] = {
            "cls": {"pearson": round(float(pearsonr(g, pc).statistic), 4),
                    "spearman": round(float(spearmanr(g, pc).statistic), 4)},
            "reg": {"pearson": round(float(pearsonr(g, pr).statistic), 4),
                    "spearman": round(float(spearmanr(g, pr).statistic), 4)}}
        for j in range(n):
            rows.append({"sample": j, "modality": m, "gate": round(float(g[j]), 5),
                         "abs_phi_cls": round(float(pc[j]), 6),
                         "abs_phi_reg": round(float(pr[j]), 6)})
    phi_mat = np.stack([np.abs(phi_cls[m][np.arange(n), pred]) for m in MODS], 1)
    agree = float((phi_mat.argmax(1) == gates.argmax(1)).mean())

    add("trend_text_spearman_positive",
        corr["text"]["cls"]["spearman"] > 0,
        {"spearman_text_cls": corr["text"]["cls"]["spearman"], "gate": ">0",
         "n": n, "all": corr, "argmax_agreement": round(agree, 4),
         "note": "g=融合权重、φ=预测贡献，趋势分析非等同验证（方案 §6）"})

    with open(OUT / "scatter.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "MRFN+ (3-seed ensemble)", "checkpoint_sha12": fps,
        "split": "valid", "n": n,
        "shapley": "M2 同款 8 联盟精确枚举（valid 全量）",
        "correlations": corr, "argmax_agreement": round(agree, 4),
        "checks": checks, "all_pass": all(c["ok"] for c in checks)}
    (OUT / "gphi_valid.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ncache → {OUT / 'gphi_valid.json'} + scatter.csv")
    print("ALL PASS" if payload["all_pass"] else "FAILED")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
