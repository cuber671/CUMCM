"""P3 M2 验收脚本：MRFN+ 3-seed 集成在附件4 全 20 条上的精确 Shapley。

检查项：
  1. 联盟枚举：每条恰好 8 个联盟、键序与 COALITIONS 冻结序一致（缓存契约）；
  2. 完备性：20/20 分类逐类 |Σφ − (p(x)−p∅)| < 1e-5 且回归同式成立；
  3. #13 哑玩家：o_vision 全零 → v(S∪{v})=v(S) → φ_v≡0（cls 三类与 reg 全零），单独标记；
  4. 锚点 sanity（仅内部，不入解释/精度声明）：5 条附件2 test 重合样本预测 vs 真实标签；
  5. 指纹：checkpoint SHA256 + 联盟值缓存 → runs/p3/m2/shapley_att4.json（M3/M4 复用）。
退出码：全过 = 0。
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import sha256_file  # noqa: E402
from src.p2.models import MODEL_REGISTRY  # noqa: E402
from src.p2.text_mask import BertTextEncoder  # noqa: E402
from src.p3.data import PREREG, load_att4  # noqa: E402
from src.p3.shapley import COALITIONS, MODS, phi_report, prepare_sample  # noqa: E402

CKPT = ROOT / "runs/p2/bft/MRFN_seed{seed}"
OUT = ROOT / "runs/p3/m2"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("✅" if ok else "❌"), name, "" if ok else detail)

    # ---- 集成（MRFN+ 采纳权重）----
    ensemble, fps = [], []
    for seed in (1, 2, 3):
        sd = Path(str(CKPT).format(seed=seed))
        be = BertTextEncoder(unfreeze_last=1).to(device)
        be.load_state_dict(torch.load(sd / "bert_checkpoint.pt", weights_only=True)); be.eval()
        m = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        m.load_state_dict(torch.load(sd / "checkpoint.pt", weights_only=True)); m.eval()
        ensemble.append((m, be))
        fps.append(hashlib.sha256(Path(sd / "checkpoint.pt").read_bytes()).hexdigest()[:12])

    # ---- 全量 20 条 ----
    samples = load_att4()
    reports, residuals = [], []
    for s in samples:
        r = phi_report(ensemble, prepare_sample(s), device)
        reports.append(r)
        residuals.append((r["sample_id"], r["completeness"]["cls_residual"],
                          r["completeness"]["reg_residual"]))
        print(f"  #{s.n:02d} pred={r['pred_class']} conf={r['confidence']:.3f} "
              f"I={r['pred_intensity']:+.3f} φ_cls(t/a/v)={r['phi_cls']} "
              f"res={r['completeness']['cls_residual']:.1e}", flush=True)

    # ---- 1. 联盟枚举 ----
    want = ["|".join(sorted(S)) if S else "∅" for S in COALITIONS]
    got_keys = [list(r["coalition_values"].keys()) for r in reports]
    add("coalition_enumeration", all(k == want for k in got_keys),
        {"n_coalitions": len(want), "order_frozen": want})

    # ---- 2. 完备性 ----
    ok_c = all(r["completeness"]["pass"] for r in reports)
    max_cls = max(r[1] for r in residuals); max_reg = max(r[2] for r in residuals)
    add("completeness_all", ok_c,
        {"n_pass": sum(r["completeness"]["pass"] for r in reports), "n": len(reports),
         "max_cls_residual": max_cls, "max_reg_residual": max_reg})

    # ---- 3. #13 哑玩家 ----
    r13 = next(r for r in reports if r["sample_id"] == "13")
    phi_v13_cls = r13["phi_cls"]["vision"]; phi_v13_reg = r13["phi_reg"]["vision"]
    dummy = (all(abs(x) == 0.0 for x in phi_v13_cls) and phi_v13_reg == 0.0)
    r13["natural_vision_missing"] = True   # 单独标记（解释卡用）
    add("dummy_player_13", dummy,
        {"phi_cls_vision": phi_v13_cls, "phi_reg_vision": phi_v13_reg,
         "note": "o_vision 全零 → v(S∪{v})≡v(S)，φ_v≡0（哑玩家公理实证）"})

    # ---- 4. 锚点 sanity（仅内部）----
    a2 = pickle.load(open(ROOT / "data/附件2-数据集特征文件/aligned_50.pkl", "rb"))
    texts = [str(t) for t in a2["test"]["raw_text"]]
    sanity = []
    for r in reports:
        s = next(x for x in samples if f"{x.n:02d}" == r["sample_id"])
        if s.n in PREREG["att2_overlap"]:
            j = texts.index(s.raw_text)
            y = int(a2["test"]["classification_labels"][j])
            sanity.append({"n": s.n, "pred": r["pred_class"], "true": y,
                           "hit": r["pred_class"] == y})
    add("anchor_sanity_internal", len(sanity) == 5,
        {"detail": sanity, "agreement": sum(x["hit"] for x in sanity),
         "discipline": "仅内部 sanity，不入解释输出与任何精度声明"})

    # ---- 5. 缓存落盘（指纹 = 权重 SHA + 时间）----
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "MRFN+ (3-seed ensemble, runs/p2/bft)",
        "checkpoint_sha12": fps,
        "coalition_order": want,
        "prereg": PREREG,
        "reports": reports,
        "checks": checks,
        "all_pass": all(c["ok"] for c in checks),
    }
    (OUT / "shapley_att4.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ncache → {OUT / 'shapley_att4.json'}")
    print("ALL PASS" if payload["all_pass"] else "FAILED")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
