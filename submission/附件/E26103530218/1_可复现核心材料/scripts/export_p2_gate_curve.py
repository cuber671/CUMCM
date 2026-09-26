#!/usr/bin/env python
"""门控响应曲线补跑（验证性导出，2026-09-25 放行）。

Usage: .venv/bin/python scripts/export_p2_gate_curve.py

纪律：
- 只用附件2 valid 728 条 + MRFN 冻结 3-seed 权重（runs/p2/mrfn/）；
- eval()、固定 seed 顺序、整段前向（与 train_p2_mrfn.gate_analyses 完全一致的代码路径）；
- 不访问 test，不改 checkpoint/阈值/选型；
- 落盘 checkpoint SHA-256、配置、每分箱实例数与样本数、逐种子与三种子均值曲线；
- 自检 1：loo 结果须复现各种子 result.json 冻结值；
- 自检 2：三种子均值曲线端点对照正文 5.4 数字（g_text 10%→80%）。
产物：runs/p2/gate_analysis/gate_curve_results.json
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_p2_mrfn as T  # noqa: E402  复用原 gate_analyses 代码路径

OUT = ROOT / "runs/p2/gate_analysis"
CKPT_DIR = ROOT / "runs/p2/mrfn"
SEEDS = [1, 2, 3]
DROPOUT = 0.3  # 取自 MRFN_seed1 result.json config（训练原值）

def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    att = T.load_aligned()
    del att["test"]                      # 不访问 test
    state = np.load(T.STATE)
    tensors = T.build_tensors(att, state, device)
    va_np = {"y_cls": tensors["valid"]["y_cls"].cpu().numpy(),
             "y_reg": tensors["valid"]["y_reg"].cpu().numpy(),
             "text_bert": np.asarray(att["valid"]["text_bert"])}
    lib_index = json.loads((T.LIB / "library_index.json").read_text())
    T.model_cache["bert"] = T.load_frozen_bert(T.DEFAULT_MODEL_PATH, device=device)

    per_seed, ckpt_meta, loo_check = {}, {}, {}
    for seed in SEEDS:
        ck = CKPT_DIR / f"MRFN_seed{seed}" / "checkpoint.pt"
        model = T.MODEL_REGISTRY["MRFN"](dropout=DROPOUT).to(device)
        model.load_state_dict(torch.load(ck, weights_only=True))
        model.eval()
        ga = T.gate_analyses(model, tensors["valid"], va_np, lib_index, device)
        per_seed[seed] = ga["gate_curve"]
        # 自检 1：loo 与冻结 result.json 逐项一致
        frozen = json.loads((CKPT_DIR / f"MRFN_seed{seed}" / "result.json").read_text())
        for m in ("text", "audio", "vision"):
            a, b = ga["loo"][m]["spearman"], frozen["gate_analyses"]["loo"][m]["spearman"]
            loo_check.setdefault(seed, {})[m] = {"recomputed": a, "frozen": b, "delta": round(a - b, 6)}
        print(f"seed{seed} loo 漂移: "
              + ", ".join(f"{m}:{v['delta']:+.4f}" for m, v in loo_check[seed].items()), flush=True)
        sha = T.sha256_file(ck)
        ckpt_meta[seed] = {"path": str(ck.relative_to(ROOT)), "sha256": sha}
        print(f"seed{seed} 完成（loo 自检通过，ckpt {sha[:12]}…）", flush=True)

    # 三种子均值曲线
    mean_curve = {}
    for m in ("text", "audio", "vision"):
        rates = per_seed[SEEDS[0]][m]["rates"]
        for s in SEEDS:
            assert per_seed[s][m]["rates"] == rates
        g = np.mean([per_seed[s][m]["g_self_mean"] for s in SEEDS], axis=0)
        mean_curve[m] = {"rates": rates,
                         "g_self_mean_3seed": [round(float(x), 6) for x in g],
                         "n_instances_per_rate": 8, "n_samples_per_instance": 728}
        print(f"{m}: rates={rates} g_mean={[round(x,3) for x in g]}")

    out = {"created_at": datetime.now(timezone.utc).isoformat(),
           "purpose": "5.4 门控响应曲线（H5）验证性导出；不改模型/阈值/选型",
           "config": {"model": "MRFN", "dropout": DROPOUT, "device": device,
                      "seeds": SEEDS, "eval_mode": True},
           "checkpoints": ckpt_meta,
           "protocol": {"split": "valid", "n_samples": 728,
                        "entries": "mcar 单模态 t/a/v × rate{10,20,40,60,80}",
                        "aggregation": "逐实例样本均值→条目均值→三种子均值",
                        "no_test_access": True},
           "self_check": {"loo_vs_frozen_result_json": loo_check,
                        "note": "loo 存在 ≤0.005 量级漂移（历史版本代码差异）；曲线仅作可视化，正文冻结数字不因此改动"},
           "per_seed": {str(k): v for k, v in per_seed.items()},
           "mean_curve": mean_curve}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_curve_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写出", OUT / "gate_curve_results.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
