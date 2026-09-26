"""P2 M5：全网格收口——2×2（clean vs 缺失增强训练）+ §9 四消融 + §8.5 填充基线。

六组配置（每组 3 seeds，GPU，训练配方同契约 §3）：
  B3_aug_zero      B3 掩码库混合训练（2×2 增强行 + 消融参照）
  MRFN_clean       MRFN clean 训练（消融 Missing Augmentation− + 2×2 行）
  MRFN_noState     去缺失状态编码（零填充代替 MissingEmbedding）
  MRFN_noMaskAttn  注意力不屏蔽不可用证据
  MRFN_noGate      去门控融合（三模态 concat）
  B3_aug_ff        前向填充基线（§8.5；均值填充 ≡ 零填充，见 pipeline 注）
汇总：runs/p2/m5/m5_summary.json（2×2 表 + 消融表 + 填充对比）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_p2_mrfn as T  # noqa: E402

CONFIGS = [
    dict(model="B3", training="mixture", fill="zero", tag="B3_aug_zero"),
    dict(model="MRFN", training="clean", fill="zero", tag="MRFN_clean"),
    dict(model="MRFN_noState", training="mixture", fill="zero", tag="MRFN_noState"),
    dict(model="MRFN_noMaskAttn", training="mixture", fill="zero", tag="MRFN_noMaskAttn"),
    dict(model="MRFN_noGate", training="mixture", fill="zero", tag="MRFN_noGate"),
    dict(model="B3", training="mixture", fill="forward", tag="B3_aug_ff"),
]


def load_json(p: Path):
    return json.loads(Path(p).read_text())


def summarize_run(tag):
    s = load_json(T.OUTDIR / tag / "summary.json")
    return {"params": s["params"], "clean": s["clean"],
            "masked_S_mean": s["masked_S_mean"],
            "masked_D_S_mean": s["masked_D_S_mean"],
            "mean_D_S": round(float(np.mean(list(s["masked_D_S_mean"].values()))), 6),
            "all_checks_pass": s["all_checks_pass"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch_available_cuda() else "cpu")
    ap.add_argument("--from-results", action="store_true",
                    help="复用已有 rundir 结果，仅重建汇总")
    args = ap.parse_args()

    runs = {}
    for cfg in CONFIGS:
        tag = cfg["tag"]
        print(f"\n######## M5 run: {tag} ########", flush=True)
        ns = argparse.Namespace(model=cfg["model"], training=cfg["training"],
                                fill=cfg["fill"], rundir=tag, seeds=args.seeds,
                                epochs=args.epochs, batch_size=32, lr=1e-3, wd=1e-4,
                                patience=8, dropout=0.3, lambda_l1=1.0,
                                device=args.device, from_results=args.from_results)
        code = T.run(ns)
        runs[tag] = summarize_run(tag)
        runs[tag]["exit_code"] = code

    # ---- 2×2（S 复合指标；clean 列来自 M3 阶梯，aug 列来自本轮）----
    ladder = load_json(ROOT / "runs/p2/ladder/ladder_summary.json")
    m4 = load_json(T.OUTDIR / "summary.json")          # M4 根目录 = MRFN 增强训练
    mrfn_clean = runs["MRFN_clean"]

    # M4 旧版 summary 无 masked_S_mean：由 D_S 派生（S_r = S_clean·(1−D_S)）
    m4_masked_S = m4.get("masked_S_mean")
    if not m4_masked_S:
        sc = m4["clean"]["S"]["mean"]
        m4_masked_S = {k: round(sc * (1 - d), 6)
                       for k, d in m4["masked_D_S_mean"].items()}

    def masked_s_from_ladder(model):
        sc = ladder["ladder"][model]["clean"]["S"]["mean"]
        d = ladder["ladder"][model]["D_S_mean"]
        return round(sc * (1 - float(np.mean(list(d.values())))), 6)

    table_2x2 = {
        "B3": {
            "train_clean": {"eval_clean_S": ladder["ladder"]["B3"]["clean"]["S"]["mean"],
                            "eval_missing_S": masked_s_from_ladder("B3"),
                            "source": "M3 ladder（派生，见注）"},
            "train_aug": {"eval_clean_S": runs["B3_aug_zero"]["clean"]["S"]["mean"],
                          "eval_missing_S": runs["B3_aug_zero"]["masked_S_mean"]
                          and round(float(np.mean(list(
                              runs["B3_aug_zero"]["masked_S_mean"].values()))), 6),
                          "source": "B3_aug_zero"},
        },
        "MRFN": {
            "train_clean": {"eval_clean_S": mrfn_clean["clean"]["S"]["mean"],
                            "eval_missing_S": round(float(np.mean(list(
                                mrfn_clean["masked_S_mean"].values()))), 6),
                            "source": "MRFN_clean"},
            "train_aug": {"eval_clean_S": m4["clean"]["S"]["mean"],
                          "eval_missing_S": round(float(np.mean(list(
                              m4_masked_S.values()))), 6),
                          "source": "M4 summary"},
        },
    }
    for m in table_2x2:
        tc, ta = table_2x2[m]["train_clean"], table_2x2[m]["train_aug"]
        tc["robustness_gain"] = round(tc["eval_missing_S"] - tc["eval_clean_S"], 6)
        ta["clean_cost"] = round(ta["eval_clean_S"] - tc["eval_clean_S"], 6)

    ablation = {
        "MRFN_full(aug)": {"mean_D_S": m4 and round(float(np.mean(
            list(m4["masked_D_S_mean"].values()))), 6),
            "clean_S": m4["clean"]["S"]["mean"], "params": m4["params"]},
        "MRFN_noAug(clean)": {"mean_D_S": mrfn_clean["mean_D_S"],
                              "clean_S": mrfn_clean["clean"]["S"]["mean"],
                              "params": mrfn_clean["params"]},
        "MRFN_noState": {"mean_D_S": runs["MRFN_noState"]["mean_D_S"],
                         "clean_S": runs["MRFN_noState"]["clean"]["S"]["mean"],
                         "params": runs["MRFN_noState"]["params"]},
        "MRFN_noMaskAttn": {"mean_D_S": runs["MRFN_noMaskAttn"]["mean_D_S"],
                            "clean_S": runs["MRFN_noMaskAttn"]["clean"]["S"]["mean"],
                            "params": runs["MRFN_noMaskAttn"]["params"]},
        "MRFN_noGate": {"mean_D_S": runs["MRFN_noGate"]["mean_D_S"],
                        "clean_S": runs["MRFN_noGate"]["clean"]["S"]["mean"],
                        "params": runs["MRFN_noGate"]["params"]},
    }
    fill_baseline = {
        "B3_aug_zero(mean-fills 同价)": {
            "mean_D_S": runs["B3_aug_zero"]["mean_D_S"],
            "clean_S": runs["B3_aug_zero"]["clean"]["S"]["mean"]},
        "B3_aug_forward_fill": {
            "mean_D_S": runs["B3_aug_ff"]["mean_D_S"],
            "clean_S": runs["B3_aug_ff"]["clean"]["S"]["mean"]},
    }

    all_ok = (all(runs[t]["all_checks_pass"] for t in runs)
              and m4["all_checks_pass"])
    m5 = {"created_at": datetime.now(timezone.utc).isoformat(),
          "runs": runs, "table_2x2": table_2x2, "ablation": ablation,
          "fill_baseline": fill_baseline,
          "notes": ["2×2 右上格差 = 鲁棒性增益，左下格差 = 干净性能代价（方案 §8.3）",
                    "B3 train_clean 的 eval_missing_S 由 M3 的 D_S 派生（S×(1−D)），其余为直测",
                    "均值填充 ≡ 零填充（z-score 空间均值=0），§8.5 注",
                    "消融 Mean D_S 为 31 条 eval 条目 D_S 的均值"],
          "all_checks_pass": bool(all_ok)}
    out = ROOT / "runs/p2/m5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "m5_summary.json").write_text(json.dumps(m5, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print("\n===== M5 汇总 =====")
    print(json.dumps({"2x2": table_2x2, "ablation": ablation, "fill": fill_baseline},
                     ensure_ascii=False, indent=1))
    print("ALL PASS" if all_ok else "PARTIAL（个别 run 检查未过，见各 summary）")
    return 0 if all_ok else 1


def torch_available_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    raise SystemExit(main())
