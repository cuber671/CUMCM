"""P2 M1-A 验收脚本：真实附件2 → 全部结构校验 + tokenizer 前缀复核 + M1-A 产物落盘。

产物（默认 runs/p2/data/）：
  normalization_stats.npz  train-only 逐维统计（canonical = content 且 o=1）
  m1a_state.npz            各 split ids / content-special-padding 分区 / o 三态 / 标签
  m1a_report.json          全部校验结果、o 率、统计量元数据、来源 sha256
退出码：全部检查通过 = 0，否则 1。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import ALIGNED_PKL, CONTRACT, Check, build_m1a, load_aligned


def tokenizer_check(att: dict) -> Check:
    """raw_text 重分词前缀复核：存储 wp 必须是完整分词的头部前缀。

    官方截断规则 = CLS + 前 48 非特殊片 + SEP（P1 方案 §4）；
    整词越界不入网格 → 存储数可 <48，但必须是前缀（前缀性质为硬门槛）。
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ROOT / "cache/hf/bert-base-uncased"))
    except Exception as e:  # 环境缺失也要显式失败，不允许静默跳过
        return Check("tokenizer_prefix_rule", False, {"skipped": True, "reason": repr(e)})

    total = n_mismatch = n_exact48 = n_boundary = 0
    examples = []
    for sp in ("train", "valid", "test"):
        tb = np.asarray(att[sp]["text_bert"])
        am = tb[:, 1, :]
        T = tb.shape[2]
        first0 = np.where(am.min(axis=1) == 1, T, np.argmax(am == 0, axis=1))
        sep_pos = np.where(am[:, -1] == 1, T - 1, first0 - 1)
        for i, raw in enumerate(att[sp]["raw_text"]):
            full = tok(str(raw), add_special_tokens=True)["input_ids"]
            wp_full = list(full[1:-1])
            stored = tb[i, 0, 1:int(sep_pos[i])].astype(int).tolist()
            total += 1
            if stored == wp_full[:len(stored)]:
                if len(stored) == 48:
                    n_exact48 += 1
                elif len(stored) < 48:
                    n_boundary += 1
            else:
                n_mismatch += 1
                if len(examples) < 5:
                    examples.append({"split": sp, "idx": i, "id": att[sp]["id"][i],
                                     "stored_head": stored[:8], "full_head": wp_full[:8]})
    return Check("tokenizer_prefix_rule", n_mismatch == 0, {
        "total": total, "mismatch": n_mismatch, "exact48": n_exact48,
        "stored_lt48": n_boundary, "examples": examples,
        "rule": "存储 wp = 完整分词头部前缀（硬门槛）；=48 即满额截断，"
                "<48 含天然短文本与整词越界剔除两种，不再细分",
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "runs/p2/data"))
    ap.add_argument("--source", default=str(ALIGNED_PKL),
                    help="att 实际来源 pkl；sha256 按此路径计算并写入报告")
    args = ap.parse_args()
    source = Path(args.source)
    print(f"来源: {source}")
    att = load_aligned(source)
    rep = build_m1a(att, Path(args.outdir), CONTRACT,
                    extra_checks=[tokenizer_check(att)], source_path=source)

    for c in rep["checks"]:
        print(("✅" if c["ok"] else "❌"), c["name"],
              json.dumps(c["detail"], ensure_ascii=False) if not c["ok"] else "")
    print("\n== o_summary ==")
    print(json.dumps(rep.get("o_summary", {}), ensure_ascii=False, indent=1))
    print("\n== stats_meta ==")
    print(json.dumps(rep.get("stats_meta", {}), ensure_ascii=False, indent=1))
    print(f"\n产物目录: {args.outdir}")
    print("ALL PASS" if rep["all_checks_pass"] else "FAILED")
    return 0 if rep["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
