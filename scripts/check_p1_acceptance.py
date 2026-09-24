"""P1 第 6 项验收——机器部分：覆盖完整性/Schema/文本一致性/行为一致性 + 论文汇总表生成。

用法：.venv/bin/python scripts/check_p1_acceptance.py [runs/p1_full]

1. Schema 一致性：100 条 pkl 的 shape/dtype、mask 互斥完备分区、SEP 位；
2. 文本一致性：7 条留出样本（附件2 test 重叠）——text_bert 逐位 + text(768) 逐位 cos；
3. 行为一致性：audio/vision 观测率对标（train 实测 99.915% / 94.473%）。
结果打印 + 落 runs/<root>/acceptance_report.json。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABEL = ROOT / "data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx"
ALIGNED = ROOT / "data/附件2-数据集特征文件/aligned_50.pkl"

EXPECT = {"text": (50, 768), "audio": (50, 25), "vision": (50, 23)}


def main(run_root: str = "runs/p1_full") -> int:
    root = ROOT / run_root
    samples = sorted((root / "samples").glob("*.pkl"))
    report = {"n_samples": len(samples), "schema_violations": [], "text_check": [], "behavior": {}, "verdict": {}}
    assert len(samples) == 100, f"期望 100 条，实际 {len(samples)}"

    # 0) 覆盖完整性（题目要求一：样本编号、模态文件与输出特征一一对应）
    labels = pd.read_excel(LABEL, sheet_name="label")
    expect_ids = [f"{r.video_id}$_${int(r.clip_id)}" for r in labels.itertuples()]
    got_ids = [pickle.load(open(p, "rb"))["id"] for p in samples]
    dup = {i for i in got_ids if got_ids.count(i) > 1}
    report["coverage"] = {
        "label_rows": len(expect_ids),
        "missing": sorted(set(expect_ids) - set(got_ids)),
        "extra": sorted(set(got_ids) - set(expect_ids)),
        "duplicates": sorted(dup),
    }
    ok_cov = (
        len(expect_ids) == 100
        and not report["coverage"]["missing"]
        and not report["coverage"]["extra"]
        and not dup
    )
    print(f"0) 覆盖完整性: {'✅ 100 条一一对应，无缺失/重复/多余' if ok_cov else '❌ ' + str(report['coverage'])}")
    report["verdict"]["coverage"] = ok_cov

    # 1) Schema
    from src.p1.assemble import validate_sample, SCHEMA_VERSION, SCHEMA_HASH
    audio_obs, vision_obs, audio_words, vision_words = [], [], [], []
    audio_nz, vision_nz = [], []
    for p in samples:
        d = pickle.load(open(p, "rb"))
        sid = d["id"]
        try:
            validate_sample(d)
        except Exception as e:
            report["schema_violations"].append(f"{sid}: validate_sample {type(e).__name__}: {e}")
        if d.get("schema_version") != SCHEMA_VERSION or d.get("schema_hash") != SCHEMA_HASH:
            report["schema_violations"].append(f"{sid}: schema 版本/hash 与当前契约不符")
        for key, shape in EXPECT.items():
            if tuple(d[key].shape) != shape:
                report["schema_violations"].append(f"{sid}:{key} shape {d[key].shape}")
        part = d["content_mask"].astype(int) + d["special_mask"] + d["padding_mask"]
        if not (part == 1).all():
            report["schema_violations"].append(f"{sid}: mask 非分区")
        am, cm = d["attention_mask"], d["content_mask"]
        if not (am[: int(am.sum())] == 1).all() or am[int(am.sum()):].any():
            report["schema_violations"].append(f"{sid}: attention 连续性")
        if d["text_bert"][0][int(am.sum()) - 1] != 102:
            report["schema_violations"].append(f"{sid}: SEP 不在最后有效位")
        # 行为统计（词级）：observed（覆盖率口径）与 非零行（官方同口径）分开记
        oa, ov = d["observed_mask_audio"], d["observed_mask_video"]
        n_content = int(cm.sum())
        audio_obs.append(int(oa[cm == 1].sum())); audio_words.append(n_content)
        vision_obs.append(int(ov[cm == 1].sum())); vision_words.append(n_content)
        audio_nz.append(int((d["audio"][cm == 1].any(axis=1)).sum()))
        vision_nz.append(int((d["vision"][cm == 1].any(axis=1)).sum()))

    # 2) 文本一致性（7 条留出）
    label_ids = set(expect_ids)
    att = pickle.load(open(ALIGNED, "rb"))
    test_overlap = {}
    for i, sid in enumerate(att["test"]["id"]):
        if sid in label_ids:
            test_overlap[sid] = i
    assert len(test_overlap) == 7, f"留出样本应 7 条，实际 {len(test_overlap)}"
    by_id = {pickle.load(open(p, "rb"))["id"]: p for p in samples}
    for sid, i in test_overlap.items():
        d = pickle.load(open(by_id[sid], "rb"))
        bitwise = bool(np.array_equal(d["text_bert"], att["test"]["text_bert"][i]))
        own = d["text"][att["test"]["text_bert"][i][1] == 1]
        ref = att["test"]["text"][i][att["test"]["text_bert"][i][1] == 1]
        cos_vec = (own * ref).sum(-1) / (np.linalg.norm(own, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-9)
        cos = float(cos_vec.min())
        report["text_check"].append({"id": sid, "text_bert_bitwise": bitwise, "text_cos_min": round(cos, 6)})

    # 3) 行为一致性
    report["behavior"] = {
        # 覆盖率口径（observed_mask；音频静音按 §4 属已观测，不与官方非零率混比）
        "audio_observed_rate": round(sum(audio_obs) / sum(audio_words), 6),
        "vision_observed_rate": round(sum(vision_obs) / sum(vision_words), 6),
        # 非零率口径（与官方 99.915%/94.473% 同一指标，可容差比较）
        "audio_nonzero_rate": round(sum(audio_nz) / sum(audio_words), 6),
        "vision_nonzero_rate": round(sum(vision_nz) / sum(vision_words), 6),
        "benchmark_audio_nonzero": 0.99915, "benchmark_vision_nonzero": 0.94473,
        "tolerance_pp": 2.0,
    }
    b = report["behavior"]
    ok_behavior = (
        abs(b["audio_nonzero_rate"] - b["benchmark_audio_nonzero"]) * 100 <= b["tolerance_pp"]
        and abs(b["vision_nonzero_rate"] - b["benchmark_vision_nonzero"]) * 100 <= b["tolerance_pp"]
    )

    # 4) 论文汇总表（题目要求：样本编号、模态类型、原始有效时长、特征维度、对齐粒度）
    mf = pd.read_csv(root / "manifest.csv")
    paper = mf[["id", "duration"]].copy()
    paper.columns = ["样本编号", "原始有效时长(s)"]
    paper["模态类型"] = "text/audio/vision"
    paper["特征维度(text)"] = "50×768"
    paper["特征维度(audio)"] = "50×25"
    paper["特征维度(vision)"] = "50×23"
    paper["对齐粒度"] = "词级·50步网格·半开区间"
    paper["词片数"] = mf.get("wp_len")
    paper["对齐失败率"] = mf.get("alignment_failure_rate").round(4)
    paper["audio观测率"] = mf.get("audio_coverage").round(4)
    paper["vision观测率"] = mf.get("vision_coverage").round(4)
    paper["处理状态"] = mf.get("status")
    paper.to_csv(root / "manifest_paper.csv", index=False, encoding="utf-8-sig")
    print(f"4) 论文汇总表: {root}/manifest_paper.csv（{len(paper)} 行，utf-8-sig 可直接入 Excel/论文）")

    ok_schema = not report["schema_violations"]
    ok_text = all(r["text_bert_bitwise"] and r["text_cos_min"] > 0.99 for r in report["text_check"])
    print(f"1) Schema: {'✅ 通过' if ok_schema else '❌ ' + str(report['schema_violations'][:3])}")
    for r in report["text_check"]:
        print(f"2) 留出 {r['id']}: text_bert bitwise={r['text_bert_bitwise']}, text cos_min={r['text_cos_min']}")
    b = report["behavior"]
    print(f"3) 行为: 非零率 audio {b['audio_nonzero_rate']} vs 官方 {b['benchmark_audio_nonzero']}"
          f" | vision {b['vision_nonzero_rate']} vs 官方 {b['benchmark_vision_nonzero']}"
          f"（容差 ±{b['tolerance_pp']}pp）→ {'✅' if ok_behavior else '❌ 超容差'}")
    print(f"   覆盖率(参考量,不设门槛): audio {b['audio_observed_rate']} | vision {b['vision_observed_rate']}")
    report["verdict"].update({"schema": ok_schema, "text": ok_text, "behavior": ok_behavior})
    (root / "acceptance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"报告: {root}/acceptance_report.json")
    return 0 if ok_cov and ok_schema and ok_text and ok_behavior else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
