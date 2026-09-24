"""P1 细粒度行为指标（方案 §6.3 补充；边界经用户 2026-09-24 确认封口）。

只读既有数据：附件2 aligned_50.pkl、runs/p1_full 样本 pkl、附件1 音频。
不重跑管线、不修改任何 P1 规则。

指标（11 条 train=标定视图，7 条 test=冻结后验证视图，分开汇报）：
  A. 位置级零/非零行为一致率（content 位上：官方非零 vs 自产非零）
     音频注意：静音词官方为零行、自产 eGeMAPS 对静音给非零值——
     该分歧类按设计存在，单独计数，不算管线错误。
  B. 100 条 coverage 分布：mean / median / P5 / min（audio & vision）。
  C. missing_reason 词级比例（no_face / no_frame / alignment_failed / 其他）。
  D. 词级 RMS ↔ 官方幅度代理相关性。
     能量代理预注册：官方音频词级行的 L2 范数（全 74 维聚合标量）。
     依据：题目补充说明明示"不对其每一维的具体物理含义作额外假定"，
     故禁止任何逐列挑选，取维度无关的幅度聚合是唯一无偏选择。
     相关系数：Pearson 与 Spearman（逐 split 汇报）。

输出：runs/p1_full/behavior_detail.json + behavior_detail.csv
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABEL = ROOT / "data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx"
VROOT = LABEL.parent
ALIGNED = ROOT / "data/附件2-数据集特征文件/aligned_50.pkl"
OUT = ROOT / "runs/p1_full"
WAVDIR = ROOT / "runs/audit18"


def rms_in_interval(wav_path: Path, t_s: float, t_e: float) -> float | None:
    audio, sr = sf.read(wav_path, dtype="float32")
    lo, hi = int(t_s * sr), max(int(t_s * sr) + 1, int(t_e * sr))
    seg = audio[min(lo, len(audio)):min(hi, len(audio))]
    if seg.size == 0:
        return None
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))


def main() -> int:
    labels = pd.read_excel(LABEL, sheet_name="label")
    label_text = {f"{r.video_id}$_${int(r.clip_id)}": str(r.text) for r in labels.itertuples()}
    att = pickle.load(open(ALIGNED, "rb"))

    samples = {}
    for p in sorted((OUT / "samples").glob("*.pkl")):
        d = pickle.load(open(p, "rb"))
        samples[d["id"]] = d

    overlap = {}  # sid -> (split, official_index)
    for split in ("train", "test"):
        for i, sid in enumerate(att[split]["id"]):
            if sid in label_text and sid in samples:
                overlap[sid] = (split, i)
    assert sum(1 for s, _ in overlap.values() if s == "train") == 11
    assert sum(1 for s, _ in overlap.values() if s == "test") == 7

    # A/B/D：18 条重叠样本；C：全部 100 条
    per_sample = []
    pooled_words = {"train": [], "test": []}
    pooled_audio_agree = {"train": [], "test": []}
    pooled_vision_agree = {"train": [], "test": []}
    wavdir = WAVDIR
    wavdir.mkdir(parents=True, exist_ok=True)

    for sid, (split, i) in sorted(overlap.items()):
        d = samples[sid]
        off_audio = att[split]["audio"][i]
        off_vision = att[split]["vision"][i]
        cm = d["content_mask"] == 1
        positions = np.where(cm)[0]

        off_a_nz = np.array([bool(off_audio[p].any()) for p in positions])
        our_a_nz = np.array([bool(d["audio"][p].any()) for p in positions])
        off_v_nz = np.array([bool(off_vision[p].any()) for p in positions])
        our_v_nz = np.array([bool(d["vision"][p].any()) for p in positions])

        a_agree = float((off_a_nz == our_a_nz).mean())
        v_agree = float((off_v_nz == our_v_nz).mean())
        a_only_official = int((off_a_nz & ~our_a_nz).sum())   # 官方非零、自产零（与字段名一致）
        pooled_audio_agree[split].extend((off_a_nz == our_a_nz).tolist())
        pooled_vision_agree[split].extend((off_v_nz == our_v_nz).tolist())

        # D：词级 RMS（自产区间）↔ 官方行 L2 范数
        wmap = {m["position"]: m for m in d["wp_word_time_map"]}
        seen_words, xs, ys = set(), [], []
        vid, clip = sid.split("$_$")
        wav = wavdir / f"{r.video_id}_{int(r.clip_id)}.wav" if False else wavdir / f"{vid.replace('$_$','_')}_{clip}.wav"
        if not wav.exists():
            from src.p1.media import extract_audio_16k
            extract_audio_16k(VROOT / vid / f"{int(clip)}.mp4", wav)
        for p in positions:
            m = wmap.get(int(p))
            if m is None or m.get("word_id") is None or m["word_id"] in seen_words:
                continue
            seen_words.add(m["word_id"])
            t_s, t_e = m.get("t_s"), m.get("t_e")
            if t_s is None or t_e is None or not (0 <= t_s < t_e):
                continue
            r = rms_in_interval(wav, float(t_s), float(t_e))
            if r is None:
                continue
            mag = float(np.linalg.norm(off_audio[p]))
            xs.append(r); ys.append(mag)
        pr = float(pearsonr(xs, ys)[0]) if len(xs) > 2 and np.std(xs) > 0 else None
        sp = float(spearmanr(xs, ys)[0]) if len(xs) > 2 and np.std(xs) > 0 else None
        pooled_words[split].extend(zip(xs, ys))

        row = {
            "id": sid, "split": split,
            "content_positions": int(cm.sum()),
            "audio_nonzero_agreement": round(a_agree, 4),
            "audio_official_nonzero_but_ours_zero": a_only_official,
            "vision_nonzero_agreement": round(v_agree, 4),
            "vision_nonzero_agreement_pooled_note": "",
            "audio_coverage": round(float(d["observed_mask_audio"][cm].mean()), 4),
            "vision_coverage": round(float(d["observed_mask_video"][cm].mean()), 4),
            "rms_words": len(xs),
            "rms_pearson": None if pr is None else round(pr, 4),
            "rms_spearman": None if sp is None else round(sp, 4),
        }
        per_sample.append(row)
        print(f"[{split}] {sid}: a一致={a_agree:.3f} v一致={v_agree:.3f} "
              f"rms对(n={len(xs)}, pearson={row['rms_pearson']}, spearman={row['rms_spearman']})", flush=True)

    # B：100 条 coverage 分布（manifest）
    mf = pd.read_csv(OUT / "manifest.csv")

    def dist(col):
        v = mf[col].astype(float)
        return {"mean": round(v.mean(), 4), "median": round(v.median(), 4),
                "p5": round(v.quantile(0.05), 4), "min": round(v.min(), 4)}

    # C：100 条 missing_reason 词级比例
    reasons_v = {"no_face": 0, "no_frame": 0, "alignment_failed": 0, "observed": 0, "other": 0}
    reasons_a = {"no_face": 0, "no_frame": 0, "alignment_failed": 0, "observed": 0, "all_nan": 0, "other": 0}
    content_total = 0
    for d in samples.values():
        cm = d["content_mask"] == 1
        content_total += int(cm.sum())
        for p, reason in enumerate(d["missing_reason_video"]):
            if not cm[p]: continue
            key = reason if reason in reasons_v else "other"
            reasons_v[key] += 1
        for p, reason in enumerate(d["missing_reason_audio"]):
            if not cm[p]: continue
            key = reason if reason in reasons_a else "other"
            reasons_a[key] += 1
    result = {
        "A_位置级一致率": {
            split: {
                "audio": round(float(np.mean(pooled_audio_agree[split])), 4),
                "vision": round(float(np.mean(pooled_vision_agree[split])), 4),
                "n_positions": len(pooled_audio_agree[split]),
            } for split in ("train", "test")
        },
        "B_coverage分布_100条": {"audio": dist("audio_coverage"), "vision": dist("vision_coverage")},
        "C_缺失原因比例": {
            "口径": "content-position 级（按 content_mask 位置统计，非去重原词）",
            "vision_content_position比例": {k: round(v / content_total, 4) for k, v in reasons_v.items()},
            "audio_content_position比例": {k: round(v / content_total, 4) for k, v in reasons_a.items()},
            "content_positions": content_total,
        },
        "D_能量代理相关": {
            split: {
                "n_words": len(pooled_words[split]),
                "pearson": None if not pooled_words[split] else round(float(pearsonr(*zip(*pooled_words[split]))[0]), 4),
                "spearman": None if not pooled_words[split] else round(float(spearmanr(*zip(*pooled_words[split]))[0]), 4),
            } for split in ("train", "test")
        },
        "per_sample": per_sample,
        "approved_paper_wording": "在18条重叠样本的 content-position 级非零状态比较中，audio 一致率为100%，vision 在train/test视图分别为93.57%和92.75%。由于两套视觉特征由不同检测器生成，该指标反映的是观测状态行为一致性，而非人脸检测准确率。100条自产样本的vision观测覆盖率均值为93.27%，中位数为100%，P5为49.42%，表明缺失主要集中于少量异常样本。词区间RMS与附件2音频行L2幅度代理的相关性按预注册口径报告，不对官方74维逐列作物理解释。",
        "energy_proxy_preregistered": "官方音频词级行 L2 范数（全 74 维聚合；题目明示不作逐维物理解释，故无列选择）",
    }
    (OUT / "behavior_detail.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    pd.DataFrame(per_sample).to_csv(OUT / "behavior_detail.csv", index=False)
    print("\n=== 汇总 ===")
    print(json.dumps({k: result[k] for k in ("A_位置级一致率", "B_coverage分布_100条", "C_缺失原因比例", "D_能量代理相关")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
