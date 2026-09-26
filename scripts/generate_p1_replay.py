"""P1 人工回放素材生成器（题目"典型样本验证"）。

选取 8–10 条覆盖边界类型的样本，每条生成三轨图：
  轨1 波形 + 词区间（含词文本标注）
  轨2 帧条带（≤10 帧，按 pts 落位，叠加 top-1 人脸框）
  轨3 映射表节选（position/token/word/时间/置信度/缺失原因）
输出：
  paper/latex/figures/q1/replay_<id>.png/.pdf
  runs/p1_replay/replay_audit.json（含 edge_types 与 human_conclusion=null 待签核）
  runs/p1_replay/replay_audit.md（签核骨架）
不修改任何 P1 规则与已有产物。
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

FIGDIR = ROOT / "paper/latex/figures/q1"


def configure_style() -> None:
    """Use the paper font and embed TrueType text in vector exports."""
    cjk_font = Path.home() / ".fonts" / "NotoSansSC-Regular.otf"
    if cjk_font.is_file():
        font_manager.fontManager.addfont(str(cjk_font))
        cjk_family = font_manager.FontProperties(fname=str(cjk_font)).get_name()
    else:
        cjk_family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [cjk_family, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def esc(text: str) -> str:
    """样本 id 含 $_$、词表含下划线，必须转义防 matplotlib mathtext 解析。"""
    return str(text).replace("\\", "\\\\").replace("$", "\\$").replace("_", "\\_")
AUDIT = ROOT / "runs/p1_replay"


def edge_types_of(manifest_row, sample: dict) -> list[str]:
    types = []
    if manifest_row["status"] == "rollback":
        types.append("rollback")
    if int(manifest_row["wp_len"]) > 48:
        types.append("头部截断")
    if abs(float(manifest_row["fps"]) - 30.0) > 0.5:
        types.append("混合FPS")
    m = sample["missing_reason_video"]
    if "no_face" in m:
        types.append("无人脸")
    if "alignment_failed" in m or float(manifest_row["alignment_failure_rate"]) > 0:
        types.append("对齐失败")
    if any(x.get("wp_count_in_word", 0) >= 3 for x in sample["wp_word_time_map"]
           if x.get("wp_count_in_word") is not None):
        types.append("多WordPiece")
    if any(x.get("attach_type") in ("head", "tail", "ambi")
           for x in sample["wp_word_time_map"] if x.get("attach_type")):
        types.append("标点继承")
    if manifest_row["status"] == "ok" and manifest_row["audio_coverage"] == 1.0:
        types.append("正常")
    return types or ["未分类"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=300, help="PNG rasterization DPI (default: 300)")
    args = parser.parse_args()
    configure_style()
    import cv2
    from compat_feat import get_detector
    from src.p1.media import extract_audio_16k, extract_frames_pts
    from src.p1.pool import _sanitize_faces

    mf = pd.read_csv(ROOT / "runs/p1_full/manifest.csv")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    vroot = ROOT / "data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条"

    # ---- 选样：pkl 全量扫描建立边界类型表，每类取 id 最前一条，去重，上限 10 ----
    import pickle as _pkl
    samples_all = {}
    for sp in sorted((ROOT / "runs/p1_full/samples").glob("*.pkl")):
        d = _pkl.load(open(sp, "rb"))
        samples_all[d["id"]] = d

    edge_table: dict[str, list[str]] = {}
    for cid2, d in samples_all.items():
        et = []
        if d.get("alignment", {}).get("status") == "rollback":
            et.append("rollback")
        if any(m.get("trunc_flag") for m in d["wp_word_time_map"]) or \
           any(w.get("dropped") for w in d.get("full_word_map", [])):
            et.append("头部截断")
        if max((len(w.get("wp_positions", [])) for w in d.get("full_word_map", [])), default=0) >= 3:
            et.append("多WordPiece")
        if any(m.get("attach_type") in ("head", "tail", "ambi") for m in d["wp_word_time_map"]):
            et.append("标点继承")
        if "no_face" in d["missing_reason_video"]:
            et.append("无人脸")
        if "alignment_failed" in d["missing_reason_audio"] or "alignment_failed" in d["missing_reason_video"]:
            et.append("对齐失败")
        mrow = mf.set_index("id").loc[cid2]
        if abs(float(mrow["fps"]) - 30.0) > 0.5:
            et.append("混合FPS")
        if float(mrow["alignment_failure_rate"]) > 0:
            et.append("对齐失败")
        if (mrow["alignment_status"] == "ok" and float(mrow["audio_coverage"]) == 1.0
                and float(mrow["vision_coverage"]) == 1.0):
            et.append("正常")
        edge_table[cid2] = et or ["未分类"]

    picks: dict[str, str] = {}
    for cat in ["rollback", "rollback", "头部截断", "多WordPiece", "标点继承",
                "混合FPS", "无人脸", "对齐失败", "正常"]:
        for cid2 in sorted(edge_table):
            if cat in edge_table[cid2] and cid2 not in picks.values():
                picks[f"{cat}#{len(picks)}"] = cid2
                break
    chosen = sorted(set(picks.values()))[:10]
    required = {"rollback", "头部截断", "多WordPiece", "标点继承", "混合FPS", "无人脸", "对齐失败", "正常"}
    got = set()
    for cat, cid2 in picks.items():
        got.update(edge_table[cid2])
    missing = required - got
    assert not missing, "选样未覆盖必备边界类型: %s" % sorted(missing)
    from compat_feat import get_detector as _gd
    det = _gd(device="cuda")

    audit = {"samples": [], "note": "human_conclusion 待人工签核后回填"}
    for cid in chosen:
        row = mf.set_index("id").loc[cid]
        video_id, clip_id = cid.split("$_$")
        video = vroot / video_id / f"{int(clip_id)}.mp4"
        sample_p = ROOT / "runs/p1_full/samples" / f"{video_id}__{clip_id}.pkl"
        import pickle
        sample = pickle.load(open(sample_p, "rb"))
        etypes = edge_table[cid]  # 展示与选样同源（manifest 列语义不同，勿用 edge_types_of）
        tags = [k for k, v in picks.items() if v == cid]

        safe = cid.replace("$_$", "_").replace("-", "")
        wav_p = AUDIT / f"{safe}.wav"
        extract_audio_16k(video, wav_p)
        audio, sr = sf.read(wav_p, dtype="float32")
        duration = len(audio) / sr

        frames = extract_frames_pts(video, AUDIT / f"{safe}_frames")

        # 帧条带取 ≤10 帧（时间均匀），仅对这些帧做人脸框检测
        idx = np.linspace(0, len(frames) - 1, min(10, len(frames))).astype(int)
        strip = [frames[j] for j in idx]
        images = [np.asarray(Image.open(f["path"]).convert("RGB")) for f in strip]
        batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        faces_raw = det.detect_faces(batch)
        faces = _sanitize_faces(faces_raw, images)

        # ---- 画图 ----
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1.4, 1.6], hspace=0.42)
        display_cid = cid.replace("$_$", "/")
        edge_disp = "、".join(e.replace("多WordPiece", "多子词").replace("混合FPS", "混合帧率")
                              for e in etypes)
        title = f"样本 {display_cid}（{edge_disp}）"
        fig.suptitle(title, fontsize=10)

        # 轨1 波形 + 词区间
        ax1 = fig.add_subplot(gs[0])
        t = np.arange(len(audio)) / sr
        ax1.plot(t, audio, lw=0.4, color="steelblue")
        wmap = sample["wp_word_time_map"]
        seen, labeled = set(), 0
        for m in wmap:
            w = m.get("word_id")
            if w is None or w in seen or m.get("t_s") is None:
                continue
            seen.add(w)
            t_s, t_e = float(m["t_s"]), float(m["t_e"])
            ax1.axvspan(t_s, t_e, color="orange", alpha=0.18)
            if labeled < 45:
                ax1.text((t_s + t_e) / 2, audio.max() * 0.85, esc(m["word_text"]),
                         rotation=90, fontsize=5.5, ha="center", va="top")
                labeled += 1
        ax1.set_xlim(0, duration)
        ax1.set_ylabel("幅值")
        ax1.set_xlabel("时间— 橙色带=词区间（强制对齐）")
        ax1.set_title("轨1: 波形 + 词区间", fontsize=9)

        # 轨2 帧条带 + 人脸框
        ax2 = fig.add_subplot(gs[1])
        ax2.set_xlim(0, duration)
        ax2.set_ylim(0, 1)
        gaps = np.diff([f["pts_time"] for f in strip]) if len(strip) > 1 else np.array([0.5])
        dt = max(0.15, float(np.median(gaps)) * 0.45)
        for (f, img, ff) in zip(strip, images, faces):
            thumb = Image.fromarray(img).copy()
            draw = ImageDraw.Draw(thumb)
            for box in (ff if isinstance(ff, list) else []):
                x1, y1, x2, y2 = box[:4]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            arr = np.asarray(thumb)
            ax2.imshow(arr, extent=(f["pts_time"] - dt, f["pts_time"] + dt, 0.1, 0.95), aspect="auto", zorder=2)
        ax2.set_yticks([])
        ax2.set_xlabel("时间— 缩略图按 pts 落位，红框=top-1 人脸")
        ax2.set_title("轨2: 帧条带（≤10 帧，pts 对位）", fontsize=9)

        # 轨3 映射表节选
        ax3 = fig.add_subplot(gs[2])
        ax3.axis("off")
        cols = ["position", "token", "word_text", "t_s", "t_e", "confidence", "attach_type", "missing_reason_video"]
        cells = [[esc(str(m.get(c))) for c in cols] for m in wmap[:16]]
        tab = ax3.table(colLabels=cols, cellText=cells, loc="center", cellLoc="center",
                        bbox=[0.0, 0.0, 1.0, 0.86])
        tab.auto_set_font_size(False)
        tab.set_fontsize(6.5)
        tab.scale(1, 1.25)
        ax3.set_title("轨3: 映射表节选（前 16 行）", fontsize=9, pad=2)

        out_png = FIGDIR / f"replay_{safe}.png"
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(FIGDIR / f"replay_{safe}.pdf", bbox_inches="tight")
        plt.close(fig)

        audit["samples"].append({
            "id": cid,
            "figure": str(out_png.relative_to(ROOT)),
            "status": row["status"],
            "failure_rate": float(row["alignment_failure_rate"]),
            "edge_types": etypes,
            "选样标签": tags,
            "frames_total": len(frames),
            "human_conclusion": None,
        })
        print(f"✅ {cid} ({','.join(etypes)})", flush=True)

    (AUDIT / "replay_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1))
    md = ["# P1 人工回放签核单", "",
          "| 样本 | 边界类型 | 图 | 人工结论 (✅/❌ + 一句话) |", "|---|---|---|---|"]
    for s in audit["samples"]:
        md.append(f"| {s['id']} | {','.join(s['edge_types'])} | {s['figure']} | 待签核 |")
    md += ["", "签核规则：全部 ✅ → 第 4 项时序回放验收闭合；任何 ❌ → 记录原因并回溯对齐/池化规则。"]
    (AUDIT / "replay_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n审计清单: {AUDIT/'replay_audit.json'} / .md；图: {FIGDIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
