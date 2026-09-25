#!/usr/bin/env python
"""图1：统一语义—时间坐标图（正文主图，三轨 + 映射连接带）。

Usage:
  .venv/bin/python scripts/figures/plot_q1_unified.py

从冻结 pkl 与回放缓存自动生成，输出 paper/latex/figures/q1/q1_unified_coordinate.{pdf,png}：
  带0 文本轨：WordPiece 网格（position 域，词分组着色，CLS/SEP 特殊位）
  带1 连接带：词 → [t_s, t_e) 映射梯形（多子词整词共享区间，插值词虚线）
  带2 音频轨：波形 + 词区间池化带（time 域）
  带3 视觉轨：关键帧按 pts_time 落位 + 全帧 pts rug（time 域）

两个横轴域独立（position / time），仅通过映射梯形连接，不做
position→time 线性拉伸——"非 50 等间隔帧"的结论由图2承担，本图不削弱它。
帧的 pts 缓存写 runs/p1_replay/<safe>_frames_pts.json 供复现复用。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/figures"))

RUN_ROOT = ROOT / "runs/p1_full"
REPLAY_DIR = ROOT / "runs/p1_replay"
VROOT = ROOT / "data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条"
OUT = ROOT / "paper/latex/figures/q1"

SAMPLE_ID = "-3g5yACwYnA$_$13"  # VFR 样本；多子词 they've/table；插值词 least
N_STRIP = 8


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    cjk_font = Path.home() / ".fonts" / "NotoSansSC-Regular.otf"
    if cjk_font.is_file():
        font_manager.fontManager.addfont(str(cjk_font))
        cjk_family = font_manager.FontProperties(fname=str(cjk_font)).get_name()
    else:
        cjk_family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [cjk_family, "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.35,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def load_assets(sid: str) -> dict:
    sample = pickle.load(open(RUN_ROOT / "samples" / f"{sid.replace('$_$', '__')}.pkl", "rb"))
    mrow = pd.read_csv(RUN_ROOT / "manifest.csv").set_index("id").loc[sid]
    video_id, clip_id = sid.split("$_$")
    video = VROOT / video_id / f"{int(clip_id)}.mp4"
    safe = f"{video_id}_{clip_id}"

    wav_p = REPLAY_DIR / f"{safe}.wav"
    if not wav_p.is_file():
        from src.p1.media import extract_audio_16k
        extract_audio_16k(video, wav_p)
    audio, sr = sf.read(wav_p, dtype="float32")

    pts_json = REPLAY_DIR / f"{safe}_frames_pts.json"
    if pts_json.is_file():
        frames = json.loads(pts_json.read_text())
    else:
        from src.p1.media import extract_frames_pts
        frames = extract_frames_pts(video, REPLAY_DIR / f"{safe}_frames")
        pts_json.write_text(json.dumps(frames, ensure_ascii=False))
    return {"sample": sample, "mrow": mrow, "audio": audio, "sr": sr, "frames": frames}


def main() -> int:
    configure_style()
    sid = SAMPLE_ID
    ast = load_assets(sid)
    sample, mrow = ast["sample"], ast["mrow"]
    wmap = sample["wp_word_time_map"]
    duration = float(sample["metadata"]["duration"])
    n_pos = len(wmap)

    # 词表（按 word_id 升序），词色循环；特殊位灰
    words = [w for w in sample["full_word_map"] if not w.get("dropped")]
    palette = sns.color_palette("colorblind", max(2, len(words)))
    wcolor = {w["word_id"]: palette[i % len(palette)] for i, w in enumerate(words)}
    special = "#5B6470"

    fig, (ax_pos, ax_mid, ax_aud, ax_vis) = plt.subplots(
        4, 1, figsize=(6.9, 7.6),
        gridspec_kw={"height_ratios": [1.0, 1.35, 1.15, 1.5]},
    )
    fig.subplots_adjust(hspace=0.55)

    # ---- 带0 文本轨：position 域（只画有效位，padding 以示意条表示） ----
    n_valid = int(np.asarray(sample["attention_mask"]).sum())
    wmap_valid = wmap[:n_valid]
    n_pad = n_pos - n_valid
    pos_xlim = (-0.55, n_valid + 1.15)
    for m in wmap_valid:
        p = m["position"]
        is_special = m["attach_type"] == "special"
        face = special if is_special else wcolor[m["word_id"]]
        ax_pos.add_patch(plt.Rectangle((p - 0.48, 0.24), 0.96, 0.68, facecolor=face,
                                       edgecolor="white", linewidth=1.0, zorder=2))
        ax_pos.text(p, 0.28, m["token"], rotation=90, ha="center", va="bottom", fontsize=6.0,
                    color="white" if is_special else "#1F2933", zorder=3)
    ax_pos.add_patch(plt.Rectangle((n_valid - 0.5 + 0.18, 0.24), 1.35, 0.68,
                                   facecolor="#E5E7EB", edgecolor="white", linewidth=1.0, zorder=2))
    ax_pos.text(n_valid + 0.36, 0.58, f"PAD ×{n_pad}", ha="center", va="center",
                fontsize=6.0, color="#6B7280", zorder=3)
    seps = [m["position"] for m in wmap_valid[1:]
            if m["word_id"] != wmap_valid[m["position"] - 1]["word_id"]]
    for i in seps:
        ax_pos.plot([i - 0.5, i - 0.5], [0.21, 0.95], color="#9AA5B1", lw=0.6, alpha=0.55, zorder=1)
    ax_pos.set_xlim(*pos_xlim)
    ax_pos.set_ylim(0, 1.05)
    ax_pos.set_yticks([])
    ax_pos.set_xticks(list(range(n_valid)))
    ax_pos.tick_params(axis="x", labelsize=5.8)
    ax_pos.grid(False)
    ax_pos.set_xlabel("语义位置（WordPiece 网格，position 域；content=18/50，含 CLS/SEP 的 attention 有效位=20/50）")
    ax_pos.set_title("文本轨：BERT 逐位置输出，词分组着色（多子词同色相连）",
                     loc="left", fontsize=8.4, pad=3)

    # ---- 带1 连接带：词 → [t_s, t_e) 映射梯形（fraction 坐标） ----
    time_xlim = (0.0, duration)
    pos_span = pos_xlim[1] - pos_xlim[0]
    t_span = time_xlim[1] - time_xlim[0]

    def xf_pos(p: float) -> float:
        return (p - pos_xlim[0]) / pos_span

    def xf_time(t: float) -> float:
        return (t - time_xlim[0]) / t_span

    interpolated = {m["word_id"] for m in wmap if m.get("alignment_status") == "interpolated"}
    for w in words:
        ps = w.get("retained_positions") or [m["position"] for m in wmap if m["word_id"] == w["word_id"]]
        if not ps or w.get("t_s") is None:
            continue
        p_lo, p_hi = min(ps) - 0.48, max(ps) + 0.48
        poly = plt.Polygon(
            [(xf_pos(p_lo), 1.0), (xf_pos(p_hi), 1.0),
             (xf_time(w["t_e"]), 0.0), (xf_time(w["t_s"]), 0.0)],
            closed=True, facecolor=wcolor[w["word_id"]], alpha=0.15, zorder=1,
            edgecolor=wcolor[w["word_id"]], linewidth=0.8,
            linestyle=":" if w["word_id"] in interpolated else "-",
        )
        ax_mid.add_patch(poly)
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.set_xticks([])
    ax_mid.set_yticks([])
    ax_mid.grid(False)
    ax_mid.set_title("映射：词 → [t_s, t_e)（音频 / 视觉在词区间池化后复制到词内全部 WordPiece）",
                     loc="left", fontsize=8.4, pad=3)
    legend_items = [
        plt.Rectangle((0, 0), 1, 1, fc="#7FB3D5", alpha=0.35, ec="#3E7CB1", lw=0.8,
                      label="词区间（整词共享）"),
        plt.Line2D([0], [0], color="#666666", lw=0.9, ls=":",
                   label="低置信度插值词"),
        plt.Rectangle((0, 0), 1, 1, fc=special, ec="none", label="特殊位 CLS / SEP（无时间映射）"),
    ]
    ax_mid.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.01, 0.01),
                  frameon=False, fontsize=6.2, handlelength=1.4)

    # ---- 带2 音频轨：time 域，波形 + 词区间 ----
    t_audio = np.arange(len(ast["audio"])) / ast["sr"]
    ax_aud.plot(t_audio, ast["audio"], lw=0.35, color="#5B8DB8", zorder=2)
    amp = float(np.abs(ast["audio"]).max()) or 1.0
    for w in words:
        if w.get("t_s") is None:
            continue
        ax_aud.axvspan(w["t_s"], w["t_e"], color=wcolor[w["word_id"]], alpha=0.15, zorder=1)
        ax_aud.text((w["t_s"] + w["t_e"]) / 2, amp * 0.95, w["word_text"], rotation=90,
                    fontsize=5.2, ha="center", va="top", color="#1F2933", zorder=3)
    ax_aud.set_xlim(*time_xlim)
    ax_aud.set_ylim(-amp * 1.15, amp * 1.15)
    ax_aud.set_yticks([-amp, 0, amp])
    ax_aud.set_yticklabels(["−A", "0", "+A"], fontsize=6.2)
    ax_aud.tick_params(axis="x", labelbottom=False)
    ax_aud.set_ylabel("波形幅度")
    ax_aud.set_title("音频轨：eGeMAPS LLD 在词区间 [t_s, t_e) 内池化（time 域）",
                     loc="left", fontsize=8.4, pad=3)

    # ---- 带3 视觉轨：time 域，关键帧按 pts_time 落位 + 全帧 pts rug ----
    frames = sorted(ast["frames"], key=lambda f: f["pts_time"])
    pts_all = np.array([f["pts_time"] for f in frames])
    ax_vis.vlines(pts_all, 0.0, 0.055, color="#444444", lw=0.4, zorder=2)
    strip_idx = np.linspace(0, len(frames) - 1, min(N_STRIP, len(frames))).astype(int)
    strip = [frames[j] for j in strip_idx]
    gaps = np.diff([f["pts_time"] for f in strip])
    dt = max(0.12, float(np.median(gaps)) * 0.42)
    for f in strip:
        img = np.asarray(Image.open(f["path"]).convert("RGB"))
        pt = f["pts_time"]
        ax_vis.imshow(img, extent=(max(pt - dt, time_xlim[0]), min(pt + dt, time_xlim[1]),
                                   0.34, 0.94),
                      aspect="auto", zorder=3)
        ha = "left" if pt < 0.3 else ("right" if pt > duration - 0.3 else "center")
        ax_vis.text(pt, 0.10, f"{pt:.2f}", ha=ha, va="bottom",
                    fontsize=5.0, color="#374151", zorder=4)
    ax_vis.set_xlim(*time_xlim)
    ax_vis.set_ylim(0, 1.32)
    ax_vis.set_yticks([])
    ax_vis.set_xlabel("物理时间 (s)（音频 / 视觉轨共享，time 域）")
    ax_vis.set_title("视觉轨：关键帧按帧时间戳 pts_time 落位（底部短刻度 = 全部 163 帧的 pts）",
                     loc="left", fontsize=8.4, pad=3)
    ax_vis.text(0.08, 1.30, "VFR 样本：底部短刻度即全部帧的 pts_time，间隔非均匀\n"
                            "（视觉时间禁用 帧号 × 固定帧率 推算）",
                ha="left", va="top", fontsize=6.4)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "q1_unified_coordinate.pdf")
    fig.savefig(OUT / "q1_unified_coordinate.png", dpi=300)
    plt.close(fig)

    n_words_all = len(sample["full_word_map"])
    multi = sorted({m["word_text"] for m in wmap if (m.get("wp_count_in_word") or 0) > 1})
    print("图1 caption 素材：")
    print(f"  样本 {sid}（VFR={'是' if bool(mrow['is_vfr']) else '否'}，{len(frames)} 帧，"
          f"时长 {duration:.2f} s，名义 fps {float(mrow['fps']):.4f}）")
    print(f"  词数 {n_words_all}（保留 {len(words)}），内容片 {sum(1 for m in wmap if m['word_id'] is not None)}，"
          f"多子词词：{', '.join(multi)}")
    print(f"  插值词：{', '.join(sorted({m['word_text'] for m in wmap if m.get('alignment_status') == 'interpolated'}))}"
          f"；对齐状态 {mrow['alignment_status']}，失败率 {float(mrow['alignment_failure_rate']):.4f}")
    print(f"  模态维度 text 50×768 / audio 50×25 / vision 50×23；"
          f"audio/vision 观测率 {float(mrow['audio_coverage']):.3f}/{float(mrow['vision_coverage']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
