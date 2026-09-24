#!/usr/bin/env python
"""Generate reproducible Question 1 figures from frozen P1 artifacts.

Usage:
  .venv/bin/python scripts/figures/plot_q1.py

The script only reads ``runs/p1_full`` and writes final assets to
``paper/latex/figures/q1``. Replay figures that require selected video frames
are intentionally handled by a separate generator.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs" / "p1_full"
DEFAULT_OUT = ROOT / "paper" / "latex" / "figures" / "q1"


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


def save_figure(fig: mpl.figure.Figure, output: Path, *, dpi: int = 600) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_pipeline(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.35))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = [
        (0.2, 3.7, 2.0, 1.0, "附件1\n视频 + 转写", "#EAF2F8"),
        (3.0, 4.35, 2.0, 0.85, "文本\nBERT WordPiece\n50格", "#DCEAF7"),
        (3.0, 3.0, 2.0, 0.85, "语音\n16 kHz WAV\n25维 LLD", "#E1F0E6"),
        (3.0, 1.65, 2.0, 0.85, "视觉\nPNG + pts_time\nAU + pose", "#FBE7DC"),
        (6.0, 3.15, 2.25, 1.5, "原词—时间锚\nforced alignment\n半开区间 [t_s,t_e)", "#FFF4D6"),
        (9.15, 3.15, 2.35, 1.5, "P1 p1.v2\n50步特征 + mask\n映射表 + quality", "#EEEAF4"),
    ]
    for x, y, w, h, label, color in nodes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.0, edgecolor="#34495E", facecolor=color,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", linespacing=1.35)

    arrows = [
        ((2.2, 4.2), (3.0, 4.78)),
        ((2.2, 4.0), (3.0, 3.43)),
        ((2.2, 3.8), (3.0, 2.08)),
        ((5.0, 4.78), (6.0, 4.12)),
        ((5.0, 3.43), (6.0, 3.9)),
        ((5.0, 2.08), (6.0, 3.35)),
        ((8.25, 3.9), (9.15, 3.9)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.0, color="#34495E", connectionstyle="arc3,rad=0.0",
        ))
    ax.text(6.0, 0.72, "WordPiece为符号坐标；语音和视觉仅通过原词时间区间映射。",
            ha="left", va="center", fontsize=7.6, color="#4D4D4D")
    ax.text(6.0, 0.38, "vision: pts_time   |   audio: (k+0.5)×10 ms   |   alignment: 20 ms CTC quantization",
            ha="left", va="center", fontsize=7.4, color="#4D4D4D")
    save_figure(fig, output / "q1_pipeline.pdf")


def plot_quality(run_root: Path, output: Path) -> None:
    manifest = pd.read_csv(run_root / "manifest.csv")
    behavior = json.loads((run_root / "behavior_detail.json").read_text(encoding="utf-8"))
    status_order = ["ok", "review", "rollback"]
    status_colors = {"ok": "#59A14F", "review": "#F2CF5B", "rollback": "#E15759"}

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.7))
    ax = axes[0, 0]
    counts = manifest["alignment_status"].value_counts().reindex(status_order, fill_value=0)
    ax.bar(counts.index, counts.values, color=[status_colors[s] for s in status_order])
    ax.set_title("100条样本的对齐状态")
    ax.set_ylabel("样本数")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 1, str(value), ha="center", va="bottom")

    ax = axes[0, 1]
    sns.histplot(manifest["vision_coverage"], bins=np.linspace(0, 1, 11), ax=ax, color="#4E79A7")
    ax.set_title("视觉观测覆盖率")
    ax.set_xlabel("content-position观测比例")
    ax.set_ylabel("样本数")

    ax = axes[1, 0]
    reason = behavior["C_缺失原因比例"]["vision_content_position比例"]
    labels = ["observed", "no_face", "no_frame", "alignment_failed", "other"]
    values = [reason.get(label, 0.0) * 100 for label in labels]
    ax.bar(labels, values, color=["#59A14F", "#F28E2B", "#B07AA1", "#E15759", "#BAB0AC"])
    ax.set_title("视觉缺失原因（content-position）")
    ax.set_ylabel("比例（%）")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 1]
    sns.scatterplot(
        data=manifest, x="alignment_failure_rate", y="vision_coverage",
        hue="alignment_status", hue_order=status_order, palette=status_colors,
        s=38, ax=ax, legend=True,
    )
    ax.set_title("对齐失败率与视觉覆盖率")
    ax.set_xlabel("对齐失败率")
    ax.set_ylabel("视觉观测覆盖率")
    ax.legend(title="状态", fontsize=6.5, title_fontsize=7.5, loc="best")
    save_figure(fig, output / "q1_quality_overview.pdf")


def _load_sample_masks(run_root: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    paths = sorted((run_root / "samples").glob("*.pkl"))
    ids, audio, vision = [], [], []
    for path in paths:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        ids.append(sample["id"])
        audio.append(sample["observed_mask_audio"].astype(np.uint8))
        vision.append(sample["observed_mask_video"].astype(np.uint8))
    return ids, np.asarray(audio), np.asarray(vision)


def plot_observation_heatmaps(run_root: Path, output: Path) -> None:
    ids, audio, vision = _load_sample_masks(run_root)
    order = np.argsort(vision.mean(axis=1))
    labels = [ids[index] for index in order]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.8), sharex=True)
    for ax, values, title in zip(axes, (audio[order], vision[order]), ("语音自然观测状态", "视觉自然观测状态")):
        xlabels = [str(index) if index % 5 == 0 else "" for index in range(50)]
        sns.heatmap(
            values, ax=ax, cmap=sns.color_palette(["#F1F1F1", "#4E79A7"], as_cmap=True),
            vmin=0, vmax=1, cbar=False, xticklabels=xlabels, yticklabels=False,
        )
        ax.set_title(title)
        ax.set_ylabel("按覆盖率排序的样本")
        ax.tick_params(axis="x", length=2, pad=2)
    axes[-1].set_xlabel("WordPiece 网格位置")
    save_figure(fig, output / "q1_observation_heatmap.pdf")


def plot_fps_distribution(run_root: Path, output: Path) -> None:
    manifest = pd.read_csv(run_root / "manifest.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    sns.histplot(manifest["fps"], bins=12, color="#4E79A7", ax=axes[0])
    axes[0].set_title("由PTS间隔估计的FPS")
    axes[0].set_xlabel("帧率（FPS）")
    axes[0].set_ylabel("样本数")
    sns.histplot(manifest["vision_coverage"], bins=np.linspace(0, 1, 11), color="#F28E2B", ax=axes[1])
    axes[1].set_title("视觉覆盖率分布")
    axes[1].set_xlabel("content-position观测比例")
    axes[1].set_ylabel("样本数")
    save_figure(fig, output / "q1_timing_coverage.pdf")


def plot_coordinate_mapping(run_root: Path, output: Path) -> None:
    """Show one real WordPiece-to-word-to-time mapping example."""
    sample_path = run_root / "samples" / "-3g5yACwYnA__13.pkl"
    with sample_path.open("rb") as handle:
        sample = pickle.load(handle)
    rows = [
        row for row in sample["wp_word_time_map"]
        if row["word_id"] is not None and row["t_s"] is not None
    ][:16]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.65), gridspec_kw={"height_ratios": [1.0, 1.45]})
    palette = sns.color_palette("colorblind", n_colors=max(2, len({row["word_id"] for row in rows})))
    word_colors = {}
    for row in rows:
        word_colors.setdefault(row["word_id"], palette[len(word_colors) % len(palette)])

    ax = axes[0]
    for row in rows:
        position = row["position"]
        color = word_colors[row["word_id"]]
        ax.add_patch(plt.Rectangle((position - 0.46, 0.15), 0.92, 0.65, facecolor=color, edgecolor="white", linewidth=0.7))
        ax.text(position, 0.48, row["token"], ha="center", va="center", fontsize=7.3)
    ax.set_xlim(rows[0]["position"] - 0.6, rows[-1]["position"] + 0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([row["position"] for row in rows])
    ax.set_xlabel("WordPiece grid position")
    ax.set_title("WordPiece positions and original-word ownership")
    ax.grid(False)

    ax = axes[1]
    shown_words = []
    for row in rows:
        if row["word_id"] in shown_words:
            continue
        shown_words.append(row["word_id"])
        start, end = float(row["t_s"]), float(row["t_e"])
        color = word_colors[row["word_id"]]
        ax.broken_barh([(start, end - start)], (len(shown_words) - 1, 0.62), facecolors=[color], edgecolors="white", linewidth=0.7)
        label = str(row["word_text"])
        if row["alignment_status"] == "interpolated":
            label += "*"
        ax.text(start + (end - start) / 2, len(shown_words) - 1 + 0.31, label, ha="center", va="center", fontsize=7.0)
    ax.set_ylim(-0.1, max(1, len(shown_words)))
    ax.set_yticks([])
    ax.set_xlabel("Physical time (s), half-open intervals [t_s, t_e)")
    ax.set_title("Word intervals used to pool audio and vision")
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Example: WordPiece → word → physical-time mapping", y=1.01, fontsize=10.5)
    save_figure(fig, output / "q1_coordinate_mapping.pdf")


def plot_behavior_detail(run_root: Path, output: Path) -> None:
    behavior = json.loads((run_root / "behavior_detail.json").read_text(encoding="utf-8"))
    agreement = behavior["A_位置级一致率"]
    reason = behavior["C_缺失原因比例"]["vision_content_position比例"]
    corr = behavior["D_能量代理相关"]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.55))

    ax = axes[0, 0]
    splits = ["train", "test"]
    x = np.arange(len(splits))
    width = 0.18
    for offset, modality, color in [(-width, "audio", "#0072B2"), (0, "vision", "#D55E00")]:
        values = [agreement[split][modality] * 100 for split in splits]
        bars = ax.bar(x + offset, values, width, label=modality, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1)
    ax.set_xticks(x - width / 2, ["train (11)", "test (7)"])
    ax.set_ylim(80, 101)
    ax.set_ylabel("Agreement (%)")
    ax.set_title("Position-level nonzero agreement")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    labels = ["observed", "no_face", "no_frame", "alignment_failed"]
    values = [reason.get(label, 0) * 100 for label in labels]
    ax.bar(labels, values, color=["#009E73", "#E69F00", "#CC79A7", "#D55E00"])
    ax.set_ylabel("Content-position share (%)")
    ax.set_title("Vision missing-reason composition")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 0]
    mf = pd.read_csv(run_root / "manifest.csv")
    sns.boxplot(data=mf[["audio_coverage", "vision_coverage"]].rename(columns={"audio_coverage": "audio", "vision_coverage": "vision"}), ax=ax, palette=["#0072B2", "#D55E00"], width=0.45)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Observed coverage")
    ax.set_title("Coverage across 100 clips")

    ax = axes[1, 1]
    corr_labels = ["Pearson\ntrain", "Pearson\ntest", "Spearman\ntrain", "Spearman\ntest"]
    corr_values = [corr["train"]["pearson"], corr["test"]["pearson"], corr["train"]["spearman"], corr["test"]["spearman"]]
    colors = ["#0072B2", "#0072B2", "#009E73", "#009E73"]
    ax.bar(corr_labels, corr_values, color=colors)
    ax.axhline(0, color="#444444", linewidth=0.7)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Correlation")
    ax.set_title("RMS vs official L2 amplitude proxy")
    ax.tick_params(axis="x", labelsize=6.8)
    fig.suptitle("Question 1: fine-grained behavior checks", y=1.01, fontsize=10.5)
    save_figure(fig, output / "q1_behavior_detail.pdf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    configure_style()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_pipeline(args.output)
    plot_quality(args.run_root, args.output)
    plot_observation_heatmaps(args.run_root, args.output)
    plot_fps_distribution(args.run_root, args.output)
    plot_coordinate_mapping(args.run_root, args.output)
    plot_behavior_detail(args.run_root, args.output)
    print(f"Generated Q1 figures in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
