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


# ---------------------------------------------------------------------------
# 图2：50 步语义网格与状态掩码（正文主图，自动从冻结 pkl 生成）
# ---------------------------------------------------------------------------
# 样本固定为触发头部截断的最长样本：65 词截断为 39 词 + 48 个内容片，
# 同图覆盖多子词 alpha、tail/mid-tail/ambi 标点继承、低置信度插值、SEP@49。
SEMANTIC_GRID_SAMPLE = "-a55Q6RWvTA$_$3"

ATTACH_STYLE = {  # attach_type -> (填充色, 文字色, hatch)
    "special": ("#5B6470", "white", None),
    "word": ("#BFD8EA", "#1F2933", None),
    "tail": ("#F5B971", "#1F2933", None),
    "mid-tail": ("#F5B971", "#1F2933", "///"),
    "ambi": ("#BFE3B4", "#1F2933", None),
}


def plot_semantic_grid(run_root: Path, output: Path, sid: str = SEMANTIC_GRID_SAMPLE) -> None:
    sample = pickle.load(open(run_root / "samples" / f"{sid.replace('$_$', '__')}.pkl", "rb"))
    mrow = pd.read_csv(run_root / "manifest.csv").set_index("id").loc[sid]
    rows = sample["wp_word_time_map"]
    meta = sample["metadata"]
    duration = float(meta["duration"])

    fig, (ax_tok, ax_mask, ax_aq, ax_time) = plt.subplots(
        4, 1, figsize=(6.9, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 2.1, 1.05, 1.6]},
    )
    fig.subplots_adjust(hspace=0.42)

    # ---- 词元行：位置 0–49 的 token，按 attach_type 着色 ----
    word_ids = [r["word_id"] for r in rows]
    for r in rows:
        pos = r["position"]
        fill, tcol, hatch = ATTACH_STYLE.get(r["attach_type"], ("#E5E7EB", "#1F2933", None))
        truncated = bool(r["trunc_flag"])
        ax_tok.add_patch(plt.Rectangle(
            (pos - 0.47, 0.04), 0.94, 0.64, facecolor=fill,
            edgecolor="#C0392B" if truncated else "white",
            linewidth=1.5 if truncated else 0.6, hatch=hatch, zorder=2,
        ))
        ax_tok.text(pos, 0.09, r["token"], rotation=90, ha="center", va="bottom",
                    fontsize=6.2, color=tcol, zorder=3)
    ax_tok.set_xlim(-0.55, 49.55)
    ax_tok.set_ylim(0, 4.1)
    ax_tok.set_yticks([])
    ax_tok.grid(False)
    ax_tok.set_title("词元行：CLS + 前 48 个内容片 + SEP@49（填充色 = attach_type，红框 = trunc_flag）",
                     loc="left", fontsize=8.4, pad=3)

    # 词边界参考线（相邻位置 word_id 变化处），贯穿词元/连续量/时间三个轴
    seps = [i for i in range(1, 50) if word_ids[i] != word_ids[i - 1]]
    for ax, y0, y1 in [(ax_tok, 0.02, 0.70), (ax_aq, -0.5, 1.5), (ax_time, 0, duration * 1.04)]:
        for i in seps:
            ax.plot([i - 0.5, i - 0.5], [y0, y1], color="#9AA5B1", lw=0.6, alpha=0.55, zorder=1)

    # 图例（attach_type 分类；tail/mid-tail 合并，斜纹示意词内继承）
    legend_items = [
        plt.Rectangle((0, 0), 1, 1, fc=ATTACH_STYLE["word"][0], ec="none", label="词片 word"),
        plt.Rectangle((0, 0), 1, 1, fc=ATTACH_STYLE["tail"][0], ec="none", hatch="///",
                      label="标点 tail / mid-tail（继承前词，斜纹 = 词内）"),
        plt.Rectangle((0, 0), 1, 1, fc=ATTACH_STYLE["ambi"][0], ec="none", label="歧义标点 ambi"),
        plt.Rectangle((0, 0), 1, 1, fc=ATTACH_STYLE["special"][0], ec="none", label="特殊位 CLS / SEP"),
        plt.Rectangle((0, 0), 1, 1, fc="none", ec="#C0392B", lw=1.4, label="trunc_flag 截断边界"),
    ]
    ax_tok.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(0.0, 1.02),
                  frameon=False, fontsize=6.2, ncol=2, handlelength=1.3, columnspacing=0.9)

    arrow = dict(arrowstyle="->", color="#374151", lw=0.7, shrinkA=1, shrinkB=1)
    # 注释按实测包围盒分带放置：带1 y∈[1.15,1.6]（短块）、带2 y∈[1.8,2.8]、带3 y∈[3.0,4.0]（图例与截断）
    # 低置信度插值词
    ax_tok.annotate("低置信度插值\nquality ≈ 0.08", xy=(2, 0.70), xytext=(0.8, 1.58),
                    ha="left", va="top", fontsize=6.4, arrowprops=arrow)
    # 多子词 + 标点继承 + alpha
    ax_tok.annotate("多子词：coupons → coup | ##ons\n标点 , attach_type=tail 继承前词\nalpha = 1/n_wp = 1/3",
                    xy=(6.5, 0.70), xytext=(7.5, 2.78), ha="left", va="top", fontsize=6.4,
                    arrowprops=arrow)
    # ambi 引号
    ax_tok.annotate("歧义标点（ambi）\n归属前后词存在歧义", xy=(40, 0.70), xytext=(24, 1.58),
                    ha="left", va="top", fontsize=6.4, arrowprops=arrow)
    # 头部截断 + SEP@49
    ax_tok.annotate("头部截断：仅保留 CLS + 前 48 个内容片\n其后 26 个词（t > 11.7 s）被丢弃\nSEP@49 固定末位，该位 observed_mask = 0",
                    xy=(48.2, 0.70), xytext=(38.5, 3.98), ha="center", va="top", fontsize=6.4,
                    arrowprops=arrow)

    # ---- 掩码行：三分区 + 三模态观测 ----
    masks = np.vstack([
        sample["content_mask"], sample["special_mask"], sample["padding_mask"],
        sample["observed_mask_text"], sample["observed_mask_audio"], sample["observed_mask_video"],
    ]).astype(int)
    ax_mask.imshow(masks, aspect="auto", cmap=mpl.colors.ListedColormap(["#F2F2F2", "#3E7CB1"]),
                   interpolation="nearest", vmin=0, vmax=1,
                   extent=(-0.5, 49.5, len(masks) - 0.5, -0.5))
    ax_mask.set_yticks(range(len(masks)))
    ax_mask.set_yticklabels(["content_mask", "special_mask", "padding_mask",
                             "observed_mask_text", "observed_mask_audio", "observed_mask_video"],
                            fontsize=6.8)
    ax_mask.set_title("状态掩码（浅色 = 0，深色 = 1；本样本三模态 observed_mask 逐位相同）",
                      loc="left", fontsize=8.4, pad=3)
    ax_mask.set_xticks(np.arange(51) - 0.5, minor=True)
    ax_mask.set_yticks(np.arange(len(masks) + 1) - 0.5, minor=True)
    ax_mask.grid(False)
    ax_mask.grid(which="minor", color="white", lw=0.35)
    ax_mask.tick_params(which="both", length=0)

    # ---- 连续状态量：alpha 与 quality ----
    aq = np.vstack([np.asarray(sample["alpha"]), np.asarray(sample["quality_audio"])])
    im_aq = ax_aq.imshow(aq, aspect="auto", cmap="viridis", interpolation="nearest",
                         vmin=0, vmax=1, extent=(-0.5, 49.5, 1.5, -0.5))
    ax_aq.set_yticks([0, 1])
    ax_aq.set_yticklabels(["alpha", "quality_audio"], fontsize=6.8)
    ax_aq.set_title("连续状态量（quality_video 与 quality_audio 逐位相同）", loc="left", fontsize=8.4, pad=3)
    ax_aq.set_xticks(np.arange(51) - 0.5, minor=True)
    ax_aq.set_yticks(np.arange(3) - 0.5, minor=True)
    ax_aq.grid(False)
    ax_aq.grid(which="minor", color="white", lw=0.35)
    ax_aq.tick_params(which="both", length=0)
    # 在 alpha 行标注 1/n_wp（仅在多子词组首格标注，避免相邻标签粘连）
    for r in rows:
        n = r.get("wp_count_in_word") or 0
        if r["attach_type"] != "special" and n > 1 and r["position"] - 1 in seps:
            ax_aq.text(r["position"], 0, f"1/{n}", ha="center", va="center",
                       fontsize=4.8, color="white")
    cbar = fig.colorbar(im_aq, ax=ax_aq, fraction=0.03, pad=0.01)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("取值 0–1", fontsize=6.5)

    # ---- 位置 → 物理时间：词区间池化，非等间隔帧 ----
    for r in rows:
        if r["t_s"] is not None:
            ax_time.bar(r["position"], r["t_e"] - r["t_s"], bottom=r["t_s"], width=0.86,
                        color="#3E7CB1", edgecolor="white", linewidth=0.25, zorder=2)
    xs = np.arange(50)
    ax_time.plot(xs, duration * (xs + 0.5) / 50, ls="--", lw=1.0, color="#888888", zorder=3,
                 label="等间隔采样参考（22.15 s ÷ 50）")
    ax_time.set_ylim(0, duration * 1.05)
    ax_time.set_ylabel("物理时间 (s)")
    ax_time.set_xlabel("语义位置（WordPiece 网格）")
    ax_time.set_title("语义位置 → 物理时间：词区间 [t_s, t_e) 池化后复制到各 WordPiece", loc="left",
                      fontsize=8.4, pad=3)
    ax_time.annotate("同一词的多个 WordPiece\n复制同一 [t_s, t_e)（coupons：位置 5–7）",
                     xy=(6, 2.6), xytext=(0.5, 12.6), ha="left", va="top", fontsize=6.4,
                     arrowprops=arrow)
    ax_time.annotate("虚线为名义等间隔时间；词区间明显偏离——\naudio / vision 并非 50 个等间隔帧",
                     xy=(40, duration * 40.5 / 50), xytext=(17, 18.4), ha="left", va="top",
                     fontsize=6.4, arrowprops=arrow)
    ax_time.legend(frameon=False, fontsize=6.4, loc="lower right")

    ax_time.set_xticks(list(range(0, 50, 5)) + [49])
    ax_time.tick_params(axis="x", labelbottom=True)
    for ax in (ax_tok, ax_mask, ax_aq):
        ax.tick_params(axis="x", labelbottom=False)

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "q1_semantic_grid.pdf")
    fig.savefig(output / "q1_semantic_grid.png", dpi=300)
    plt.close(fig)

    kept_words = sum(1 for w in sample["full_word_map"] if not w.get("dropped"))
    print("图2 caption 素材：")
    print(f"  样本 {sid}（{meta['video_id']} 片段 {meta['clip_id']}，{mrow['split']}）")
    print(f"  词数 {len(sample['full_word_map'])} → 保留 {kept_words}（截断丢弃 "
          f"{len(sample['full_word_map']) - kept_words}），内容片 48，CLS@0 + SEP@49，"
          f"trunc_flag 位于位置 47–48")
    print(f"  时长 {duration:.2f} s，fps {float(mrow['fps']):.4f}"
          f"（{'VFR' if bool(mrow['is_vfr']) else 'CFR'}），模态维度 text 50×768 / audio 50×25 / vision 50×23")
    print(f"  对齐状态 {mrow['alignment_status']}，对齐失败率 {float(mrow['alignment_failure_rate']):.4f}，"
          f"audio/vision 观测率 {float(mrow['audio_coverage']):.3f}/{float(mrow['vision_coverage']):.3f}")


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
    plot_semantic_grid(args.run_root, args.output)
    print(f"Generated Q1 figures in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
