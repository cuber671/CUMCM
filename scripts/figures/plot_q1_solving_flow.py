#!/usr/bin/env python
"""Q1 章首求解流程图（图 3.2，main.tex:§4 章首）。

六步压缩（多分支 + 收束）：
  ① 输入（附件1） → ② 三路并行特征提取 → ③ CTC 强制对齐
  → ④ 词区间池化 → ⑤ 统一时序坐标（4 类状态 + R1–R5）
  → ⑥ 四类机器指标验收

风格对齐：与 q2 / q3 求解流程图同构（figstyle.configure + FancyBboxPatch），
但改为多分支收束结构以体现三模态并行处理。
字号 ≥ 6.0pt，pdftotext 反查通过。

Usage:
  .venv/bin/python scripts/figures/plot_q1_solving_flow.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figstyle import configure

ROOT = Path(__file__).resolve().parents[2]
FIG_OUT = ROOT / "paper/latex/figures/q1"


def save_figure(fig: mpl.figure.Figure, output: Path, *, dpi: int = 300) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_q1_solving_flow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # ============ ① 输入 ============
    ax.add_patch(FancyBboxPatch(
        (0.15, 4.30), 1.95, 1.10,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor="#34495E", facecolor="#F4F4F4",
    ))
    ax.text(1.13, 4.85, "① 原始素材\n附件1\n100 条视频", ha="center", va="center",
            linespacing=1.30, fontsize=6.4)

    # ============ ② 三路并行特征提取 ============
    # 文本轨
    ax.add_patch(FancyBboxPatch(
        (2.85, 6.10), 2.50, 1.00,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor="#34495E", facecolor="#EAF2F8",
    ))
    ax.text(4.10, 6.60, "②a 文本\n转写 + BERT\n子词级 768 维", ha="center", va="center",
            linespacing=1.30, fontsize=6.4)

    # 音频轨
    ax.add_patch(FancyBboxPatch(
        (2.85, 4.30), 2.50, 1.00,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor="#34495E", facecolor="#E1F0E6",
    ))
    ax.text(4.10, 4.80, "②b 音频\neGeMAPSv02\n帧级 25 维", ha="center", va="center",
            linespacing=1.30, fontsize=6.4)

    # 视觉轨
    ax.add_patch(FancyBboxPatch(
        (2.85, 2.50), 2.50, 1.00,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor="#34495E", facecolor="#FBE7DC",
    ))
    ax.text(4.10, 3.00, "②c 视觉\nPy-Feat\n帧级 23 维", ha="center", va="center",
            linespacing=1.30, fontsize=6.4)

    # ① → ② 三路箭头
    for y in (6.60, 4.80, 3.00):
        ax.add_patch(FancyArrowPatch(
            (2.10, 4.85), (2.85, y),
            arrowstyle="-|>", mutation_scale=10,
            linewidth=0.9, color="#34495E",
            connectionstyle="arc3,rad=0.0",
        ))

    # ============ ③ CTC 强制对齐 ============
    ax.add_patch(FancyBboxPatch(
        (6.10, 4.30), 2.40, 1.10,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor="#34495E", facecolor="#FFF4D6",
    ))
    ax.text(7.30, 4.85, "③ CTC 强制对齐\n词区间 $I_w$\nC1 / C2 / C3 检查", ha="center", va="center",
            linespacing=1.30, fontsize=6.4)

    # ② → ③ 收束（文本 + 音频进入 CTC，视觉仅作特征）
    ax.add_patch(FancyArrowPatch(
        (5.35, 6.60), (6.10, 4.95),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=0.9, color="#34495E",
    ))
    ax.add_patch(FancyArrowPatch(
        (5.35, 4.80), (6.10, 4.85),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=0.9, color="#34495E",
    ))

    # ============ ④ 池化与映射 ============
    ax.add_patch(FancyBboxPatch(
        (9.20, 4.30), 2.40, 1.10,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor="#34495E", facecolor="#EEEAF4",
    ))
    ax.text(10.40, 4.85, "④ 池化与映射\n文本：不池化\n音/视：区间池化 + 复制", ha="center", va="center",
            linespacing=1.30, fontsize=6.4)

    # ③ → ④
    ax.add_patch(FancyArrowPatch(
        (8.50, 4.85), (9.20, 4.85),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.0, color="#34495E",
    ))
    # ②c 视觉 → ④（旁路）
    ax.add_patch(FancyArrowPatch(
        (5.35, 3.00), (9.20, 4.70),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=0.9, color="#34495E",
        connectionstyle="arc3,rad=-0.18",
    ))

    # ============ ⑤ 统一时序坐标 ============
    ax.add_patch(FancyBboxPatch(
        (12.30, 4.30), 2.50, 1.10,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.2, edgecolor="#1A5490", facecolor="#DCE6F1",
    ))
    ax.text(13.55, 4.85, "⑤ 统一时序坐标\n50 位网格\n4 类状态量 + R1–R5", ha="center", va="center",
            linespacing=1.30, fontsize=6.4, fontweight="bold")

    # ④ → ⑤
    ax.add_patch(FancyArrowPatch(
        (11.60, 4.85), (12.30, 4.85),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.0, color="#34495E",
    ))

    # ============ ⑥ 四类机器指标验收 ============
    metric_boxes = [
        (0.55, 0.55, "覆盖完整性\n100 / 100", "#EAF2F8"),
        (4.05, 0.55, "行为一致性\n非零率 ±2pp", "#E1F0E6"),
        (7.55, 0.55, "接口同源性\n逐位余弦 $\\geq 0.99$", "#FFF4D6"),
        (11.05, 0.55, "对齐质量\nok/review/rollback", "#FBE7DC"),
    ]
    for x, y, label, color in metric_boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), 2.95, 0.95,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.0, edgecolor="#34495E", facecolor=color,
        ))
        ax.text(x + 1.475, y + 0.475, label, ha="center", va="center",
                linespacing=1.30, fontsize=6.2)

    # ⑤ → ⑥ 单线到中央，再分四扇出
    ax.add_patch(FancyArrowPatch(
        (13.55, 4.30), (13.55, 1.85),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, color="#34495E",
    ))
    # 中央横线
    ax.plot([2.025, 12.525], [1.85, 1.85], color="#34495E", linewidth=1.0)
    # 中央到四个验收框
    for x_mid in (2.025, 5.525, 9.025, 12.525):
        ax.add_patch(FancyArrowPatch(
            (x_mid, 1.85), (x_mid, 1.50),
            arrowstyle="-|>", mutation_scale=10,
            linewidth=1.0, color="#34495E",
        ))

    # ============ 阶段标题 ============
    ax.text(1.13, 5.55, "① 输入", ha="center", va="center",
            fontsize=6.4, color="#4D4D4D", fontweight="bold")
    ax.text(4.10, 5.55, "② 三路并行特征提取", ha="center", va="center",
            fontsize=6.4, color="#4D4D4D", fontweight="bold")
    ax.text(7.30, 5.55, "③ CTC 对齐", ha="center", va="center",
            fontsize=6.4, color="#4D4D4D", fontweight="bold")
    ax.text(10.40, 5.55, "④ 池化与映射", ha="center", va="center",
            fontsize=6.4, color="#4D4D4D", fontweight="bold")
    ax.text(13.55, 5.55, "⑤ 统一坐标", ha="center", va="center",
            fontsize=6.4, color="#4D4D4D", fontweight="bold")
    ax.text(7.50, 2.30, "⑥ 四类机器指标验收（并行）",
            ha="center", va="center",
            fontsize=6.4, color="#4D4D4D", fontweight="bold")

    # ============ 顶部与底部脚注 ============
    ax.text(14.0, 7.05,
            "多分支收束：三模态并行 → CTC 收束 → 坐标统一 → 多指标并行验收",
            ha="right", va="center", fontsize=6.4, color="#6B7280", style="italic")
    ax.text(8.0, 0.18,
            "§4.1 数据事实 → §4.2 统一坐标 → §4.3 边界规则 → §4.4 CTC 对齐 → §4.5 验收",
            ha="center", va="center", fontsize=6.2, color="#6B7280")

    save_figure(fig, output / "q1_solving_flow.pdf")


def main() -> int:
    configure()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    plot_q1_solving_flow(FIG_OUT)
    print(f"Generated Q1 solving flow: {FIG_OUT / 'q1_solving_flow.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())