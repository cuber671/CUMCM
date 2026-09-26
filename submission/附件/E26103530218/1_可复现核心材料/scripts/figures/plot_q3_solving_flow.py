#!/usr/bin/env python
"""Q3 章首求解流程图（图 9 章首，main.tex:§6 章首）。

五步压缩：输入 → Shapley → IG → 可信性验证 → 解释结果。
术语与字号遵循 figure-polish-standard-v3：所有文字在 main.tex 中可命中，
字号 ≥ 6.0pt，pdftotext 反查通过。

Usage:
  .venv/bin/python scripts/figures/plot_q3_solving_flow.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figstyle import configure

ROOT = Path(__file__).resolve().parents[2]
FIG_OUT = ROOT / "paper/latex/figures/q3"


def save_figure(fig: mpl.figure.Figure, output: Path, *, dpi: int = 300) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_q3_solving_flow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = [
        # ① 输入
        (0.15, 2.85, 1.95, 1.30,
         "附件4\n+ 冻结\nMRFN+ 预测器", "#EAF2F8"),
        # ② 模态级 Shapley
        (2.85, 2.85, 2.20, 1.30,
         "模态级 Shapley\n8 联盟全枚举\n带符号贡献 $\\varphi_m$\n主要参考模态", "#EEEAF4"),
        # ③ 位置级 IG
        (5.85, 2.85, 2.20, 1.30,
         "位置级 IG\n50 位 $\\times$ 64 步中点\n$\\mathrm{IG}_j$ + 词级 $A_w$\n条件特征基线", "#FFF4D6"),
        # ④ 可信性验证
        (8.85, 2.85, 2.20, 1.30,
         "可信性验证\nA--E 通过\nF 部分成立\n门控—贡献一致性", "#E1F0E6"),
        # ⑤ 解释结果
        (11.00, 2.85, 2.10, 1.30,
         "解释结果\n解释卡\n证据回溯至原词\n$+ [t_{\\mathrm{s}}, t_{\\mathrm{e}})$", "#FBE7DC"),
    ]
    for x, y, w, h, label, color in nodes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.0, edgecolor="#34495E", facecolor=color,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                linespacing=1.30, fontsize=6.4)

    arrows = [
        ((2.10, 3.50), (2.85, 3.50)),
        ((5.05, 3.50), (5.85, 3.50)),
        ((8.05, 3.50), (8.85, 3.50)),
        ((11.05, 3.50), (11.00, 3.50)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12,
            linewidth=1.0, color="#34495E", connectionstyle="arc3,rad=0.0",
        ))

    # 阶段标注
    stage_labels = [
        (1.12, 5.55, "① 输入"),
        (3.95, 5.55, "② Shapley"),
        (6.95, 5.55, "③ IG"),
        (9.95, 5.55, "④ 可信性"),
        (12.05, 5.55, "⑤ 解释"),
    ]
    for x, y, label in stage_labels:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=6.4, color="#4D4D4D", fontweight="bold")

    # 底部脚注
    ax.text(6.6, 0.78,
            "§6.1 形式化 → §6.2 Shapley → §6.3 IG → §6.4 可信性 → §6.5 解释结果",
            ha="center", va="center", fontsize=6.6, color="#4D4D4D")
    ax.text(6.6, 0.38,
            "复用问题二 MRFN+ 冻结预测器，不重新训练；证据经问题一映射链回溯",
            ha="center", va="center", fontsize=6.4, color="#6B7280")

    save_figure(fig, output / "q3_solving_flow.pdf")


def main() -> int:
    configure()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    plot_q3_solving_flow(FIG_OUT)
    print(f"Generated Q3 solving flow: {FIG_OUT / 'q3_solving_flow.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
