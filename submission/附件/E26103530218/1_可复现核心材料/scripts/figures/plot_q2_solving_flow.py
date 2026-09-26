#!/usr/bin/env python
"""Q2 章首求解流程图（图 5 章首，main.tex:§5 章首）。

五步压缩：输入 → 缺失空间 → MRFN → 训练与改进 → 评测与行为。
术语与字号遵循 figure-polish-standard-v3：所有文字在 main.tex 中可命中，
字号 ≥ 6.0pt，pdftotext 反查通过。

Usage:
  .venv/bin/python scripts/figures/plot_q2_solving_flow.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figstyle import configure

ROOT = Path(__file__).resolve().parents[2]
FIG_OUT = ROOT / "paper/latex/figures/q2"


def save_figure(fig: mpl.figure.Figure, output: Path, *, dpi: int = 300) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_q2_solving_flow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = [
        # ① 输入
        (0.15, 2.85, 1.95, 1.30,
         "附件2\n官方特征\n74 / 35", "#EAF2F8"),
        # ② 缺失空间
        (2.85, 2.85, 2.20, 1.30,
         "缺失空间\n$(M, P, r, \\bar{l})$\n合成缺失算子\n31 场景网格", "#EEEAF4"),
        # ③ MRFN
        (5.85, 2.85, 2.20, 1.30,
         "MRFN\n缺失状态编码\n+ 可用性约束注意力\n+ 可靠性门控", "#FFF4D6"),
        # ④ 训练与改进
        (8.85, 2.85, 2.20, 1.30,
         "训练与改进\n五级阶梯 B0$\\to$…$\\to$MRFN\n八轮预注册 $\\to$ MRFN+", "#E1F0E6"),
        # ⑤ 评测 + 行为
        (11.00, 2.85, 2.10, 1.30,
         "评测与行为\n31 场景退化\n错误归因\n附件3 行为分析", "#FBE7DC"),
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
        (3.95, 5.55, "② 缺失空间"),
        (6.95, 5.55, "③ MRFN"),
        (9.95, 5.55, "④ 训练与改进"),
        (12.05, 5.55, "⑤ 评测与行为"),
    ]
    for x, y, label in stage_labels:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=6.4, color="#4D4D4D", fontweight="bold")

    # 底部脚注：与正文 §5 六节顺序对应 + 输出接口
    ax.text(6.6, 0.78,
            "§5.1 缺失空间 → §5.2 MRFN → §5.3 训练阶梯 → §5.4 鲁棒性 → §5.5 终评 → §5.6 行为",
            ha="center", va="center", fontsize=6.6, color="#4D4D4D")
    ax.text(6.6, 0.38,
            "输出：极性 + 强度 + 门控 $g_m$；MRFN+ 冻结预测器供问题三反事实解释复用",
            ha="center", va="center", fontsize=6.4, color="#6B7280")

    save_figure(fig, output / "q2_solving_flow.pdf")


def main() -> int:
    configure()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    plot_q2_solving_flow(FIG_OUT)
    print(f"Generated Q2 solving flow: {FIG_OUT / 'q2_solving_flow.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
