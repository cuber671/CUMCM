#!/usr/bin/env python
"""§5.3 模型阶梯图（main.tex 第 446 行附近插入，fig:q2-ladder 图）。

五级阶梯：B0 → B1 → B2 → B3 → MRFN + MRFN+（八轮筛选）。
每级标注新增机制 + 关键性能（参数量、Acc、综合效用 S、31 场景退化）。
最终节点标注"验证集选型、测试集冻结"。

数据来源：paper/latex/tables/p2_ladder.tex 表格（手工摘录，保证与正文一致）。

术语与字号遵循 figure-polish-standard-v3。

Usage:
  .venv/bin/python scripts/figures/plot_q2_ladder.py
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


def plot_q2_ladder(output: Path) -> None:
    """阶梯图：B0→B1→B2→B3→MRFN 五级 + MRFN+ 终点。

    节点高度统一 h=2.20，y 坐标依次递增以体现阶梯感。
    内容起始 y+h-1.10，确保 5 行内容入盒。
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-1.4, 6.0)
    ax.axis("off")

    # 节点高度统一为 h=2.20，y 递增体现阶梯感（每级 +0.45）
    nodes = [
        # (x, y, w, h, 名称, 关键内容, 颜色)
        (0.20, 0.20, 1.85, 2.20,
         "B0\n地板基线",
         "逐模态线性投影\n+ 内容位掩码池化\n+ 拼接 + 双头\n无缺失状态",
         "#F5F5F5"),
        (2.50, 0.65, 1.85, 2.20,
         "B1\n+ 独立模态编码",
         "各模态独立编码\n回答：增益是否\n来自独立模态编码？",
         "#DCEAF7"),
        (4.80, 1.10, 1.85, 2.20,
         "B2\n+ 时序建模",
         "模态内换双向 GRU\n回答：增益是否\n来自时序建模？",
         "#E1F0E6"),
        (7.10, 1.55, 1.85, 2.20,
         "B3\n+ 跨模态注意力",
         "六方向跨模态注意力\n证据侧可用性屏蔽\n回答：增益是否\n来自跨模态交互？",
         "#FFF4D6"),
        (9.40, 2.00, 1.85, 2.20,
         "MRFN\n+ 缺失感知",
         "缺失状态编码\n（机制一）\n+ 可靠性门控\n（机制三）",
         "#EEEAF4"),
        (11.65, 2.45, 1.45, 2.20,
         "MRFN+\n（八轮改进）",
         "解冻编码器\n末 1 层\n+ 抽样权重\n+ 选型分早停",
         "#FBE7DC"),
    ]

    for x, y, w, h, name, content, color in nodes:
        # 主体框
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.0, edgecolor="#34495E", facecolor=color,
        ))
        # 标题（框内顶部）
        ax.text(x + w / 2, y + h - 0.32, name, ha="center", va="top",
                fontsize=7.4, fontweight="bold", color="#1F2933")
        # 内容（标题下方）
        ax.text(x + w / 2, y + h - 1.10, content, ha="center", va="top",
                linespacing=1.30, fontsize=6.4, color="#374151")

    # 箭头连接（B0→B1→B2→B3→MRFN→MRFN+），箭头从源框右中点到目标框左中点
    arrow_pairs = [
        ((2.05, 1.30), (2.50, 1.75)),   # B0 → B1
        ((4.35, 1.75), (4.80, 2.20)),   # B1 → B2
        ((6.65, 2.20), (7.10, 2.65)),   # B2 → B3
        ((8.95, 2.65), (9.40, 3.10)),   # B3 → MRFN
        ((11.25, 3.10), (11.65, 3.55)), # MRFN → MRFN+
    ]
    for start, end in arrow_pairs:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14,
            linewidth=1.2, color="#34495E", connectionstyle="arc3,rad=0.0",
        ))

    # 顶部标题（小型，作为图内提示，不冲突 v3 纪律）
    ax.text(6.6, 5.65, "模型阶梯：五级机制递增 + 八轮预注册筛选",
            ha="center", va="center", fontsize=8.8, color="#1F2933",
            fontweight="bold")

    # 性能条表头（统一放在 y=-0.05）
    ax.text(6.6, -0.05, "参数    Acc    $\\mathcal{S}$    $\\overline{D_{\\mathcal{S}}}$",
            ha="center", va="center", fontsize=6.8, color="#4D4D4D",
            fontweight="bold")
    # 性能条数据（每个节点正下方 y=-0.40）
    perf_data = [
        (0.20, "81k", "0.637", "0.736", "0.029"),
        (2.50, "94k", "0.657", "0.749", "0.035"),
        (4.80, "256k", "0.658", "0.748", "0.034"),
        (7.10, "429k", "0.650", "0.748", "0.035"),
        (9.40, "417k", "0.667", "0.756", "0.028"),
        (11.65, "—", "0.675", "+0.0225", "—"),
    ]
    for x, params, acc, s, deg in perf_data:
        text = f"{params}    {acc}    {s}    {deg}"
        ax.text(x + 0.92, -0.40, text, ha="center", va="center",
                fontsize=6.2, color="#374151")

    # 底部脚注（统一放在 y=-0.80 和 y=-1.15）
    ax.text(6.6, -0.85,
            "参数量（千）/ 干净准确率 / 综合效用 $\\mathcal{S}$ / 31 场景平均退化 $\\overline{D_{\\mathcal{S}}}$（附件2 test，三种子均值）",
            ha="center", va="center", fontsize=6.2, color="#6B7280")
    ax.text(6.6, -1.20,
            "末级 MRFN 同时引入两个组件，组件级归因由 §5.4 消融拆分（H6）；MRFN+ 经 §5.5 八轮预注册筛选唯一通过全部门槛",
            ha="center", va="center", fontsize=6.0, color="#6B7280")

    save_figure(fig, output / "q2_ladder.pdf")


def main() -> int:
    configure()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    plot_q2_ladder(FIG_OUT)
    print(f"Generated Q2 ladder: {FIG_OUT / 'q2_ladder.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
