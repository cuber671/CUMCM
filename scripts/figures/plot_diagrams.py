#!/usr/bin/env python
"""结构图：G1 全文总体框架 + P2-F1 缺失空间与 MRFN 结构（非数据图，与代码契约一致）。

Usage:
  .venv/bin/python scripts/figures/plot_diagrams.py

产物：
  paper/latex/figures/g1_framework.{pdf,png}
  paper/latex/figures/q2/q2_mrfn_architecture.{pdf,png}

内容与 src/p2/models.py::MRFN 逐条对应：§6.1 z = p_i + x_eff + (1−δ)·e^m；
6 方向 AvailabilityAttention（K/V 屏蔽不可用证据、全空方向严格置零）；
c_m = Σ(content∧avail)/Σcontent；g = softmax(MLP([h̃;c]))；双头 cls/reg。
P2 输入维度用附件2 官方 74/35（区别于 P1 自产 25/23）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIG_OUT_G = ROOT / "paper/latex/figures"
FIG_OUT_Q2 = ROOT / "paper/latex/figures/q2"

EDGE = "#34495E"
C_P1, C_P2, C_P3 = "#DCEAF7", "#FBE7DC", "#EEEAF4"
C_IO, C_OP = "#EAF2F8", "#FFF4D6"


def configure_style() -> None:
    cjk = Path.home() / ".fonts/NotoSansSC-Regular.otf"
    if cjk.is_file():
        font_manager.fontManager.addfont(str(cjk))
        family = font_manager.FontProperties(fname=str(cjk)).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 8.0, "axes.unicode_minus": False,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    })


def box(ax, x, y, w, h, label, color, fs=7.4, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=lw, edgecolor=EDGE, facecolor=color))
    t = ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                linespacing=1.3, fontsize=fs)
    BOX_TEXTS.append((t, (x, y, w, h)))
    return (x, y, w, h)


BOX_TEXTS: list = []


def arrow(ax, start, end, rad=0.0, style="-|>", color=EDGE, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def lane_title(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=9.0,
            fontweight="bold", color="#1F2933")


def plot_g1() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 3.6))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    lane_title(ax, 2.3, 7.0, "P1 统一语义—时间坐标系")
    lane_title(ax, 7.2, 7.0, "P2 缺失鲁棒预测")
    lane_title(ax, 12.1, 7.0, "P3 反事实解释")

    box(ax, 0.3, 5.3, 4.0, 1.2, "附件1 原始多模态样本\n视频 / 音频 / 转写（100 条）", C_IO)
    box(ax, 0.3, 3.5, 4.0, 1.2, "原词—时间锚\nforced alignment，[t_s, t_e)", C_OP)
    box(ax, 0.3, 1.5, 4.0, 1.6,
        "P1 统一 50 步表示\ntext 768 / audio 25 / vision 23\nmask + α + quality + 映射表", C_P1)

    box(ax, 5.2, 5.3, 4.0, 1.2, "缺失空间 (M, P, R, L)\n31 场景网格 + 附件3（无标签）", C_IO)
    box(ax, 5.2, 3.5, 4.0, 1.2, "MRFN\n缺失状态编码/掩码注意力/门控", C_P2)
    box(ax, 5.2, 1.5, 4.0, 1.6, "双头：极性 + 强度\nMRFN+\n（BFT-lite，预注册追加）", C_P2, fs=7.0)

    box(ax, 10.1, 5.3, 4.0, 1.2, "附件4 可解释专项\n20 条 + 特征文件", C_IO)
    box(ax, 10.1, 3.5, 4.0, 1.2, "MCEF\nShapley v(∅) + IG 条件基线", C_P3)
    box(ax, 10.1, 1.5, 4.0, 1.6, "证据回溯：原词 + [t_s, t_e)\n模态贡献 / TOP-k 文本证据", C_P3)

    arrow(ax, (2.3, 5.3), (2.3, 4.7))
    arrow(ax, (2.3, 3.5), (2.3, 3.1))
    arrow(ax, (7.2, 5.3), (7.2, 4.7))
    arrow(ax, (7.2, 3.5), (7.2, 3.1))
    arrow(ax, (12.1, 5.3), (12.1, 4.7))
    arrow(ax, (12.1, 3.5), (12.1, 3.1))

    arrow(ax, (4.3, 2.6), (5.2, 4.0), rad=0.12)
    ax.text(4.75, 3.05, "统一表示\n模型输入", ha="center", va="top", fontsize=6.4, color="#4D4D4D")
    arrow(ax, (4.3, 4.1), (5.2, 5.5), rad=0.12)
    ax.text(4.75, 5.0, "词位 w\n缺失定义", ha="center", va="top", fontsize=6.4, color="#4D4D4D")
    arrow(ax, (9.2, 4.1), (10.1, 4.1))
    ax.text(9.65, 4.45, "模型 +\ngates", ha="center", va="center", fontsize=6.4, color="#4D4D4D")
    arrow(ax, (2.3, 1.5), (10.6, 1.5), rad=-0.18)
    ax.text(6.2, 0.42, "P1 映射表：WordPiece → 原词 → 物理时间（P3 证据回溯的坐标底座）",
            ha="center", fontsize=6.8, color="#4D4D4D")

    fig.savefig(FIG_OUT_G / "g1_framework.pdf")
    fig.savefig(FIG_OUT_G / "g1_framework.png", dpi=300)
    plt.close(fig)


def plot_mrfn_arch() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 5.0))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 11.2)
    ax.axis("off")

    # ---- 左列：缺失空间 (M,P,R,L) ----
    lane_title(ax, 2.2, 10.8, "缺失空间 (M, P, R, L)")
    box(ax, 0.3, 8.9, 3.8, 1.5,
        "M ⊆ {t,a,v}：7 组合\nP：头 / 中 / 尾 / 随机\nR：缺失率  |S|/w\nL：平均段长（短散 / 连续块）", C_IO, fs=6.8)
    box(ax, 0.3, 7.3, 3.8, 1.2,
        "可用状态 a = o·(1−b)\no 自然观测，b 合成抹除\nCLS/SEP/padding 不抹", C_OP, fs=6.8)
    box(ax, 0.3, 5.9, 3.8, 1.0,
        "text：BERT 编码前替换 [MASK]\n（保留位置与 50 步坐标）", "#FDEBD0", fs=6.8)
    box(ax, 0.3, 4.5, 3.8, 1.0,
        "audio/vision：x[S]=0, b[S]=1\n（z 中不进计算图）", "#FDEBD0", fs=6.8)
    arrow(ax, (2.2, 8.9), (2.2, 8.5))
    arrow(ax, (2.2, 7.3), (2.2, 6.9))
    arrow(ax, (2.2, 5.9), (2.2, 5.5))

    # ---- 右列：MRFN 数据流 ----
    lane_title(ax, 9.5, 10.8, "MRFN 数据流（§6，与代码逐条对应）")
    x0, w = 5.6, 8.5
    ws = w / 3 - 0.18
    for i, (lab, col) in enumerate([
        ("text\n50×768", C_IO), ("audio\n50×74", C_IO), ("vision\n50×35", C_IO)]):
        box(ax, x0 + i * (ws + 0.18), 9.6, ws, 1.05, lab + "\n+ avail", col, fs=6.8)
    box(ax, x0, 8.15, w, 0.85, "投影 Linear→ReLU→Dropout，d=64（3 模态独立）", C_OP, fs=6.8)
    box(ax, x0, 6.65, w, 1.15,
        "缺失状态编码（§6.1）\n"
        "$z = p_i + x_{eff} + (1-\\delta)\\cdot e^m$　（$p_i$ 位置嵌入，$e^m$ 模态缺失嵌入）\n"
        "a=0 位不进计算图（受控差异点 1）", C_P2, fs=6.8)
    box(ax, x0, 5.1, w, 1.2,
        "content 窗口 BiGRU（2d=128）+ 6 方向可用性注意力\nQ←源模态，K/V←证据模态；K/V 屏蔽不可用证据\n全空证据方向输出严格置零（差异点 2）", C_P2, fs=6.8)
    box(ax, x0, 3.6, w, 1.05,
        "masked_pool → $\\tilde{h}_m$（每模态 256 = 128 + 2×64）\n"
        "覆盖率 $c_m$ = Σ(content∧avail) / Σcontent", C_P2, fs=6.8)
    box(ax, x0, 2.0, w, 1.3,
        "门控融合（§6.3）：$g = \\mathrm{softmax}(\\mathrm{MLP}([\\tilde{h};c]))$\n"
        "$h = \\sum_m g_m \\tilde{h}_m$　（$[\\tilde{h};c]$ 为三模态池化与覆盖率拼接）", C_P2, fs=6.6)
    box(ax, x0, 0.5, w, 1.05,
        "双头输出：cls（3 类 logits）+ reg（强度，3·tanh）\n附加输出 gates / coverage（门控行为验证）", C_OP, fs=6.8)

    for y0, y1 in ((9.6, 9.0), (8.15, 7.8), (6.65, 6.3), (5.1, 4.65), (3.6, 3.3), (2.0, 1.55)):
        arrow(ax, (x0 + w / 2, y0), (x0 + w / 2, y1))
    arrow(ax, (4.1, 9.35), (5.6, 9.35))
    ax.text(4.85, 9.62, "场景生成", ha="center", fontsize=6.6, color="#4D4D4D")

    fig.savefig(FIG_OUT_Q2 / "q2_mrfn_architecture.pdf")
    fig.savefig(FIG_OUT_Q2 / "q2_mrfn_architecture.png", dpi=300)
    plt.close(fig)


def main() -> int:
    configure_style()
    FIG_OUT_Q2.mkdir(parents=True, exist_ok=True)
    plot_g1()
    plot_mrfn_arch()
    print("g1_framework / q2_mrfn_architecture → paper/latex/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
