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
    """G1 总体框架：三泳道 × 五节点（输入→核心处理→输出），一级方法节点。

    节点命名与 main.tex §2"各问技术路线"定稿文字逐条对应：
    P1 提取→CTC 对齐→区间映射→坐标资产；P2 = MRFN 三机制（缺失状态编码/
    可用性约束跨模态注意力/可靠性门控融合）→双头；P3 预测→归因→验证→回溯。
    工具级/参数级细节（BERT/GRU/维度/联盟数）不进本图。
    """
    fig, ax = plt.subplots(figsize=(6.9, 4.75))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 9.9)
    ax.axis("off")

    lane_title(ax, 2.3, 9.45, "P1 统一语义—时间坐标系")
    lane_title(ax, 7.2, 9.45, "P2 缺失鲁棒预测")
    lane_title(ax, 12.1, 9.45, "P3 反事实解释")

    # 每泳道五节点：输入(y8.05) → 处理1(6.55) → 处理2(5.05) → 处理3(3.55) → 输出(1.95)
    LANE_X = (0.3, 5.2, 10.1)
    Y1, Y2, Y3, Y4, Y5 = 8.05, 6.55, 5.05, 3.55, 1.95
    H1, H, H5 = 1.15, 1.1, 1.25

    # ---- P1：数据 → 提取 → 对齐 → 映射 → 坐标资产 ----
    x = LANE_X[0]
    box(ax, x, Y1, 4.0, H1, "附件1 原始多模态样本\n视频 / 音频 / 转写文本（100 条）", C_IO)
    box(ax, x, Y2, 4.0, H, "多模态特征提取\n文本子词化 · 音视逐帧特征", C_OP)
    box(ax, x, Y3, 4.0, H, "CTC 强制对齐\n原词时间区间 $[t_s, t_e)$", C_OP)
    box(ax, x, Y4, 4.0, H, "词区间 → 子词位置映射\n音视特征按区间聚合", C_OP)
    box(ax, x, Y5, 4.0, H5,
        "统一 50 步坐标资产\n序列位置体系 · 观测状态\n位置—物理时间映射", C_P1)

    # ---- P2：缺失 → 编码 → 注意力 → 门控 → 预测 ----
    x = LANE_X[1]
    box(ax, x, Y1, 4.0, H1, "附件2 标准化特征\n缺失空间构造 (M, P, R, L)\n附件3 无标签专项推理", C_IO)
    box(ax, x, Y2, 4.0, H, "缺失状态编码\n缺失位置显式进入表示", C_P2)
    box(ax, x, Y3, 4.0, H, "可用性约束跨模态注意力\n不可用证据不进注意力", C_P2)
    box(ax, x, Y4, 4.0, H, "可靠性门控融合\n按模态可靠性加权", C_P2)
    box(ax, x, Y5, 4.0, H5, "双头预测：极性 + 强度\n（MRFN；附件3 推理用 MRFN+）", C_P2, fs=7.0)

    # ---- P3：输入 → 预测 → 归因 → 验证 → 证据 ----
    x = LANE_X[2]
    box(ax, x, Y1, 4.0, H1, "附件4 可解释专项\n三模态完整（20 条）", C_IO)
    box(ax, x, Y2, 4.0, H, "冻结预测器\n复用问题二模型与门控", C_P3)
    box(ax, x, Y3, 4.0, H, "反事实归因\n模态级精确 Shapley 值\n位置级积分梯度 IG", C_P3, fs=7.0)
    box(ax, x, Y4, 4.0, H, "保真度验证\n删除 / 插入操作性检验", C_P3)
    box(ax, x, Y5, 4.0, H5, "证据回溯\n原词 + $[t_s, t_e)$ 时间区间", C_P3)

    for lx in LANE_X:
        cx = lx + 2.0
        for y_top, y_bot in ((Y1, Y2 + H), (Y2, Y3 + H), (Y3, Y4 + H), (Y4, Y5 + H5)):
            arrow(ax, (cx, y_top), (cx, y_bot))

    # ---- 跨问题耦合（保留原三条）----
    arrow(ax, (4.3, Y5 + H5 / 2), (5.2, Y1 + H1 / 2), rad=0.03)
    ax.text(4.75, 6.2, "坐标接口\n观测状态\n位置体系",
            ha="center", va="center", fontsize=6.2, color="#4D4D4D")
    arrow(ax, (9.2, Y5 + H5 / 2), (10.1, Y2 + H / 2), rad=0.04)
    ax.text(9.42, 5.3, "模型 +\n门控值", ha="center", va="center", fontsize=6.2, color="#4D4D4D")
    arrow(ax, (2.3, Y5), (12.1, Y5), rad=-0.12)
    ax.text(7.2, 0.35, "位置—原词—物理时间映射（旁路直达问题三：证据时间定位不经预测模型，可独立核验）",
            ha="center", fontsize=6.6, color="#4D4D4D")

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
