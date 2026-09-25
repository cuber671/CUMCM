#!/usr/bin/env python
"""结构图：P2-F1 缺失空间构造与 MRFN 前向数据流（非数据图，与代码契约一致）。

Usage:
  .venv/bin/python scripts/figures/plot_diagrams.py

产物：
  paper/latex/figures/q2/q2_mrfn_architecture.{pdf,png}

版式（2026-09-25 v3，按用户两轮规范定稿，风格与图1 g1_framework 统一）：
- 左栏「缺失场景构造流程」三级递进，右栏「MRFN 前向数据流」四阶段，其中
  缺失鲁棒编码四模块（状态编码→BiGRU+可用性注意力→掩码池化→门控融合）
  由等边距虚线组框包裹（组框标签 6.5pt 粗，置框顶左上，与模块标题分层）；
- 跨栏映射全直角走线并与右栏层级水平对齐：场景生成（L1 中心→输入行中心）、
  可用状态输入（L2 中心→组框左缘中点，组内 x_eff/K·V 屏蔽/覆盖率三处共用）、
  掩码后特征输入（L3 右缘中点→框外左侧通道 90° 肘形→输入层下缘；通道与
  「可用状态输入」交叉处作半圆线桥 hop）；
- 元素语义：直角矩形，输入/定义=#ECECEC、处理=白、输出=#FFF0C2（三档亮度
  92/100/95，2026-09-25 用户终值——输出用色相而非灰度区分，图例色样同常量）；
  四级线条：边框/主流程 0.75pt 黑实线、辅助映射 0.5pt #666 虚线、
  组框 0.5pt #999 虚线；右栏模块垂直间距全列统一 0.30；
  底部图例条五项（字形与实物同比例）；图内不画总标题；
- 字号四级：栏标题 9pt 粗、模块标题 7.5pt 粗、正文/公式 6.5pt、备注 6pt 灰。
  变量斜体、函数/缩写正体（mathtext cm）。填充语义与图例逐项对应
  （左栏仅参数定义为输入灰）。正文未定义的内部术语（受控差异点等）不进图。

内容与 src/p2/models.py::MRFN 逐条对应：§6.1 z = p_i + x_eff + (1−δ)·e^m；
6 方向 AvailabilityAttention（K/V 屏蔽不可用证据、全空方向严格置零）；
c_m = Σ(content∧avail)/Σcontent；g = softmax(MLP([h̃;c]))；双头 cls/reg。
P2 输入维度用附件2 官方 74/35（区别于 P1 自产 25/23）；缺失空间记号
随正文符号表用 (M, P, r, l̄)（main.tex:173/336）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
FIG_OUT_Q2 = ROOT / "paper/latex/figures/q2"

INK, AUX, FRAME_C = "#000000", "#666666", "#999999"
FILL_IN, FILL_PROC, FILL_OUT = "#ECECEC", "#FFFFFF", "#FFF0C2"  # 三档亮度 92/100/95：输入灰/处理白/输出淡玉米（用户定稿，勿"统一"回灰阶）
X_MAX, Y_TOP, Y_BOT = 14.9, 11.0, -1.05        # 逻辑画布（底部留给图例条）
FIGSIZE = (6.9, 6.9 * (Y_TOP - Y_BOT) / X_MAX)

LX, LW, CX_L = 0.4, 3.9, 2.35                  # 左栏（右缘 4.3）
RX, RW, CX_R = 6.1, 8.25, 10.225               # 右栏模块（右缘 14.35）
GAP = 0.30                                     # 右栏模块垂直间距（全列统一）
FRAME = (5.85, 1.79, 8.75, 6.02)               # 虚线组框：等边距 0.25 包裹 M1–M4
LANE_X, HOP_Y = 4.55, 7.02                     # 掩码通道竖段 x；线桥所在高度
IN_GAP = 0.25                                  # 输入行三小盒水平间距

# (key, x, y, w, h, 填充, 模块标题, 正文)；右栏自下而上按 GAP=0.30 逐层定位，
# L2 中心=7.02 与 M1 编码层中心严格水平共线，L1 中心=9.345 与输入行中心共线。
# v3.1 文字瘦身：限定语/内部维度/括号补充移交正文，模块内只留接口与公式。
MODULES = [
    ("L1", LX, 8.555, LW, 1.58, FILL_IN, "缺失空间参数",
     "$M \\subseteq \\{t,a,v\\}$：7 种组合\n"
     "$P$：头 / 中 / 尾 / 随机\n"
     "$r$：缺失率 $|S|/w$；$\\bar{l}$：平均段长"),
    ("L2", LX, 6.445, LW, 1.15, FILL_PROC, "可用状态计算",
     "$a = o \\cdot (1-b)$\n"
     "$o$ 自然观测，$b$ 合成抹除"),
    ("L3", LX, 4.29, LW, 1.15, FILL_PROC, "分模态掩码处理",
     "text：编码前替换 [MASK]\n"
     "audio / vision：$x[S]{=}0,\\ b[S]{=}1$"),
    ("IN1", RX, 8.97, 2.5833, 0.75, FILL_IN, None, "text\n50×768"),
    ("IN2", RX + 2.8333, 8.97, 2.5833, 0.75, FILL_IN, None, "audio\n50×74"),
    ("IN3", RX + 5.6667, 8.97, 2.5833, 0.75, FILL_IN, None, "vision\n50×35"),
    ("PRJ", RX, 7.86, RW, 0.81, FILL_PROC, "特征投影层",
     "Linear → ReLU → Dropout，$d = 64$（3 模态独立）"),
    ("M1", RX, 6.48, RW, 1.08, FILL_PROC, "缺失状态编码",
     "$z = p_i + x_{\\mathrm{eff}} + (1-\\delta)\\cdot e^m$\n"
     "$a = 0$ 位不进计算图"),
    ("M2", RX, 5.10, RW, 1.08, FILL_PROC, "双向 GRU + 可用性注意力",
     "Q←源模态，K / V←证据模态（屏蔽不可用证据）\n"
     "全空证据方向输出严格置零"),
    ("M3", RX, 3.72, RW, 1.08, FILL_PROC, "掩码池化",
     "$c_m = \\sum(\\mathrm{content}\\wedge\\mathrm{avail})\\,/\\,\\sum\\mathrm{content}$"),
    ("M4", RX, 2.04, RW, 1.38, FILL_PROC, "门控融合",
     "$g = \\mathrm{softmax}(\\mathrm{MLP}([\\,\\tilde{h};\\, c\\,]))$\n"
     "$h = \\sum_m g_m\\,\\tilde{h}_m$"),
    ("OUT", RX, 0.66, RW, 1.08, FILL_OUT, "双头输出",
     "cls（3 类 logits）+ reg（强度，3·tanh）\n"
     "附加 gates / coverage 输出"),
]
IN_CX = [RX + 1.2917, RX + 4.125, RX + 6.9583]  # 三个输入小盒中心
MAIN_ARROWS = [((CX_L, 8.555), (CX_L, 7.595)), ((CX_L, 6.445), (CX_L, 5.44)),
               ((CX_R, 7.86), (CX_R, 7.56)), ((CX_R, 6.48), (CX_R, 6.18)),
               ((CX_R, 5.10), (CX_R, 4.80)), ((CX_R, 3.72), (CX_R, 3.42)),
               ((CX_R, 2.04), (CX_R, 1.74))]
# 图例（样式键, 标签, 字形起点 x），基线 y=-0.55；字形 0.9×0.32 与实物同比例
LEG_Y, LEG_GLYPH, LEG_H = -0.55, 0.9, 0.32
LEGEND = [("in", "输入模块", 2.3), ("proc", "处理模块", 4.75),
          ("out", "输出模块", 7.2), ("solid", "主流程", 9.65),
          ("aux", "辅助映射", 12.1)]


def configure_style() -> None:
    for name in ("NotoSansSC-Regular.otf", "NotoSansSC-Bold.otf"):
        p = Path.home() / ".fonts" / name
        if p.is_file():
            font_manager.fontManager.addfont(str(p))
    family = "Noto Sans SC" if any(
        f.name == "Noto Sans SC" for f in font_manager.fontManager.ttflist
        if "NotoSansSC" in f.fname) else "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    })


def main_arrow(ax, start, end) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=0.75, color=INK, shrinkA=0, shrinkB=0,
                                 zorder=4))


def aux_line(ax, start, end) -> None:
    ax.add_line(Line2D((start[0], end[0]), (start[1], end[1]),
                       linewidth=0.5, color=AUX, linestyle=(0, (4, 3)), zorder=3))


def aux_arrow(ax, start, end) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=0.5, color=AUX,
                                 linestyle=(0, (4, 3)), shrinkA=0, shrinkB=0,
                                 zorder=3))


def module(ax, x, y, w, h, fill, title, body):
    """直角矩形模块：标题 7.5pt 粗置顶，正文 6.5pt 居中；返回两个文本对象。"""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK,
                           linewidth=0.75, zorder=2))
    cx = x + w / 2
    if title:
        t_title = ax.text(cx, y + h - 0.16, title, ha="center", va="center",
                          fontsize=7.5, fontweight="bold", color=INK, zorder=5)
        t_body = ax.text(cx, y + h - 0.36, body, ha="center", va="top",
                         multialignment="center", linespacing=1.2,
                         fontsize=6.5, color=INK, zorder=5)
    else:
        t_title = None
        t_body = ax.text(cx, y + h / 2, body, ha="center", va="center",
                         multialignment="center", linespacing=1.2,
                         fontsize=6.5, color=INK, zorder=5)
    return t_title, t_body


def draw() -> tuple[plt.Figure, plt.Axes, dict, list]:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_axis_off()

    module_texts: dict[str, tuple] = {}
    other_texts: list[tuple[str, mtext.Text]] = []
    for key, x, y, w, h, fill, title, body in MODULES:
        module_texts[key] = module(ax, x, y, w, h, fill, title, body)

    # 栏标题（9pt 粗）
    for label, cx in (("缺失场景构造流程", CX_L), ("MRFN 前向数据流", CX_R)):
        t = ax.text(cx, 10.55, label, ha="center", va="center", fontsize=9.0,
                    fontweight="bold", color=INK, zorder=5)
        other_texts.append(("栏标题:" + label[:4], t))
    # 虚线组框（0.5pt #999）：标签 6.5pt 粗置框顶左上，白底压住框线
    fx, fy, fw, fh = FRAME
    ax.add_patch(Rectangle((fx, fy), fw, fh, facecolor="none", edgecolor=FRAME_C,
                           linewidth=0.5, linestyle=(0, (4, 3)), zorder=1))
    t = ax.text(fx + 0.18, fy + fh, "缺失鲁棒编码模块", ha="left", va="center",
                fontsize=6.5, fontweight="bold", color=INK, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    other_texts.append(("组框标签", t))

    # 主流程（严格垂直，间距全列统一 0.30）
    for a, b in MAIN_ARROWS:
        main_arrow(ax, a, b)
    for cx in IN_CX:
        main_arrow(ax, (cx, 8.97), (cx, 8.67))

    # 跨栏辅助映射（0.5pt #666 虚线，全直角走线）
    aux_arrow(ax, (4.30, 9.345), (RX, 9.345))    # 场景生成：L1 中心→输入行中心
    t = ax.text(5.02, 9.48, "场景生成", ha="center", va="center",
                fontsize=6.0, color=AUX, zorder=5)
    other_texts.append(("注:场景生成", t))
    aux_arrow(ax, (4.30, HOP_Y), (fx, HOP_Y))    # 可用状态输入：L2 中心→组框左缘
    t = ax.text(5.20, 7.24, "可用状态输入", ha="center", va="center",
                fontsize=6.0, color=AUX, zorder=5)
    other_texts.append(("注:可用状态输入", t))
    # 掩码后特征输入：L3 右缘中点→框外通道→输入层下缘；交叉处半圆线桥
    aux_line(ax, (4.30, 4.865), (LANE_X, 4.865))
    aux_line(ax, (LANE_X, 4.865), (LANE_X, HOP_Y - 0.12))
    ax.add_patch(Arc((LANE_X, HOP_Y), 0.24, 0.24, theta1=90, theta2=270,
                     linewidth=0.5, color=AUX, linestyle=(0, (4, 3)), zorder=3))
    aux_line(ax, (LANE_X, HOP_Y + 0.12), (LANE_X, 9.12))
    aux_arrow(ax, (LANE_X, 9.12), (RX, 9.12))
    t = ax.text(5.20, 5.60, "掩码后特征输入", ha="center", va="center",
                fontsize=6.0, color=AUX, zorder=5)
    other_texts.append(("注:掩码后特征输入", t))

    # 底部图例条（字形与实物同填充/同线型）
    for kind, name, gx in LEGEND:
        y0, y1 = LEG_Y - LEG_H / 2, LEG_Y + LEG_H / 2
        if kind in ("in", "proc", "out"):
            ax.add_patch(Rectangle((gx, y0), LEG_GLYPH, LEG_H,
                                   facecolor={"in": FILL_IN, "proc": FILL_PROC,
                                              "out": FILL_OUT}[kind],
                                   edgecolor=INK, linewidth=0.75, zorder=3))
        elif kind == "solid":
            main_arrow(ax, (gx, LEG_Y), (gx + LEG_GLYPH, LEG_Y))
        else:
            aux_arrow(ax, (gx, LEG_Y), (gx + LEG_GLYPH, LEG_Y))
        t = ax.text(gx + LEG_GLYPH + 0.18, LEG_Y, name, ha="left", va="center",
                    fontsize=6.5, color=INK, zorder=5)
        other_texts.append(("图例:" + name, t))
    return fig, ax, module_texts, other_texts


def audit(fig, ax, module_texts: dict, other_texts: list) -> bool:
    """验收：①模块文字入盒（四向留白 ≥0.03u）②全部文字两两无重叠（间隙 ≥0.02u）。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def data_bbox(t: mtext.Text):
        bb = mtext.Text.get_window_extent(t, renderer=renderer)
        (x0, y0), (x1, y1) = inv.transform(((bb.x0, bb.y0), (bb.x1, bb.y1)))
        return x0, y0, x1, y1

    ok = True
    spec = {k: (x, y, w, h) for k, x, y, w, h, *_ in MODULES}
    print("== 验收① 入盒审计（模块标题与正文分别对盒四向留白）==")
    for key, texts in module_texts.items():
        bx, by, bw, bh = spec[key]
        for role, t in zip(("标题", "正文"), texts):
            if t is None:
                continue
            x0, y0, x1, y1 = data_bbox(t)
            margins = (x0 - bx, y0 - by, bx + bw - x1, by + bh - y1)
            worst = min(margins)
            flag = "PASS" if worst >= 0.03 else "FAIL"
            ok &= flag == "PASS"
            print(f"  {key}-{role}: 左{margins[0]:+.2f} 下{margins[1]:+.2f} "
                  f"右{margins[2]:+.2f} 上{margins[3]:+.2f}  最小 {worst:+.2f}  {flag}")

    print("== 验收② 两两无重叠（含栏标题、副标题、组框标签、旁注、备注与图例）==")
    items = [(f"{k}-{r}", t) for k, ts in module_texts.items()
             for r, t in zip(("标题", "正文"), ts) if t is not None] + other_texts
    boxes = [(name, data_bbox(t)) for name, t in items]
    n_overlap = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (na, (ax0, ay0, ax1, ay1)), (nb, (bx0, by0, bx1, by1)) = boxes[i], boxes[j]
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox > -0.02 and oy > -0.02:
                n_overlap += 1
                ok = False
                print(f"  OVERLAP: {na} × {nb}  (dx={ox:+.2f}, dy={oy:+.2f})")
    print(f"  重叠对数：{n_overlap}  {'PASS' if n_overlap == 0 else 'FAIL'}")
    return ok


def main() -> int:
    configure_style()
    fig, ax, module_texts, other_texts = draw()
    if not audit(fig, ax, module_texts, other_texts):
        print("审计未通过，不落盘")
        return 1
    fig.savefig(FIG_OUT_Q2 / "q2_mrfn_architecture.pdf")
    fig.savefig(FIG_OUT_Q2 / "q2_mrfn_architecture.png", dpi=300)
    from PIL import Image
    w, h = Image.open(FIG_OUT_Q2 / "q2_mrfn_architecture.png").size
    target = X_MAX / (Y_TOP - Y_BOT)
    print(f"== 验收 PNG {w}×{h}px  宽高比 {w / h:.3f}"
          f"（画布目标 {target:.3f} ±5%）==")
    print("q2_mrfn_architecture.{pdf,png} →", FIG_OUT_Q2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
