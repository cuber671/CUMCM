#!/usr/bin/env python
"""图1 三问统一技术路线图 g1_framework（fig:g1，main.tex:93）——出版规范版。

Usage:
  .venv/bin/python scripts/figures/plot_g1_framework.py

版式（2026-09-25 出版规范定稿，黑白灰方案一）：形状语义——平行四边形
（倾斜 15°）=输入/输出、直角矩形=处理/模型、圆柱=数据集；实线 0.75pt=主流程、
灰 #666 虚线 0.5pt=辅助流程；跨列连线全直角化（横→竖→横，双通道上下分层）；
旁路改直角折线沿底部走；三级模块严格行共线（行距统一 0.6u）；图例条五项；
图内不画总标题（caption 归 main.tex）。dpi 沿用项目 300 约定（数据图实测）。
DIAGRAM_STYLE 为结构图填色族单源——figstyle.py 明文排除本族，故内置于此。

口径纪律（v3 冻结，块文本一字不改）：
- 语义锁：P2②「MRFN」承载主干结构（缺失状态编码 + 掩码注意力 + 门控）；
  P2③「MRFN+」及其末行「解冻末层 + 早停 + 3-seed 集成」是输出端补充变体，
  以 P2②→P2③ 上下层级表达「主干 → 追加变体」，详细口径归正文 5.5 节。
- P1③ 的 25/23 是 P1 自产特征维（openSMILE eGeMAPS 25 / Py-Feat 23），
  不得改成附件2 官方 74/35；「无标签」等实验属性与任何精度暗示不进图；
  「TOP-k 文本证据」「预注册追加」「双头」等已删字样不回填。
- 接线按 main.tex:90 叙事：P1②→P2①（坐标资产定义缺失空间）、
  P1③→P2②（统一表示作模型输入）、P2②→P3②（模型 + 门控值交接）、
  P1③⇢P3³ 虚线旁路 = 映射链（证据时间定位不经预测模型）。
- 数学符号：变量斜体（mathtext cm），下标 s/e 为 start/end 缩写用正体
  （$t_{\\mathrm{s}}$）；Shapley 联盟值为 $v(\\emptyset)$（非 v(θ)，口径锁）。

七项验收：①块文字全部入盒（含圆柱按筒身）②文字两两无重叠（含旁注/图例）
②b 旁路折线避让 ②c 连线—旁注净距 ③中文全 Noto Sans SC ④pdffonts 全嵌入
⑤PNG 宽高比 ±5% ⑥xelatex 无新增 Overfull ⑦无已删字样残留。
①②（含 ②b/②c）由本脚本内置几何审计自动执行。
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
FIG_OUT = ROOT / "paper/latex/figures"

DIAGRAM_STYLE = {   # 结构图填色族单源（figstyle.py 明文排除本族，故内置于此）
    "fill_io": "#F5F5F5",    # 输入/输出（平行四边形）
    "fill_proc": "#FFFFFF",  # 处理/模型（矩形）
    "fill_data": "#EAEAEA",  # 数据集（圆柱）
    "ink": "#000000", "aux": "#666666",
    "lw_main": 0.75, "lw_aux": 0.5, "lw_border": 0.75,
    "arrow_scale": 12, "dash": (0, (5, 3)),
    "fs_lane": 9.0, "fs_title": 7.5, "fs_body": 6.5,
    "fs_note": 6.0, "fs_legend": 6.5,
    "linespacing": 1.2, "skew_deg": 15.0,
}
S = DIAGRAM_STYLE

# 画布：三级行距统一 0.96u = 0.8×h（评审硬要求，实测落地；底行锚定 1.5）；底部留图例条
X_MAX, Y_TOP, Y_BOT = 14.6, 8.2, -0.95
U_IN = 6.9 / X_MAX
FIGSIZE = (6.9, 6.9 * (Y_TOP - Y_BOT) / X_MAX)
SKEW = math.tan(math.radians(S["skew_deg"]))      # 斜移量 = h·tan15°（角度统一）
CYL_EH = 0.28                                     # 圆柱椭圆半高
CYL_W = 3.0                                       # 圆柱宽（居中于列；1:2 需 2.4 与入盒冲突，取可达最窄）
TITLE_H = S["fs_title"] * S["linespacing"] / 72 / U_IN
BODY_H = S["fs_body"] * S["linespacing"] / 72 / U_IN

# 行基线：输入 6.22 / 处理 4.06 / 输出 1.5（行距 0.96；列沿用 v3 冻结值 0.3/5.2/10.1）
R_IN, R_MID, R_OUT = 6.22, 4.06, 1.5
H_IN, H_MID, H_OUT = 1.2, 1.2, 1.6
CY_IN, CY_MID = R_IN + H_IN / 2, R_MID + H_MID / 2   # 输入/处理层中心线 6.82 / 4.66

# (key, x, y, h, 文本)：三列 × 三层；w 默认 4.0，圆柱 CYL_W
BLOCKS = [
    ("P1①", 0.3, R_IN, H_IN, "附件1 原始多模态样本\n视频 / 音频 / 转写（100 条）"),
    ("P1②", 0.3, R_MID, H_MID, "词—时对齐\nCTC 强制对齐，$[t_{\\mathrm{s}},t_{\\mathrm{e}})$"),
    ("P1③", 0.3, R_OUT, H_OUT, "统一 50 步表示\ntext 768 / audio 25 / vision 23\nmask + $\\alpha$ + 质量 + 映射"),
    ("P2①", 5.2 + (4.0 - CYL_W) / 2, R_IN, H_IN, "缺失空间 $(M,P,R,L)$\n31 场景 + 附件3"),
    ("P2②", 5.2, R_MID, H_MID, "MRFN\n缺失状态编码 + 掩码注意力 + 门控"),
    ("P2③", 5.2, R_OUT, H_OUT, "极性 + 强度\nMRFN+\n解冻末层 + 早停 + 3-seed 集成"),
    ("P3①", 10.1, R_IN, H_IN, "附件4 可解释专项\n20 条 + 特征文件"),
    ("P3②", 10.1, R_MID, H_MID, "MCEF\nShapley $v(\\emptyset)$ + 条件基线 IG"),
    ("P3③", 10.1, R_OUT, H_OUT, "证据回溯\n原词 + $[t_{\\mathrm{s}},t_{\\mathrm{e}})$ + 模态贡献"),
]
LANES = [("P1 统一语义—时间坐标", 2.3), ("P2 缺失鲁棒预测", 7.2), ("P3 反事实解释", 12.1)]
LANE_Y = 7.82

# 连线（全直角，端点均为模块边中点）：列内主干 ×6 + 跨列双通道（A 内 4.75 / B 外 4.95）
MAIN_ARROWS = [
    [(2.3, R_IN), (2.3, R_MID + H_MID)], [(2.3, R_MID), (2.3, R_OUT + H_OUT)],
    [(7.2, R_IN), (7.2, R_MID + H_MID)], [(7.2, R_MID), (7.2, R_OUT + H_OUT)],
    [(12.1, R_IN), (12.1, R_MID + H_MID)], [(12.1, R_MID), (12.1, R_OUT + H_OUT)],
]
CROSS_A = [(4.3, CY_MID), (4.75, CY_MID), (4.75, CY_IN), (5.2 + (4.0 - CYL_W) / 2, CY_IN)]  # P1②→P2①
CROSS_B = [(4.514, 2.3), (4.95, 2.3), (4.95, CY_MID), (5.2, CY_MID)]                        # P1③→P2②
CROSS_C = [(9.2, CY_MID), (10.1, CY_MID)]                                                   # P2②→P3②
BYPASS = [(2.3, R_OUT), (2.3, 0.95), (10.6, 0.95), (10.6, R_OUT)]  # P1③⇢P3³ 映射链（直角折线）

# 旁注（文本, x, y, 对齐）：竖线段旁右侧对齐 / 横线段上方居中，一律水平排列
NOTES = [
    ("词位 $w$ / 缺失定义", 4.60, 5.74, "right"),
    ("统一表示 / 模型输入", 4.80, 3.48, "right"),
    ("模型 +\n门控 $g_m$", 9.65, 4.96, "center"),
]
BYPASS_NOTE = ("子词 → 原词 → 物理时间：P3 回溯坐标底座", 6.2, 1.2)

# 图例：占位 x=0，实际位置由 draw() 按文字实测量宽等间距分布并整体居中
LEG_Y, LEG_GLYPH, LEG_H = -0.5, 0.9, 0.32
LEGEND = [("solid", "主流程", 0.0), ("dash", "辅助流程", 0.0),
          ("io", "输入 / 输出", 0.0), ("proc", "处理 / 模型", 0.0),
          ("data", "数据集", 0.0)]


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


def poly_arrow(ax, pts, dashed=False) -> None:
    verts = [(x, y) for x, y in pts]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    ax.add_patch(FancyArrowPatch(
        path=MplPath(verts, codes), arrowstyle="-|>",
        mutation_scale=S["arrow_scale"],
        linewidth=S["lw_aux"] if dashed else S["lw_main"],
        color=S["aux"] if dashed else S["ink"],
        linestyle=S["dash"] if dashed else "solid",
        shrinkA=0, shrinkB=0, zorder=4))


def block_w(key: str) -> float:
    return CYL_W if key == "P2①" else 4.0


def shape_patch(ax, key, x, y, w, h) -> None:
    if key == "P2①":                                   # 数据集：圆柱
        cx = x + w / 2
        ax.add_patch(Rectangle((x, y + CYL_EH), w, h - 2 * CYL_EH,
                               facecolor=S["fill_data"], edgecolor=S["ink"],
                               linewidth=S["lw_border"], zorder=2))
        ax.add_patch(Ellipse((cx, y + CYL_EH), w, 2 * CYL_EH,
                             facecolor=S["fill_data"], edgecolor=S["ink"],
                             linewidth=S["lw_border"], zorder=2.1))
        ax.add_patch(Ellipse((cx, y + h - CYL_EH), w, 2 * CYL_EH,
                             facecolor=S["fill_data"], edgecolor=S["ink"],
                             linewidth=S["lw_border"], zorder=2.2))
    elif key[-1] in "①③":                             # 输入/输出：平行四边形（15°）
        s = h * SKEW
        ax.add_patch(Polygon(
            [(x, y), (x + w, y), (x + w + s, y + h), (x + s, y + h)],
            closed=True, facecolor=S["fill_io"], edgecolor=S["ink"],
            linewidth=S["lw_border"], zorder=2))
    else:                                              # 处理/模型：直角矩形
        ax.add_patch(Rectangle((x, y), w, h, facecolor=S["fill_proc"],
                               edgecolor=S["ink"], linewidth=S["lw_border"],
                               zorder=2))


def block_texts(ax, key, x, y, w, h, label) -> list[tuple[str, mtext.Text]]:
    """模块文字两级排版：首行 7.5pt 加粗主标题，其余 6.5pt 说明，双居中、行距 1.2。"""
    lines = label.split("\n")
    nb = len(lines) - 1
    cx, cy = x + w / 2, y + h / 2
    total = TITLE_H + nb * BODY_H
    parts = [(lines[0][:8], ax.text(cx, cy + total / 2 - TITLE_H / 2, lines[0],
                                    ha="center", va="center", fontsize=S["fs_title"],
                                    fontweight="bold", color=S["ink"], zorder=5))]
    if nb:
        parts.append((lines[1][:8], ax.text(
            cx, cy + total / 2 - TITLE_H - nb * BODY_H / 2, "\n".join(lines[1:]),
            ha="center", va="center", fontsize=S["fs_body"],
            linespacing=S["linespacing"], color=S["ink"], zorder=5)))
    return parts


def legend_glyph(ax, kind: str, gx: float) -> None:
    y0, y1 = LEG_Y - LEG_H / 2, LEG_Y + LEG_H / 2
    if kind == "solid":
        poly_arrow(ax, [(gx, LEG_Y), (gx + LEG_GLYPH, LEG_Y)])
    elif kind == "dash":
        poly_arrow(ax, [(gx, LEG_Y), (gx + LEG_GLYPH, LEG_Y)], dashed=True)
    elif kind == "io":
        s = 0.15
        ax.add_patch(Polygon(
            [(gx, y0), (gx + LEG_GLYPH - s, y0), (gx + LEG_GLYPH, y1), (gx + s, y1)],
            closed=True, facecolor=S["fill_io"], edgecolor=S["ink"],
            linewidth=0.9, zorder=3))
    elif kind == "proc":
        ax.add_patch(Rectangle((gx, y0), LEG_GLYPH, LEG_H, facecolor=S["fill_proc"],
                               edgecolor=S["ink"], linewidth=0.9, zorder=3))
    else:  # data：标准圆柱侧面（筒身 + 上下椭圆）
        ax.add_patch(Rectangle((gx, y0 + 0.08), LEG_GLYPH, LEG_H - 0.16,
                               facecolor=S["fill_data"], edgecolor=S["ink"],
                               linewidth=0.9, zorder=3))
        ax.add_patch(Ellipse((gx + LEG_GLYPH / 2, y0 + 0.08), LEG_GLYPH, 0.16,
                             facecolor=S["fill_data"], edgecolor=S["ink"],
                             linewidth=0.9, zorder=3.1))
        ax.add_patch(Ellipse((gx + LEG_GLYPH / 2, y1 - 0.08), LEG_GLYPH, 0.16,
                             facecolor=S["fill_data"], edgecolor=S["ink"],
                             linewidth=0.9, zorder=3.2))


def draw() -> tuple[plt.Figure, plt.Axes, dict, list]:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_axis_off()

    texts: dict[str, list[tuple[str, mtext.Text]]] = {}
    for key, x, y, h, label in BLOCKS:
        shape_patch(ax, key, x, y, block_w(key), h)
        texts[key] = block_texts(ax, key, x, y, block_w(key), h, label)

    other: list[tuple[str, mtext.Text]] = []
    for label, cx in LANES:
        t = ax.text(cx, LANE_Y, label, ha="center", va="center",
                    fontsize=S["fs_lane"], fontweight="bold", color=S["ink"], zorder=5)
        other.append(("泳道标题:" + label[:2], t))

    for pts in MAIN_ARROWS:
        poly_arrow(ax, pts)
    for pts in (CROSS_A, CROSS_B, CROSS_C):
        poly_arrow(ax, pts)
    poly_arrow(ax, BYPASS, dashed=True)

    for label, x, y, align in NOTES:
        t = ax.text(x, y, label, ha=align, va="center", fontsize=S["fs_note"],
                    color=S["ink"], linespacing=S["linespacing"], zorder=6)
        other.append((label.split("\n")[0][:6], t))
    label, x, y = BYPASS_NOTE
    t = ax.text(x, y, label, ha="center", va="center", fontsize=S["fs_note"],
                color=S["ink"], zorder=6)
    other.append(("旁路注:" + label[:6], t))

    # 图例：先量文字实宽 → 等间距均匀分布、整体居中 → 再按最终位置画字形
    legend_items = []
    for kind, name, _ in LEGEND:
        t = ax.text(0, LEG_Y, name, ha="left", va="center",
                    fontsize=S["fs_legend"], color=S["ink"], zorder=5)
        legend_items.append((kind, name, t))
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    inv_ = ax.transData.inverted()

    def text_w(t: mtext.Text) -> float:
        bb = t.get_window_extent(renderer=ren)
        return inv_.transform((bb.x1, 0))[0] - inv_.transform((bb.x0, 0))[0]

    item_w = [LEG_GLYPH + 0.15 + text_w(t) for _, _, t in legend_items]
    gap = (X_MAX - 1.0 - sum(item_w)) / (len(legend_items) - 1)
    gx = 0.5
    for (kind, name, t), w in zip(legend_items, item_w):
        legend_glyph(ax, kind, gx)
        t.set_position((gx + LEG_GLYPH + 0.15, LEG_Y))
        other.append(("图例:" + name, t))
        gx += w + gap
    return fig, ax, texts, other


def audit(fig: plt.Figure, ax: plt.Axes,
          texts: dict, other: list) -> bool:
    """验收①②：块文字入盒（留白 ≥0.04u）；全部文字两两无重叠（间隙 ≥0.02u）。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def data_bbox(t: mtext.Text) -> tuple[float, float, float, float]:
        bb = mtext.Text.get_window_extent(t, renderer=renderer)
        (x0, y0), (x1, y1) = inv.transform(((bb.x0, bb.y0), (bb.x1, bb.y1)))
        return x0, y0, x1, y1

    ok = True
    spec = {k: (x, y, block_w(k), h) for k, x, y, h, _ in BLOCKS}
    spec["P2①"] = (spec["P2①"][0], R_IN + CYL_EH, CYL_W, H_IN - 2 * CYL_EH)  # 圆柱按筒身审计
    print("== 验收① 入盒审计（块内文字四向留白，单位=逻辑坐标）==")
    flat = []
    for key, parts in texts.items():
        bx, by, bw, bh = spec[key]
        for pi, (tag, t) in enumerate(parts):
            x0, y0, x1, y1 = data_bbox(t)
            margins = [x0 - bx, y0 - by, bx + bw - x1, by + bh - y1]
            if key == "P2①" and pi == 0:
                # 圆柱顶边按椭圆弧审计（矩形审计对弧顶过保守；弧最低点在文字角落 x 处）
                cxe, a = bx + bw / 2, bw / 2
                arc = min(by + bh + CYL_EH * math.sqrt(max(0.0, 1 - ((xx - cxe) / a) ** 2))
                          for xx in (x0, x1))
                margins[3] = arc - y1
            worst = min(margins)
            flag = "PASS" if worst >= 0.04 else "FAIL"
            ok &= flag == "PASS"
            flat.append((f"块{key}·{tag}", t))
            print(f"  {key}·{tag}: 左{margins[0]:+.2f} 下{margins[1]:+.2f} "
                  f"右{margins[2]:+.2f} 上{margins[3]:+.2f}  最小 {worst:+.2f}  {flag}")

    print("== 验收② 两两无重叠（含泳道标题、旁注与图例）==")
    items = flat + other
    boxes = [(name, data_bbox(t)) for name, t in items]
    n_overlap = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (na, (ax0, ay0, ax1, ay1)), (nb_, (bx0, by0, bx1, by1)) = boxes[i], boxes[j]
            if na.startswith("块") and na.split("·")[0] == nb_.split("·")[0]:
                continue  # 同块「标题×说明」为构造分层，行框内边距贴连非重叠
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox > -0.02 and oy > -0.02:
                n_overlap += 1
                ok = False
                print(f"  OVERLAP: {na} × {nb_}  (dx={ox:+.2f}, dy={oy:+.2f})")
    print(f"  重叠对数：{n_overlap}  {'PASS' if n_overlap == 0 else 'FAIL'}")

    print("== 验收②b 旁路折线避让 ==")
    hy = BYPASS[1][1]
    note_bb = next(b for name, b in boxes if name.startswith("旁路注"))
    note_gap = min(abs(note_bb[1] - hy), abs(hy - note_bb[3]))
    ok_b = hy <= 1.42 and note_gap >= 0.05
    ok &= ok_b
    print(f"  底边横线 y={hy:.2f}（阈 ≤1.42）；旁注—横线净距={note_gap:.2f}u（阈 ≥0.05）  "
          f"{'PASS' if ok_b else 'FAIL'}")

    print("== 验收②c 连线—旁注净距（双通道与交接段 ≥0.10u）==")
    bbs = dict(boxes)
    checks = [
        ("注1—通道A(4.75)", bbs["词位 $w$"][2], 4.75),
        ("注2—通道B(4.95)", bbs["统一表示 /"][2], 4.95),
    ]
    ok_c = True
    for name, right, chan in checks:
        gap = chan - right
        ok_c &= gap >= 0.10
        print(f"  {name}: 净距 {gap:.2f}u")
    n3 = bbs["模型 +"]
    l_gap, r_gap = n3[0] - 9.2, 10.1 - n3[2]
    b_gap = n3[1] - 4.3
    ok_c &= l_gap >= 0.10 and r_gap >= 0.10 and b_gap >= 0.05
    print(f"  注3—交接段: 左{l_gap:.2f} 右{r_gap:.2f} 下{b_gap:.2f}u")
    ok &= ok_c
    print(f"  {'PASS' if ok_c else 'FAIL'}")
    return ok


def main() -> int:
    configure_style()
    fig, ax, texts, other = draw()
    if not audit(fig, ax, texts, other):
        print("审计未通过，不落盘")
        return 1
    fig.savefig(FIG_OUT / "g1_framework.pdf")
    fig.savefig(FIG_OUT / "g1_framework.png", dpi=300)
    from PIL import Image
    w, h = Image.open(FIG_OUT / "g1_framework.png").size
    target = X_MAX / (Y_TOP - Y_BOT)
    print(f"== 验收⑤ PNG {w}×{h}px  宽高比 {w / h:.3f}"
          f"（画布目标 {target:.3f} ±5% → [{target * 0.95:.3f}, {target * 1.05:.3f}]）==")
    print("g1_framework.{{pdf,png}} →", FIG_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
