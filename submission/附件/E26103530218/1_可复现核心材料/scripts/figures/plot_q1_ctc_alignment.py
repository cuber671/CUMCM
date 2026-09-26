#!/usr/bin/env python
"""Q1 §4.4 CTC 强制对齐示意图（图 main.tex:§4.4）。

三层横向映射：
  1. CTC 目标序列（字符级 + | 词分隔符）
  2. CTC 路径（每帧发射；ε 为空白符；维特比最优路径）
  3. 50 位统一网格 + 词区间 [t_s, t_e)

数据来源：从 runs/p1_full/samples 抽取一条 5-8 词、全部对齐的样本；
构字级帧分布按词区间均分（word 级 t_s/t_e 来自 pkl 真实数据）。

术语与字号遵循 figure-polish-standard-v3：所有文字在 main.tex 中可命中，
字号 ≥ 6pt，pdftotext 反查通过；图内不画总标题（caption 归 main.tex）。

Usage:
  .venv/bin/python scripts/figures/plot_q1_ctc_alignment.py
"""
from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figstyle import configure, FONT_SIZE

ROOT = Path(__file__).resolve().parents[2]
FIG_OUT = ROOT / "paper/latex/figures/q1"
SAMPLE_DIR = ROOT / "runs/p1_full/samples"

# 结构图填色族（与 figstyle.py 数据图色板分离；q1_solving_flow 同族低饱和淡彩）
C = {
    "frame_bg":   "#FAF7F0",  # 米白背景（网格单元）
    "blank":      "#E8E4DA",  # 空白符 ε
    "blank_edge": "#9C9685",
    "char":       "#DCE9F1",  # 字符发射（淡蓝）
    "char_edge":  "#6F8AA6",
    "sep":        "#1F2937",  # | 词分隔符（深灰，强语义）
    "sep_face":   "#374151",
    "ink":        "#374151",  # 主文字
    "aux":        "#6B7280",  # 辅助文字
    "rule":       "#4B5563",  # 边线
    "grid_line":  "#D1D5DB",  # 网格线
    "special":    "#BFBFBF",
    "pad":        "#F0EFEC",
}
# 词区间色板（淡彩，与 q1_solving_flow 同族：蓝/绿/黄/紫/粉，五色循环）
WORD_FILL = ["#EAF2F8", "#E1F0E6", "#FFF4D6", "#EEEAF4", "#FBE7DC"]
WORD_EDGE = ["#3B6E8F", "#3F7B57", "#A8853B", "#6B5B95", "#9C5A45"]


def find_short_ok_sample():
    """选一条 5-8 词的 OK 样本（failure_rate==0）。"""
    for pkl in sorted(SAMPLE_DIR.glob("*.pkl")):
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        fwm = data.get("full_word_map", [])
        align = data.get("alignment", {})
        if 5 <= len(fwm) <= 8 and align.get("failure_rate", 1) == 0:
            return data
    return None


def build_per_frame_symbols(words, last_end_frame, tail_blank=8):
    """按词区间均分构建每帧发射符号（character-level frame path）。"""
    path = []
    last_end = 0
    for w in words:
        ws, we = int(w["start_frame"]), int(w["end_frame"])
        for f in range(last_end, ws):
            path.append((f, "ε"))
        text = w["ctc_text"].lower()
        n_chars = len(text)
        n_frames = we - ws
        per_char = max(1, n_frames // n_chars)
        remainder = max(0, n_frames - per_char * n_chars)
        f_cursor = ws
        for i, ch in enumerate(text):
            n_this = per_char + (1 if i < remainder else 0)
            for _ in range(n_this):
                if f_cursor < we:
                    path.append((f_cursor, ch))
                    f_cursor += 1
        last_end = we
    tail_end = last_end_frame + tail_blank
    for f in range(last_end, tail_end):
        path.append((f, "ε"))
    return path


def compress_runs(path):
    """运行长度压缩：list[(symbol, n_frames, start_frame)]。"""
    runs = []
    if not path:
        return runs
    sym = path[0][1]
    cnt = 1
    start = path[0][0]
    for f, s in path[1:]:
        if s == sym:
            cnt += 1
        else:
            runs.append((sym, cnt, start))
            sym, cnt, start = s, 1, f
    runs.append((sym, cnt, start))
    return runs


def pick_axis_ticks(words, total_frames, min_gap=6):
    """只挑不会重叠的关键帧刻度：每个 word 选 start/end；额外加 0 与 total-1。"""
    keys = [0]
    for w in words:
        keys.append(int(w["start_frame"]))
        keys.append(int(w["end_frame"]))
    keys.append(total_frames - 1)
    keys = sorted(set(keys))
    out = []
    for k in keys:
        if not out or k - out[-1] >= min_gap:
            out.append(k)
    if out[0] != 0:
        out = [0] + out
    if out[-1] != total_frames - 1:
        out = out + [total_frames - 1]
    return out


def main():
    configure()
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    data = find_short_ok_sample()
    if data is None:
        print("No suitable sample found.")
        return 1

    words = data["alignment"]["words"]
    duration = data["metadata"]["duration"]
    wp = data["wp_word_time_map"]
    fwm = data["full_word_map"]
    sample_id = data["id"]
    norm_text = data["normalized_text"]

    # 总帧数 = 末词 end_frame + 尾空白
    last_end_frame = max(int(w["end_frame"]) for w in words)
    path = build_per_frame_symbols(words, last_end_frame, tail_blank=8)
    runs = compress_runs(path)

    # ========== 画布 ==========
    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    # 四层行基线（自上而下）
    Y_META = 10.20      # 示例元数据
    Y_TARGET = 9.10     # CTC 目标
    Y_PATH = 6.65       # CTC 路径
    Y_BAR = 4.10        # 词区间水平条
    Y_GRID = 2.45       # 50 位网格
    Y_AXIS = 0.95       # 帧时间轴

    # 共用水平刻度（基于帧号 0..total_frames）
    total_frames = last_end_frame + 8
    X_LEFT = 0.50
    X_RIGHT = 12.55
    span = X_RIGHT - X_LEFT
    x_of = lambda f: X_LEFT + span * f / total_frames

    # ========== 顶部示例元数据 ==========
    safe_id = sample_id.replace("$", r"\$")
    ax.text(X_LEFT, Y_META,
            f"示例：``{norm_text}''（{len(words)} 词，{safe_id}，时长 {duration:.2f} s）",
            ha="left", va="center", fontsize=6.6, color=C["ink"])

    # ========== Layer 1：CTC 目标序列 ==========
    target_chars = []
    target_groups = []
    for wi, w in enumerate(words):
        text = w["ctc_text"].lower()
        for ch in text:
            target_chars.append(ch)
            target_groups.append(wi)
        target_chars.append("|")
        target_groups.append(wi)

    n_chars = len(target_chars)
    char_w = span / n_chars
    char_h = 0.50

    for i, (ch, gid) in enumerate(zip(target_chars, target_groups)):
        x0 = X_LEFT + i * char_w
        xc = x0 + char_w / 2
        if ch == "|":
            ax.add_patch(Rectangle(
                (x0 + 0.02, Y_TARGET - char_h / 2), char_w - 0.04, char_h,
                facecolor=C["sep_face"], edgecolor=C["sep"], linewidth=0.8))
            ax.text(xc, Y_TARGET, "|", ha="center", va="center",
                    fontsize=8.5, color="#FFFFFF", fontweight="bold")
        else:
            ax.add_patch(Rectangle(
                (x0 + 0.02, Y_TARGET - char_h / 2), char_w - 0.04, char_h,
                facecolor=WORD_FILL[gid % len(WORD_FILL)],
                edgecolor=WORD_EDGE[gid % len(WORD_EDGE)], linewidth=0.5))
            ax.text(xc, Y_TARGET, ch, ha="center", va="center",
                    fontsize=7.0 if char_w > 0.42 else 6.2, color=C["ink"])

    # Layer 1 左侧标签
    ax.text(X_LEFT - 0.30, Y_TARGET + 0.20, "① 字符目标",
            ha="right", va="center", fontsize=7.4, color=C["ink"], fontweight="bold")
    ax.text(X_LEFT - 0.30, Y_TARGET - 0.20, r"($\mathrm{target}$)",
            ha="right", va="center", fontsize=6.4, color=C["aux"])

    # Layer 1 右侧注解
    ax.text(X_RIGHT + 0.10, Y_TARGET + 0.20,
            r"$\mathrm{target}=w_1\,|\,w_2\,|\,\cdots\,|\,w_K$",
            ha="left", va="center", fontsize=6.4, color=C["aux"])
    ax.text(X_RIGHT + 0.10, Y_TARGET - 0.20,
            r"$\mathcal{B}(\pi)=\mathrm{target}$",
            ha="left", va="center", fontsize=6.4, color=C["aux"])

    # ========== Layer 2：CTC 路径（每帧发射） ==========
    y_path = Y_PATH
    run_h = 0.50
    for sym, cnt, start in runs:
        x0 = x_of(start)
        x1 = x_of(start + cnt)
        w_run = max(0.015, x1 - x0)
        if sym == "ε":
            face = C["blank"]
            edge = C["blank_edge"]
        else:
            face = C["char"]
            edge = C["char_edge"]
        ax.add_patch(Rectangle(
            (x0, y_path - run_h / 2), w_run, run_h,
            facecolor=face, edgecolor=edge, linewidth=0.35))
        if w_run > 0.30:
            label = sym if sym != "ε" else r"$\varepsilon$"
            ax.text(x0 + w_run / 2, y_path, label, ha="center", va="center",
                    fontsize=6.4, color=C["ink"] if sym != "ε" else C["aux"])
        elif w_run > 0.16 and sym != "ε":
            ax.text(x0 + w_run / 2, y_path, sym, ha="center", va="center",
                    fontsize=6.0, color=C["ink"])

    # Layer 2 左侧标签
    ax.text(X_LEFT - 0.30, y_path + 0.20, r"② 字符级路径 $\pi$",
            ha="right", va="center", fontsize=7.4, color=C["ink"], fontweight="bold")
    ax.text(X_LEFT - 0.30, y_path - 0.20, "(维特比求解；每帧取值)",
            ha="right", va="center", fontsize=6.4, color=C["aux"])

    # ========== Layer 3a：词区间水平条 ==========
    y_bar = Y_BAR
    bar_h = 0.30
    for w in words:
        wid = int(w["word_id"])
        ws_x = x_of(int(w["start_frame"]))
        we_x = x_of(int(w["end_frame"]))
        # 最小可视宽度 0.18u（极短词如 "i" 在 1 帧内仍可见）
        bar_width = max(0.18, we_x - ws_x)
        ax.add_patch(Rectangle(
            (ws_x, y_bar - bar_h / 2), bar_width, bar_h,
            facecolor=WORD_FILL[wid % len(WORD_FILL)],
            edgecolor=WORD_EDGE[wid % len(WORD_EDGE)], linewidth=0.7))
        ax.text(ws_x + bar_width / 2, y_bar, w["word_text"],
                ha="center", va="center", fontsize=6.6, color=C["ink"],
                fontweight="bold")

    # 仅首末词显示 t_s / t_e 标注（避免密集词区间处的文字重叠）
    t_s_first = words[0]["t_s"]
    t_e_first = words[0]["t_e"]
    t_s_last = words[-1]["t_s"]
    t_e_last = words[-1]["t_e"]
    ws0_x = x_of(int(words[0]["start_frame"]))
    we0_x = x_of(int(words[0]["end_frame"]))
    wsn_x = x_of(int(words[-1]["start_frame"]))
    wen_x = x_of(int(words[-1]["end_frame"]))

    ax.text(ws0_x - 0.04, y_bar + bar_h / 2 + 0.18,
            f"$t_{{\\mathrm{{s}}}}={t_s_first:.2f}\\,\\mathrm{{s}}$",
            ha="right", va="bottom", fontsize=6.0, color=C["aux"])
    ax.text(we0_x + 0.04, y_bar + bar_h / 2 + 0.18,
            f"$t_{{\\mathrm{{e}}}}={t_e_first:.2f}\\,\\mathrm{{s}}$",
            ha="left", va="bottom", fontsize=6.0, color=C["aux"])
    ax.text(wsn_x - 0.04, y_bar + bar_h / 2 + 0.18,
            f"$t_{{\\mathrm{{s}}}}={t_s_last:.2f}\\,\\mathrm{{s}}$",
            ha="right", va="bottom", fontsize=6.0, color=C["aux"])
    ax.text(wen_x + 0.04, y_bar + bar_h / 2 + 0.18,
            f"$t_{{\\mathrm{{e}}}}={t_e_last:.2f}\\,\\mathrm{{s}}$",
            ha="left", va="bottom", fontsize=6.0, color=C["aux"])

    # Layer 3a 左侧标签
    ax.text(X_LEFT - 0.30, y_bar + 0.05, "③ 词区间",
            ha="right", va="center", fontsize=7.4, color=C["ink"], fontweight="bold")
    ax.text(X_LEFT - 0.30, y_bar - 0.22, r"$[t_{\mathrm{s}},t_{\mathrm{e}})$",
            ha="right", va="center", fontsize=6.4, color=C["aux"])

    # ========== Layer 3b：50 位统一网格 ==========
    y_grid = Y_GRID
    grid_h = 0.55

    # 7 个活动位 + 43 个填充位；活动位给宽，填充位紧凑并标注 ×43
    n_active = sum(1 for e in wp[:7] if True)  # CLS + 5 content + SEP
    # 活动位宽按"span * 0.45 / 7"分配；剩余宽度给 padding 段
    active_total = span * 0.40
    pad_total = span - active_total - 0.15
    active_w = active_total / 7
    pad_cell_w = pad_total / 43

    cur_x = X_LEFT
    # 7 个活动位
    for i in range(7):
        e = wp[i]
        x0 = cur_x
        xc = x0 + active_w / 2
        if i == 0:
            face = C["special"]
            edge = "#6B7280"
            label = "CLS"
        elif e.get("token") == "[SEP]":
            face = C["special"]
            edge = "#6B7280"
            label = "SEP"
        else:
            wid = e.get("word_id")
            face = WORD_FILL[wid % len(WORD_FILL)]
            edge = WORD_EDGE[wid % len(WORD_EDGE)]
            label = str(wid + 1)
        ax.add_patch(Rectangle(
            (x0 + 0.01, y_grid - grid_h / 2), active_w - 0.02, grid_h,
            facecolor=face, edgecolor=edge, linewidth=0.5))
        ax.text(xc, y_grid, label, ha="center", va="center",
                fontsize=6.4, color=C["ink"], fontweight="bold")
        cur_x += active_w

    # 留 0.15u 间隔
    cur_x += 0.15

    # 43 个填充位（连续显示为一条淡色带，不画线）
    ax.add_patch(Rectangle(
        (cur_x, y_grid - grid_h / 2), pad_total, grid_h,
        facecolor=C["pad"], edgecolor="#D1D5DB", linewidth=0.5))
    # 标注 ×43
    ax.text(cur_x + pad_total / 2, y_grid, r"$\times 43$",
            ha="center", va="center", fontsize=7.0, color=C["aux"])
    cur_x += pad_total

    # 活动位下方：1..K 编号（占位说明，单元格本身已显示数字）
    # 此处不再添加文字以避免与下方网格/连接线重叠

    # Layer 3b 左侧标签
    ax.text(X_LEFT - 0.30, y_grid + 0.05, "④ 50 位网格",
            ha="right", va="center", fontsize=7.4, color=C["ink"], fontweight="bold")
    ax.text(X_LEFT - 0.30, y_grid - 0.22, "(统一时序网格)",
            ha="right", va="center", fontsize=6.4, color=C["aux"])

    # 活动位上方的词归属短线（连向词区间条）
    for i in range(7):
        e = wp[i]
        if e.get("word_id") is None:
            continue
        xc = X_LEFT + (i + 0.5) * active_w
        # 虚线从条带底部到网格顶部
        ax.plot([xc, xc], [y_bar - bar_h / 2 - 0.04, y_grid + grid_h / 2 + 0.04],
                color=WORD_EDGE[e["word_id"] % len(WORD_EDGE)],
                linewidth=0.5, linestyle=(0, (2, 2)), alpha=0.6)

    # ========== 帧时间轴 ==========
    ax.plot([X_LEFT, X_RIGHT], [Y_AXIS, Y_AXIS], color=C["rule"], linewidth=0.7)
    ax.add_patch(FancyArrowPatch(
        (X_RIGHT - 0.02, Y_AXIS), (X_RIGHT + 0.18, Y_AXIS),
        arrowstyle="-|>", mutation_scale=8, color=C["rule"], linewidth=0.7))
    tick_frames = pick_axis_ticks(words, total_frames)
    for tf in tick_frames:
        xt = x_of(tf)
        ax.plot([xt, xt], [Y_AXIS - 0.04, Y_AXIS + 0.04], color=C["rule"], linewidth=0.7)
        ax.text(xt, Y_AXIS - 0.18, f"{tf}", ha="center", va="top",
                fontsize=6.0, color=C["ink"])
    ax.text(X_RIGHT + 0.32, Y_AXIS, f"≈ {duration:.2f} s",
            ha="left", va="center", fontsize=6.4, color=C["aux"])
    ax.text(X_LEFT - 0.30, Y_AXIS, "帧时间戳",
            ha="right", va="center", fontsize=6.4, color=C["aux"])

    # 帧移提示（轴线下方一行）
    ax.text(X_LEFT + span / 2, Y_AXIS - 0.42,
            r"帧移 $\Delta = 20$ ms（16 kHz @ 320 采样；C2 时长下界）",
            ha="center", va="top", fontsize=6.2, color=C["aux"])

    # ========== 底部脚注（与帧移分行，最末行） ==========
    ax.text(X_LEFT + span / 2, Y_AXIS - 0.78,
            "维特比精确求解；字符帧段按 ``|'' 切分归词；"
            "词内全部字符帧段的发射后验均值即词置信度（截断于 [0, 1]）",
            ha="center", va="top", fontsize=6.0, color=C["aux"])

    out = FIG_OUT / "q1_ctc_alignment.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"Generated: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())