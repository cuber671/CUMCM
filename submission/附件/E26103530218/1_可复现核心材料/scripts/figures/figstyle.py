#!/usr/bin/env python
"""全局图表样式单源配置（数据图共用；结构图填色族不入此文件）。

语义纪律：
1. 四条独立语义轴：模态色 / 极性色 / 模型色 / 状态色，跨轴可复用同色值；
2. 同一张图内，一个颜色不得同时承载两种语义；
3. 不纳入全局统一、保持图族独立：
   - 结构图填色族（plot_diagrams.py / plot_g1_framework.py 框线图）；
   - MODEL_COLOR 仅图6（31 场景退化曲线）与相关表使用；
   - 图3 专用：二值掩码色 / quality 轨 viridis（plot_q1.py 内）；
   - 图14 专用：三带分区色 C_Z0/Z1/Z2（plot_q2_perf.py 内）。

Usage（各绘图脚本）：
    from figstyle import MOD_COLOR, configure
    configure()
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ========== 全局色板 ==========
# 模态色（全图唯一语义：text=蓝 / audio=绿 / vision=朱红）
MOD_COLOR = {"text": "#0072B2", "audio": "#009E73", "vision": "#D55E00"}

# 极性色（与模态色跨轴复用，同图禁双语义）
POLARITY = {"负": "#0072B2", "中": "#999999", "正": "#D55E00"}

# 对齐质量状态色
STATUS = {"ok": "#009E73", "review": "#E69F00", "rollback": "#999999"}

# 模型对比色（仅图6 曲线族）
MODEL_COLOR = {"B0": "#999999", "B1": "#CC79A7", "B2": "#009E73",
               "B3": "#0072B2", "MRFN": "#D55E00"}

HEAT_CMAP = "Blues"         # 性能类矩阵热力图（图6b、图14a）
CTRL_STYLE = dict(ls="--")  # 对照组（随机序/均值线）一律虚线
ANNOT_GRAY = "#374151"      # 旁注与参考线灰
NOTE_GRAY = "#6B7280"       # 脚注灰

# ========== 字号层级（按现状锁死，不调整数值） ==========
FONT_SIZE = {"SUPTITLE": 9.5, "TITLE": 8.5, "LABEL": 8.0, "TICK": 7.0, "NOTE": 6.2}


def configure() -> None:
    """统一 rcParams：字体注册、嵌入合规（fonttype 42）、公式字体 cm、导出参数。"""
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    cjk = Path.home() / ".fonts/NotoSansSC-Regular.otf"
    if cjk.is_file():
        font_manager.fontManager.addfont(str(cjk))
        family = font_manager.FontProperties(fname=str(cjk)).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9.0, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "grid.linewidth": 0.45, "grid.alpha": 0.35,
        "axes.unicode_minus": False,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    })
