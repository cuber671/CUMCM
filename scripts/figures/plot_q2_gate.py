#!/usr/bin/env python
"""图：门控响应曲线（5.4，H5）——g_m 分箱均值 vs 缺失率（MRFN，valid，3-seed 均值）。

Usage: .venv/bin/python scripts/figures/plot_q2_gate.py
数据：runs/p2/gate_analysis/gate_curve_results.json（export_p2_gate_curve.py 产物）
产物：paper/latex/figures/q2/q2_gate_curve.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
J = json.loads((ROOT / "runs/p2/gate_analysis/gate_curve_results.json").read_text())
OUT = ROOT / "paper/latex/figures/q2"

from figstyle import MOD_COLOR as COLOR, configure
GATE_MARK = {"text": "o", "audio": "s", "vision": "^"}  # 灰度打印：形状区分（语音/视觉灰度差仅 0.04）

configure()

MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}
COLOR = {"text": "#D55E00", "audio": "#0072B2", "vision": "#009E73"}

fig, ax = plt.subplots(figsize=(3.6, 2.5))
for m in ("text", "audio", "vision"):
    c = J["mean_curve"][m]
    r = np.array(c["rates"])
    g = np.array(c["g_self_mean_3seed"])
    ax.plot(r, g, f"{GATE_MARK[m]}-", ms=3.5, lw=1.2, color=COLOR[m], label=MOD_CN[m])
    ax.annotate(MOD_CN[m], xy=(r[-1], g[-1]), xytext=(4, 0), textcoords="offset points",
                color=COLOR[m], fontsize=7.0, va="center")
ax.set_xlabel("缺失率 r（%）")
ax.set_ylabel("门控信任 $g_m$")
ax.set_xticks([10, 20, 40, 60, 80])
ax.set_xlim(5, 97)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "q2_gate_curve.pdf")
fig.savefig(OUT / "q2_gate_curve.png", dpi=300)
for m in ("text", "audio", "vision"):
    c = J["mean_curve"][m]["g_self_mean_3seed"]
    print(m, [round(x, 3) for x in c])
print("已输出", OUT / "q2_gate_curve.pdf")
