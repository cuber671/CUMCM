#!/usr/bin/env python
"""图：MRFN+ 双输出性能与误差结构（附件2 valid 728 条，3-seed）。

Usage:
  .venv/bin/python scripts/figures/plot_q2_perf.py

产物：
  paper/latex/figures/q2/q2_perf_analysis.{pdf,png}

口径（2026-09-25 拍板）：
- Panel A 混淆矩阵：三种子概率集成 argmax（逐 seed softmax 后平均），行归一化，格注 n+%;
- Panel B 强度散点：三种子回归均值 vs 真实，[-3,3]，y=x、零线、分箱均值线，图内注 MAE/Pearson;
- Panel C 分区条形：三种子分区准确率均值（冻结值，直读 error_attribution_mrfn_plus.json）。
自检：npz 按同协议重算的分区均值须逐位复现 json 冻结值（<1e-6），否则拒绝出图。
不访问 test；不改动任何模型/阈值/交付物。
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
OUT = ROOT / "paper/latex/figures/q2"
EA = ROOT / "runs/p2/error_attribution"

# ── 冻结数据 ──
z = np.load(EA / "per_seed_predictions_mrfn_plus.npz")
y_cls, rl = z["y_cls"].astype(int), z["rl"].astype(float)
SEEDS = [1, 2, 3]
logits = np.stack([z[f"seed{i}_logits"] for i in SEEDS])       # (3,728,3)
regs = np.stack([z[f"seed{i}_reg"] for i in SEEDS])            # (3,728)
j = json.loads((EA / "error_attribution_mrfn_plus.json").read_text())
ZONES = j["B_model"]["zones"]

def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

# 类序：0=负 1=中 2=正（A_label_structure oracle 规则）
assert (y_cls == 1).sum() == ZONES["Z0_neutral"]["n"] == 184, "类序或样本数不符"

# ── 集成（A/B 口径）──
p_ens = softmax(logits).mean(0)          # 三种子概率均值
pred = p_ens.argmax(1)
reg = regs.mean(0)                        # 三种子回归均值
mae = float(np.abs(reg - rl).mean())
pear = float(np.corrcoef(rl, reg)[0, 1])
acc = float((pred == y_cls).mean())

# ── 自检：per-seed 分区均值 vs json 冻结值 ──
masks = {"Z0_neutral": rl == 0, "Z1_weak": (np.abs(rl) > 0) & (np.abs(rl) < 0.5),
         "Z2_strong": np.abs(rl) >= 0.5}
for zk, m in masks.items():
    per_seed = [float((logits[s].argmax(1) == y_cls)[m].mean()) for s in range(3)]
    rec = float(np.mean(per_seed))
    assert abs(rec - ZONES[zk]["acc_mean"]) < 1e-6, f"{zk}: 重算 {rec} ≠ 冻结 {ZONES[zk]['acc_mean']}"
print(f"自检通过；集成 acc={acc:.4f}  MAE={mae:.4f}  Pearson={pear:.4f}")

# ── 样式（figstyle 单源；三带分区色为本图专用语义轴）──
from figstyle import FONT_SIZE, HEAT_CMAP, configure

configure()
C_Z0, C_Z1, C_Z2 = "#999999", "#E69F00", "#0072B2"
CLS = ["负", "中", "正"]

fig = plt.figure(figsize=(7.0, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.62], wspace=0.30)

# Panel A：混淆矩阵（集成 argmax，行归一化）
axA = fig.add_subplot(gs[0])
cm = np.zeros((3, 3), int)
for t, p in zip(y_cls, pred):
    cm[t, p] += 1
im = axA.imshow(cm / cm.sum(1, keepdims=True), cmap=HEAT_CMAP, vmin=0, vmax=1)
for t in range(3):
    for p in range(3):
        axA.text(p, t, f"{cm[t, p]}\n{cm[t, p] / cm[t].sum():.1%}",
                 ha="center", va="center", fontsize=7.5,
                 color="white" if cm[t, p] / cm[t].sum() > 0.55 else "#222222")
axA.set_xticks(range(3), CLS); axA.set_yticks(range(3), CLS)
axA.set_xlabel("预测类"); axA.set_ylabel("真实类")
axA.set_title("(a) 混淆矩阵", loc="left", fontsize=FONT_SIZE["TITLE"])
axA.grid(False)

# Panel B：强度散点（seed 均值回归，三带着色 + 分箱均值线）
axB = fig.add_subplot(gs[1])
band = np.where(masks["Z2_strong"], C_Z2, np.where(masks["Z1_weak"], C_Z1, C_Z0))
axB.scatter(rl, reg, s=7, c=band, alpha=0.45, linewidths=0)
xs = np.linspace(-3, 3, 2)
axB.plot(xs, xs, "--", color="#555555", lw=0.9, label="y = x")
axB.axhline(0, color="#BBBBBB", lw=0.6); axB.axvline(0, color="#BBBBBB", lw=0.6)
edges = np.linspace(-3, 3, 13)
mids, means = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (rl >= lo) & (rl < hi) if hi < 3 else (rl >= lo) & (rl <= hi)
    if m.sum() >= 3:
        mids.append((lo + hi) / 2); means.append(reg[m].mean())
axB.plot(mids, means, "-", color="#D55E00", lw=1.4, label="分箱均值")
axB.set_xlim(-3, 3); axB.set_ylim(-3, 3)
axB.set_xlabel("真实强度 $y$"); axB.set_ylabel("预测强度 $\\hat y$")
axB.set_title("(b) 强度预测", loc="left", fontsize=FONT_SIZE["TITLE"])
axB.text(0.03, 0.965, f"MAE = {mae:.3f}\nPearson = {pear:.3f}", transform=axB.transAxes,
         va="top", ha="left", fontsize=7.5,
         bbox=dict(fc="white", ec="#CCCCCC", alpha=0.9, pad=2))
h1, l1 = axB.get_legend_handles_labels()
from matplotlib.lines import Line2D
pts = [Line2D([], [], marker="o", ls="", ms=4, color=c) for c in (C_Z0, C_Z1, C_Z2)]
axB.legend(h1 + pts, l1 + ["真中性", "弱极性", "强极性"], fontsize=6.6,
           loc="lower right", framealpha=0.9, handletextpad=0.4)

# Panel C：分区准确率（三 seed 均值，冻结值）
axC = fig.add_subplot(gs[2])
vals = [ZONES[k]["acc_mean"] for k in ("Z0_neutral", "Z1_weak", "Z2_strong")]
ns = [ZONES[k]["n"] for k in ("Z0_neutral", "Z1_weak", "Z2_strong")]
axC.bar(range(3), vals, color=[C_Z0, C_Z1, C_Z2], width=0.62)
for i, (v, n) in enumerate(zip(vals, ns)):
    axC.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5)
    axC.text(i, 0.03, f"n={n}", ha="center", fontsize=6.8, color="white")
axC.set_xticks(range(3), ["真中性", "弱极性", "强极性"], fontsize=7.2)
axC.set_ylim(0, 1.0); axC.set_ylabel("准确率")
axC.set_title("(c) 分区准确率", loc="left", fontsize=FONT_SIZE["TITLE"])

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "q2_perf_analysis.pdf")
fig.savefig(OUT / "q2_perf_analysis.png", dpi=300)
print("已输出", OUT / "q2_perf_analysis.pdf")
print("混淆矩阵（行=真实 负/中/正）：")
print(cm)
