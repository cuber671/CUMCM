#!/usr/bin/env python
"""P3 正文图表：3 图 + 4 表（数据全部来自冻结 json/csv，只读不重跑）。

Usage:
  .venv/bin/python scripts/figures/plot_q3_results.py

产物：
  paper/latex/figures/q3/q3_del_ins.{pdf,png}        P3-F1 删除/插入保真度曲线（20 条均值）
  paper/latex/figures/q3/q3_gphi.{pdf,png}           P3-F2 门控–贡献散点（valid 728，三模态）
  paper/latex/figures/q3/q3_att4.{pdf,png}           P3-F3 附件4 带符号 φ 堆叠条（20 条）
  paper/latex/tables/p3_gphi.tex                     P3-T3 g–φ 相关表
  （p3_completeness/fidelity/ig_validation/shapley_direction 由并行会话维护，本脚本不覆写）
  paper/latex/tables/p3_att4.tex                     P3-T4 附件4 全量汇总（20 行）
口径纪律：只读 runs/p3/{m2,m3,m4,m5,m6} 冻结产物；F 检查按预注册结果原样呈现（未过）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FIG_OUT = ROOT / "paper/latex/figures/q3"
TAB_OUT = ROOT / "paper/latex/tables"
RUNS = ROOT / "runs/p3"

M2 = json.loads((RUNS / "m2/shapley_att4.json").read_text())
M3 = json.loads((RUNS / "m3/ig_att4.json").read_text())
M4 = json.loads((RUNS / "m4/fidelity_att4.json").read_text())
DIAG = json.loads((RUNS / "m4/loo_diagnosis.json").read_text())
M5 = json.loads((RUNS / "m5/gphi_valid.json").read_text())
ATT4 = pd.read_csv(ROOT / "runs/p3/m6/附件4_预测与模态贡献.csv", encoding="utf-8-sig")

MODS = ("text", "audio", "vision")
MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}
CLS_CN = {0: "负", 1: "中", 2: "正"}
from figstyle import FONT_SIZE, MOD_COLOR as COLOR, configure


def configure_style() -> None:
    configure()


def fig_del_ins() -> None:
    fracs = sorted({r["frac"] for r in M4["per_sample"][0]["deletion"]})
    curves = {c: {k: ([], [], []) for k in ("attr", "rand", "std")}
              for c in ("deletion", "insertion")}
    for f in fracs:
        for c in ("deletion", "insertion"):
            key = "deletion" if c == "deletion" else "insertion"
            a = np.array([next(r["p_attr"] for r in s[key] if r["frac"] == f)
                          for s in M4["per_sample"]])
            m = np.array([next(r["p_rand_mean"] for r in s[key] if r["frac"] == f)
                          for s in M4["per_sample"]])
            sd = np.array([next(r["p_rand_std"] for r in s[key] if r["frac"] == f)
                           for s in M4["per_sample"]])
            curves[c]["attr"][0].append(a.mean()); curves[c]["attr"][1].append(a.std())
            curves[c]["attr"][2].append(0)
            curves[c]["rand"][0].append(m.mean()); curves[c]["rand"][1].append(m.std())
            curves[c]["rand"][2].append(sd.mean())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)
    titles = {"deletion": "(a) 删除曲线：归因序 vs 随机序",
              "insertion": "(b) 插入曲线：归因序 vs 随机序"}
    for ax, c in zip(axes, ("deletion", "insertion")):
        for k, ls, col, lab in (("attr", "-", "#D55E00", "按 |IG| 归因序"),
                                ("rand", "--", "#0072B2", "随机序（5 组均值）")):
            mu = np.array(curves[c][k][0]); sd = np.array(curves[c][k][1])
            ax.plot(fracs, mu, ls, color=col, lw=1.4, label=lab)
            ax.fill_between(fracs, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.set_xlabel("删除/插入的证据比例 $f$")
        ax.set_title(titles[c], fontsize=FONT_SIZE["TITLE"])
    axes[0].set_ylabel("预测类概率 $p$（20 条均值）")
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"q3_del_ins.{ext}", dpi=300 if ext == "png" else "figure")
    plt.close(fig)


def fig_gphi() -> None:
    df = pd.read_csv(RUNS / "m5/scatter.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), constrained_layout=True)
    for ax, m in zip(axes, MODS):
        d = df[df["modality"] == m]
        ax.scatter(d["gate"], d["abs_phi_cls"], s=2.5, alpha=0.30, lw=0,
                   color=COLOR[m], rasterized=True)
        rho = M5["correlations"][m]["cls"]["spearman"]
        ax.set_title(f"{MOD_CN[m]}：Spearman $\\rho$={rho:.2f}", loc="left", fontsize=FONT_SIZE["TITLE"])
        ax.set_xlabel(f"门控 $g_{{{m}}}$")
    axes[0].set_ylabel("$|\\varphi_m^{\\,cls}|$（预测类）")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"q3_gphi.{ext}", dpi=300 if ext == "png" else "figure")
    plt.close(fig)


def fig_att4() -> None:
    d = ATT4.sort_values(["pred_polarity_id", "sample_id"], ascending=[False, True])
    ids = [f"{int(i):02d}" for i in d["sample_id"]]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    left = np.zeros(len(d))
    for m in MODS:
        v = d[f"phi_{m[0]}"].to_numpy(float)
        ax.barh(y, v, left=left, color=COLOR[m], label=MOD_CN[m],
                height=0.68, edgecolor="white", lw=0.3)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{i}({CLS_CN[int(c)]})" for i, c in
                        zip(ids, d["pred_polarity_id"])], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("带符号 $\\varphi_m$（预测类）")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"q3_att4.{ext}", dpi=300 if ext == "png" else "figure")
    plt.close(fig)


def tab_shapley() -> None:
    res = np.array([r["completeness"]["cls_residual"] for r in M2["reports"]])
    reg = np.array([r["completeness"]["reg_residual"] for r in M2["reports"]])
    rows = [
        r"联盟枚举完备性 & 20 条 $\times$ 恰好 8 联盟、键序一致 & 通过 \\",
        r"效率公理（分类，逐类） & $\max_i \|\sum_m \varphi_m^{cls} - (p(x)-p_\emptyset)\|_i$ & "
        f"{res.max():.1e} \\",
        r"效率公理（回归） & $\max_i |\sum_m \varphi_m^{r} - (\hat y-\hat y_\emptyset)|$ & "
        f"{reg.max():.1e} \\",
        r"哑玩家（\#13 视觉整模态缺失） & $\varphi_v^{cls}\equiv 0$、$\varphi_v^{r}=0$ & 实证 \\",
        r"锚点自检（附件2 重合 5 条） & 预测 vs 真实标签一致数 & 5/5 \\",
    ]
    (TAB_OUT / "p3_shapley.tex").write_text(
        "\\begin{table}[htbp]\n  \\centering\n"
        "  \\caption{模态级精确 Shapley 的公理校验（附件4 全 20 条，3-seed 集成）}\n"
        "  \\label{tab:p3-shapley}\n"
        "  \\begin{tabular}{llr}\n    \\toprule\n    校验项 & 口径 & 结果 \\\\\n"
        "    \\midrule\n    " + "\n    ".join(rows) +
        "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n", encoding="utf-8")


def tab_fidelity() -> None:
    a = next(c["detail"] for c in M4["checks"] if c["name"] == "A_deletion_superiority")
    b = next(c["detail"] for c in M4["checks"] if c["name"] == "B_insertion_superiority")
    cD = next(c["detail"] for c in M4["checks"] if c["name"] == "C_compr_suff")
    d = M4["randomization"]["median_rho"]
    e = next(c["detail"] for c in M4["checks"] if c["name"] == "E_stability_text")
    fck = next(c["detail"] for c in M4["checks"] if c["name"] == "F_ig_loo_text")
    dg = DIAG["summary_mean"]
    rows = [
        f"A 删除优越性 & AOPC 均值 {a['mean_aopc']:.3f}，逐样本胜 & {a['wins']}/20 通过 \\\\",
        f"B 插入优越性 & $\\Delta$AUC 均值 {b['mean_dauc']:.3f}，逐样本胜 & {b['wins']}/20 通过 \\\\",
        f"C Compr/Suff@0.2 & {cD['mean_compr_at_20']:.3f} / {cD['mean_abs_suff_at_20']:.3f} & 通过 \\\\",
        f"D 权重随机化坍缩 & 中位 $\\rho$（t/a/v）= {d['text']}/{d['audio']}/{d['vision']} & 通过 \\\\",
        f"E 扰动稳定性（text） & $\\bar\\rho$={e['text_mean']:.2f}（a/v {e['av_report']['audio']}/"
        f"{e['av_report']['vision']}） & 通过 \\\\",
        f"F IG--LOO（text） & $\\bar\\rho$={fck['text_mean']:.2f} & 未通过 \\\\",
        f"\\quad F 诊断 & 局部干预 a/v $\\rho_{{|\\Delta p|}}$={dg['audio']['rho_abs']:.2f}/"
        f"{dg['vision']['rho_abs']:.2f}；text {dg['text']['rho_abs']:.2f} & 结构性差异 \\\\",
    ]
    (TAB_OUT / "p3_fidelity.tex").write_text(
        "\\begin{table}[htbp]\n  \\centering\n"
        "  \\caption{解释保真度与稳定性验证（预注册门槛，附件4 全 20 条；"
        "随机对照 5 组种子化）}\n  \\label{tab:p3-fidelity}\n"
        "  \\begin{tabular}{lll}\n    \\toprule\n    检验 & 指标 & 判定 \\\\\n"
        "    \\midrule\n    " + "\n    ".join(rows) +
        "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n", encoding="utf-8")


def tab_gphi() -> None:
    rows = []
    for m in MODS:
        c = M5["correlations"][m]
        rows.append(f"{MOD_CN[m]} & {c['cls']['pearson']:.3f} & {c['cls']['spearman']:.3f} & "
                    f"{c['reg']['pearson']:.3f} & {c['reg']['spearman']:.3f} \\\\")
    (TAB_OUT / "p3_gphi.tex").write_text(
        "\\begin{table}[htbp]\n  \\centering\n"
        "  \\caption{门控--贡献一致性（valid 全量 $n=728$；argmax 一致率 "
        f"{M5['argmax_agreement']:.3f}）}}\n  \\label{{tab:p3-gphi}}\n"
        "  \\begin{tabular}{lcccc}\n    \\toprule\n"
        "    模态 & cls Pearson & cls Spearman & reg Pearson & reg Spearman \\\\\n"
        "    \\midrule\n    " + "\n    ".join(rows) +
        "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n", encoding="utf-8")


def tab_att4() -> None:
    rows = []
    for _, r in ATT4.iterrows():
        rows.append(
            f"\\#{int(r['sample_id']):02d} & {r['pred_polarity']} & "
            f"{r['confidence']:.2f} & {r['pred_intensity']:+.2f} & "
            f"{r['main_modality']} & {r['phi_t']:+.3f} & {r['phi_a']:+.3f} & "
            f"{r['phi_v']:+.3f} \\\\")
    (TAB_OUT / "p3_att4.tex").write_text(
        "\\begin{table}[htbp]\n  \\centering\n"
        "  \\caption{附件4 全量解释汇总（极性/校准置信度/强度/主要参考模态/"
        "带符号 $\\varphi$；\\#13 视觉整模态自然缺失故 $\\varphi_v=0$）}\n"
        "  \\label{tab:p3-att4}\n  \\small\n"
        "  \\begin{tabular}{llcccccc}\n    \\toprule\n"
        "    样本 & 极性 & 置信度 & 强度 & 主模态 & $\\varphi_t$ & $\\varphi_a$ & "
        "$\\varphi_v$ \\\\\n    \\midrule\n    " + "\n    ".join(rows) +
        "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n", encoding="utf-8")


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    fig_del_ins(); fig_gphi(); fig_att4()
    tab_gphi(); tab_att4()
    print(f"figures → {FIG_OUT}\ntables  → {TAB_OUT}")


if __name__ == "__main__":
    main()
