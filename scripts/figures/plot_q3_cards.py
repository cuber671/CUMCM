#!/usr/bin/env python
"""P3 图表（本会话版本）：F1/F2 合并图、F3 曲线、F4 验证图、F6 解释卡 + T1–T4。

Usage:
  .venv/bin/python scripts/figures/plot_q3_cards.py

与 scripts/figures/plot_q3_results.py（并行会话版本，产出 q3_del_ins/q3_gphi/
q3_att4 + p3_shapley/p3_gphi/p3_att4）共存：本脚本产物名不与其重叠，仅
p3_fidelity.tex 同名同义（两版内容等价，本版含 F 诊断行）。

口径纪律：
- 数字统一用第二轮 M3（commit 6fa6c56）：逐样本 40/40、max_rel 0.62%（64 步中点）；
- M4：A–E 通过，F text IG–LOO 0.1701 < 0.3 未通过并完成诊断；A 删除胜数 19/20；
- v(∅)（Shapley 联盟级，avail=0）与 IG 条件特征基线（avail=o）分开命名；
- 附件4 无时间戳：词级回溯用 P1 网格（20/20 对齐），[t_s,t_e) 回溯仅 P1 域内；
- 解释卡不做精度声明（附件4 无标签）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIG_OUT = ROOT / "paper/latex/figures/q3"
TAB_OUT = ROOT / "paper/latex/tables"
RUNS = ROOT / "runs/p3"
ATT4 = ROOT / "data/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"

M2 = json.loads((RUNS / "m2/shapley_att4.json").read_text())
M3 = json.loads((RUNS / "m3/ig_att4.json").read_text())
M4 = json.loads((RUNS / "m4/fidelity_att4.json").read_text())
LOO = json.loads((RUNS / "m4/loo_diagnosis.json").read_text())
CURVES = pd.read_csv(RUNS / "m4/curves.csv", encoding="utf-8-sig")

MODS = ("text", "audio", "vision")
CLS_NAME = {0: "负", 1: "中", 2: "正"}
MOD_COLOR = {"text": "#0072B2", "audio": "#E69F00", "vision": "#009E73"}
CARD_SAMPLES = ("09", "14", "13")  # 强文本主导 / 代表性 vision 主导 / vision 全缺哑玩家


def configure_style() -> None:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    cjk = Path.home() / ".fonts/NotoSansSC-Regular.otf"
    if cjk.is_file():
        font_manager.fontManager.addfont(str(cjk))
        family = font_manager.FontProperties(fname=str(cjk)).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9.0, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
        "mathtext.fontset": "cm",
    })


def phi_table() -> pd.DataFrame:
    rows = []
    for r in M2["reports"]:
        tc = r["pred_class"]
        rec = {"id": r["sample_id"], "pred": tc, "conf": r["confidence"],
               "intensity": r["pred_intensity"]}
        for m in MODS:
            rec[f"phi_{m}"] = r["phi_cls"][m][tc]
        rec["gates"] = tuple(r["gates_full"])
        rec["dom"] = max(MODS, key=lambda m: abs(rec[f"phi_{m}"]))
        rows.append(rec)
    return pd.DataFrame(rows)


def word_evidence(sample_id: str, topk: int = 8) -> list[tuple[str, float]]:
    """主基线（missing，avail=o 条件特征基线）IG_cls 位置归因 → 词级质量守恒求和。"""
    from src.p3.ig import p1_word_grid
    rep = next(r for r in M3["reports"] if r["sample_id"] == sample_id)
    ig_pos = np.asarray(rep["baselines"]["missing"]["ig"]["text"]["cls"], dtype=float)
    import pickle
    raw = pickle.load(open(ATT4 / f"{sample_id}.pkl", "rb"))["raw_text"]
    g = p1_word_grid(raw)
    wsum: dict[int, float] = {}
    for j, w in enumerate(g["word_by_position"]):
        if w is not None:
            wsum[w] = wsum.get(w, 0.0) + ig_pos[j]
    items = sorted(wsum.items(), key=lambda kv: -abs(kv[1]))[:topk]
    return [(g["word_texts"][w], v) for w, v in items]


def plot_shapley_evidence() -> None:
    df = phi_table().sort_values("phi_text").reset_index(drop=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.9, 4.3),
                                     gridspec_kw={"width_ratios": [1.25, 1.0]})
    y = np.arange(len(df))
    for m, dx in (("phi_vision", 1.4), ("phi_audio", 0.7), ("phi_text", 0.0)):
        ax_l.barh(y + dx, df[m].to_numpy(), height=0.62,
                  color=MOD_COLOR[m.split("_")[1]], label=m.split("_")[1])
    for i, row in df.iterrows():
        ax_l.plot(row[f"phi_{row['dom']}"],
                  y[i] + {"text": 0.0, "audio": 0.7, "vision": 1.4}[row["dom"]],
                  marker="D", ms=3.4, color="#1F2933", zorder=5)
    ax_l.set_yticks(y + 0.7)
    ax_l.set_yticklabels([f"#{i}" for i in df["id"]], fontsize=6.0)
    ax_l.axvline(0, color="#374151", lw=0.8)
    ax_l.set_xlabel(r"带符号 Shapley 贡献 $\varphi_m$（预测类，联盟级 $v(\emptyset)$ 基准）")
    ax_l.set_title("20 条样本模态贡献与主导模态（◆）", loc="left", fontsize=8.4)
    ax_l.legend(frameon=False, fontsize=6.6, loc="lower right")
    dom_counts = df["dom"].value_counts()
    ax_l.text(0.02, 0.02, "主导：" + "，".join(f"{m} {dom_counts.get(m, 0)}" for m in MODS),
              transform=ax_l.transAxes, fontsize=6.2, color="#374151")

    words = word_evidence("09", topk=8)
    names = [w for w, _ in words][::-1]
    vals = [v for _, v in words][::-1]
    ax_r.barh(range(len(vals)), vals, height=0.62,
              color=[MOD_COLOR["text"] if v >= 0 else "#CC79A7" for v in vals])
    ax_r.set_yticks(range(len(vals)))
    ax_r.set_yticklabels(names, fontsize=7.2)
    ax_r.axvline(0, color="#374151", lw=0.8)
    ax_r.set_xlabel("词级 IG 贡献（条件特征基线，质量守恒求和）")
    ax_r.set_title("TOP-8 原词证据（#09，text 主导）", loc="left", fontsize=8.4)
    ax_r.text(0.5, -0.16, "附件4 无时间戳：词回溯用 P1 网格（20/20 对齐），\n"
                          "$[t_s,t_e)$ 时间回溯仅 P1 100 条域内可用",
              transform=ax_r.transAxes, ha="center", va="top", fontsize=6.2, color="#6B7280")
    fig.suptitle("模态贡献—证据回溯（MRFN+ 3-seed 集成，附件4，20 条）", y=0.995, fontsize=9.5)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIG_OUT / "q3_shapley_evidence.pdf")
    fig.savefig(FIG_OUT / "q3_shapley_evidence.png", dpi=300)
    plt.close(fig)


def plot_fidelity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), sharey=True)
    for ax, curve, title in zip(axes, ("deletion", "insertion"),
                                ("删除实验（按归因删除，越降越有效）",
                                 "插入实验（按归因插入，越升越有效）")):
        d = CURVES[CURVES.curve == curve]
        agg = d.groupby("frac").agg(pa=("p_attr", "mean"), pr=("p_rand_mean", "mean"),
                                    ps=("p_rand_std", "mean")).reset_index()
        ax.plot(agg.frac, agg.pa, marker="o", ms=3.2, lw=1.4, color="#D55E00", label="归因排序")
        ax.plot(agg.frac, agg.pr, marker="s", ms=2.8, lw=1.2, ls="--", color="#0072B2",
                label="随机排序（均值）")
        ax.fill_between(agg.frac, agg.pr - agg.ps, agg.pr + agg.ps,
                        color="#0072B2", alpha=0.15, linewidth=0)
        ax.set_xlabel("删除/插入比例")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_title(title, loc="left", fontsize=8.4)
        ax.legend(frameon=False, fontsize=6.6,
                  loc="upper left" if curve == "deletion" else "lower right")
    axes[0].set_ylabel("目标类概率（20 条均值）")
    ck = {c["name"]: c for c in M4["checks"]}
    a, b = ck["A_deletion_superiority"]["detail"], ck["B_insertion_superiority"]["detail"]
    axes[0].text(0.035, 0.03, f"AOPC={a['mean_aopc']:.4f}\n胜 {a['wins']}/{a['n']}",
                 transform=axes[0].transAxes, ha="left", va="bottom", fontsize=6.4,
                 color="#374151", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    axes[1].text(0.97, 0.90, f"ΔAUC={b['mean_dauc']:.4f}\n胜 {b['wins']}/20",
                 transform=axes[1].transAxes, ha="right", fontsize=6.4, color="#374151")
    fig.suptitle("删除/插入保真度（M3 主基线 IG_cls；A–E 通过，F 见验证表）", y=1.02, fontsize=9.5)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIG_OUT / "q3_fidelity.pdf")
    fig.savefig(FIG_OUT / "q3_fidelity.png", dpi=300)
    plt.close(fig)


def plot_ig_validation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
    ck3 = {c["name"]: c for c in M3["checks"]}
    ax = axes[0]
    sens = ck3["baseline_sensitivity"]["detail"]["spearman_mean"]
    pairs = [("mz", "missing↔zero"), ("mm", "missing↔mean"), ("zm", "zero↔mean")]
    xpos, vals, cols = [], [], []
    for i, (k, _) in enumerate(pairs):
        for j, m in enumerate(MODS):
            xpos.append(i + (j - 1) * 0.26)
            vals.append(sens[k][m])
            cols.append(MOD_COLOR[m])
    ax.bar(xpos, vals, width=0.24, color=cols)
    for x, v in zip(xpos, vals):
        ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=5.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab for _, lab in pairs], fontsize=7.2)
    ax.set_ylim(0.90, 1.045)
    ax.set_ylabel("排序 Spearman ρ（20 条均值）")
    ax.set_title("基线敏感性：三基线排序高度一致", loc="left", fontsize=8.4)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MOD_COLOR[m], ec="none") for m in MODS]
    ax.legend(handles, list(MODS), frameon=False, fontsize=6.2, ncol=3,
              loc="upper left", handlelength=1.1, columnspacing=0.8)
    ax = axes[1]
    sm = LOO["summary_mean"]
    vals = [sm[m]["rho_abs"] for m in MODS]
    bars = ax.bar(list(MODS), vals, width=0.55,
                  color=[MOD_COLOR[m] for m in MODS])
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.axhline(0.3, color="#C0392B", lw=1.0, ls="--")
    ax.text(0.99, 0.315, "F 门槛 ρ≥0.3", ha="left", fontsize=6.4, color="#C0392B",
            transform=ax.get_yaxis_transform())
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("IG–LOO 一致性 |ρ|（20 条均值）")
    ax.set_title("F 诊断：a/v 局部干预强一致，text 未达门槛", loc="left", fontsize=8.4)
    fig.suptitle("IG 验证：64 步中点积分 40/40（max 0.62%），结构位泄漏 0", y=1.02, fontsize=9.5)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIG_OUT / "q3_ig_validation.pdf")
    fig.savefig(FIG_OUT / "q3_ig_validation.png", dpi=300)
    plt.close(fig)


def plot_explain_cards() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(6.9, 6.6),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})
    reps = {r["sample_id"]: r for r in M2["reports"]}
    for row, sid in enumerate(CARD_SAMPLES):
        r = reps[sid]
        tc = r["pred_class"]
        ax = axes[row, 0]
        ax.axis("off")
        g = r["gates_full"]
        lines = [f"样本 #{sid}",
                 f"预测：{CLS_NAME[tc]}（强度 {r['pred_intensity']:+.2f}，置信 {r['confidence']:.2f}）",
                 ""]
        for i, m in enumerate(MODS):
            lines.append(f"$\\varphi_{{{m}}}$ = {r['phi_cls'][m][tc]:+.3f}   （gate {g[i]:.2f}）")
        lines += ["", r"$\Sigma\varphi = v(N)-v(\emptyset)$，残差 $\leq 10^{-5}$"]
        if sid == "13":
            lines.append("vision 全模态自然缺失：哑玩家 $\\varphi_v \\equiv 0$")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=7.6,
                linespacing=1.55, transform=ax.transAxes)
        ax.add_patch(plt.Rectangle((0.0, 0.02), 1.0, 0.96, transform=ax.transAxes,
                                   fill=False, edgecolor="#9AA5B1", lw=0.8))
        ax = axes[row, 1]
        words = word_evidence(sid, topk=5)
        names = [w for w, _ in words][::-1]
        vals = [v for _, v in words][::-1]
        ax.barh(range(len(vals)), vals, height=0.6,
                color=[MOD_COLOR["text"] if v >= 0 else "#CC79A7" for v in vals])
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=7.4)
        ax.axvline(0, color="#374151", lw=0.8)
        if row == 0:
            ax.set_title("TOP-5 原词证据（IG，条件特征基线）", loc="left", fontsize=7.8)
        ax.set_xlim(min(min(vals), 0) * 1.25 - 1e-3, max(max(vals), 0) * 1.25 + 1e-3)
        if row == 2:
            ax.set_xlabel("词级 IG 贡献（P1 网格聚合，质量守恒求和）")
    fig.suptitle("解释卡（附件4，MRFN+ 集成；无标签 → 不做精度声明，仅展示预测/贡献/证据回溯）",
                 y=0.985, fontsize=9.0)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7, rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_OUT / "q3_explain_cards.pdf")
    fig.savefig(FIG_OUT / "q3_explain_cards.png", dpi=300)
    plt.close(fig)


def write_tables() -> None:
    TAB_OUT.mkdir(parents=True, exist_ok=True)
    df = phi_table()
    ck3 = {c["name"]: c for c in M3["checks"]}
    ck4 = {c["name"]: c for c in M4["checks"]}

    dom = df["dom"].value_counts()
    pos = {m: int((df[f"phi_{m}"] > 0).sum()) for m in MODS}
    rows = [f"    {m} & {pos[m]} & {20 - pos[m]} & {dom.get(m, 0)} \\\\" for m in MODS]
    (TAB_OUT / "p3_shapley_direction.tex").write_text("\n".join([
        "\\begin{table}[htbp]", "  \\centering",
        "\\caption{模态 Shapley 方向表（预测类上带符号 $\\varphi$；$v(\\emptyset)$ 联盟级基准）}",
        "  \\label{tab:p3-direction}", "  \\begin{tabular}{lccc}", "    \\toprule",
        "    模态 & $\\varphi{>}0$ & $\\varphi{<}0$ & 主导 \\\\", "    \\midrule", *rows,
        "    \\bottomrule", "  \\end{tabular}", "\\end{table}",
    ]), encoding="utf-8")

    cls_res = max(r["completeness"]["cls_residual"] for r in M2["reports"])
    reg_res = max(r["completeness"]["reg_residual"] for r in M2["reports"])
    n_pass = sum(r["completeness"]["pass"] for r in M2["reports"])
    (TAB_OUT / "p3_completeness.tex").write_text("\n".join([
        "\\begin{table}[htbp]", "  \\centering",
        "\\caption{Shapley 完备性（$\\sum_m \\varphi_m = v(N)-v(\\emptyset)$，逐样本）}",
        "  \\label{tab:p3-completeness}", "  \\begin{tabular}{lccc}", "    \\toprule",
        "    输出头 & 最大残差 & 门槛 & 通过 \\\\", "    \\midrule",
        f"    分类 & ${sci(cls_res)}$ & $1{{\\times}}10^{{-5}}$ & {n_pass}/20 \\\\",
        f"    强度 & ${sci(reg_res)}$ & $1{{\\times}}10^{{-5}}$ & {n_pass}/20 \\\\",
        "    \\bottomrule", "  \\end{tabular}", "\\end{table}",
    ]), encoding="utf-8")

    integ = ck3["integration_error_primary_per_sample"]["detail"]
    sens = ck3["baseline_sensitivity"]["detail"]["spearman_mean"]
    sens_min = min(v for k in sens for v in sens[k].values())
    e_det = ck4["E_stability_text"]["detail"]
    rows = [
        f"    积分误差（64 步中点） & {integ['n_pass']}/{integ['n_checks']} & max rel {integ['max_rel']*100:.2f}\\% & 通过 \\\\",
        f"    阻断位泄漏 & max $|IG|$={sci(ck3['zero_ig_outside_evidence']['detail']['max_abs_ig_on_blocked_positions'])}（逐位精确） & 3基线$\\times$2输出$\\times$3模态 & 通过 \\\\",
        f"    \\#13 哑玩家 & $\\max|IG_v|$={sci(ck3['ig_zero_vision_13']['detail']['max_abs_ig_vision'])}（逐位精确） & $\\varphi_v\\equiv 0$ & 通过 \\\\",
        f"    问题一网格映射 & {ck3['p1_grid_alignment']['detail']['aligned']}/20 & 截断词 {ck3['p1_grid_alignment']['detail']['uncovered_words_total']} 个未覆盖 & 通过 \\\\",
        f"    基线排序敏感性 & $\\bar\\rho_{{min}}$={sens_min:.3f} & $\\ge 0.96$ & 通过 \\\\",
        f"    稳定性 t/a/v & {e_det['text_mean']:.3f}/{e_det['av_report']['audio']:.3f}/{e_det['av_report']['vision']:.3f} & text $\\ge 0.6$ & 通过 \\\\",
    ]
    (TAB_OUT / "p3_ig_validation.tex").write_text("\n".join([
        "\\begin{table}[htbp]", "  \\centering\\small",
        "\\caption{IG 验证（条件特征基线，avail 固定 $o$；与 $v(\\emptyset)$ 严格区分）}",
        "  \\label{tab:p3-ig-validation}",
        "  \\begin{tabular}{p{2.9cm}p{3.3cm}p{3.6cm}l}", "    \\toprule",
        "    验证项 & 实测 & 门槛/口径 & 判定 \\\\", "    \\midrule", *rows,
        "    \\bottomrule", "  \\end{tabular}", "\\end{table}",
    ]), encoding="utf-8")

    sm = LOO["summary_mean"]
    rows = []
    for name, cn in (("A_deletion_superiority", "A 删除"), ("B_insertion_superiority", "B 插入")):
        d = ck4[name]["detail"]
        key = "mean_aopc" if name.startswith("A") else "mean_dauc"
        rows.append(f"    {cn} & {d[key]:.3f}（胜 {d['wins']}/{d.get('n', 20)}） & 均值$>$0 且 $\\ge$15/20 & 通过 \\\\")
    c = ck4["C_compr_suff"]["detail"]
    rows.append(f"    C 全面/充分 & compr={c['mean_compr_at_20']:.3f}，|suff|={c['mean_abs_suff_at_20']:.3f} & compr$>$0，$|$suff$|<$compr & 通过 \\\\")
    dmax = max(abs(v) for v in M4["randomization"]["median_rho"].values())
    rows.append(f"    D 随机化坍缩 & median $\\rho$ max {dmax:.3f} & $\\le 0.3$ & 通过 \\\\")
    rows.append(f"    E 稳定性（text） & {ck4['E_stability_text']['detail']['text_mean']:.3f} & $\\ge 0.6$ & 通过 \\\\")
    rows.append(f"    F text IG--LOO & {ck4['F_ig_loo_text']['detail']['text_mean']:.3f} & $\\ge 0.3$ & \\textbf{{未通过}} \\\\")
    note = (f"注：F 项结构性诊断——a/v 局部干预 $|\\rho|$={sm['audio']['rho_abs']:.2f}/"
            f"{sm['vision']['rho_abs']:.2f} 强一致；text [MASK] 重编码为非局部干预，与嵌入层 IG 属不同"
            f"反事实，故 IG--LOO 一致性低（详见正文 6.4 节）。")
    (TAB_OUT / "p3_fidelity.tex").write_text("\n".join([
        "\\begin{table}[htbp]", "  \\centering\\small",
        "\\caption{解释保真度六项检验结果（判定与预注册门槛）}",
        "  \\label{tab:p3-fidelity}",
        "  \\begin{tabular}{p{1.8cm}p{4.3cm}p{3.2cm}l}", "    \\toprule",
        "    门槛 & 实测 & 预注册规则 & 判定 \\\\", "    \\midrule", *rows,
        "    \\bottomrule", "  \\end{tabular}",
        "  \\par \\smallskip", "  {\\small " + note + "}", "\\end{table}",
    ]), encoding="utf-8")
    print("表: p3_shapley_direction / p3_completeness / p3_ig_validation / p3_fidelity →", TAB_OUT)



def sci(x: float) -> str:
    """科学计数 LaTeX 化；精确 0 返回 '0'。"""
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def main() -> int:
    configure_style()
    plot_shapley_evidence()
    plot_fidelity()
    plot_ig_validation()
    plot_explain_cards()
    write_tables()
    ck4 = {c["name"]: c for c in M4["checks"]}
    a = ck4["A_deletion_superiority"]["detail"]
    print("caption 素材：")
    print(f"  A AOPC={a['mean_aopc']:.4f} 胜 {a['wins']}/{a['n']}；F text IG–LOO 未通过（诊断闭环）")
    print(f"  解释卡样本：{'/'.join(CARD_SAMPLES)}（强 text 主导 / 代表性 vision 主导 / vision 全缺哑玩家）")
    print("  F1/F2 图注要点：左图按 phi_text 升序排列（vision 主导为 #14/#19，18+0+2=20 对账）；")
    print("  TOP-8 中 not 出现两次 = word_id 3/11 两个不同词位实例（非去重错误）；")
    print("  功能词贡献高属条件特征基线下的正常现象，正文预说明。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
