#!/usr/bin/env python
"""P2 正文图表：2 图 + 4 表（数据全部来自冻结 json，只读不重跑）。

Usage:
  .venv/bin/python scripts/figures/plot_q2_results.py

产物：
  paper/latex/figures/q2/q2_degradation.{pdf,png}      P2-F2 缺失率—性能退化曲线
  paper/latex/figures/q2/q2_scenario_heatmap.{pdf,png}  P2-F3 31 场景鲁棒性热力图
  paper/latex/figures/q2/q2_fj3_behavior.{pdf,png}     P2-F5 附件3 行为分析图
  paper/latex/tables/p2_ladder.tex                      P2-T1 模型阶梯表（test 口径统一）
  paper/latex/tables/p2_transfer.tex                    P2-T2 2×2 迁移表（m5 变体口径）
  paper/latex/tables/p2_ablation.tex                    P2-T3 消融与填充基线表
  paper/latex/tables/p2_final_test.tex                  P2-T4 最终 test 表（含评估批次列）

口径纪律：
- 主图/主表只用 test_final_results.json（B0–MRFN，附件2 test 唯一一次评估）；
- MRFN+ 仅来自 test_second（预注册追加，clean 口径），只进 T4，不构造其鲁棒性曲线；
- 单模型均值与三 seed 集成不同口径，分列不混比。
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
FIG_OUT = ROOT / "paper/latex/figures/q2"
TAB_OUT = ROOT / "paper/latex/tables"
RUNS = ROOT / "runs/p2"

TF = json.loads((RUNS / "test_final/test_final_results.json").read_text())
M5 = json.loads((RUNS / "m5/m5_summary.json").read_text())
TS = json.loads((RUNS / "test_second/test_second_results.json").read_text())

MODELS = ["B0", "B1", "B2", "B3", "MRFN"]
MODEL_COLOR = {"B0": "#999999", "B1": "#CC79A7", "B2": "#009E73",
               "B3": "#0072B2", "MRFN": "#D55E00"}
RATES = [10, 20, 40, 60, 80]
MOD_CN = {"t": "text", "a": "audio", "v": "vision"}


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
    })


def scenario_grid_keys() -> list[str]:
    """31 场景固定排序：MCAR 单模态 → MCAR 双模态 → Block → Joint。"""
    keys = [f"mcar/{m}/rate{r}/random" for m in "tav" for r in RATES]
    keys += [f"mcar/{mm}/rate40/random" for mm in ("ta", "tv", "av")]
    keys += [f"block/{m}/rate40/{pos}" for m in "tav" for pos in ("head", "mid", "tail", "random")]
    keys += ["joint/av/rate40/random"]
    assert len(keys) == 31
    return keys


def short_label(key: str) -> str:
    fam, mods, rate, pos = key.split("/")
    if fam == "mcar":
        return f"{mods}{int(rate[4:])}"
    if fam == "block":
        return f"b-{mods[:1]}-{pos[0]}"
    return "joint-av"


def plot_degradation() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7), sharey=True)
    for ax, mod in zip(axes, "tav"):
        for model in MODELS:
            rec = TF["models"][model]
            xs = [0] + RATES
            ys = [rec["clean"]["S"]["mean"]] + [
                rec["masked_S_mean"][f"mcar/{mod}/rate{r}/random"] for r in RATES]
            ax.plot(xs, ys, marker="o", ms=3.0, lw=1.2,
                    color=MODEL_COLOR[model], label=model)
        ax.set_xticks([0] + RATES)
        ax.set_xticklabels(["0", "10", "20", "40", "60", "80"])
        ax.set_xlabel(f"{MOD_CN[mod]} 缺失率（%，0 = clean）")
        ax.set_title(f"{MOD_CN[mod]} 缺失", fontsize=8.6)
    axes[0].set_ylabel("综合指标 S（3-seed 均值）")
    axes[0].set_ylim(0.46, 0.80)
    axes[0].legend(frameon=False, fontsize=6.4, ncol=1, loc="lower left",
                   handlelength=1.4, labelspacing=0.3)
    fig.suptitle("缺失率—性能退化曲线（附件2 test，31 网格冻结协议）", y=1.03, fontsize=9.5)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIG_OUT / "q2_degradation.pdf")
    fig.savefig(FIG_OUT / "q2_degradation.png", dpi=300)
    plt.close(fig)


def plot_scenario_heatmap() -> None:
    keys = scenario_grid_keys()
    ds = np.array([[TF["models"][m]["masked_D_S_mean"][k] for k in keys] for m in MODELS])
    fig, ax = plt.subplots(figsize=(6.9, 2.3))
    im = ax.imshow(ds, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.22, ds.max()))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([short_label(k) for k in keys], rotation=90, fontsize=5.2)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=7.5)
    # 场景分组分隔线
    for x in (14.5, 17.5, 29.5):
        ax.axvline(x, color="white", lw=1.6)
    for txt, x in [("MCAR 单模态", 6.5), ("双模态", 16), ("Block（40%）", 23.5), ("Joint", 30)]:
        ax.text(x, -0.9, txt, ha="center", va="bottom", fontsize=6.2, color="#374151")
    ax.set_title("31 缺失场景下的性能退化 $D_{\\mathcal{S}}$（3-seed 均值；颜色越深退化越大）",
                 loc="left", fontsize=8.6, pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("$D_{\\mathcal{S}}$", fontsize=6.5)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIG_OUT / "q2_scenario_heatmap.pdf")
    fig.savefig(FIG_OUT / "q2_scenario_heatmap.png", dpi=300)
    plt.close(fig)


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def write_tables() -> None:
    TAB_OUT.mkdir(parents=True, exist_ok=True)

    # ---- P2-T1 阶梯表（test_final，统一 test 口径）----
    rows = []
    for m in MODELS:
        rec = TF["models"][m]
        c = rec["clean"]
        rows.append(f"    {m} & {rec['param_count']:,} & {fmt3(c['acc']['mean'])} & "
                    f"{fmt3(c['macro_f1']['mean'])} & {fmt3(c['mae']['mean'])} & "
                    f"{fmt3(c['pearson']['mean'])} & {fmt3(c['S']['mean'])} & "
                    f"{fmt3(rec['mean_D_S'])} \\\\")
    (TAB_OUT / "p2_ladder.tex").write_text("\n".join([
        "\\begin{table}[htbp]", "  \\centering",
        "  \\caption{模型阶梯评估结果（附件2 test；3 种子均值）}",
        "  \\label{tab:p2-ladder}",
        "  \\begin{tabular}{lrrrrrrr}", "    \\toprule",
        "    模型 & 参数量 & Acc & macro-F1 & MAE & Pearson & $\mathcal{S}$ & $\\overline{D_{\\mathcal{S}}}$ \\\\",
        "    \\midrule", *rows, "    \\bottomrule", "  \\end{tabular}",
        "  \\par \\smallskip",
        "  {\\small 注：Acc 与 macro-F1 为百分数；$\\mathcal{S}$ 为综合效用分，"
        "$\\overline{D_{\\mathcal{S}}}$ 为 31 场景等权平均退化（先场景内 8 实例均值、后 31 场景等权）。}",
        "\\end{table}",
    ]), encoding="utf-8")

    # ---- P2-T2 2×2 迁移表（m5 变体：MRFN_clean vs MRFN 主模型）----
    mc = M5["runs"]["MRFN_clean"]
    mf = TF["models"]["MRFN"]
    cell = lambda rec, kind: (fmt3(rec["clean"]["S"]["mean"]) if kind == "clean"
                              else fmt3(sum(rec["masked_S_mean"].values()) / 31))
    (TAB_OUT / "p2_transfer.tex").write_text("\n".join([
        "\\begin{table}[htbp]", "  \\centering",
        "  \\caption{2$\\times$2 迁移表（同架构只变训练方式，$\\mathcal{S}$）}",
        "  \\label{tab:p2-transfer}",
        "  \\begin{tabular}{lcc}", "    \\toprule",
        "    训练方式$\\backslash$测试 & clean test & 31 场景均值 \\\\", "    \\midrule",
        f"    clean 训练 & {cell(mc, 'clean')} & {cell(mc, 'masked')} \\\\",
        f"    缺失增强训练 & {cell(mf, 'clean')} & {cell(mf, 'masked')} \\\\",
        "    \\bottomrule", "  \\end{tabular}", "\\end{table}",
    ]), encoding="utf-8")

    # ---- P2-T3 消融与填充基线表（m5）----
    name_cn = {
        "MRFN_noState": "去除缺失状态编码",
        "MRFN_noMaskAttn": "去除掩码注意力",
        "MRFN_noGate": "去除门控",
        "MRFN_clean": "去除缺失增强训练",
        "B3_aug_zero": "B3 + zero 填充基线",
        "B3_aug_ff": "B3 + 前向填充基线",
    }
    rows = [f"    MRFN（完整） & {TF['models']['MRFN']['param_count']:,} & "
            f"{fmt3(TF['models']['MRFN']['clean']['S']['mean'])} & "
            f"{fmt3(sum(TF['models']['MRFN']['masked_S_mean'].values()) / 31)} & — \\\\"]
    order = ["MRFN_noState", "MRFN_noMaskAttn", "MRFN_noGate", "MRFN_clean",
             "B3_aug_zero", "B3_aug_ff"]
    base31 = sum(TF["models"]["MRFN"]["masked_S_mean"].values()) / 31
    for k in order:
        rec = M5["runs"][k]
        s31 = sum(rec["masked_S_mean"].values()) / 31
        rows.append(f"    {name_cn[k]} & {rec['params']:,} & "
                    f"{fmt3(rec['clean']['S']['mean'])} & {fmt3(s31)} & "
                    f"{s31 - base31:+.3f} \\\\")
    (TAB_OUT / "p2_ablation.tex").write_text("\n".join([
        "\\begin{table}[htbp]", "  \\centering",
        "  \\caption{组件消融与填充基线（$\\mathcal{S}$，$\\Delta$ 相对完整 MRFN）}",
        "  \\label{tab:p2-ablation}",
        "  \\begin{tabular}{lrccc}", "    \\toprule",
        "    变体 & 参数量 & clean $\mathcal{S}$ & 31 场景均值 $\mathcal{S}$ & $\\Delta$ \\\\", "    \\midrule",
        *rows, "    \\bottomrule", "  \\end{tabular}", "\\end{table}",
    ]), encoding="utf-8")

    # ---- P2-T4 最终 test 表（含评估批次列）----
    mrfn = TF["models"]["MRFN"]["clean"]
    single = TS["mrfn_plus"]["single_mean"]
    ens = TS["mrfn_plus"]["ensemble_3seed"]
    def r(x): return f"{100 * x:.2f}\\%"
    (TAB_OUT / "p2_final_test.tex").write_text("\n".join([
        "\\begin{table}[htbp]", "  \\centering",
        "\\caption{最终测试集结果（附件2 test，clean 口径）}",
        "  \\label{tab:p2-final}",
        "  \\begin{tabular}{llrrrr}", "    \\toprule",
        "    模型 & 评估批次 & Acc & macro-F1 & MAE & $\mathcal{S}$ \\\\", "    \\midrule",
        f"    MRFN & 主表首次 test & {r(mrfn['acc']['mean'])} & {r(mrfn['macro_f1']['mean'])} & "
        f"{fmt3(mrfn['mae']['mean'])} & {fmt3(mrfn['S']['mean'])} \\\\",
        f"    MRFN+（单模型均值） & 预注册追加 test & {r(single['acc']['mean'])} & "
        f"{r(single['macro_f1']['mean'])} & {fmt3(single['mae']['mean'])} & {fmt3(single['S']['mean'])} \\\\",
        f"    MRFN+（3-seed 集成） & 预注册追加 test & {r(ens['acc'])} & {r(ens['macro_f1'])} & "
        f"{fmt3(ens['mae'])} & {fmt3(ens['S'])} \\\\",
        "    \\bottomrule", "  \\end{tabular}",
        "  \\par \\smallskip",
        "  {\\small 注：MRFN 为主表冻结正式模型；MRFN+ 为预注册追加评估的补充变体，未做 31 场景评估、不构造鲁棒性曲线；单模型均值与三种子集成为不同口径，不可直接比较。}",
        "\\end{table}",
    ]), encoding="utf-8")
    print("表: p2_ladder / p2_transfer / p2_ablation / p2_final_test →", TAB_OUT)


def plot_fj3_behavior() -> None:
    """P2-F5 附件3 行为分析：无标签测试集，只做行为图（不做任何精度声明）。

    gate / text-only 两组结论按修正后的 note 如实报告"未观察到预期单调
    重分配 / 趋同"，不沿用旧的正向表述。
    """
    df = pd.read_csv(RUNS / "fj3/附件3_预测结果.csv")
    fj = json.loads((RUNS / "fj3/附件3_行为分析.json").read_text())
    ga, tc, mo = fj["gate_analysis"], fj["text_only_convergence"], fj["missing_overview"]
    hard = set(fj["hard_samples"]["samples"])

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.9))

    # A：逐样本 joint 缺失率（排序，困难样本描边）
    ax = axes[0, 0]
    d = df.sort_values("joint_miss_rate", ascending=False).reset_index(drop=True)
    colors = ["#C0392B" if s in hard else "#3E7CB1" for s in d["sample_id"]]
    ax.bar(range(len(d)), d["joint_miss_rate"], color=colors, width=0.72)
    ax.axhline(mo["joint_rate_mean"], color="#374151", lw=0.9, ls="--")
    ax.text(len(d) - 0.5, mo["joint_rate_mean"] + 0.012, f"均值 {mo['joint_rate_mean']:.3f}",
            ha="right", fontsize=6.2, color="#374151")
    ax.set_xlabel("样本（按 joint 缺失率降序，红 = 困难样本 top-5）")
    ax.set_ylabel("joint 缺失率")
    ax.set_ylim(0, 0.56)
    ax.set_title("30 条样本 joint 缺失率（max 0.50）", loc="left", fontsize=8.4)

    # B：g_text vs a/v 平均缺失率（如实：无单调重分配）
    ax = axes[0, 1]
    av = (df["miss_rate_audio"] + df["miss_rate_vision"]) / 2
    ax.scatter(av, df["gate_text"], s=12, color="#3E7CB1", alpha=0.75, zorder=3)
    k, b = np.polyfit(av, df["gate_text"], 1)
    xs = np.linspace(0, av.max(), 50)
    ax.plot(xs, k * xs + b, color="#C0392B", lw=1.0, zorder=2)
    ax.axhline(ga["gate_text_hi_av_missing"], xmin=0.52, xmax=0.98, color="#E69F00", lw=1.0)
    ax.axhline(ga["gate_text_lo_av_missing"], xmin=0.52, xmax=0.98, color="#009E73", lw=1.0)
    ax.text(av.max() * 1.02, ga["gate_text_hi_av_missing"] - 0.018, f"hi {ga['gate_text_hi_av_missing']:.3f}",
            fontsize=6.0, va="center", color="#E69F00")
    ax.text(av.max() * 1.02, ga["gate_text_lo_av_missing"] + 0.012, f"lo {ga['gate_text_lo_av_missing']:.3f}",
            fontsize=6.0, va="center", color="#009E73")
    ax.set_xlabel("audio+vision 平均缺失率")
    ax.set_ylabel("gate_text")
    ax.set_xlim(-0.02, 0.46)
    ax.set_ylim(0.30, 0.85)
    ax.set_title("g_text 随 a/v 缺失：未见单调重分配", loc="left", fontsize=8.4)
    ax.text(0.03, 0.05, f"Spearman ρ={ga['spearman(av缺失率, gate_text)']:.3f}",
            transform=ax.transAxes, fontsize=6.2, color="#374151")

    # C：text-only 一致率 hi/lo（如实：未见趋同）
    ax = axes[1, 0]
    vals = [tc["agree_rate_low_av_missing"], tc["agree_rate_overall"], tc["agree_rate_high_av_missing"]]
    bars = ax.bar(["低 a/v 缺失组", "全体", "高 a/v 缺失组"], vals,
                  color=["#009E73", "#999999", "#E69F00"], width=0.55)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("全模态 vs text-only 预测一致率")
    ax.set_title("一致率高缺失组更低：未见趋同", loc="left", fontsize=8.4)

    # D：附件3 预测极性分布（30 条样本口径，非附件2 test）
    ax = axes[1, 1]
    vc = df["pred_polarity"].value_counts().reindex([0, 1, 2], fill_value=0)
    ax.bar(["负", "中", "正"], vc.values, color=["#0072B2", "#999999", "#D55E00"], width=0.55)
    for i, v in enumerate(vc.values):
        ax.text(i, v + 0.3, str(int(v)), ha="center", fontsize=7)
    ax.set_ylabel("样本数")
    ax.set_ylim(0, max(vc.values) * 1.25 + 1)
    ax.set_title("预测极性分布（30 条，无标签仅行为）", loc="left", fontsize=8.4)
    cal = fj["calibration"]
    ax.text(0.03, 0.90, f"温度 T={cal['temperature']:.4f}（valid NLL 拟合）\n"
                        f"ECE {cal['ece'].split('→')[0]}→{cal['ece'].split('→')[1]}，"
                        f"argmax 不变", transform=ax.transAxes, fontsize=6.2, color="#374151")

    fig.suptitle("附件3 行为分析（MRFN+ 集成，无标签测试集：不做精度声明）", y=0.99, fontsize=9.5)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(FIG_OUT / "q2_fj3_behavior.pdf")
    fig.savefig(FIG_OUT / "q2_fj3_behavior.png", dpi=300)
    plt.close(fig)


def main() -> int:
    configure_style()
    plot_degradation()
    plot_scenario_heatmap()
    plot_fj3_behavior()
    write_tables()
    mf = TF["models"]["MRFN"]
    print("caption 素材：")
    print(f"  MRFN clean S={mf['clean']['S']['mean']:.4f}±{mf['clean']['S']['std']:.4f}, "
          f"31 场景 D_S 均值={mf['mean_D_S']:.4f}（B0={TF['models']['B0']['mean_D_S']:.4f}）")
    print(f"  MRFN text80: S={mf['masked_S_mean']['mcar/t/rate80/random']:.4f} vs "
          f"B3 {TF['models']['B3']['masked_S_mean']['mcar/t/rate80/random']:.4f}（MRFN 双优）")
    print(f"  MRFN+ 单模型 Acc={100*TS['mrfn_plus']['single_mean']['acc']['mean']:.2f}%，"
          f"集成 Acc={100*TS['mrfn_plus']['ensemble_3seed']['acc']:.2f}%（预注册追加批次）")
    print("  图注建议：audio/vision 缺失下各基线退化差异 <0.01、曲线高度重合（数值本就接近，非绘制错误）；"
          "text 缺失为主导风险，MRFN 退化最平缓。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
