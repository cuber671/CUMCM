#!/usr/bin/env python
"""全章节数据分布与密度分析图（7 张，对应 §2.1/§4.1/§4.2/§5.1/§5.3/§6.2/§7.1）。

Usage:
  .venv/bin/python scripts/figures/plot_distributions.py

产物（paper/latex/figures/dist/）：
  d_missing_runs.{pdf,png}     §2.1  附件3 联合零行连续缺失长度 PMF
  d_offset_kde.{pdf,png}        §4.1  三模态时间偏移量核密度对比
  d_length_error.{pdf,png}      §4.2  对齐后特征长度误差分模态密度
  d_miss_rate.{pdf,png}         §5.1  训练/验证/测试缺失率分布
  d_ladder_box.{pdf,png}        §5.3  B1/B2/B3/MRFN 三种子分布箱线图
  d_shapley_kde.{pdf,png}       §6.2  三模态 Shapley 贡献核密度对比
  d_baseline_violin.{pdf,png}   §7.1  本文 vs 三基线性能分布小提琴图

口径纪律：
- 全部只读冻结 json/pkl/csv，不重跑训练/对齐；
- §5.3 用 ladder 3 种子（B0 仅单点不画箱），图注明确写"3 种子"；
- §7.1 用 m5 单点 + m5 3 种子（MRFN_clean/noState/noMaskAttn）做小提琴；
- 与既有图（q2_scenario_heatmap / q2_degradation / q1_acceptance_summary / q3_shapley_evidence）不重复；
- 命名走 dist/ 子目录，与 q1/q2/q3 并列。
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from figstyle import FONT_SIZE, MOD_COLOR, POLARITY, configure

ROOT = Path(__file__).resolve().parents[2]
FIG_OUT = ROOT / "paper/latex/figures/dist"
FIG_OUT.mkdir(parents=True, exist_ok=True)

P1_FULL = ROOT / "runs/p1_full"
P2 = ROOT / "runs/p2"
P3 = ROOT / "runs/p3"

MOD_CN = {"t": "文本", "a": "语音", "v": "视觉"}
PAIRS = [("t", "a"), ("t", "v"), ("a", "v")]


# ============================================================================
# 图 1 (§2.1)：附件3 联合零行连续缺失长度分布
# ============================================================================
def plot_missing_runs() -> None:
    """§2.1 三重挑战「模态局部缺失」真实分布证据。

    数据：runs/p2/mask_library/library_index.json::att3_pmf_frozen
    - 30 条附件3 样本 × 605 内容位扫描出的 131 段联合零行的段长分布
    - pmf[0..3] 对应段长 1..4；>4 段合计计入"4+"桶
    """
    d = json.loads((P2 / "mask_library/library_index.json").read_text())
    pmf = d["att3_pmf_frozen"]           # [p1, p2, p3, p4]
    raw = d["att3_pmf_measurement"]["raw_counts"]
    mean_seg_len = d["att3_pmf_measurement"]["mean_seg_len"]
    n_files = d["att3_pmf_measurement"]["n_files"]
    n_content = d["att3_pmf_measurement"]["n_content_positions"]
    n_joint = d["att3_pmf_measurement"]["n_joint_zero"]

    # 段长桶：1, 2, 3, 4+（pmf 索引 0..3 即段长 1..4）
    labels = ["1", "2", "3", "≥4"]
    counts = [raw.get(str(k), 0) for k in range(1, 5)]
    # raw_counts 没记录 ≥5；按 pmf 推导（pmf[3] 即段长 ≥4 的累计；这里展示段长 4 的占比作为 ≥4 标记）
    # 但 pmf[3] = 0.0421 是段长 4 的概率，>4 段数 = 0（测量时 n_runs=95 全部 ≤4）
    # 修正：labels 与 pmf 一一对应即可，counts 用 raw；>4 段数为 0 时附注
    p4plus = sum(p for k, p in enumerate(pmf, start=1) if k > 4)

    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
    bars = ax.bar(labels, pmf, color=colors, width=0.62, edgecolor="white",
                  linewidth=0.6)
    for b, p in zip(bars, pmf):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{p*100:.1f}%", ha="center", fontsize=FONT_SIZE["TICK"])
    # 期望段长参考线
    ax.axvline(x=mean_seg_len - 1 + 0.5, color="#374151", lw=0.7, ls="--")
    ax.text(len(labels) - 0.5, 0.78,
            f"期望段长 $\\bar l={mean_seg_len:.3f}$",
            ha="right", fontsize=FONT_SIZE["NOTE"], color="#374151")
    ax.set_ylim(0, 0.92)
    ax.set_xlabel("连续缺失段长（位置数）")
    ax.set_ylabel("段数占比")
    ax.set_title(
        f"附件3 联合零行连续缺失段长分布（{n_files} 条 × {n_content} 位，{n_joint} 段）",
        loc="left", fontsize=FONT_SIZE["TITLE"],
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_OUT / "d_missing_runs.pdf")
    fig.savefig(FIG_OUT / "d_missing_runs.png", dpi=300)
    plt.close(fig)
    print(f"[§2.1] d_missing_runs：pmf={pmf} mean_seg_len={mean_seg_len}")


# ============================================================================
# 图 2 (§4.1)：三模态时间偏移量核密度对比
# ============================================================================
def plot_offset_kde() -> None:
    """§4.1 词持续时间按位置分位的核密度分布。

    三条 KDE 曲线：头段位置 / 中段位置 / 尾段位置的词持续时间（t_e - t_s）。
    数据：runs/p1_full/samples/*.pkl alignment.words[*].t_s, t_e, alignment_status。
    论证目标：CTC 词区间在 50 位序列上的非均匀性，即非等间隔对齐粒度的实证依据。
    """
    buckets = {"头段 (≤K/3)": [], "中段 (K/3~2K/3)": [], "尾段 (>2K/3)": []}
    n_samples = 0
    n_words_total = 0
    for pkl in sorted((P1_FULL / "samples").glob("*.pkl")):
        with pkl.open("rb") as f:
            d = pickle.load(f)
        if d.get("schema_version") in (None, "", "0", 0):
            continue
        align = d.get("alignment", {})
        if align.get("status") == "rollback":
            continue
        kept = [w for w in align.get("words", [])
                if w.get("alignment_status") == "aligned"]
        if not kept:
            continue
        n_samples += 1
        K = len(kept)
        n_words_total += K
        for j, w in enumerate(kept, start=1):
            dur = w["t_e"] - w["t_s"]
            if j <= K / 3:
                buckets["头段 (≤K/3)"].append(dur)
            elif j <= 2 * K / 3:
                buckets["中段 (K/3~2K/3)"].append(dur)
            else:
                buckets["尾段 (>2K/3)"].append(dur)

    print(f"[§4.1] d_offset_kde：{n_samples} 样本，共 {n_words_total} 词，按位置分位：")
    for k, v in buckets.items():
        a = np.array(v)
        print(f"  {k}: n={len(a)} mean={a.mean():.4f}s std={a.std():.4f}s")

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    color_map = {"头段 (≤K/3)": "#0072B2", "中段 (K/3~2K/3)": "#009E73",
                 "尾段 (>2K/3)": "#D55E00"}
    # 横轴 0~1.5 s（绝大多数词持续时间落于此区间）
    xs = np.linspace(0, 1.5, 200)
    for k, arr in buckets.items():
        a = np.array(arr)
        if len(a) < 2:
            continue
        try:
            kde = gaussian_kde(a[a <= 1.5], bw_method=0.30)
            ys = kde(xs)
        except np.linalg.LinAlgError:
            ys = np.zeros_like(xs)
        ax.plot(xs, ys, color=color_map[k], lw=1.5, label=f"{k}（n={len(a)}）")
        ax.fill_between(xs, ys, color=color_map[k], alpha=0.12)
        ax.axvline(a.mean(), color=color_map[k], lw=0.6, ls=":")
    # 参考线：CTC 帧移 20 ms（对齐量化粒度）
    ax.axvline(0.02, color="#374151", lw=0.6, ls="--")
    ax.text(0.025, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.9,
            "CTC 帧移 20 ms", fontsize=6.0, color="#374151",
            rotation=90, va="top")
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("词持续时间（秒；CTC 对齐区间 $t_e-t_s$）")
    ax.set_ylabel("概率密度")
    ax.legend(frameon=False, fontsize=FONT_SIZE["TICK"], loc="upper right")
    ax.set_title(
        f"词持续时间按位置分位的核密度（{n_samples} 样本 × {n_words_total} 词）",
        loc="left", fontsize=FONT_SIZE["TITLE"],
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_OUT / "d_offset_kde.pdf")
    fig.savefig(FIG_OUT / "d_offset_kde.png", dpi=300)
    plt.close(fig)


# ============================================================================
# 图 3 (§4.2)：对齐后特征长度误差分模态密度
# ============================================================================
def plot_length_error() -> None:
    """§4.2 对齐后各模态长度误差分布。

    对每个 P1 样本：
      - text: |wp_len - 50|
      - audio: |audio_missing_counts|（=50 - audio_observed）
      - vision: |vision_missing_counts|（=50 - vision_observed）
    数据来自 manifest.csv 的 audio_missing_counts / vision_missing_counts 列；
    wp_len 由 manifest.csv 直接读取。
    """
    df = pd.read_csv(P1_FULL / "manifest.csv")
    df = df[df["status"] == "ok"].copy()

    # §4.2 「对齐后内容位填充误差」分模态：
    # - text：内容位空位数 = max(0, 48 - wp_len)
    # - audio：内容位零行数 ≈ (1 - audio_coverage) * wp_len
    # - vision：内容位零行数 ≈ (1 - vision_coverage) * wp_len
    wp_len = df["wp_len"].fillna(0).astype(int).clip(lower=0)
    text_err = (48 - wp_len).clip(lower=0).to_numpy()
    audio_cov = df["audio_coverage"].fillna(1.0).astype(float)
    vision_cov = df["vision_coverage"].fillna(1.0).astype(float)
    audio_err = np.round((1.0 - audio_cov) * wp_len).astype(int).to_numpy()
    vision_err = np.round((1.0 - vision_cov) * wp_len).astype(int).to_numpy()

    # 限制横轴，避免 KDE 失真（少量离群）
    cap = 25
    print(f"[§4.2] d_length_error：n={len(df)}，text/audio/vision 误差均值：")
    print(f"  text err: mean={text_err.mean():.2f}, max={text_err.max()}")
    print(f"  audio err: mean={audio_err.mean():.2f}, max={audio_err.max()}")
    print(f"  vision err: mean={vision_err.mean():.2f}, max={vision_err.max()}")

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    bins = np.arange(0, cap + 1, 1)
    series = [
        ("文本", text_err, MOD_COLOR["text"]),
        ("语音", audio_err, MOD_COLOR["audio"]),
        ("视觉", vision_err, MOD_COLOR["vision"]),
    ]
    xs_grid = np.linspace(0, cap, 200)
    for name, arr, color in series:
        arr_clip = arr[arr <= cap]
        # KDE 仅在数据非退化时启用；否则用直方图（归一化密度）
        if len(arr_clip) > 5 and arr_clip.std() > 1e-3:
            try:
                kde = gaussian_kde(arr_clip, bw_method=0.25)
                ys = kde(xs_grid)
                ax.plot(xs_grid, ys, color=color, lw=1.5,
                        label=f"{name}（n={len(arr)}，max={int(arr.max())}）")
                ax.fill_between(xs_grid, ys, color=color, alpha=0.12)
                continue
            except np.linalg.LinAlgError:
                pass
        # 退化数据：直方图替代
        ax.hist(arr_clip, bins=bins, density=True, alpha=0.45, color=color,
                label=f"{name}（n={len(arr)}，max={int(arr.max())}）", edgecolor="white", lw=0.4)
    ax.set_xlim(0, cap)
    ax.set_xlabel("对齐后内容位空位数（48 位内容位内）")
    ax.set_ylabel("概率密度")
    ax.legend(frameon=False, fontsize=FONT_SIZE["TICK"], loc="upper right")
    ax.set_title(
        "对齐后内容位空位数分模态密度（附件1，100 条 ok 样本）",
        loc="left", fontsize=FONT_SIZE["TITLE"],
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_OUT / "d_length_error.pdf")
    fig.savefig(FIG_OUT / "d_length_error.png", dpi=300)
    plt.close(fig)


# ============================================================================
# 图 4 (§5.1)：训练/验证/测试缺失率分布
# ============================================================================
def plot_miss_rate() -> None:
    """§5.1 31 场景缺失率分布按 train/valid/test 三划分。

    数据：mask_library/library_index.json entries[*].actual_rate_mean
    - 43 条 train / 43 条 valid / 43 条 test
    - 每条 entry 含 1~3 个模态的实际缺失率（已聚合 8 实例均值）
    """
    d = json.loads((P2 / "mask_library/library_index.json").read_text())
    by_split = defaultdict(list)
    for e in d["entries"]:
        for rate in e["actual_rate_mean"].values():
            by_split[e["split"]].append(float(rate))

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    bins = np.linspace(0, 0.85, 36)
    split_color = {"train": "#0072B2", "valid": "#009E73", "test": "#D55E00"}
    split_label = {"train": "训练", "valid": "验证", "test": "测试"}
    for sp in ["train", "valid", "test"]:
        arr = np.array(by_split[sp])
        ax.hist(arr, bins=bins, density=True, alpha=0.32, color=split_color[sp],
                label=f"{split_label[sp]}（n={len(arr)}）")
        # KDE
        if len(arr) > 1:
            try:
                kde = gaussian_kde(arr, bw_method=0.25)
                xs = np.linspace(0, 0.85, 200)
                ax.plot(xs, kde(xs), color=split_color[sp], lw=1.4)
            except np.linalg.LinAlgError:
                pass
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("实际缺失率（31 场景 × 模态，8 实例均值）")
    ax.set_ylabel("概率密度")
    ax.legend(frameon=False, fontsize=FONT_SIZE["TICK"], loc="upper right")
    ax.set_title(
        "31 缺失场景实际缺失率分布（train / valid / test）",
        loc="left", fontsize=FONT_SIZE["TITLE"],
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_OUT / "d_miss_rate.pdf")
    fig.savefig(FIG_OUT / "d_miss_rate.png", dpi=300)
    plt.close(fig)
    for sp in ["train", "valid", "test"]:
        arr = np.array(by_split[sp])
        print(f"[§5.1] d_miss_rate：{sp} n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f}")


# ============================================================================
# 图 5 (§5.3)：B1/B2/B3/MRFN 三种子分布箱线图
# ============================================================================
def plot_ladder_box() -> None:
    """§5.3 阶梯消融三种子性能分布（B0 单点 + B1/B2/B3/MRFN_clean 三种子）。

    数据：runs/p2/ladder/{model}_seed{N}/result.json valid_metrics.S
    """
    models = ["B1", "B2", "B3"]
    per_model_seeds = {m: [] for m in models}
    for m in models:
        for sd in [1, 2, 3]:
            p = P2 / "ladder" / f"{m}_seed{sd}" / "result.json"
            r = json.loads(p.read_text())
            per_model_seeds[m].append(r["valid_metrics"]["S"])
    # MRFN_clean 同样 3 种子
    mrfn_seeds = []
    for sd in [1, 2, 3]:
        p = P2 / "mrfn" / "MRFN_clean" / f"MRFN_seed{sd}" / "result.json"
        r = json.loads(p.read_text())
        mrfn_seeds.append(r["valid_metrics"]["S"])
    per_model_seeds["MRFN_clean"] = mrfn_seeds
    # B0 单点：test_final B0 clean S（test 口径主表值）
    tf = json.loads((P2 / "test_final" / "test_final_results.json").read_text())
    b0_S = tf["models"]["B0"]["clean"]["S"]["mean"]

    labels = ["B0", "B1", "B2", "B3", "MRFN"]
    data_box = [None, per_model_seeds["B1"], per_model_seeds["B2"],
                per_model_seeds["B3"], per_model_seeds["MRFN_clean"]]

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    # B0 作为单点散点（不在箱体内）
    ax.scatter([0], [b0_S], color="#999999", s=42, marker="D", zorder=4,
               edgecolor="white", linewidth=0.6, label="B0（test 单点）")
    # 箱线图（B1..MRFN）
    bp = ax.boxplot(
        [d for d in data_box if d is not None],
        positions=[1, 2, 3, 4],
        widths=0.55,
        patch_artist=True,
        showmeans=False,
        medianprops=dict(color="#374151", lw=1.1),
        whiskerprops=dict(color="#374151", lw=0.8),
        capprops=dict(color="#374151", lw=0.8),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#D55E00",
                        markeredgecolor="none"),
    )
    colors = [MODEL_COLOR_B := {"B1": "#CC79A7", "B2": "#009E73", "B3": "#0072B2",
                                  "MRFN_clean": "#D55E00"}[m]
              for m in ["B1", "B2", "B3", "MRFN_clean"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)

    # 叠加 3 颗种子散点
    rng = np.random.default_rng(42)
    for i, m in enumerate(["B1", "B2", "B3", "MRFN_clean"]):
        x_jitter = rng.normal(loc=i + 1, scale=0.04, size=3)
        ax.scatter(x_jitter, per_model_seeds[m], color=colors[i],
                   s=22, edgecolor="white", linewidth=0.4, zorder=5)

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, 4.7)
    ax.set_ylabel("验证集 $\\mathcal{S}$")
    ax.set_title(
        "模型阶梯三种子性能分布（valid；B0 为 test 单点对照）",
        loc="left", fontsize=FONT_SIZE["TITLE"],
    )
    # 数值标注
    for i, m in enumerate(["B1", "B2", "B3", "MRFN_clean"], start=1):
        mean_S = np.mean(per_model_seeds[m])
        ax.text(i + 0.32, mean_S, f"均 {mean_S:.3f}",
                fontsize=6.0, va="center", color="#374151")
    ax.text(0 + 0.32, b0_S, f"test {b0_S:.3f}",
            fontsize=6.0, va="center", color="#999999")
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_OUT / "d_ladder_box.pdf")
    fig.savefig(FIG_OUT / "d_ladder_box.png", dpi=300)
    plt.close(fig)
    print(f"[§5.3] d_ladder_box：B0 test S={b0_S:.4f}")
    for m in ["B1", "B2", "B3", "MRFN_clean"]:
        print(f"  {m}: seeds={['%.4f'%s for s in per_model_seeds[m]]}")


# ============================================================================
# 图 6 (§6.2)：三模态 Shapley 贡献核密度对比
# ============================================================================
def plot_shapley_kde() -> None:
    """§6.2 三模态 Shapley 贡献分布。

    数据：runs/p3/m2/shapley_att4.json reports[*]（20 条样本）
    - 每个样本的精确 Shapley 值按 8 联盟全枚举计算
    - 取预测类对应的分类带符号贡献 φ_m^{cls}[ŷ]
    """
    d = json.loads((P3 / "m2" / "shapley_att4.json").read_text())
    per_mod = {"text": [], "audio": [], "vision": []}
    per_mod_abs = {"text": [], "audio": [], "vision": []}
    for rep in d["reports"]:
        phi_cls = rep.get("phi_cls", {})
        pred_class = rep.get("pred_class", 0)
        for m in ["text", "audio", "vision"]:
            cls_vec = phi_cls.get(m)
            if cls_vec is None or len(cls_vec) <= pred_class:
                continue
            val = float(cls_vec[pred_class])
            per_mod[m].append(val)
            per_mod_abs[m].append(abs(val))

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    # 左：带符号密度
    ax = axes[0]
    xs = np.linspace(-0.7, 0.7, 240)
    for m, color in [("text", MOD_COLOR["text"]), ("audio", MOD_COLOR["audio"]),
                     ("vision", MOD_COLOR["vision"])]:
        arr = np.array(per_mod[m])
        if len(arr) < 2:
            continue
        kde = gaussian_kde(arr, bw_method=0.35)
        ys = kde(xs)
        ax.plot(xs, ys, color=color, lw=1.4, label=f"{MOD_CN[m[0]] if m!='vision' else '视觉'}（n={len(arr)}）")
        ax.fill_between(xs, ys, color=color, alpha=0.12)
        ax.axvline(arr.mean(), color=color, lw=0.6, ls=":")
    ax.axvline(0, color="#374151", lw=0.6)
    ax.set_xlim(-0.7, 0.7)
    ax.set_xlabel("带符号 Shapley 贡献 $\\varphi_m^{\\mathrm{cls}}[\\hat y]$")
    ax.set_ylabel("概率密度")
    ax.legend(frameon=False, fontsize=FONT_SIZE["TICK"], loc="upper right")
    ax.set_title("(a) 带符号贡献密度", loc="left", fontsize=FONT_SIZE["TITLE"])

    # 右：绝对值密度
    ax = axes[1]
    xs = np.linspace(0, 0.7, 200)
    for m, color in [("text", MOD_COLOR["text"]), ("audio", MOD_COLOR["audio"]),
                     ("vision", MOD_COLOR["vision"])]:
        arr = np.array(per_mod_abs[m])
        if len(arr) < 2:
            continue
        kde = gaussian_kde(arr, bw_method=0.35)
        ys = kde(xs)
        ax.plot(xs, ys, color=color, lw=1.4, label=f"{MOD_CN[m[0]] if m!='vision' else '视觉'}（n={len(arr)}）")
        ax.fill_between(xs, ys, color=color, alpha=0.12)
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("$|\\varphi_m^{\\mathrm{cls}}[\\hat y]|$")
    ax.set_ylabel("概率密度")
    ax.legend(frameon=False, fontsize=FONT_SIZE["TICK"], loc="upper right")
    ax.set_title("(b) 贡献绝对值密度", loc="left", fontsize=FONT_SIZE["TITLE"])

    fig.suptitle(
        f"三模态 Shapley 贡献分布（附件4，{len(d['reports'])} 条样本，预测类带符号）",
        x=0.02, y=1.02, ha="left", fontsize=FONT_SIZE["SUPTITLE"],
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_OUT / "d_shapley_kde.pdf")
    fig.savefig(FIG_OUT / "d_shapley_kde.png", dpi=300)
    plt.close(fig)
    for m in ["text", "audio", "vision"]:
        arr = np.array(per_mod[m])
        arr_abs = np.array(per_mod_abs[m])
        print(f"[§6.2] d_shapley_kde {m}: signed mean={arr.mean():.4f} |·|mean={arr_abs.mean():.4f} median|={np.median(arr_abs):.4f}")


# ============================================================================
# 图 7 (§7.1)：本文 vs 三基线性能分布小提琴图
# ============================================================================
def plot_baseline_violin() -> None:
    """§7.1 本文 MRFN vs 三消融基线性能分布小提琴图。

    数据：runs/p2/mrfn/{MRFN_clean,MRFN_noState,MRFN_noMaskAttn,MRFN_noGate}
    - 每个变体 3 种子 valid_metrics.S
    - 31 场景均值 S 取自 m5_summary.json masked_S_mean 各 31 场景等权平均
    """
    variants = ["MRFN_clean", "MRFN_noState", "MRFN_noMaskAttn", "MRFN_noGate"]
    label_cn = {
        "MRFN_clean": "去除缺失增强训练",
        "MRFN_noState": "去除缺失状态编码",
        "MRFN_noMaskAttn": "去除掩码注意力",
        "MRFN_noGate": "去除门控",
    }
    color_cn = {
        "MRFN_clean": MOD_COLOR["vision"],   # 朱红主色
        "MRFN_noState": "#B0B0B0",
        "MRFN_noMaskAttn": "#0072B2",
        "MRFN_noGate": "#009E73",
    }
    m5 = json.loads((P2 / "m5" / "m5_summary.json").read_text())

    valid_S = {v: [] for v in variants}
    scen31_S = {v: [] for v in variants}
    for v in variants:
        for sd in [1, 2, 3]:
            # MRFN_clean 用 MRFN_seed{N} 子目录，其他用 {v}_seed{N}
            if v == "MRFN_clean":
                p = P2 / "mrfn" / v / f"MRFN_seed{sd}" / "result.json"
            else:
                p = P2 / "mrfn" / v / f"{v}_seed{sd}" / "result.json"
            r = json.loads(p.read_text())
            valid_S[v].append(r["valid_metrics"]["S"])
        rec = m5["runs"][v]
        scen31_S[v].append(sum(rec["masked_S_mean"].values()) / 31)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))

    # 左：验证集 S 分布（3 种子小提琴）
    ax = axes[0]
    data = [valid_S[v] for v in variants]
    parts = ax.violinplot(data, positions=range(len(variants)), widths=0.7,
                          showmeans=True, showmedians=False, showextrema=True)
    for i, v in enumerate(variants):
        body = parts["bodies"][i]
        body.set_facecolor(color_cn[v])
        body.set_alpha(0.55)
        body.set_edgecolor(color_cn[v])
    # 3 颗种子散点
    rng = np.random.default_rng(7)
    for i, v in enumerate(variants):
        xj = rng.normal(loc=i, scale=0.05, size=3)
        ax.scatter(xj, valid_S[v], color=color_cn[v], s=22,
                   edgecolor="white", linewidth=0.4, zorder=5)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([label_cn[v] for v in variants], fontsize=6.4,
                       rotation=15, ha="right")
    ax.set_ylabel("验证集 $\\mathcal{S}$")
    ax.set_title("(a) valid（3 种子）", loc="left", fontsize=FONT_SIZE["TITLE"])
    ax.set_ylim(0.71, 0.76)

    # 右：31 场景 S（单点 → 柱状图）
    ax = axes[1]
    vals = [scen31_S[v][0] for v in variants]
    bars = ax.bar(range(len(variants)), vals,
                  color=[color_cn[v] for v in variants],
                  width=0.55, edgecolor="white", linewidth=0.4)
    # 数值标注
    for i, (b, val) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.0015,
                f"{val:.3f}",
                ha="center", fontsize=6.4, color="#374151",
                transform=ax.transData)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([label_cn[v] for v in variants], fontsize=6.4,
                       rotation=15, ha="right")
    ax.set_ylabel("31 场景均值 $\\mathcal{S}$")
    ax.set_title("(b) 31 场景（m5 单点）", loc="left", fontsize=FONT_SIZE["TITLE"])
    ax.set_ylim(0.69, 0.755)

    fig.suptitle(
        "本文 MRFN vs 三消融基线性能分布",
        x=0.02, y=1.04, ha="left", fontsize=FONT_SIZE["SUPTITLE"],
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_OUT / "d_baseline_violin.pdf")
    fig.savefig(FIG_OUT / "d_baseline_violin.png", dpi=300)
    plt.close(fig)
    for v in variants:
        print(f"[§7.1] d_baseline_violin {v}: valid seeds={['%.4f'%s for s in valid_S[v]]}, 31 scen S={scen31_S[v][0]:.4f}")


# ============================================================================
# 图 8 (§6.3)：TOP-20 关键词 IG 贡献条形图
# ============================================================================
def plot_top20_keywords() -> None:
    """§6.3 关键词贡献梯度。

    数据：runs/p3/m3/ig_att4.json reports[*].baselines.words.{word_attr, word_texts}
    - 20 条样本 × 每样本若干词，word_attr 按 missing 基线聚合的词级 IG 绝对值
    - 跨样本按词文本聚合 |attr| 总和，取 TOP-20
    """
    from collections import defaultdict
    d = json.loads((P3 / "m3" / "ig_att4.json").read_text())
    word_sum = defaultdict(float)
    word_count = defaultdict(int)
    n_samples_with_words = 0
    for r in d["reports"]:
        wa = r["baselines"]["words"].get("word_attr", {})
        wt = r["baselines"]["words"].get("word_texts", {})
        if not wa:
            continue
        n_samples_with_words += 1
        for wid in wa:
            if wid not in wt:
                continue
            w = wt[wid]
            word_sum[w] += abs(float(wa[wid]))
            word_count[w] += 1
    # 按 total_|attr| 排序
    sorted_words = sorted(word_sum.items(), key=lambda x: x[1], reverse=True)[:20]
    words = [w for w, _ in sorted_words]
    vals = np.array([v for _, v in sorted_words])
    counts = np.array([word_count[w] for w in words])

    # 颜色按出现样本数着色：高频词（n≥3）朱红、单样本词灰
    colors = ["#D55E00" if c >= 3 else ("#0072B2" if c == 2 else "#999999")
              for c in counts]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    y_pos = np.arange(len(words))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
    # 数值标注（右端）
    for b, v, c in zip(bars, vals, counts):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2,
                f"{v:.3f}（n={int(c)}）", va="center",
                fontsize=FONT_SIZE["TICK"], color="#374151")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=FONT_SIZE["TICK"])
    ax.invert_yaxis()
    ax.set_xlabel("跨样本 $\\sum|\\mathrm{IG}_w|$（missing 基线，词级聚合）")
    ax.set_xlim(0, vals.max() * 1.22)
    ax.set_title(
        f"TOP-20 关键词 IG 贡献（附件4，{n_samples_with_words} 样本词级聚合）",
        loc="left", fontsize=FONT_SIZE["TITLE"],
    )
    # 图例说明
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#D55E00", label="≥3 样本"),
        Patch(color="#0072B2", label="2 样本"),
        Patch(color="#999999", label="1 样本"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=FONT_SIZE["TICK"],
              loc="lower right")
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_OUT / "d_top20_keywords.pdf")
    fig.savefig(FIG_OUT / "d_top20_keywords.png", dpi=300)
    plt.close(fig)
    print(f"[§6.3] d_top20_keywords：{n_samples_with_words} 样本聚合，TOP 词：{words[:5]}...")


# ============================================================================
# 图 9（附录D）：31 场景缺失类型占比环形图
# ============================================================================
def plot_missing_type_donut() -> None:
    """附录D 31 场景缺失类型占比总览。

    数据：runs/p2/mask_library/library_index.json entries[*]
    按 plot_q2_results.scenario_grid_keys 同款筛选：仅取 31 评测场景，
    类别：MCAR 单模态 (15) / MCAR 双模态 (3) / Block (12) / 联合 (1)。
    """
    d = json.loads((P2 / "mask_library" / "library_index.json").read_text())
    counts = {"MCAR 单模态": 0, "MCAR 双模态": 0, "Block（连续块）": 0, "联合缺失": 0}
    seen_scenes = set()
    for e in d["entries"]:
        mech, mods, rate, pos = e["mechanism"], e["modalities"], e["rate"], e["position"]
        # 31 评测场景筛选（与 plot_q2_results.scenario_grid_keys 一致）
        keep = False
        if mech == "mcar" and len(mods) == 1 and rate in [10, 20, 40, 60, 80] and pos == "random":
            keep = True
        elif mech == "mcar" and len(mods) == 2 and rate == 40 and pos == "random":
            keep = True
        elif mech == "block" and len(mods) == 1 and rate == 40 and pos in ("head", "mid", "tail", "random"):
            keep = True
        elif mech == "joint" and mods == "av" and rate == 40 and pos == "random":
            keep = True
        if not keep:
            continue
        scene_key = (mech, mods, rate, pos)
        if scene_key in seen_scenes:
            continue
        seen_scenes.add(scene_key)
        if mech == "mcar" and len(mods) == 1:
            counts["MCAR 单模态"] += 1
        elif mech == "mcar" and len(mods) == 2:
            counts["MCAR 双模态"] += 1
        elif mech == "block":
            counts["Block（连续块）"] += 1
        elif mech == "joint":
            counts["联合缺失"] += 1

    labels = list(counts.keys())
    vals = list(counts.values())
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, autopct=lambda p: f"{p:.1f}%\n({int(round(p*sum(vals)/100))})",
        startangle=90, colors=colors,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.6),
        textprops=dict(fontsize=FONT_SIZE["TICK"]),
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(FONT_SIZE["NOTE"])
        at.set_fontweight("bold")
    ax.set_title(
        f"31 缺失场景类型占比（合计 {sum(vals)} 场景）",
        loc="left", fontsize=FONT_SIZE["TITLE"],
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_OUT / "d_missing_type_donut.pdf")
    fig.savefig(FIG_OUT / "d_missing_type_donut.png", dpi=300)
    plt.close(fig)
    print(f"[附录D] d_missing_type_donut：{counts} (合计 {sum(vals)} 场景)")


# ============================================================================
# main
# ============================================================================
def main() -> int:
    configure()
    plot_missing_runs()
    plot_offset_kde()
    plot_length_error()
    plot_miss_rate()
    plot_ladder_box()
    plot_shapley_kde()
    plot_baseline_violin()
    plot_top20_keywords()
    plot_missing_type_donut()
    print(f"\n9 张分布图 → {FIG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())