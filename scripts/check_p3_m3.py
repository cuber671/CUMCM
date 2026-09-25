"""P3 M3 验收脚本（第二轮审阅修正版）：附件4 全 20 条 IG + 三基线敏感性。

检查项：
  1. 积分近似误差（主基线 missing，逐样本门槛）：每样本每输出
     |ΣIG − ΔF| ≤ max(5%·|ΔF|, 1e-3)，报告通过数/最大相对误差/失败清单；
     zero/mean 基线同式逐样本报告（不设门槛）；
  2. 证据边界（架构级断言，不再只看结果事实）：全部基线 × 双输出 × 三模态，
     IG 在不可用位（非 content ∪ content∧o=0）最大绝对值 ≤ 1e-6；
     #13 vision 自然缺失（全模态实例）单列复检；
  3. P1 对齐：build_grid input_ids 与附件4 存储网格逐 token 对齐（20/20）；
     词聚合按 P1 alpha（词内均匀）质量守恒求和；
  4. 三基线排序敏感性：content 位 Spearman ρ（missing↔zero / missing↔mean / zero↔mean），
     a/v 的 missing≡mean 为构造性 ρ=1（z 空间重合留痕）；
  5. 可微层级声明 + 基线语义声明（avail 固定 o，与 v(∅) 区分）+ checkpoint SHA 落盘。
积分：中点 64 步（右端点 32 步在 6 样本超阈值的参数修正，见 docs M3 修订记录）。
产物：runs/p3/m3/ig_att4.json。退出码：全过 = 0。
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import load_aligned  # noqa: E402
from src.p2.models import MODEL_REGISTRY  # noqa: E402
from src.p2.pipeline import load_stats  # noqa: E402
from src.p2.text_mask import BertTextEncoder  # noqa: E402
from src.p1.grid import build_grid  # noqa: E402
from src.p3.data import load_att4  # noqa: E402
from src.p3.ig import (BASELINES, N_STEPS, aggregate_words, av_baseline,  # noqa: E402
                       endpoints, ig_modality, p1_word_grid, text_baseline,
                       train_mean_text_embedding)
from src.p3.shapley import MODS, prepare_sample  # noqa: E402

CKPT = ROOT / "runs/p2/bft/MRFN_seed{seed}"
STATE = ROOT / "runs/p2/data/m1a_state.npz"
OUT = ROOT / "runs/p3/m3"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("✅" if ok else "❌"), name, "" if ok else json.dumps(detail, ensure_ascii=False))

    m2 = json.loads((ROOT / "runs/p3/m2/shapley_att4.json").read_text())
    target = {r["sample_id"]: r["pred_class"] for r in m2["reports"]}

    att = load_aligned()
    state = np.load(STATE)
    stats = load_stats()

    ensemble, mean_embs, fps = [], {}, []
    for seed in (1, 2, 3):
        sd = Path(str(CKPT).format(seed=seed))
        be = BertTextEncoder(unfreeze_last=1).to(device)
        be.load_state_dict(torch.load(sd / "bert_checkpoint.pt", weights_only=True)); be.eval()
        m = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        m.load_state_dict(torch.load(sd / "checkpoint.pt", weights_only=True)); m.eval()
        ensemble.append((m, be))
        fps.append(hashlib.sha256(Path(sd / "checkpoint.pt").read_bytes()).hexdigest()[:12])
        mean_embs[seed] = train_mean_text_embedding(
            be, np.asarray(att["train"]["text_bert"]).astype(np.int64),
            state["train_content"].astype(bool), device)
        print(f"seed {seed} mean-emb ready", flush=True)
    del att

    samples = load_att4()
    reports, rho_acc = [], {p: {m: [] for m in MODS}
                            for p in ("mz", "mm", "zm")}
    for s in samples:
        st = prepare_sample(s, stats)
        tc = target[f"{s.n:02d}"]
        avail = {m: torch.from_numpy(st.o[m]).to(device) for m in MODS}
        per_base = {}
        for kind in BASELINES:
            ig_sum, d_f = {m: {o: [] for o in ("cls", "reg")} for m in MODS}, {"cls": [], "reg": []}
            per_seed_zero = {}
            for (model, be), seed in zip(ensemble, (1, 2, 3)):
                with torch.no_grad():
                    x_feats = {"text": be(torch.from_numpy(st.text_bert).to(device)),
                               "audio": torch.from_numpy(st.audio).to(device),
                               "vision": torch.from_numpy(st.vision).to(device)}
                    base_feats = {"text": text_baseline(kind, st, be, device, mean_embs[seed]),
                                  "audio": torch.from_numpy(
                                      av_baseline(kind, "audio", stats)[None]).to(device),
                                  "vision": torch.from_numpy(
                                      av_baseline(kind, "vision", stats)[None]).to(device)}
                ep = endpoints(model, st, device, tc, base_feats, x_feats, avail)
                d_f["cls"].append(ep["cls"]); d_f["reg"].append(ep["reg"])
                ig = ig_modality(model, st, device, tc, base_feats, x_feats, avail)
                for m in MODS:
                    for o in ("cls", "reg"):
                        ig_sum[m][o].append(ig[m][o])
                    # 审计修正：逐 seed（非均值后）检查阻断位归因
                    per_seed_zero[m] = max(per_seed_zero.get(m, 0.0), float(max(
                        np.abs(ig[m][o][~st.o[m][0]]).max() for o in ("cls", "reg"))))
            per_base[kind] = {
                "ig": {m: {o: np.mean(ig_sum[m][o], axis=0).tolist() for o in ("cls", "reg")}
                       for m in MODS},
                "dF": {o: float(np.mean(d_f[o])) for o in ("cls", "reg")}}
            # ΣIG 完备性（该基线，逐样本值直接进 gate）
            for o in ("cls", "reg"):
                tot = sum(np.mean(ig_sum[m][o], axis=0).sum() for m in MODS)
                per_base[kind][f"gap_{o}"] = float(abs(tot - per_base[kind]["dF"][o]))
            # 证据边界：不可用位（非 content ∪ content∧o=0）IG 应恒零（架构级，逐 seed）
            per_base[kind]["zero_ig_outside_evidence"] = per_seed_zero

        # 敏感性 Spearman（content 位）
        c = st.content[0]          # 只比 content 位（结构位恒 0，纳入只会制造平秩）
        for m in MODS:
            r = {k: np.array(per_base[k]["ig"][m]["cls"])[c] for k in BASELINES}
            for tag, (a, b) in (("mz", ("missing", "zero")), ("mm", ("missing", "mean")),
                                ("zm", ("zero", "mean"))):
                if np.allclose(r[a], r[b]):
                    rho = 1.0
                elif np.ptp(r[a]) == 0 or np.ptp(r[b]) == 0:   # 常数向量（如 #13 vision）
                    rho = float("nan")
                else:
                    rho = float(spearmanr(r[a], r[b]).statistic)
                rho_acc[tag][m].append(rho)

        # 词聚合（主基线，分类输出）：P1 build_grid 映射 + alpha（词内均匀）
        wg = p1_word_grid(s.raw_text)
        words = aggregate_words(np.array(per_base["missing"]["ig"]["text"]["cls"]), wg,
                                st.content)
        per_base["words"] = {"n_mapped": words["n_words"],
                             "n_total_retained": wg["n_words_retained"],
                             "n_total_all": wg["n_words_all"],
                             "uncovered": wg["n_words_all"] - wg["n_words_retained"],
                             "word_attr": {str(k): round(v, 6) for k, v in
                                           sorted(words["words"].items(),
                                                  key=lambda kv: -abs(kv[1]))[:10]},
                             "word_texts": {str(k): t for k, t in
                                            list(wg["word_texts"].items())[:24]},
                             "n_pieces": {str(k): v for k, v in
                                          list(wg["n_pieces"].items())[:24]},
                             "structural_leak": round(words["structural_leak"], 6),
                             "alignment": "build_grid input_ids ≡ 附件4 存储网格（check 3）"}
        reports.append({"sample_id": f"{s.n:02d}", "target_cls": tc,
                        "vision_natural_missing": not st.o["vision"][0].any(),
                        "baselines": per_base})
        print(f"  #{s.n:02d} done (gap_cls={per_base['missing']['gap_cls']:.2e})", flush=True)

    # ---- 1. 积分误差（主基线逐样本门槛；其余逐样本报告不设门槛）----
    def per_sample(k):
        rows = []
        for r in reports:
            b = r["baselines"][k]
            for o in ("cls", "reg"):
                gap, df = b[f"gap_{o}"], abs(b["dF"][o])
                rows.append({"id": r["sample_id"], "out": o, "gap": round(gap, 6),
                             "dF": round(df, 6),
                             "rel": round(gap / df, 4) if df > 1e-12 else None,
                             "pass": gap <= max(0.05 * df, 1e-3)})
        return rows
    ms = per_sample("missing")
    fails = [x for x in ms if not x["pass"]]
    add("integration_error_primary_per_sample", not fails,
        {"rule": "|ΣIG−ΔF| ≤ max(5%·|ΔF|, 1e-3)，逐样本逐输出",
         "n_checks": len(ms), "n_pass": len(ms) - len(fails),
         "max_rel": max((x["rel"] or 0) for x in ms), "fails": fails,
         "others_report": {k: {"n_pass": sum(x["pass"] for x in per_sample(k)),
                               "max_rel": max((x["rel"] or 0) for x in per_sample(k))}
                           for k in ("zero", "mean")}})

    # ---- 2. 证据边界（架构级，全基线×输出×模态）+ #13 全模态实例 ----
    worst = 0.0
    for r in reports:
        for k in BASELINES:
            worst = max(worst, max(r["baselines"][k]["zero_ig_outside_evidence"].values()))
    add("zero_ig_outside_evidence", worst <= 1e-6,
        {"max_abs_ig_on_blocked_positions": worst, "tol": 1e-6,
         "scope": "3 基线 × 2 输出 × 3 模态 × 20 样本，不可用位 = 非content ∪ content∧o=0"})
    r13 = next(r for r in reports if r["sample_id"] == "13")
    v13 = [r13["baselines"][k]["ig"]["vision"][o] for k in BASELINES for o in ("cls", "reg")]
    add("ig_zero_vision_13", r13["vision_natural_missing"]
        and all(np.max(np.abs(v)) == 0.0 for v in v13),
        {"max_abs_ig_vision": float(max(np.max(np.abs(v)) for v in v13)),
         "note": "#13 = 证据边界检查的全模态自然实例"})

    # ---- 3. P1 网格对齐与词覆盖 ----
    n_align = 0
    for smp in samples:
        g = build_grid(smp.raw_text)
        stored = smp.text_bert[0, 0].astype(int)
        L = int((stored != 0).sum())
        n_align += int(np.array_equal(np.asarray(g["input_ids"])[:L], stored[:L]))
    cov = [r["baselines"]["words"]["uncovered"] for r in reports]
    add("p1_grid_alignment", n_align == len(samples),
        {"aligned": n_align, "total": len(samples),
         "uncovered_words_total": int(sum(cov)),
         "note": "build_grid ≡ 存储网格逐 token 对齐（含 #07/#18 截断）；"
                 "词聚合 = P1 alpha 映射 + 质量守恒求和；截断整词 uncovered"})

    # ---- 4. 基线敏感性 ----
    def _clean(v):
        return [x for x in v if not np.isnan(x)]
    rho_mean = {tag: {m: round(float(np.mean(_clean(v))), 4) for m, v in d.items()}
                for tag, d in rho_acc.items()}
    rho_min = {tag: {m: round(float(np.min(_clean(v))), 4) for m, v in d.items()}
               for tag, d in rho_acc.items()}
    add("baseline_sensitivity", True,
        {"spearman_mean": rho_mean, "spearman_min": rho_min,
         "note": "a/v 的 missing≡mean（z 空间重合）→ ρ=1 为构造性；"
                 "实质对比在 text 三基线与 missing↔zero"})

    # ---- 落盘 ----
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "MRFN+ (3-seed ensemble)", "checkpoint_sha12": fps,
        "config": {"n_steps": N_STEPS, "rule": "midpoint",
                   "baselines": list(BASELINES),
                   "integration": f"中点 Riemann α=(k-0.5)/{N_STEPS}",
                   "baseline_semantics": (
                       "全缺失特征值基线，avail 固定自然观测 o（条件特征归因）。"
                       "与 v(∅)（Shapley 联盟级反事实，avail=0）严格区分："
                       "avail=0 时架构阻断 ∂F/∂x≡0，实测 IG 全零（退化平凡博弈）"),
                   "differentiable_layer": {
                       "text": "BERT last_hidden_state (50,768)，冻结编码器 eval；"
                               "α 路径作用在连续嵌入空间，绝不对 token ID 数值插值",
                       "audio/vision": "z-score 标准化连续特征"},
                   "word_aggregation": "P1 build_grid alpha（词内均匀 1/n_w）+ "
                                       "质量守恒求和 A_w=Σ_{j∈w}a_j",
                   "target_cls_source": "M2 集成预测类（runs/p3/m2/shapley_att4.json）"},
        "reports": reports, "checks": checks,
        "all_pass": all(c["ok"] for c in checks)}
    (OUT / "ig_att4.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\ncache → {OUT / 'ig_att4.json'}")
    print("ALL PASS" if payload["all_pass"] else "FAILED")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
