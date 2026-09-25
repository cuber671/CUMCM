"""P3 M3 验收脚本：附件4 全 20 条 IG 位置归因（主基线 missing）+ 三基线敏感性。

检查项：
  1. 积分近似误差（主基线）：分类（M2 集成预测类 logit）与回归分别
     |ΣIG − ΔF| ≤ max(5%·|ΔF|, 1e-3)；zero/mean 基线同式报告（不设门槛）；
  2. #13 vision 自然缺失：IG_vision ≡ 0（全基线、双输出，架构级阻断实证）；
  3. 证据边界：special/padding 不作证据（structural_leak 单列），词聚合覆盖差 = 截断词；
  4. 三基线排序敏感性：content 位 Spearman ρ（missing↔zero / missing↔mean / zero↔mean），
     a/v 的 missing≡mean 为构造性 ρ=1（z 空间重合留痕）；
  5. 可微层级声明 + checkpoint SHA + 配置落盘（M4/M6 复用缓存）。
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
from src.p1.grid import load_tokenizer  # noqa: E402
from src.p3.data import load_att4  # noqa: E402
from src.p3.ig import (BASELINES, N_STEPS, aggregate_words, av_baseline,  # noqa: E402
                       endpoints, ig_modality, text_baseline,
                       train_mean_text_embedding, word_piece_map)
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
    tok = load_tokenizer()

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
            per_base[kind] = {
                "ig": {m: {o: np.mean(ig_sum[m][o], axis=0).tolist() for o in ("cls", "reg")}
                       for m in MODS},
                "dF": {o: float(np.mean(d_f[o])) for o in ("cls", "reg")}}
            # ΣIG 完备性（该基线）
            for o in ("cls", "reg"):
                tot = sum(np.mean(ig_sum[m][o], axis=0).sum() for m in MODS)
                per_base[kind][f"gap_{o}"] = float(abs(tot - per_base[kind]["dF"][o]))

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

        # 词聚合（主基线，分类输出）
        wp = word_piece_map(s.raw_text, tok)
        words = aggregate_words(np.array(per_base["missing"]["ig"]["text"]["cls"]), wp,
                                st.content)
        # 全量分词（不截断）的词数：与 word_ids 同口径（标点/缩写独立成词），
        # whitespace split 会低估词数（曾致 uncovered<0）
        full_wids = tok(s.raw_text, return_offsets_mapping=True).word_ids()
        n_total_words = len({w for w in full_wids if w is not None})
        per_base["words"] = {"n_mapped": words["n_words"],
                             "n_total": n_total_words,
                             "uncovered": n_total_words - words["n_words"],
                             "word_attr": {str(k): round(v, 6) for k, v in
                                           sorted(words["words"].items(),
                                                  key=lambda kv: -abs(kv[1]))[:10]},
                             "structural_leak": round(words["structural_leak"], 6)}
        reports.append({"sample_id": f"{s.n:02d}", "target_cls": tc,
                        "vision_natural_missing": not st.o["vision"][0].any(),
                        "baselines": per_base})
        print(f"  #{s.n:02d} done (gap_cls={per_base['missing']['gap_cls']:.2e})", flush=True)

    # ---- 1. 积分误差（主基线门槛；其余报告）----
    gaps = {k: {o: max(r["baselines"][k][f"gap_{o}"] for r in reports)
                for o in ("cls", "reg")} for k in BASELINES}
    dfs = {k: {o: max(abs(r["baselines"][k]["dF"][o]) for r in reports)
               for o in ("cls", "reg")} for k in BASELINES}
    ok1 = all(gaps["missing"][o] <= max(0.05 * dfs["missing"][o], 1e-3)
              for o in ("cls", "reg"))
    add("integration_error_primary", ok1,
        {"missing": {k: float(v) for k, v in gaps["missing"].items()},
         "dF_scale": {k: float(v) for k, v in dfs["missing"].items()},
         "others_report": {k: gaps[k] for k in ("zero", "mean")},
         "rule": "|ΣIG−ΔF| ≤ max(5%·|ΔF|, 1e-3)"})

    # ---- 2. #13 哑玩家 ----
    r13 = next(r for r in reports if r["sample_id"] == "13")
    v13 = [r13["baselines"][k]["ig"]["vision"][o] for k in BASELINES for o in ("cls", "reg")]
    add("ig_zero_vision_13", r13["vision_natural_missing"]
        and all(np.max(np.abs(v)) == 0.0 for v in v13),
        {"max_abs_ig_vision": float(max(np.max(np.abs(v)) for v in v13))})

    # ---- 3. 证据边界与词覆盖 ----
    leaks = [r["baselines"]["words"]["structural_leak"] for r in reports]
    cov = [r["baselines"]["words"]["uncovered"] for r in reports]
    add("evidence_boundary", all(u >= 0 for u in cov),
        {"structural_leak_mean": round(float(np.mean(np.abs(leaks))), 6),
         "uncovered_words_total": int(sum(cov)),
         "note": "special/padding 不作证据；截断词标记 uncovered"})

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
        "config": {"n_steps": N_STEPS, "baselines": list(BASELINES),
                   "integration": "右端点 Riemann (k/32, k=1..32)",
                   "differentiable_layer": {
                       "text": "BERT last_hidden_state (50,768)，冻结编码器 eval；"
                               "α 路径作用在连续嵌入空间，绝不对 token ID 数值插值",
                       "audio/vision": "z-score 标准化连续特征"},
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
