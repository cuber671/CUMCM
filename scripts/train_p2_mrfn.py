"""P2 M4：MRFN——B3 + 缺失状态编码 + 可靠性门控，掩码库混合训练。

训练（契约 §2）：每 step 抽分量 A35/B40/C15/D10 → 该分量条目均匀抽一 → 均匀抽实例 k，
以该实例的 b 行作用于整个 batch；D(clean) 即 a=o。b_t≠0 的 batch 文本按 M1-D 机制
在线 [MASK] 重编码；audio/vision 用 zscored 值（缺失位零占位，模型内部门控处理）。
评测：clean + 掩码库 31 条（复用 M3 的 eval_masked）。
门控验证（§8.4，H5 为待验证假设，只报告不断言）：
  ① g–缺失率响应曲线（mcar {t,a,v} × 率 → 自身 g_m，Spearman）；
  ② g–LOO 贡献：I_{i,m} = p_true(clean) − p_true(去模态 m)，与 g_{i,m} 的
     Spearman + 1000 次 bootstrap 95% CI。
产物：runs/p2/mrfn/{summary.json, MRFN_seed*/...}。硬门槛：3 seed、clean Acc>49.19%、
gates 合法、LOO 计算齐备；鲁棒性改善为报告项。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_p2_b0 import MAJORITY_ACC, ensure_text_cache, metrics  # noqa: E402
from train_p2_ladder import (build_tensors, eval_masked, forward_model)  # noqa: E402

from src.p2.data import CONTRACT, load_aligned  # noqa: E402
from src.p2.models import MODEL_REGISTRY, count_params  # noqa: E402
from src.p2.pipeline import load_stats, zscore_reset  # noqa: E402
from src.p2.text_mask import (DEFAULT_MODEL_PATH, encode_text,  # noqa: E402
                              load_frozen_bert, mask_text_tokens)

STATE = ROOT / "runs/p2/data/m1a_state.npz"
LIB = ROOT / "runs/p2/mask_library"
OUTDIR = ROOT / "runs/p2/mrfn"
MODS = ("text", "audio", "vision")
O_KEY = {"text": "o_text", "audio": "o_audio", "vision": "o_vision"}
MIXTURE = {"A": 0.35, "B": 0.40, "C": 0.15, "D": 0.10}


def preload_train_masks(lib_index) -> dict:
    """train split 的 train_A/B/C 条目实例预载：{comp: [entry: [k: {mod: b_uint8}]]}。"""
    comps = {"A": [], "B": [], "C": []}
    for e in lib_index["entries"]:
        if e["split"] != "train":
            continue
        for comp in ("A", "B", "C"):
            if f"train_{comp}" in e["tags"]:
                z = np.load(LIB / e["path"])
                inst = [{mm: z[f"b_{k}_{letter}"] for mm, letter in
                         (("text", "t"), ("audio", "a"), ("vision", "v"))
                         if f"b_{k}_{letter}" in z.files}
                        for k in range(e["k"])]
                comps[comp].append(inst)
    return comps


def make_batch_avail(rng, idx_np, idx, tr, comps):
    """按混合比抽分量/条目/实例 → batch 的可用性 a（bool GPU，已切到 batch）+ b_t。"""
    comps_list = list(MIXTURE)
    comp = comps_list[int(rng.choice(4, p=[MIXTURE[c] for c in comps_list]))]
    if comp == "D":
        return {m: tr[f"o_{m}"][idx] for m in MODS}, None, comp
    entry = comps[comp][int(rng.integers(len(comps[comp])))]
    b_map = entry[int(rng.integers(len(entry)))]
    avail = {}
    b_t = None
    for m in MODS:
        letter = {"text": "t", "audio": "a", "vision": "v"}[m]
        if m in b_map:
            b = torch.from_numpy(b_map[m][idx_np]).to(tr["content"].device)
            avail[m] = tr[f"o_{m}"][idx] & ~b.bool()
            if m == "text":
                b_t = b
        else:
            avail[m] = tr[f"o_{m}"][idx]
    return avail, b_t, comp


def train_one_seed(seed, tensors, tb_train_np, comps, args, device, out_dir: Path, bert):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MODEL_REGISTRY["MRFN"](dropout=args.dropout).to(device)
    tr, va = tensors["train"], tensors["valid"]
    n = tr["content"].shape[0]
    counts = torch.bincount(tr["y_cls"], minlength=3).float()
    w = (counts.sum() / (3 * counts)).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_state, bad = -1.0, None, 0
    log, comp_counts = [], {"A": 0, "B": 0, "C": 0, "D": 0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(seed * 100000 + epoch)
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx_np = perm[i:i + args.batch_size]
            idx = torch.from_numpy(idx_np).long().to(device)
            avail, b_t, comp = make_batch_avail(rng, idx_np, idx, tr, comps)
            comp_counts[comp] += 1
            if b_t is not None and bool(b_t.any()):
                ids = mask_text_tokens(tb_train_np[idx_np], b_t.cpu().numpy(), CONTRACT)
                tf = torch.from_numpy(encode_text(ids, bert, CONTRACT,
                                                  batch_size=len(idx_np))).to(device)
            else:
                tf = tr["text"][idx]
            feats = {"text": tf, "audio": tr["audio"][idx], "vision": tr["vision"][idx]}
            out = model(feats, tr["content"][idx], avail)
            loss = ce(out["logits"], tr["y_cls"][idx]) + \
                args.lambda_l1 * (out["reg"] - tr["y_reg"][idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        vm = eval_clean(model, va, device)
        log.append({"epoch": epoch, "train_loss": round(tot / nb, 6), **vm})
        if vm["S"] > best:
            best, bad = vm["S"], 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    sd = out_dir / f"MRFN_seed{seed}"
    sd.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, sd / "checkpoint.pt")
    final = eval_clean(model, va, device)
    return {"model": "MRFN", "seed": seed, "param_count": count_params(model),
            "epochs_run": len(log),
            "best_epoch": int(np.argmax([x["S"] for x in log]) + 1),
            "best_S": best, "valid_metrics": final, "per_epoch": log,
            "mixture_realized": comp_counts}


@torch.no_grad()
def eval_clean(model, va, device) -> dict:
    model.eval()
    logits, regs, gates = [], [], []
    for i in range(0, va["content"].shape[0], 256):
        idx = slice(i, i + 256)
        feats = {k: va[k][idx] for k in MODS}
        av = {m: va[f"o_{m}"][idx] for m in MODS}
        o = model(feats, va["content"][idx], av)
        logits.append(o["logits"]); regs.append(o["reg"]); gates.append(o["gates"])
    lo, rg, g = torch.cat(logits), torch.cat(regs), torch.cat(gates)
    m = metrics(va["y_cls"].cpu().numpy(), va["y_reg"].cpu().numpy(),
                lo.cpu().numpy(), rg.cpu().numpy())
    m["gate_mean"] = [round(float(x), 6) for x in g.mean(0)]
    m["gate_std_over_samples"] = [round(float(x), 6) for x in g.std(0)]
    return m


@torch.no_grad()
def gate_analyses(model, t_va, va_np, lib_index, device) -> dict:
    """① g–缺失率响应（mcar 自模态曲线）；② g–LOO 贡献 Spearman + bootstrap CI。"""
    model.eval()
    out = {"gate_curve": {}, "loo": {}}
    # ① 曲线：mcar {t,a,v} × 率，取自模态 g
    for m, letter in (("text", "t"), ("audio", "a"), ("vision", "v")):
        rates, gmeans = [], []
        for e in lib_index["entries"]:
            if e["split"] != "valid" or e["mechanism"] != "mcar" or e["modalities"] != letter:
                continue
            z = np.load(LIB / e["path"])
            gs = []
            for k in range(e["k"]):
                a_map = {mm: t_va[f"o_{mm}"] for mm in MODS}
                b = torch.from_numpy(z[f"b_{k}_{letter}"]).to(device)
                a_map[m] = t_va[f"o_{m}"] & ~b.bool()
                feats = {mm: (t_va[mm] if mm != m else
                              torch.where(a_map[m][..., None], t_va[mm],
                                          torch.zeros_like(t_va[mm])))
                         for mm in MODS}
                # 文本 b≠0 时需重编码
                if m == "text" and bool(b.any()):
                    ids = mask_text_tokens(va_np["text_bert"], z[f"b_{k}_t"], CONTRACT)
                    feats["text"] = torch.from_numpy(
                        encode_text(ids, model_cache["bert"], CONTRACT,
                                    batch_size=256)).to(device)
                o = model(feats, t_va["content"], a_map)
                gs.append(o["gates"][:, MODS.index(m)].mean().item())
            rates.append(e["rate"]); gmeans.append(float(np.mean(gs)))
        order = np.argsort(rates)
        r_arr = np.array(rates)[order]; g_arr = np.array(gmeans)[order]
        out["gate_curve"][m] = {"rates": r_arr.tolist(),
                                "g_self_mean": [round(x, 6) for x in g_arr],
                                "spearman_rate": round(float(spearmanr(r_arr, g_arr).statistic), 6)}
    # ② LOO 贡献 vs clean gate
    logits, probs_true = [], None
    feats0 = {m: t_va[m] for m in MODS}
    av0 = {m: t_va[f"o_{m}"] for m in MODS}
    o0 = model(feats0, t_va["content"], av0)
    p = torch.softmax(o0["logits"], dim=-1)
    p_true0 = p[torch.arange(len(p)), va_np["y_cls"]].cpu().numpy()
    g0 = o0["gates"].cpu().numpy()
    for mi, m in enumerate(MODS):
        a_loo = {mm: t_va[f"o_{mm}"] for mm in MODS}
        a_loo[m] = torch.zeros_like(a_loo[m])           # 整模态不可用
        feats = {mm: (torch.where(a_loo[mm][..., None], t_va[mm],
                                  torch.zeros_like(t_va[mm])) if mm != m else t_va[mm])
                 for mm in MODS}
        # 被去模态的特征全零（文本不再重编码——content 位全被门控关闭）
        om = model(feats, t_va["content"], a_loo)
        pm = torch.softmax(om["logits"], dim=-1)
        p_true_m = pm[torch.arange(len(pm)), va_np["y_cls"]].cpu().numpy()
        I = p_true0 - p_true_m
        rho = float(spearmanr(I, g0[:, mi]).statistic)
        rng = np.random.default_rng(2026)
        n = len(I)
        boots = []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            boots.append(spearmanr(I[idx], g0[idx, mi]).statistic)
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out["loo"][m] = {"spearman": round(rho, 6),
                         "ci95": [round(x, 6) for x in ci],
                         "mean_contribution": round(float(I.mean()), 6)}
    return out


model_cache = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lambda-l1", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from-results", action="store_true",
                    help="跳过训练，从已有 MRFN_seed*/result.json 重建汇总与门控分析")
    args = ap.parse_args()
    out_dir = OUTDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device={device}")

    att = load_aligned()
    del att["test"]
    state = np.load(STATE)
    tensors = build_tensors(att, state, device)
    tb_train_np = np.asarray(att["train"]["text_bert"])
    va_np = {"y_cls": tensors["valid"]["y_cls"].cpu().numpy(),
             "y_reg": tensors["valid"]["y_reg"].cpu().numpy(),
             "text_bert": np.asarray(att["valid"]["text_bert"])}
    lib_index = json.loads((LIB / "library_index.json").read_text())
    comps = preload_train_masks(lib_index)
    bert = load_frozen_bert(DEFAULT_MODEL_PATH, device=device)
    model_cache["bert"] = bert
    print(f"库分量条目数: A={len(comps['A'])} B={len(comps['B'])} C={len(comps['C'])}")

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed, checks = [], []
    if args.from_results:
        per_seed = [json.loads((out_dir / f"MRFN_seed{s}" / "result.json").read_text())
                    for s in seeds]
        for s_i, r in zip(seeds, per_seed):
            model = MODEL_REGISTRY["MRFN"](dropout=args.dropout).to(device)
            model.load_state_dict(torch.load(out_dir / f"MRFN_seed{s_i}" / "checkpoint.pt",
                                             weights_only=True))
            model.eval()
            r["gate_analyses"] = gate_analyses(model, tensors["valid"], va_np,
                                               lib_index, device)
        print("已从 result.json 复载 per-seed 结果并重算门控分析")
    else:
        for seed in seeds:
            print(f"== MRFN seed {seed} ==")
            r = train_one_seed(seed, tensors, tb_train_np, comps, args, device, out_dir, bert)
            model = MODEL_REGISTRY["MRFN"](dropout=args.dropout).to(device)
            model.load_state_dict(torch.load(out_dir / f"MRFN_seed{seed}" / "checkpoint.pt",
                                             weights_only=True))
            model.eval()
            entries_out = {}
            for e in [e for e in lib_index["entries"] if e["split"] == "valid"
                      and any(t.startswith("eval") for t in e["tags"])]:
                z = np.load(LIB / e["path"])
                key = f"{e['mechanism']}/{e['modalities']}/rate{e['rate']}/{e['position']}"
                entries_out[key] = eval_masked(model, "MRFN", tensors["valid"], va_np,
                                               e, z, r["best_S"], bert, device)
            r["masked_entries"] = entries_out
            r["gate_analyses"] = gate_analyses(model, tensors["valid"], va_np,
                                               lib_index, device)
            per_seed.append(r)
            (out_dir / f"MRFN_seed{seed}" / "result.json").write_text(
                json.dumps({"config": vars(args), **r}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"  best_epoch={r['best_epoch']} S={r['best_S']} "
                  f"acc={r['valid_metrics']['acc']:.4f} mixture={r['mixture_realized']}")

    accs = [r["valid_metrics"]["acc"] for r in per_seed]
    checks.append({"name": "three_seeds_completed", "ok": len(per_seed) == 3,
                   "detail": {"n": len(per_seed)}})
    checks.append({"name": "clean_acc_above_majority",
                   "ok": all(a > MAJORITY_ACC for a in accs),
                   "detail": {"accs": [round(a, 4) for a in accs]}})
    gates_valid = all(len(r["valid_metrics"]["gate_mean"]) == 3 for r in per_seed)
    checks.append({"name": "gates_valid", "ok": gates_valid,
                   "detail": {"gate_mean": per_seed[0]["valid_metrics"]["gate_mean"]}})
    checks.append({"name": "loo_computed",
                   "ok": all(len(r["gate_analyses"]["loo"]) == 3 for r in per_seed),
                   "detail": {m: per_seed[0]["gate_analyses"]["loo"][m]["spearman"]
                              for m in MODS}})

    # 报告项（非硬门槛）：与 B3 的 t 缺失鲁棒性对比
    ladder = json.loads((ROOT / "runs/p2/ladder/ladder_summary.json").read_text())
    obs = []
    for key, d in ladder["ladder"]["B3"]["D_S_mean"].items():
        dm = float(np.mean([r["masked_entries"][key]["D_S"] for r in per_seed]))
        obs.append({"entry": key, "B3_D_S": d, "MRFN_D_S": round(dm, 6),
                    "improvement": round(d - dm, 6)})
    gc = {}
    for m in MODS:
        rates = per_seed[0]["gate_analyses"]["gate_curve"][m]["rates"]
        cols = list(zip(*[r["gate_analyses"]["gate_curve"][m]["g_self_mean"]
                          for r in per_seed]))
        gc[m] = {"rates": rates,
                 "g_self_mean": [round(float(np.mean(c)), 6) for c in cols]}
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mixture": MIXTURE,
        "params": per_seed[0]["param_count"],
        "clean": {k: {"mean": round(float(np.mean([r["valid_metrics"][k] for r in per_seed])), 6),
                      "std": round(float(np.std([r["valid_metrics"][k] for r in per_seed])), 6)}
                  for k in ("acc", "macro_f1", "weighted_f1", "mae", "pearson", "S")},
        "masked_D_S_mean": {},
        "gate_curve_seed_mean": gc,
        "loo_seed_mean": {m: {"spearman_mean": round(float(np.mean(
            [r["gate_analyses"]["loo"][m]["spearman"] for r in per_seed])), 6)}
            for m in MODS},
        "checks": checks,
        "observations_vs_B3": obs,
        "all_checks_pass": all(c["ok"] for c in checks),
        "notes": ["H5 门控单调性为待验证假设，仅报告 Spearman/曲线，不作硬断言（契约 §3）",
                  "鲁棒性改善为报告项：若 t 缺失条目 D_S 未改善，需回滚检查门控输入"],
    }
    keys = list(per_seed[0]["masked_entries"].keys())
    summary["masked_D_S_mean"] = {key: round(float(np.mean(
        [r["masked_entries"][key]["D_S"] for r in per_seed])), 6) for key in keys}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    for c in checks:
        print(("✅" if c["ok"] else "❌"), c["name"], c.get("detail", ""))
    tkeys = [k for k in summary["masked_D_S_mean"] if k.startswith("mcar/t")]
    print("t 缺失条目 D_S（MRFN vs B3）：")
    for k in tkeys:
        b3 = next(o["B3_D_S"] for o in obs if o["entry"] == k)
        print(f"  {k}: {summary['masked_D_S_mean'][k]:.4f} vs {b3:.4f}")
    print("ALL PASS" if summary["all_checks_pass"] else "FAILED")
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
