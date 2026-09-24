"""P2 M4/M5 训练器：MRFN 及其消融变体、B3-aug、填充基线。

训练模式（--training）：
  mixture  掩码库混合 A35/B40/C15/D10（契约 §2），每 step 抽分量/条目/实例；
  clean    恒为 a=o（无合成缺失）。
填充（--fill）：zero（缺省，零填充；MRFN 系模型另带内部门控，预填充幂等无害）
  / forward（前向填充，§8.5 基线；仅对无内部门控的模型有语义）。
产物：--rundir 指定 runs/p2/mrfn/ 下子目录（缺省 = 根目录，保持 M4 布局）。
硬门槛：3 seed、clean Acc > 49.19%、gates 合法（带门控模型）、LOO 齐备。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_p2_b0 import MAJORITY_ACC, ensure_text_cache, metrics  # noqa: E402
from train_p2_ladder import build_tensors, eval_masked  # noqa: E402

from src.p2.data import CONTRACT, load_aligned, sha256_file  # noqa: E402
from src.p2.models import MODEL_REGISTRY, count_params  # noqa: E402
from src.p2.pipeline import forward_fill, load_stats, zero_fill  # noqa: E402
from src.p2.text_mask import (DEFAULT_MODEL_PATH, encode_text,  # noqa: E402
                              load_frozen_bert, mask_text_tokens)

STATE = ROOT / "runs/p2/data/m1a_state.npz"
LIB = ROOT / "runs/p2/mask_library"
OUTDIR = ROOT / "runs/p2/mrfn"
MODS = ("text", "audio", "vision")
O_KEY = {"text": "o_text", "audio": "o_audio", "vision": "o_vision"}
MIXTURE = {"A": 0.35, "B": 0.40, "C": 0.15, "D": 0.10}
LETTER = {"text": "t", "audio": "a", "vision": "v"}
model_cache = {}
SCHEDULE_SEED = 20260924
TEXT_K0_CACHE = ROOT / "runs/p2/text_encode"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_WEIGHT_SHA = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
SEL_ENTRIES = {"text40": "mcar/t/rate40/random",
               "text80": "mcar/t/rate80/random",
               "joint40": "joint/av/rate40/random"}


def s_select_compute(s_clean, s_t40, s_t80, s_j40):
    """Round 1 鲁棒早停准则（契约 §8 预注册公式）。"""
    return 0.50 * s_clean + 0.25 * s_t40 + 0.15 * s_t80 + 0.10 * s_j40


def build_k0_masked_cache(entry, tb_valid, ids_valid, bert, device):
    """text 条目 k=0 的 [MASK] 重编码缓存（带五元指纹，指纹不符即重建）。"""
    tag = f"valid_{entry['mechanism']}_{entry['modalities']}_rate{entry['rate']}_k0"
    path = TEXT_K0_CACHE / f"{tag}.npy"
    meta_path = path.with_suffix(".meta.json")
    z = np.load(LIB / entry["path"])
    b_t = z["b_0_t"]
    ids_sha = hashlib.sha256("\n".join(ids_valid).encode()).hexdigest()
    lib_sha = sha256_file(LIB / "library_index.json")
    aligned_sha = json.loads((ROOT / "runs/p2/data/m1a_report.json").read_text())[
        "source"]["sha256"]
    meta = {"ids_sha": ids_sha, "lib_index_sha": lib_sha, "aligned_sha": aligned_sha,
            "entry": {"mechanism": entry["mechanism"], "modalities": entry["modalities"],
                      "rate": entry["rate"], "position": entry["position"]}, "k": 0,
            "bert_revision": BERT_REVISION, "bert_weight_sha256": BERT_WEIGHT_SHA}
    if path.exists() and meta_path.exists() \
            and json.loads(meta_path.read_text()) == meta:
        return np.load(path), b_t
    masked = mask_text_tokens(tb_valid, b_t, CONTRACT)
    feats = encode_text(masked, bert, CONTRACT, batch_size=256)
    np.save(path, feats)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return feats, b_t


def preload_train_masks(lib_index):
    """返回 (comps, comp_mods)：分分量条目实例 + 各条目模态串（B 分量加权用）。"""
    comps = {"A": [], "B": [], "C": []}
    comp_mods = {"A": [], "B": [], "C": []}
    for e in lib_index["entries"]:
        if e["split"] != "train":
            continue
        for comp in ("A", "B", "C"):
            if f"train_{comp}" in e["tags"]:
                z = np.load(LIB / e["path"])
                inst = [{mm: z[f"b_{k}_{LETTER[mm]}"] for mm in MODS
                         if f"b_{k}_{LETTER[mm]}" in z.files}
                        for k in range(e["k"])]
                comps[comp].append(inst)
                comp_mods[comp].append(e["modalities"])
    return comps, comp_mods


def make_batch_avail(rng, idx_np, idx, tr, comps, training):
    if training == "clean":
        return {m: tr[f"o_{m}"][idx] for m in MODS}, None, "D"
    names = list(MIXTURE)
    comp = names[int(rng.choice(4, p=[MIXTURE[c] for c in names]))]
    if comp == "D":
        return {m: tr[f"o_{m}"][idx] for m in MODS}, None, comp
    entry = comps[comp][int(rng.integers(len(comps[comp])))]
    b_map = entry[int(rng.integers(len(entry)))]
    avail, b_t = {}, None
    for m in MODS:
        if m in b_map:
            b = torch.from_numpy(b_map[m][idx_np]).to(tr["content"].device)
            avail[m] = tr[f"o_{m}"][idx] & ~b.bool()
            if m == "text":
                b_t = b
        else:
            avail[m] = tr[f"o_{m}"][idx]
    return avail, b_t, comp


def build_feats(t, idx, avail, b_t, fill_fn, tb_np, bert):
    """batch 特征：text（b_t≠0 → 在线 [MASK] 重编码）→ 统一按 fill_fn 填充。"""
    if b_t is not None and bool(b_t.any()):
        ids = mask_text_tokens(tb_np[idx.cpu().numpy()], b_t.cpu().numpy(), CONTRACT)
        tf = torch.from_numpy(encode_text(ids, bert, CONTRACT,
                                          batch_size=len(idx))).to(t["content"].device)
    else:
        tf = t["text"][idx]
    return {m: fill_fn(t[m][idx], avail[m]) if m != "text" else fill_fn(tf, avail[m])
            for m in MODS}


@torch.no_grad()
def eval_robust_entry(model, va, spec, device) -> float:
    """选择条目的 k=0 一次前向 → S。
    spec = {"b": {letter: b_arr}, "feats_text": 被掩码文本的缓存特征或 None}。
    text 条目用缓存特征 + a_text 屏蔽；joint 条目只屏蔽 a/v。"""
    model.eval()
    bmap = spec["b"]
    avail, feats = {}, {}
    for m in MODS:
        letter = LETTER[m]
        b_arr = bmap.get(letter)
        if b_arr is None:
            avail[m] = va[f"o_{m}"]
            if m == "text" and spec.get("feats_text") is not None:
                feats[m] = torch.from_numpy(
                    np.ascontiguousarray(spec["feats_text"])).to(va["content"].device)
            else:
                feats[m] = va[m]
        else:
            b = torch.from_numpy(b_arr).to(va["content"].device)
            avail[m] = va[f"o_{m}"] & ~b.bool()
            if m == "text" and spec.get("feats_text") is not None:
                base = torch.from_numpy(
                    np.ascontiguousarray(spec["feats_text"])).to(va["content"].device)
            else:
                base = va[m]
            feats[m] = zero_fill(base, avail[m])
    o = model(feats, va["content"], avail)
    return metrics(va["y_cls"].cpu().numpy(), va["y_reg"].cpu().numpy(),
                   o["logits"].cpu().numpy(), o["reg"].cpu().numpy())["S"]


def epoch_assignment(rng, n):
    """每 epoch 严格 A35/B40/C15/D10（round 计数，D 找平），shuffle 后顺序切块。
    返回 (assign, perm)：assign 为逐样本分量，perm 供批顺序（调度种子驱动，跨模型共享）。"""
    counts = {"A": int(round(0.35 * n)), "B": int(round(0.40 * n)),
              "C": int(round(0.15 * n))}
    counts["D"] = n - sum(counts.values())
    assign = np.empty(n, dtype="<U1")
    perm = rng.permutation(n)
    pos = 0
    for c in ("A", "B", "C", "D"):
        assign[perm[pos:pos + counts[c]]] = c
        pos += counts[c]
    return assign, perm


def make_per_sample_avail(rng, idx_np, idx, tr, comps, assign, w_b):
    """逐样本独立抽分量/条目/实例（epoch 内严格配比）；返回 batch avail + b_t 行堆叠。"""
    b_stacks = {m: [] for m in MODS}
    comps_used = []
    for i_global in idx_np:
        c = assign[i_global]
        comps_used.append(c)
        if c == "D":
            for m in MODS:
                b_stacks[m].append(np.zeros(50, dtype=np.uint8))
            continue
        entries = comps[c]
        if c == "B":
            ei = int(rng.choice(len(entries), p=w_b))
        else:
            ei = int(rng.integers(len(entries)))
        inst = entries[ei][int(rng.integers(len(entries[ei])))]   # k ∈ [0, K=8)
        for m in MODS:
            letter = LETTER[m]
            row = inst[letter][i_global] if letter in inst \
                else np.zeros(50, dtype=np.uint8)
            b_stacks[m].append(row)
    avail = {}
    b_t = None
    for m in MODS:
        b = torch.from_numpy(np.stack(b_stacks[m])).to(tr["content"].device)
        if m == "text":
            b_t = b
        avail[m] = tr[f"o_{m}"][idx] & ~b.bool()
    return avail, b_t, comps_used


def train_one_seed(name, seed, tensors, tb_train_np, comps, comp_mods, args, device,
                   out_dir: Path, bert, fill_fn, sel_specs=None):  # noqa: C901
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MODEL_REGISTRY[name](dropout=args.dropout).to(device)
    tr, va = tensors["train"], tensors["valid"]
    n = tr["content"].shape[0]
    counts = torch.bincount(tr["y_cls"], minlength=3).float()
    w = (counts.sum() / (3 * counts)).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_state, bad = -1.0, None, 0
    log, comp_counts = [], {"A": 0, "B": 0, "C": 0, "D": 0}
    aug_sample = getattr(args, "aug_sampling", "batch") == "sample"
    w_b = None
    if aug_sample:
        w_b = np.array([2.0 if "t" in md else 1.0
                        for md in comp_mods["B"]], dtype=np.float64)
        w_b /= w_b.sum()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if aug_sample:
            sched_rng = np.random.default_rng([SCHEDULE_SEED, epoch])
            assign, perm_np = epoch_assignment(sched_rng, n)
        else:
            rng = np.random.default_rng(seed * 100000 + epoch)
            perm_np = rng.permutation(n)
        idx_all = torch.from_numpy(perm_np).long().to(device)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx_np = perm_np[i:i + args.batch_size]
            idx = idx_all[i:i + args.batch_size]
            if aug_sample:
                avail, b_t, comps_used = make_per_sample_avail(
                    sched_rng, idx_np, idx, tr, comps, assign, w_b)
                for c, v in Counter(comps_used).items():
                    comp_counts[c] += v
            else:
                avail, b_t, comp = make_batch_avail(rng, idx_np, idx, tr, comps,
                                                    args.training)
                comp_counts[comp] += 1
            feats = build_feats(tr, idx, avail, b_t, fill_fn, tb_train_np, bert)
            out = model(feats, tr["content"][idx], avail)
            loss = ce(out["logits"], tr["y_cls"][idx]) + \
                args.lambda_l1 * (out["reg"] - tr["y_reg"][idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        vm = eval_clean(model, va, device)
        if sel_specs is not None:
            s_t40 = eval_robust_entry(model, va, sel_specs["text40"], device)
            s_t80 = eval_robust_entry(model, va, sel_specs["text80"], device)
            s_j40 = eval_robust_entry(model, va, sel_specs["joint40"], device)
            s_sel = s_select_compute(vm["S"], s_t40, s_t80, s_j40)
        else:
            s_sel = vm["S"]
        log.append({"epoch": epoch, "train_loss": round(tot / nb, 6),
                    "S_select": round(s_sel, 6), **vm})
        if s_sel > best:
            best, best_epoch, bad = s_sel, epoch, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    sd = out_dir / f"{name}_seed{seed}"
    sd.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, sd / "checkpoint.pt")
    final = eval_clean(model, va, device)
    return {"model": name, "seed": seed, "param_count": count_params(model),
            "epochs_run": len(log), "selection": getattr(args, "selection", "clean"),
            "best_epoch": best_epoch,
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
        logits.append(o["logits"]); regs.append(o["reg"])
        gates.append(o.get("gates"))
    lo, rg = torch.cat(logits), torch.cat(regs)
    m = metrics(va["y_cls"].cpu().numpy(), va["y_reg"].cpu().numpy(),
                lo.cpu().numpy(), rg.cpu().numpy())
    if gates[0] is not None:
        g = torch.cat(gates)
        m["gate_mean"] = [round(float(x), 6) for x in g.mean(0)]
        m["gate_std_over_samples"] = [round(float(x), 6) for x in g.std(0)]
    return m


@torch.no_grad()
def gate_analyses(model, t_va, va_np, lib_index, device) -> dict:
    out = {"gate_curve": {}, "loo": {}}
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
                feats = {mm: zero_fill(t_va[mm], a_map[mm]) for mm in MODS}
                if m == "text" and bool(b.any()):
                    ids = mask_text_tokens(va_np["text_bert"], z[f"b_{k}_t"], CONTRACT)
                    feats["text"] = torch.from_numpy(
                        encode_text(ids, model_cache["bert"], CONTRACT,
                                    batch_size=256)).to(device)
                o = model(feats, t_va["content"], a_map)
                gs.append(o["gates"][:, MODS.index(m)].mean().item())
            rates.append(e["rate"]); gmeans.append(float(np.mean(gs)))
        order = np.argsort(rates)
        r_arr, g_arr = np.array(rates)[order], np.array(gmeans)[order]
        out["gate_curve"][m] = {"rates": r_arr.tolist(),
                                "g_self_mean": [round(x, 6) for x in g_arr],
                                "spearman_rate": round(
                                    float(spearmanr(r_arr, g_arr).statistic), 6)}
    feats0 = {m: t_va[m] for m in MODS}
    av0 = {m: t_va[f"o_{m}"] for m in MODS}
    o0 = model(feats0, t_va["content"], av0)
    p0 = torch.softmax(o0["logits"], dim=-1)
    p_true0 = p0[torch.arange(len(p0)), va_np["y_cls"]].cpu().numpy()
    g0 = o0["gates"].cpu().numpy()
    for mi, m in enumerate(MODS):
        a_loo = {mm: t_va[f"o_{mm}"] for mm in MODS}
        a_loo[m] = torch.zeros_like(a_loo[m])
        feats = {mm: zero_fill(t_va[mm], a_loo[mm]) for mm in MODS}
        om = model(feats, t_va["content"], a_loo)
        pm = torch.softmax(om["logits"], dim=-1)
        p_true_m = pm[torch.arange(len(pm)), va_np["y_cls"]].cpu().numpy()
        I = p_true0 - p_true_m
        rho = float(spearmanr(I, g0[:, mi]).statistic)
        rng = np.random.default_rng(2026)
        boots = [spearmanr(I[idx], g0[idx, mi]).statistic
                 for idx in (rng.integers(0, len(I), len(I)) for _ in range(1000))]
        out["loo"][m] = {"spearman": round(rho, 6),
                         "ci95": [round(float(np.percentile(boots, q)), 6)
                                  for q in (2.5, 97.5)],
                         "mean_contribution": round(float(I.mean()), 6)}
    return out


def agg(key, per_seed):
    vals = [r["valid_metrics"][key] for r in per_seed]
    return {"mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals)), 6)}


def run(args) -> int:
    out_dir = OUTDIR if not args.rundir else OUTDIR / args.rundir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device={device} model={args.model} training={args.training} "
          f"fill={args.fill} rundir={args.rundir}", flush=True)

    att = load_aligned()
    del att["test"]
    state = np.load(STATE)
    tensors = build_tensors(att, state, device)
    va_np = {"y_cls": tensors["valid"]["y_cls"].cpu().numpy(),
             "y_reg": tensors["valid"]["y_reg"].cpu().numpy(),
             "text_bert": np.asarray(att["valid"]["text_bert"])}
    lib_index = json.loads((LIB / "library_index.json").read_text())
    fill_fn = forward_fill if args.fill == "forward" else zero_fill
    has_gates = args.model.startswith("MRFN") and args.model != "MRFN_noGate"

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = []
    have_results = all((out_dir / f"{args.model}_seed{s}" / "result.json").exists()
                       for s in seeds)
    if args.from_results or have_results:      # 断点续跑：已有结果直接复载
        if not args.from_results:
            print("检测到已完成的结果，跳过训练直接复载", flush=True)
        per_seed = [json.loads((out_dir / f"{args.model}_seed{s}" / "result.json").read_text())
                    for s in seeds]
        if has_gates:
            if "bert" not in model_cache:       # 续跑路径未经过训练分支，需补载
                model_cache["bert"] = load_frozen_bert(DEFAULT_MODEL_PATH, device=device)
            for s_i, r in zip(seeds, per_seed):
                model = MODEL_REGISTRY[args.model](dropout=args.dropout).to(device)
                model.load_state_dict(torch.load(
                    out_dir / f"{args.model}_seed{s_i}" / "checkpoint.pt",
                    weights_only=True))
                model.eval()
                r["gate_analyses"] = gate_analyses(model, tensors["valid"], va_np,
                                                   lib_index, device)
        print("已从 result.json 复载 per-seed 结果", flush=True)
    else:
        tb_train_np = np.asarray(att["train"]["text_bert"])
        comps, comp_mods = preload_train_masks(lib_index)
        bert = load_frozen_bert(DEFAULT_MODEL_PATH, device=device)
        model_cache["bert"] = bert
        sel_specs = None
        if getattr(args, "selection", "clean") == "robust":
            by_key = {f"{e['mechanism']}/{e['modalities']}/rate{e['rate']}/{e['position']}": e
                      for e in lib_index["entries"] if e["split"] == "valid"}
            ids_valid = list(att["valid"]["id"])
            tb_valid = np.asarray(att["valid"]["text_bert"])
            sel_specs = {}
            for tag_, key in SEL_ENTRIES.items():
                e = by_key[key]
                if tag_ == "joint40":
                    z = np.load(LIB / e["path"])
                    sel_specs[tag_] = {"b": {"a": z["b_0_a"], "v": z["b_0_v"]},
                                       "feats_text": None}
                else:
                    feats, b_t = build_k0_masked_cache(e, tb_valid, ids_valid,
                                                       bert, device)
                    sel_specs[tag_] = {"b": {"t": b_t}, "feats_text": feats}
            print("鲁棒早停已启用（契约 §8 Round 1 公式，k=0 选择实例）", flush=True)
        for seed in seeds:
            print(f"== {args.model} seed {seed} ==", flush=True)
            r = train_one_seed(args.model, seed, tensors, tb_train_np, comps, comp_mods, args,
                               device, out_dir, bert, fill_fn, sel_specs)
            model = MODEL_REGISTRY[args.model](dropout=args.dropout).to(device)
            model.load_state_dict(torch.load(
                out_dir / f"{args.model}_seed{seed}" / "checkpoint.pt", weights_only=True))
            model.eval()
            entries_out = {}
            for e in [e for e in lib_index["entries"] if e["split"] == "valid"
                      and any(t.startswith("eval") for t in e["tags"])]:
                z = np.load(LIB / e["path"])
                key = f"{e['mechanism']}/{e['modalities']}/rate{e['rate']}/{e['position']}"
                entries_out[key] = eval_masked(model, args.model, tensors["valid"], va_np,
                                               e, z, r["best_S"], bert, device,
                                               fill_fn=fill_fn)
            r["masked_entries"] = entries_out
            if has_gates:
                r["gate_analyses"] = gate_analyses(model, tensors["valid"], va_np,
                                                   lib_index, device)
            per_seed.append(r)
            (out_dir / f"{args.model}_seed{seed}" / "result.json").write_text(
                json.dumps({"config": vars(args), **r}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"  best_epoch={r['best_epoch']} S={r['best_S']} "
                  f"acc={r['valid_metrics']['acc']:.4f} "
                  f"mixture={r.get('mixture_realized')}", flush=True)

    accs = [r["valid_metrics"]["acc"] for r in per_seed]
    checks = [
        {"name": "three_seeds_completed", "ok": len(per_seed) == 3, "detail": len(per_seed)},
        {"name": "clean_acc_above_majority", "ok": all(a > MAJORITY_ACC for a in accs),
         "detail": {"accs": [round(a, 4) for a in accs]}},
    ]
    if has_gates:
        checks.append({"name": "gates_valid",
                       "ok": all(len(r["valid_metrics"].get("gate_mean", [])) == 3
                                 for r in per_seed),
                       "detail": per_seed[0]["valid_metrics"].get("gate_mean")})
        checks.append({"name": "loo_computed",
                       "ok": all(len(r.get("gate_analyses", {}).get("loo", {})) == 3
                                 for r in per_seed),
                       "detail": {m: per_seed[0]["gate_analyses"]["loo"][m]["spearman"]
                                  for m in MODS}})

    keys = list(per_seed[0].get("masked_entries", {}).keys())
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "training": args.training, "fill": args.fill,
        "params": per_seed[0]["param_count"],
        "clean": {k: agg(k, per_seed) for k in
                  ("acc", "macro_f1", "weighted_f1", "mae", "pearson", "S")},
        "masked_S_mean": {key: round(float(np.mean(
            [r["masked_entries"][key]["S"] for r in per_seed])), 6) for key in keys},
        "masked_D_S_mean": {key: round(float(np.mean(
            [r["masked_entries"][key]["D_S"] for r in per_seed])), 6) for key in keys},
        "checks": checks,
        "all_checks_pass": all(c["ok"] for c in checks),
        "notes": ["z-score 空间中均值填充 ≡ 零填充（标准化后均值=0），不另设实验（§8.5 注）",
                  "H5 门控单调性为待验证假设，仅报告不断言（契约 §3）"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in checks:
        print(("✅" if c["ok"] else "❌"), c["name"], c.get("detail", ""), flush=True)
    print("ALL PASS" if summary["all_checks_pass"] else "FAILED", flush=True)
    return 0 if summary["all_checks_pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MRFN")
    ap.add_argument("--training", default="mixture", choices=["mixture", "clean"])
    ap.add_argument("--fill", default="zero", choices=["zero", "forward"])
    ap.add_argument("--rundir", default=None)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lambda-l1", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--selection", default="clean", choices=["clean", "robust"],
                    help="robust = 契约 §8 Round 1 的 S_select 早停准则")
    ap.add_argument("--aug-sampling", default="batch", choices=["batch", "sample"],
                    help="sample = Round 2 逐样本分量抽样（调度种子独立）")
    ap.add_argument("--schedule-seed", type=int, default=SCHEDULE_SEED)
    ap.add_argument("--from-results", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
