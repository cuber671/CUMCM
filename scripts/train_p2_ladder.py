"""P2 M3：B1→B3 阶梯——参数量表 + 同掩码库对比。

训练配方与 B0 完全一致（固定超参、clean 训练、3 seeds、早停 S），
使"架构阶梯"在单一训练策略下可比（缺失增强训练属 §9 消融，M5 处理）。
评测 = clean valid + 掩码库全部 eval_* 条目（valid，31 条 × K8 实例）：
  audio/vision = zscored 后按 a=o·(1−b) 零填充；
  text = 库实例 b_t 驱动 [MASK] 重编码（M1-D 机制）后按 a_text 零填充。
L2：D_S = (S_clean − S_r) / max(S_clean, ε)，绝对差同时记录。
产物：runs/p2/ladder/{summary.json, <model>_seed<k>/{result.json, checkpoint.pt}}。
退出码全过 = 0。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_p2_b0 import MAJORITY_ACC, ensure_text_cache, metrics  # noqa: E402

from src.p2.data import CONTRACT, load_aligned
from src.p2.models import MODEL_REGISTRY, count_params
from src.p2.pipeline import load_stats, zero_fill, zscore_reset
from src.p2.text_mask import DEFAULT_MODEL_PATH, encode_text, load_frozen_bert, mask_text_tokens

STATE = ROOT / "runs/p2/data/m1a_state.npz"
LIB = ROOT / "runs/p2/mask_library"
OUTDIR = ROOT / "runs/p2/ladder"
MODALITIES = ("text", "audio", "vision")
O_KEY = {"text": "o_text", "audio": "o_audio", "vision": "o_vision"}


def build_tensors(att: dict, state, device: str) -> dict:
    """与 B0 相同的 clean 张量 + 各模态 o（bool，content 域，兼作 clean 可用性）。"""
    stats = load_stats()
    text_all = ensure_text_cache(att, device)
    n_tr = len(att["train"]["id"])
    out = {}
    for split, sl in (("train", slice(0, n_tr)), ("valid", slice(n_tr, None))):
        t = {"content": torch.from_numpy(state[f"{split}_content"]).bool().to(device),
             "y_cls": torch.from_numpy(state[f"{split}_cls"]).long().to(device),
             "y_reg": torch.from_numpy(state[f"{split}_rl"]).float().to(device),
             "text": torch.from_numpy(np.ascontiguousarray(text_all[sl])).to(device)}
        for m in MODALITIES:
            t[f"o_{m}"] = torch.from_numpy(state[f"{split}_{O_KEY[m]}"]).bool().to(device)
        for mod in ("audio", "vision"):
            s = stats[mod]
            xn = zscore_reset(att[split][mod], s["mean"], s["std"],
                              state[f"{split}_{O_KEY[mod]}"].astype(bool), name=f"{split}.{mod}")
            t[mod] = torch.from_numpy(xn).to(device)
        out[split] = t
    return out


def forward_model(model, name, feats, content, avail):
    if name in ("B3", "MRFN"):                  # 可用性约束模型（MRFN 必传）
        return model(feats, content, avail)
    return model(feats, content)


@torch.no_grad()
def predict_l(model, name, t, avail, batch=256):
    model.eval()
    logits, regs = [], []
    for i in range(0, t["content"].shape[0], batch):
        idx = slice(i, i + batch)
        feats = {k: t[k][idx] for k in MODALITIES}
        av = {m: t[f"o_{m}"][idx] for m in MODALITIES}
        o = forward_model(model, name, feats, t["content"][idx], av)
        logits.append(o["logits"]); regs.append(o["reg"])
    return torch.cat(logits), torch.cat(regs)


def train_one_seed(name, seed, tensors, args, device, out_dir: Path):
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
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        g = torch.Generator().manual_seed(seed * 100000 + epoch)
        perm = torch.randperm(n, generator=g).to(device)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            feats = {k: tr[k][idx] for k in MODALITIES}
            av = {m: tr[f"o_{m}"][idx] for m in MODALITIES}
            out = forward_model(model, name, feats, tr["content"][idx], av)
            loss = ce(out["logits"], tr["y_cls"][idx]) + \
                args.lambda_l1 * (out["reg"] - tr["y_reg"][idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
        logits, reg = predict_l(model, name, va, None)
        vm = metrics(va["y_cls"].cpu().numpy(), va["y_reg"].cpu().numpy(),
                     logits.cpu().numpy(), reg.cpu().numpy())
        log.append({"epoch": epoch, "train_loss": round(tot / nb, 6), **vm})
        if vm["S"] > best:
            best, bad = vm["S"], 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    sd = out_dir / f"{name}_seed{seed}"
    sd.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, sd / "checkpoint.pt")
    logits, reg = predict_l(model, name, va, None)
    final = metrics(va["y_cls"].cpu().numpy(), va["y_reg"].cpu().numpy(),
                    logits.cpu().numpy(), reg.cpu().numpy())
    return {"model": name, "seed": seed, "param_count": count_params(model),
            "epochs_run": len(log),
            "best_epoch": int(np.argmax([x["S"] for x in log]) + 1),
            "best_S": best, "valid_metrics": final, "per_epoch": log}


@torch.no_grad()
def eval_masked(model, name, t_va, va_np, entry, z, clean_S, bert, device):
    per_k = []
    for k in range(entry["k"]):
        a_map, need_enc = {}, False
        for m, letter in (("text", "t"), ("audio", "a"), ("vision", "v")):
            key = f"b_{k}_{letter}"
            if key not in z.files:          # 单/双模态组合未抹该模态 → a = o
                a_map[m] = t_va[f"o_{m}"]
                continue
            b = torch.from_numpy(z[key]).to(device)
            a_map[m] = t_va[f"o_{m}"] & ~b
            if m == "text" and bool(b.any()):
                need_enc = True
        feats = {"audio": zero_fill(t_va["audio"], a_map["audio"]),
                 "vision": zero_fill(t_va["vision"], a_map["vision"])}
        if need_enc:
            ids = mask_text_tokens(va_np["text_bert"], z[f"b_{k}_t"], CONTRACT)
            tf = torch.from_numpy(encode_text(ids, bert, CONTRACT, batch_size=256)).to(device)
        else:
            tf = t_va["text"]
        feats["text"] = zero_fill(tf, a_map["text"])
        o = forward_model(model, name, feats, t_va["content"], a_map)
        per_k.append(metrics(va_np["y_cls"], va_np["y_reg"],
                             o["logits"].cpu().numpy(), o["reg"].cpu().numpy()))
    avg = {k: round(float(np.mean([p[k] for p in per_k])), 6) for k in per_k[0]}
    avg["D_S"] = round((clean_S - avg["S"]) / max(clean_S, 1e-6), 6)
    avg["dS_abs"] = round(clean_S - avg["S"], 6)
    return avg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="B1,B2,B3")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lambda-l1", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = OUTDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device={device}")

    att = load_aligned()
    del att["test"]
    state = np.load(STATE)
    tensors = build_tensors(att, state, device)
    va_np = {"y_cls": tensors["valid"]["y_cls"].cpu().numpy(),
             "y_reg": tensors["valid"]["y_reg"].cpu().numpy(),
             "text_bert": np.asarray(att["valid"]["text_bert"])}
    lib_index = json.loads((LIB / "library_index.json").read_text())
    entries = [e for e in lib_index["entries"]
               if e["split"] == "valid" and any(t.startswith("eval") for t in e["tags"])]
    bert = load_frozen_bert(DEFAULT_MODEL_PATH, device=device)
    print(f"评测条目 {len(entries)} 条（期望 31）")

    seeds = [int(s) for s in args.seeds.split(",")]
    names = args.models.split(",")
    table, checks = {}, []
    for name in names:
        model = MODEL_REGISTRY[name](dropout=args.dropout)
        params = count_params(model)
        del model
        per_seed = []
        for seed in seeds:
            print(f"== {name} seed {seed} ==")
            r = train_one_seed(name, seed, tensors, args, device, out_dir)
            per_seed.append(r)
            (out_dir / f"{name}_seed{seed}" / "result.json").write_text(
                json.dumps({"config": vars(args), **r}, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"  best_epoch={r['best_epoch']} S={r['best_S']} "
                  f"acc={r['valid_metrics']['acc']:.4f}")
            # 掩码库评测（最佳权重已载回）
            model = MODEL_REGISTRY[name](dropout=args.dropout).to(device)
            model.load_state_dict(torch.load(out_dir / f"{name}_seed{seed}" / "checkpoint.pt",
                                             weights_only=True))
            model.eval()
            entries_out = {}
            for e in entries:
                z = np.load(LIB / e["path"])
                key = f"{e['mechanism']}/{e['modalities']}/rate{e['rate']}/{e['position']}"
                entries_out[key] = eval_masked(model, name, tensors["valid"], va_np,
                                               e, z, r["best_S"], bert, device)
            r["masked_entries"] = entries_out
        accs = [r["valid_metrics"]["acc"] for r in per_seed]
        checks.append({"name": f"{name}_clean_acc_above_majority",
                       "ok": all(a > MAJORITY_ACC for a in accs),
                       "detail": {"accs": [round(a, 4) for a in accs]}})
        checks.append({"name": f"{name}_masked_grid_complete",
                       "ok": all(len(r.get("masked_entries", {})) == len(entries)
                                 for r in per_seed),
                       "detail": {"entries": len(entries), "seeds": len(per_seed)}})
        table[name] = {
            "param_count": params,
            "clean": {k: {"mean": round(float(np.mean([r["valid_metrics"][k] for r in per_seed])), 6),
                          "std": round(float(np.std([r["valid_metrics"][k] for r in per_seed])), 6)}
                      for k in ("acc", "macro_f1", "weighted_f1", "mae", "pearson", "S")},
            "masked_mean": {},       # seed 均值下的逐条目指标
            "D_S_mean": {},
        }
        keys = list(per_seed[0]["masked_entries"].keys())
        for key in keys:
            ms = {k: float(np.mean([r["masked_entries"][key][k] for r in per_seed]))
                  for k in ("acc", "macro_f1", "mae", "pearson", "S", "D_S")}
            table[name]["masked_mean"][key] = {k: round(v, 6) for k, v in ms.items()}
            table[name]["D_S_mean"][key] = round(float(
                np.mean([r["masked_entries"][key]["D_S"] for r in per_seed])), 6)
        print(f"[{name}] params={params} clean_acc={table[name]['clean']['acc']['mean']} "
              f"mean|D_S|={np.mean([abs(v) for v in table[name]['D_S_mean'].values()]):.4f}")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training": "clean（阶梯架构对比；缺失增强训练属 §9 消融，M5）",
        "eval": "clean + 掩码库 eval_* 条目（valid，K=8 实例指标平均）",
        "ladder": table,
        "checks": checks,
        "all_checks_pass": all(c["ok"] for c in checks),
        "notes": ["同掩码库对比：所有模型评同一批库实例；L2 的 D_S=(S_clean−S_r)/S_clean",
                  "B0 参照：runs/p2/b0/summary.json（Acc 63.19%±0.74，81,284 参数）"],
    }
    (out_dir / "ladder_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in checks:
        print(("✅" if c["ok"] else "❌"), c["name"], c.get("detail", ""))
    print("ALL PASS" if summary["all_checks_pass"] else "FAILED")
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
