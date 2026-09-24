"""P2 M2：B0 端到端 clean 训练（逐模态投影→masked_pool→concat→双头 MLP）。

契约 §3/§5 固定超参：AdamW lr 1e-3 / wd 1e-4、batch 32、≤40 epoch、早停 patience 8
（选择指标 S=0.5·U_cls+0.5·U_reg）、dropout 0.3、clip 1.0、λ=1.0、model_seed {1,2,3}；
类权重逆频率 {0:1.170, 1:1.493, 2:0.678}；文本 = text_bert 现场重编码（clean 缓存一次）。
验收：过拟合冒烟 + 3 seed 完成 + clean valid Acc > 49.19%（多数类）+ 指标齐备；
test 全程不碰（加载后即从内存移除）。
产物：runs/p2/b0/seed{k}/{result.json, checkpoint.pt} + summary.json。退出码全过=0。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import CONTRACT, load_aligned
from src.p2.models import B0, count_params
from src.p2.pipeline import load_stats, zscore_reset
from src.p2.text_mask import DEFAULT_MODEL_PATH, encode_text, load_frozen_bert

STATE = ROOT / "runs/p2/data/m1a_state.npz"
TEXT_CACHE = ROOT / "runs/p2/text_encode/clean_text_trainvalid.npy"
OUTDIR = ROOT / "runs/p2/b0"
MAJORITY_ACC = 1670 / 3395
MOD_KEYS = {"text": "o_text", "audio": "o_audio", "vision": "o_vision"}


def ensure_text_cache(att: dict, device: str) -> np.ndarray:
    """clean 文本特征缓存（train+valid，test 不编码不缓存）。"""
    ids = list(att["train"]["id"]) + list(att["valid"]["id"])
    sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    meta_p = TEXT_CACHE.with_suffix(".meta.json")
    if TEXT_CACHE.exists() and meta_p.exists():
        if json.loads(meta_p.read_text()).get("ids_sha") == sha:
            return np.load(TEXT_CACHE)
    print("构建 clean 文本缓存（现场重编码）...")
    model = load_frozen_bert(DEFAULT_MODEL_PATH, device=device)
    tb = np.concatenate([att["train"]["text_bert"], att["valid"]["text_bert"]], axis=0)
    feats = encode_text(tb, model, CONTRACT, batch_size=128)
    del model
    TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(TEXT_CACHE, feats)
    meta_p.write_text(json.dumps({"ids_sha": sha, "n": int(feats.shape[0]),
                                  "created_at": datetime.now(timezone.utc).isoformat()}))
    return feats


def build_tensors(att: dict, state, device: str) -> dict:
    """train/valid 张量（GPU）：标准化 audio/vision（o=0 位归零）+ 文本 + 标签 + content。"""
    stats = load_stats()
    text_all = ensure_text_cache(att, device)
    n_tr = len(att["train"]["id"])
    out = {}
    for split, sl in (("train", slice(0, n_tr)), ("valid", slice(n_tr, None))):
        t = {"content": torch.from_numpy(state[f"{split}_content"]).bool().to(device),
             "y_cls": torch.from_numpy(state[f"{split}_cls"]).long().to(device),
             "y_reg": torch.from_numpy(state[f"{split}_rl"]).float().to(device),
             "text": torch.from_numpy(np.ascontiguousarray(text_all[sl])).to(device)}
        for mod, key in (("audio", "o_audio"), ("vision", "o_vision")):
            s = stats[mod]
            xn = zscore_reset(att[split][mod], s["mean"], s["std"],
                              state[f"{split}_{key}"].astype(bool), name=f"{split}.{mod}")
            t[mod] = torch.from_numpy(xn).to(device)
        out[split] = t
    return out


@torch.no_grad()
def predict(model, t, batch=256):
    model.eval()
    logits, regs = [], []
    for i in range(0, t["content"].shape[0], batch):
        idx = slice(i, i + batch)
        feats = {k: t[k][idx] for k in ("text", "audio", "vision")}
        o = model(feats, t["content"][idx])
        logits.append(o["logits"]); regs.append(o["reg"])
    return torch.cat(logits), torch.cat(regs)


def metrics(y_cls: np.ndarray, y_reg: np.ndarray, logits: np.ndarray, reg: np.ndarray) -> dict:
    pred = logits.argmax(1)
    acc = float((pred == y_cls).mean())
    f1s, supports = [], []
    for c in np.unique(y_cls):
        tp = int(((pred == c) & (y_cls == c)).sum())
        fp = int(((pred == c) & (y_cls != c)).sum())
        fn = int(((pred != c) & (y_cls == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        supports.append(int((y_cls == c).sum()))
    weighted = sum(f * s for f, s in zip(f1s, supports)) / max(sum(supports), 1)
    mae = float(np.abs(reg - y_reg).mean())
    pearson = float(np.corrcoef(reg, y_reg)[0, 1]) if np.std(reg) > 0 else 0.0
    u_mae = min(max(1 - mae / 6, 0.0), 1.0)
    u_p = (pearson + 1) / 2
    u_cls = (acc + float(np.mean(f1s))) / 2
    u_reg = (u_mae + u_p) / 2
    return {"acc": acc, "macro_f1": round(float(np.mean(f1s)), 6),
            "weighted_f1": round(weighted, 6), "mae": round(mae, 6),
            "pearson": round(pearson, 6), "S": round(0.5 * u_cls + 0.5 * u_reg, 6)}


def train_one_seed(seed: int, tensors: dict, args, device: str, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = B0(dropout=args.dropout).to(device)
    tr, va = tensors["train"], tensors["valid"]
    n = tr["content"].shape[0]
    counts = torch.bincount(tr["y_cls"], minlength=3).float()
    w = (counts.sum() / (3 * counts)).to(device)          # 逆频率 = 契约 {1.170,1.493,0.678}
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
            feats = {k: tr[k][idx] for k in ("text", "audio", "vision")}
            out = model(feats, tr["content"][idx])
            loss = ce(out["logits"], tr["y_cls"][idx]) + \
                args.lambda_l1 * (out["reg"] - tr["y_reg"][idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
        logits, reg = predict(model, va)
        vm = metrics(va["y_cls"].cpu().numpy(), va["y_reg"].cpu().numpy(),
                     logits.cpu().numpy(), reg.cpu().numpy())
        log.append({"epoch": epoch, "train_loss": round(tot / nb, 6), **vm})
        if vm["S"] > best:
            best, bad = vm["S"], 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    sd = out_dir / f"seed{seed}"
    sd.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, sd / "checkpoint.pt")
    logits, reg = predict(model, va)
    final = metrics(va["y_cls"].cpu().numpy(), va["y_reg"].cpu().numpy(),
                    logits.cpu().numpy(), reg.cpu().numpy())
    return {"seed": seed, "param_count": count_params(model), "epochs_run": len(log),
            "best_epoch": int(np.argmax([x["S"] for x in log]) + 1),
            "best_S": best, "valid_metrics": final, "per_epoch": log}


def overfit_smoke(tensors: dict, args, device: str) -> dict:
    """32 样本全批 300 epoch：训练 Acc→100% 且 loss 大降（反向传播通路验证）。"""
    torch.manual_seed(99)
    model = B0(dropout=args.dropout).to(device)
    tr = tensors["train"]
    n = 32
    feats = {k: tr[k][:n] for k in ("text", "audio", "vision")}
    content, y, yr = tr["content"][:n], tr["y_cls"][:n], tr["y_reg"][:n]
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for ep in range(300):
        model.train()
        out = model(feats, content)
        loss = ce(out["logits"], y) + (out["reg"] - yr).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep == 0:
            first = float(loss)
        last = float(loss)
    acc = float((model(feats, content)["logits"].argmax(1) == y).float().mean())
    ok = acc == 1.0 and last < first * 0.3
    return {"ok": bool(ok), "first_loss": round(first, 4), "last_loss": round(last, 4),
            "train_acc": acc}


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
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--overfit-smoke", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device={device}")

    att = load_aligned()
    del att["test"]                      # 契约：B0 阶段 test 不读不用
    state = np.load(STATE)
    tensors = build_tensors(att, state, device)
    print("张量就绪：", {k: {kk: tuple(vv.shape) for kk, vv in t.items()} for k, t in tensors.items()})

    checks = []
    if args.overfit_smoke:
        smoke = overfit_smoke(tensors, args, device)
        checks.append({"name": "overfit_smoke", "ok": smoke["ok"], "detail": smoke})
        print("过拟合冒烟:", smoke)

    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"== seed {seed} ==")
        r = train_one_seed(seed, tensors, args, device, out_dir)
        results.append(r)
        print(f"  best_epoch={r['best_epoch']} S={r['best_S']} acc={r['valid_metrics']['acc']}")
        (out_dir / f"seed{seed}" / "result.json").write_text(
            json.dumps({"seed": seed, "config": vars(args) | {"seeds": None}, **r},
                       ensure_ascii=False, indent=1), encoding="utf-8")

    accs = [r["valid_metrics"]["acc"] for r in results]
    checks.append({"name": "three_seeds_completed", "ok": len(results) == 3,
                   "detail": {"n": len(results)}})
    checks.append({"name": "valid_acc_above_majority",
                   "ok": bool(accs) and all(a > MAJORITY_ACC for a in accs),
                   "detail": {"accs": [round(a, 4) for a in accs],
                              "majority": round(MAJORITY_ACC, 4)}})
    checks.append({"name": "metrics_complete",
                   "ok": all(all(k in r["valid_metrics"]
                                 for k in ("macro_f1", "weighted_f1", "mae", "pearson"))
                             for r in results),
                   "detail": "macro-F1 / weighted-F1 / MAE / Pearson"})
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "B0", "contract": "问题二实施步骤.md §3/§5",
        "config": {k: v for k, v in vars(args).items()},
        "per_seed": [{k: r[k] for k in ("seed", "param_count", "epochs_run",
                                        "best_epoch", "best_S", "valid_metrics")}
                     for r in results],
        "aggregate": {k: {"mean": round(float(np.mean([r["valid_metrics"][k] for r in results])), 6),
                          "std": round(float(np.std([r["valid_metrics"][k] for r in results])), 6)}
                      for k in ("acc", "macro_f1", "weighted_f1", "mae", "pearson", "S")},
        "checks": checks, "all_checks_pass": all(c["ok"] for c in checks),
        "notes": ["clean valid Acc 门槛 = 多数类 1670/3395 = 49.19%",
                  "test 未参与训练/验证/调参（加载后从内存移除）",
                  "M1-B 遗留项'标准化后无 NaN/Inf'由 pipeline.zscore_reset 断言闭合"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    for c in checks:
        print(("✅" if c["ok"] else "❌"), c["name"], c.get("detail", ""))
    print("ALL PASS" if summary["all_checks_pass"] else "FAILED")
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
