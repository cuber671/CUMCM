"""P2 Round 7：BERT 解冻最后 1–2 层微调（预注册验收同前五条，基线 = M4）。

- BertTextEncoder 解冻最后 N 层（默认 2），低学习率 5e-5 参数组；其余 MRFN 1e-3；
- 文本特征每 batch 现场编码（梯度反传至解冻层）；BERT 内部保持 eval（无 dropout）；
- 训练/评测流程与 M4 一致（batch 级混合抽样 A35/B40/C15/D10、早停 clean S）；
- 评测：clean valid + 掩码库 31 条（每个 seed 用自己的微调后 BERT 编码）。
产物：runs/p2/bft/{summary.json, MRFN_seed*/...}。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_p2_b0 import MAJORITY_ACC, metrics  # noqa: E402
from train_p2_mrfn import (LIB, LETTER, MODS, O_KEY, STATE,  # noqa: E402
                           make_batch_avail, preload_train_masks)

from src.p2.data import CONTRACT, load_aligned  # noqa: E402
from src.p2.models import MODEL_REGISTRY, count_params  # noqa: E402
from src.p2.pipeline import load_stats, zscore_reset  # noqa: E402
from src.p2.text_mask import BertTextEncoder, mask_text_tokens  # noqa: E402

OUTDIR = ROOT / "runs/p2/bft"


def build_tensors(att, state, device):
    """train/valid 张量（audio/vision 已标准化；文本改为逐 batch 现场编码）。"""
    stats = load_stats()
    out = {}
    for split in ("train", "valid"):
        t = {"content": torch.from_numpy(state[f"{split}_content"]).bool().to(device),
             "y_cls": torch.from_numpy(state[f"{split}_cls"]).long().to(device),
             "y_reg": torch.from_numpy(state[f"{split}_rl"]).float().to(device),
             "text_bert": torch.from_numpy(np.asarray(att[split]["text_bert"])).to(device)}
        for m in MODS:
            t[f"o_{m}"] = torch.from_numpy(state[f"{split}_{O_KEY[m]}"]).bool().to(device)
        for mod in ("audio", "vision"):
            s = stats[mod]
            xn = zscore_reset(att[split][mod], s["mean"], s["std"],
                              state[f"{split}_{O_KEY[mod]}"].astype(bool),
                              name=f"{split}.{mod}")
            t[mod] = torch.from_numpy(xn).to(device)
        out[split] = t
    return out


@torch.no_grad()
def predict(model, bert_enc, t, batch=64):
    model.eval(); bert_enc.eval()
    logits, regs = [], []
    for i in range(0, t["content"].shape[0], batch):
        idx = slice(i, i + batch)
        feats = {"text": bert_enc(t["text_bert"][idx]),
                 "audio": t["audio"][idx], "vision": t["vision"][idx]}
        avail = {m: t[f"o_{m}"][idx] for m in MODS}
        o = model(feats, t["content"][idx], avail)
        logits.append(o["logits"]); regs.append(o["reg"])
    return torch.cat(logits), torch.cat(regs)


def eval_clean(model, bert_enc, t, device):
    logits, regs = predict(model, bert_enc, t)
    return metrics(t["y_cls"].cpu().numpy(), t["y_reg"].cpu().numpy(),
                   logits.cpu().numpy(), regs.cpu().numpy())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-layers", type=int, default=2)
    ap.add_argument("--bert-lr", type=float, default=5e-5)
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
    print(f"device={device} ft_layers={args.ft_layers} bert_lr={args.bert_lr}", flush=True)

    att = load_aligned()
    del att["test"]
    state = np.load(STATE)
    tb_train_t = torch.as_tensor(np.asarray(att["train"]["text_bert"]))
    tensors = build_tensors(att, state, device)
    lib_index = json.loads((ROOT / "runs/p2/mask_library/library_index.json").read_text())
    comps, comp_mods = preload_train_masks(lib_index)
    w_b = np.array([2.0 if "t" in md else 1.0 for md in comp_mods["B"]], dtype=np.float64)
    w_b /= w_b.sum()
    entries = [e for e in lib_index["entries"] if e["split"] == "valid"
               and any(t_.startswith("eval") for t_ in e["tags"])]

    def batch_aug(rng, idx_np, idx):
        return make_batch_avail(rng, idx_np, idx, tensors["train"], comps, "mixture",
                                w_b=w_b)

    # ---- Round 8 S_select 选点条目（valid，k=0）----
    def ventry(key):
        e = next(e for e in lib_index["entries"] if e["split"] == "valid"
                 and f"{e['mechanism']}/{e['modalities']}/rate{e['rate']}/{e['position']}" == key)
        return np.load(ROOT / "runs/p2/mask_library" / e["path"])
    z40 = ventry("mcar/t/rate40/random"); z80 = ventry("mcar/t/rate80/random")
    zj = ventry("joint/av/rate40/random")
    b_t40 = torch.from_numpy(z40["b_0_t"]).to(device)
    b_t80 = torch.from_numpy(z80["b_0_t"]).to(device)
    b_ja = torch.from_numpy(zj["b_0_a"]).to(device)
    b_jv = torch.from_numpy(zj["b_0_v"]).to(device)

    @torch.no_grad()
    def eval_sel_entry(model, be, t_va, device, text_mask=None, b_a=None, b_v=None):
        avail = {m: t_va[f"o_{m}"] for m in MODS}
        if text_mask is not None:
            tm = text_mask.to(device).bool()
            avail["text"] = t_va["o_text"] & ~tm
            ids = mask_text_tokens(t_va["text_bert"].cpu().numpy(),
                                   text_mask.cpu().numpy(), CONTRACT)
            feats_text = be(torch.from_numpy(ids).to(device))
        else:
            feats_text = be(t_va["text_bert"])
        feats = {"text": feats_text, "audio": t_va["audio"], "vision": t_va["vision"]}
        if b_a is not None:
            avail["audio"] = t_va["o_audio"] & ~b_a.to(device).bool()
        if b_v is not None:
            avail["vision"] = t_va["o_vision"] & ~b_v.to(device).bool()
        o = model(feats, t_va["content"], avail)
        return metrics(t_va["y_cls"].cpu().numpy(), t_va["y_reg"].cpu().numpy(),
                       o["logits"].cpu().numpy(), o["reg"].cpu().numpy())["S"]

    results = []
    for seed in (1, 2, 3):
        print(f"== seed {seed} ==", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        bert_enc = BertTextEncoder(unfreeze_last=args.ft_layers).to(device)
        model = MODEL_REGISTRY["MRFN"](dropout=args.dropout).to(device)
        tr, va = tensors["train"], tensors["valid"]
        n = tr["content"].shape[0]
        counts = torch.bincount(tr["y_cls"], minlength=3).float()
        w = (counts.sum() / (3 * counts)).to(device)
        ce = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW([
            {"params": bert_enc.unfrozen_parameters(), "lr": args.bert_lr},
            {"params": model.parameters(), "lr": args.lr}], weight_decay=args.wd)
        best, best_state, best_bstate, bad = -1.0, None, None, 0
        log = []
        for epoch in range(1, args.epochs + 1):
            model.train(); bert_enc.train()
            rng = np.random.default_rng(seed * 100000 + epoch)
            perm = rng.permutation(n)
            tot, nb = 0.0, 0
            for i in range(0, n, args.batch_size):
                idx_np = perm[i:i + args.batch_size]
                idx = torch.from_numpy(idx_np).long().to(device)
                avail, b_t, comp = batch_aug(rng, idx_np, idx)
                ids = tb_train_t[idx_np].to(device)
                if b_t is not None and bool(b_t.any()):
                    ids = torch.from_numpy(
                        mask_text_tokens(ids.cpu().numpy(), b_t.cpu().numpy(),
                                         CONTRACT)).to(device)
                feats = {"text": bert_enc(ids),
                         "audio": tr["audio"][idx], "vision": tr["vision"][idx]}
                out = model(feats, tr["content"][idx], avail)
                loss = ce(out["logits"], tr["y_cls"][idx]) + \
                    args.lambda_l1 * (out["reg"] - tr["y_reg"][idx]).abs().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(bert_enc.parameters()) + list(model.parameters()), 1.0)
                opt.step()
                tot += float(loss.detach()); nb += 1
            vm = eval_clean(model, bert_enc, va, device)
            with torch.no_grad():
                s_t40 = eval_sel_entry(model, bert_enc, va, device, text_mask=b_t40)
                s_t80 = eval_sel_entry(model, bert_enc, va, device, text_mask=b_t80)
                s_j40 = eval_sel_entry(model, bert_enc, va, device,
                                       b_a=b_ja, b_v=b_jv)
            s_sel = 0.50 * vm["S"] + 0.25 * s_t40 + 0.15 * s_t80 + 0.10 * s_j40
            log.append({"epoch": epoch, "train_loss": round(tot / nb, 6),
                        "S_select": round(s_sel, 6), **vm})
            if s_sel > best:
                best, bad = s_sel, 0
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_bstate = {k: v.cpu() for k, v in bert_enc.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
            print(f"  ep{epoch} S={vm['S']:.4f} S_select={s_sel:.4f} loss={tot/nb:.4f}",
                  flush=True)
        model.load_state_dict(best_state)
        bert_enc.load_state_dict(best_bstate)
        sd = out_dir / f"MRFN_seed{seed}"
        sd.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, sd / "checkpoint.pt")
        torch.save(best_bstate, sd / "bert_checkpoint.pt")
        vm = eval_clean(model, bert_enc, va, device)
        results.append({"model": "MRFN_BFT", "seed": seed,
                        "param_count": count_params(model),
                        "bert_trainable": int(sum(p.numel()
                                                  for p in bert_enc.unfrozen_parameters())),
                        "epochs_run": len(log),
                        "best_epoch": int(np.argmax([x["S"] for x in log]) + 1),
                        "best_S": best, "valid_metrics": vm, "per_epoch": log})
        (out_dir / f"MRFN_seed{seed}" / "result.json").write_text(
            json.dumps({"config": vars(args), **results[-1]}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    # ---- 掩码库 31 条评测（每 seed 用自己的微调后 BERT 编码）----
    masked_D_S = {}
    for seed in (1, 2, 3):
        model = MODEL_REGISTRY["MRFN"](dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(out_dir / f"MRFN_seed{seed}" / "checkpoint.pt",
                                         weights_only=True))
        bert_enc = BertTextEncoder(unfreeze_last=args.ft_layers).to(device)
        bert_enc.load_state_dict(torch.load(out_dir / f"MRFN_seed{seed}" / "bert_checkpoint.pt",
                                            weights_only=True))
        model.eval(); bert_enc.eval()
        clean_S = next(r["best_S"] for r in results if r["seed"] == seed)
        t_va = tensors["valid"]
        for e in entries:
            z = np.load(ROOT / "runs/p2/mask_library" / e["path"])
            key = f"{e['mechanism']}/{e['modalities']}/rate{e['rate']}/{e['position']}"
            per_k = []
            with torch.no_grad():
                for k in range(8):
                    avail, feats = {}, {}
                    for m in MODS:
                        key_b = f"b_{k}_{LETTER[m]}"
                        if key_b not in z.files:
                            avail[m] = t_va[f"o_{m}"]
                            feats[m] = (bert_enc(t_va["text_bert"]) if m == "text"
                                        else t_va[m])
                            continue
                        b = torch.from_numpy(z[key_b]).to(device)
                        avail[m] = t_va[f"o_{m}"] & ~b.bool()
                        if m == "text":
                            ids = mask_text_tokens(t_va["text_bert"].cpu().numpy(),
                                                   b.cpu().numpy(), CONTRACT)
                            feats[m] = bert_enc(torch.from_numpy(ids).to(device))
                        else:
                            feats[m] = t_va[m]
                    o = model(feats, t_va["content"], avail)
                    per_k.append(metrics(t_va["y_cls"].cpu().numpy(),
                                         t_va["y_reg"].cpu().numpy(),
                                         o["logits"].cpu().numpy(),
                                         o["reg"].cpu().numpy()))
            mean_S_entry = float(np.mean([p["S"] for p in per_k]))
            masked_D_S.setdefault(key, []).append(
                (clean_S - mean_S_entry) / max(clean_S, 1e-6))

    mean_D_S = {k: round(float(np.mean(v)), 6) for k, v in masked_D_S.items()}
    accs = [r["valid_metrics"]["acc"] for r in results]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "MRFN_BFT", "ft_layers": args.ft_layers, "bert_lr": args.bert_lr,
        "params": {"mrfn": results[0]["param_count"],
                   "bert_trainable": results[0]["bert_trainable"]},
        "clean": {k: round(float(np.mean([r["valid_metrics"][k] for r in results])), 6)
                  for k in ("acc", "macro_f1", "weighted_f1", "mae", "pearson", "S")},
        "clean_std": {k: round(float(np.std([r["valid_metrics"][k] for r in results])), 6)
                      for k in ("acc", "macro_f1", "pearson", "S")},
        "masked_D_S_mean": mean_D_S,
        "majority_pass": bool(all(a > MAJORITY_ACC for a in accs)),
        "per_seed": [{k: r[k] for k in ("seed", "param_count", "bert_trainable",
                                        "epochs_run", "best_epoch", "best_S",
                                        "valid_metrics")} for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("\n== clean（3 seeds 均值）==")
    print(json.dumps(summary["clean"], ensure_ascii=False, indent=1))
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
