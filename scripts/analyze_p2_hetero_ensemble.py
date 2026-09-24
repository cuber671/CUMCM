"""P2 第一梯队收尾：异构集成（MRFN+×3 + B2×3 + B3×3 概率平均）+ 温度标定。

纪律：纯推理分析，零训练、不碰 test；基线 = fj3 现任交付模型
MRFN+ 3-seed 集成（valid Acc 0.6401，error_attribution_mrfn_plus.json）。
预注册验收（修正版，fj3 无标签故不设 fj3 门槛）：
  A1 异构集成 valid Acc > 0.6401 且 macro-F1 ≥ 0.6083（不输现任）；
  A2 弱情感带（Z0+Z1 合并 acc）不低于 MRFN+ 集成基线 −0.005；
  A3 温度标定 argmax 逐位不变、ECE 下降 ≥10%。
产物：runs/p2/ensemble_check/hetero_ensemble_temp.json。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_p2_b0 import metrics  # noqa: E402
from train_p2_ladder import build_tensors  # noqa: E402

from src.p2.data import load_aligned  # noqa: E402
from src.p2.models import MODEL_REGISTRY  # noqa: E402
from src.p2.text_mask import BertTextEncoder  # noqa: E402

STATE = ROOT / "runs/p2/data/m1a_state.npz"
OUT = ROOT / "runs/p2/ensemble_check"
MODS = ("text", "audio", "vision")
SEEDS = (1, 2, 3)
LADDER = ROOT / "runs/p2/ladder"
BFT = ROOT / "runs/p2/bft"


@torch.no_grad()
def probs_and_reg(model, name, va, text_f):
    """clean valid 推理：B2 只吃 (feats, content)，B3/MRFN 吃 (feats, content, avail)。"""
    model.eval()
    logits, regs = [], []
    for i in range(0, va["content"].shape[0], 256):
        idx = slice(i, i + 256)
        feats = {"text": text_f[idx], "audio": va["audio"][idx], "vision": va["vision"][idx]}
        if name == "B2":
            o = model(feats, va["content"][idx])
        else:
            av = {m: va[f"o_{m}"][idx] for m in MODS}
            o = model(feats, va["content"][idx], av)
        logits.append(o["logits"]); regs.append(o["reg"])
    lo = torch.cat(logits)
    return torch.softmax(lo, -1).cpu().numpy(), torch.cat(regs).cpu().numpy()


def macro_f1(y, pred):
    f1s = []
    for c in range(3):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s))


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def zone_stats(y, rl, pred):
    abs_rl = np.abs(rl)
    zones = {"Z0": rl == 0.0, "Z1": (rl != 0.0) & (abs_rl < 0.5), "Z2": abs_rl >= 0.5}
    out = {}
    for z, m in zones.items():
        out[z] = {"n": int(m.sum()), "acc": round(float((pred[m] == y[m]).mean()), 6)}
    weak = zones["Z0"] | zones["Z1"]
    out["weak_pooled"] = {"n": int(weak.sum()),
                          "acc": round(float((pred[weak] == y[weak]).mean()), 6)}
    out["weak_err_share"] = round(float((pred[weak] != y[weak]).sum()
                                        / (pred != y).sum()), 6)
    return out


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    att = load_aligned()
    del att["test"]
    state = np.load(STATE)
    va = build_tensors(att, state, device)["valid"]
    y = va["y_cls"].cpu().numpy()
    rl = va["y_reg"].cpu().numpy()
    tb_valid = torch.as_tensor(np.asarray(att["valid"]["text_bert"])).to(device)

    # ---- 文本特征两条通路：冻结缓存（B2/B3 训练口径）+ 逐 seed 微调编码（MRFN+）----
    text_frozen = va["text"]
    bft_probs, ladder_probs = {"B2": [], "B3": []}, {"B2": [], "B3": []}
    mrfn_reg = []
    for seed in SEEDS:
        enc = BertTextEncoder(unfreeze_last=1).to(device)
        enc.load_state_dict(torch.load(
            BFT / f"MRFN_seed{seed}" / "bert_checkpoint.pt", weights_only=True))
        enc.eval()
        with torch.no_grad():
            text_ft = enc(tb_valid)
        m = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        m.load_state_dict(torch.load(
            BFT / f"MRFN_seed{seed}" / "checkpoint.pt", weights_only=True))
        p, r = probs_and_reg(m, "MRFN", va, text_ft)
        ladder_probs["_mrfn"] = ladder_probs.get("_mrfn", [])
        ladder_probs["_mrfn"].append(p)
        mrfn_reg.append(r)
        for name in ("B2", "B3"):
            mm = MODEL_REGISTRY[name](dropout=0.3).to(device)
            mm.load_state_dict(torch.load(
                LADDER / f"{name}_seed{seed}" / "checkpoint.pt", weights_only=True))
            pp, _ = probs_and_reg(mm, name, va, text_frozen)
            ladder_probs[name].append(pp)
        print(f"seed{seed}: MRFN+/B2/B3 前向完成", flush=True)

    mrfn_ens = np.mean(ladder_probs["_mrfn"], 0)          # 现任：MRFN+ 3-seed 集成
    hetero9 = np.mean(ladder_probs["_mrfn"] + ladder_probs["B2"] + ladder_probs["B3"], 0)
    reg_ens = np.mean(mrfn_reg, 0)

    results = {"created_at": datetime.now(timezone.utc).isoformat(),
               "discipline": "纯推理；基线=MRFN+ 3-seed 集成（fj3 现任）；test 不碰"}
    for tag, probs in (("mrfn_plus_ens", mrfn_ens), ("hetero_9", hetero9)):
        pred = probs.argmax(1)
        m = metrics(y, rl, np.log(np.clip(probs, 1e-9, 1)), reg_ens)
        results[tag] = {"acc": round(m["acc"], 6), "macro_f1": round(m["macro_f1"], 6),
                        "S": round(m["S"], 6), "zones": zone_stats(y, rl, pred)}

    # ---- 温度标定（对 hetero9 若过验收、否则对现任 mrfn_ens）----
    target = "hetero_9" if results["hetero_9"]["acc"] > results["mrfn_plus_ens"]["acc"] \
        else "mrfn_plus_ens"
    probs = hetero9 if target == "hetero_9" else mrfn_ens
    logp = torch.from_numpy(np.log(np.clip(probs, 1e-9, 1)))
    yt = torch.from_numpy(y).long()
    T = torch.nn.Parameter(torch.ones(1) * 1.0)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logp / T.clamp(1e-3), yt)
        loss.backward()
        return loss
    opt.step(closure)
    t_val = float(T.item())
    conf_raw = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    cal = probs ** (1.0 / t_val)
    cal /= cal.sum(1, keepdims=True)
    conf_cal = cal.max(1)
    results["temperature_scaling"] = {
        "applied_to": target, "T": round(t_val, 4),
        "ece_before": round(ece(conf_raw, correct), 6),
        "ece_after": round(ece(conf_cal, correct), 6),
        "argmax_invariant": bool((cal.argmax(1) == probs.argmax(1)).all()),
        "conf_mean_before": round(float(conf_raw.mean()), 6),
        "conf_mean_after": round(float(conf_cal.mean()), 6)}

    # ---- 预注册验收 ----
    b = results["mrfn_plus_ens"]
    h = results["hetero_9"]
    checks = [
        {"name": "A1_acc_beats_incumbent", "ok": h["acc"] > b["acc"],
         "detail": f"{h['acc']} vs {b['acc']}"},
        {"name": "A1_macro_f1_not_worse", "ok": h["macro_f1"] >= b["macro_f1"] - 1e-9,
         "detail": f"{h['macro_f1']} vs {b['macro_f1']}"},
        {"name": "A2_weak_band_no_regression",
         "ok": h["zones"]["weak_pooled"]["acc"] >=
               b["zones"]["weak_pooled"]["acc"] - 0.005,
         "detail": f"{h['zones']['weak_pooled']['acc']} vs {b['zones']['weak_pooled']['acc']}"},
        {"name": "A3_temp_argmax_invariant",
         "ok": results["temperature_scaling"]["argmax_invariant"]},
        {"name": "A3_ece_drop_10pct",
         "ok": results["temperature_scaling"]["ece_after"] <=
               0.9 * results["temperature_scaling"]["ece_before"],
         "detail": f"{results['temperature_scaling']['ece_before']} -> "
                   f"{results['temperature_scaling']['ece_after']}"}]
    results["checks"] = checks
    results["all_pass"] = all(c["ok"] for c in checks)
    (OUT / "hetero_ensemble_temp.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in checks:
        print(("✅" if c["ok"] else "❌"), c["name"], c.get("detail", ""), flush=True)
    print("ALL PASS" if results["all_pass"] else "FAILED（按契约不采纳异构集成）",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
