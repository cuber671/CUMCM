"""P2 第二批 test 终评（预注册追加，契约 §8"第二批 test 终评预注册"条款执行）。

对象：MRFN+（R8，runs/p2/bft 3 seeds）单模型 + 3-seed 集成，clean 口径 L1+S。
校验锚：M4（runs/p2/mrfn 3 seeds）重推理均值须落在封版 0.6671±0.015 内，
超出即中止不产出 MRFN+ 数字（只落校验失败报告）。
产物：runs/p2/test_second/test_second_results.json。
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_p2_b0 import metrics  # noqa: E402

from src.p2.data import CONTRACT, load_aligned  # noqa: E402
from src.p2.models import MODEL_REGISTRY  # noqa: E402
from src.p2.pipeline import load_stats, zscore_reset  # noqa: E402
from src.p2.text_mask import (DEFAULT_MODEL_PATH, BertTextEncoder,  # noqa: E402
                              encode_text, load_frozen_bert)

STATE = ROOT / "runs/p2/data/m1a_state.npz"
OUT = ROOT / "runs/p2/test_second"
MODS = ("text", "audio", "vision")
SEAL_M4_ACC = 0.6671          # 封版主表 MRFN clean acc 均值
ANCHOR_TOL = 0.015


@torch.no_grad()
def infer(model, t, text_f, batch=256):
    model.eval()
    logits, regs = [], []
    for i in range(0, t["content"].shape[0], batch):
        idx = slice(i, i + batch)
        feats = {"text": text_f[idx], "audio": t["audio"][idx], "vision": t["vision"][idx]}
        av = {m: t[f"o_{m}"][idx] for m in MODS}
        o = model(feats, t["content"][idx], av)
        logits.append(o["logits"]); regs.append(o["reg"])
    return torch.cat(logits), torch.cat(regs)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    att = load_aligned()
    state = np.load(STATE)
    stats = load_stats()
    t = {"content": torch.from_numpy(state["test_content"]).bool().to(device),
         "y_cls": torch.from_numpy(state["test_cls"]).long().to(device),
         "y_reg": torch.from_numpy(state["test_rl"]).float().to(device)}
    for m in MODS:
        t[f"o_{m}"] = torch.from_numpy(state[f"test_o_{m}"]).bool().to(device)
    for mod in ("audio", "vision"):
        s = stats[mod]
        t[mod] = torch.from_numpy(zscore_reset(
            att["test"][mod], s["mean"], s["std"],
            state[f"test_o_{mod}"].astype(bool), name=f"test.{mod}")).to(device)
    y = t["y_cls"].cpu().numpy()
    rl = t["y_reg"].cpu().numpy()
    tb_t = torch.as_tensor(np.ascontiguousarray(
        np.asarray(att["test"]["text_bert"]))).to(device)

    # ---- 校验锚：M4 3-seed（冻结 BERT 现场编码，与第一批终评同路径）----
    bert = load_frozen_bert(DEFAULT_MODEL_PATH, device=device)
    text_frozen = torch.from_numpy(np.ascontiguousarray(
        encode_text(np.asarray(att["test"]["text_bert"]), bert, CONTRACT,
                    batch_size=128))).to(device)
    m4_accs = []
    for seed in (1, 2, 3):
        m = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        m.load_state_dict(torch.load(
            ROOT / "runs/p2/mrfn" / f"MRFN_seed{seed}" / "checkpoint.pt",
            weights_only=True))
        lo, rg = infer(m, t, text_frozen)
        m4_accs.append(float((lo.argmax(1).cpu().numpy() == y).mean()))
    anchor_mean = float(np.mean(m4_accs))
    anchor_ok = abs(anchor_mean - SEAL_M4_ACC) <= ANCHOR_TOL
    print(f"[锚] M4 重推理 per-seed {[round(a,4) for a in m4_accs]} 均值 {anchor_mean:.4f} "
          f"(封版 {SEAL_M4_ACC}, 容差 ±{ANCHOR_TOL}) -> {'✅' if anchor_ok else '❌ 中止'}",
          flush=True)

    results = {"created_at": datetime.now(timezone.utc).isoformat(),
               "commit": commit,
               "protocol": "第二批 test 终评（预注册追加）；clean 口径；无条件报告；此后封存",
               "anchor": {"m4_reinf_per_seed": [round(a, 6) for a in m4_accs],
                          "m4_reinf_mean": round(anchor_mean, 6),
                          "sealed": SEAL_M4_ACC, "tol": ANCHOR_TOL, "ok": anchor_ok}}
    if not anchor_ok:
        results["aborted"] = "锚点超出容差，按预注册中止，不产出 MRFN+ test 数字"
        (OUT / "test_second_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1

    # ---- 主体：MRFN+ 单模型 + 集成（逐 seed 微调后编码器现场编码）----
    per_seed, probs_all, regs_all = [], [], []
    for seed in (1, 2, 3):
        enc = BertTextEncoder(unfreeze_last=1).to(device)
        enc.load_state_dict(torch.load(
            ROOT / "runs/p2/bft" / f"MRFN_seed{seed}" / "bert_checkpoint.pt",
            weights_only=True))
        enc.eval()
        with torch.no_grad():
            text_ft = enc(tb_t)
        m = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        m.load_state_dict(torch.load(
            ROOT / "runs/p2/bft" / f"MRFN_seed{seed}" / "checkpoint.pt",
            weights_only=True))
        lo, rg = infer(m, t, text_ft)
        probs_all.append(torch.softmax(lo, -1).cpu().numpy())
        regs_all.append(rg.cpu().numpy())
        per_seed.append({"seed": seed,
                         **{k: round(v, 6) for k, v in
                            metrics(y, rl, lo.cpu().numpy(), rg.cpu().numpy()).items()}})
        print(f"[MRFN+] seed{seed}: {per_seed[-1]}", flush=True)
    p_ens = np.mean(probs_all, 0)
    r_ens = np.mean(regs_all, 0)
    ens = {k: round(v, 6) for k, v in
           metrics(y, rl, np.log(np.clip(p_ens, 1e-9, 1)), r_ens).items()}
    pred = p_ens.argmax(1)
    abs_rl = np.abs(rl)
    zones = {"Z0": rl == 0, "Z1": (rl != 0) & (abs_rl < 0.5), "Z2": abs_rl >= 0.5}
    zone_acc = {z: round(float((pred[m] == y[m]).mean()), 6) for z, m in zones.items()}

    def agg(key):
        v = [r[key] for r in per_seed]
        return {"mean": round(float(np.mean(v)), 6), "std": round(float(np.std(v)), 6)}

    results["mrfn_plus"] = {
        "per_seed": per_seed,
        "single_mean": {k: agg(k) for k in
                        ("acc", "macro_f1", "weighted_f1", "mae", "pearson", "S")},
        "ensemble_3seed": ens,
        "ensemble_zones": zone_acc,
        "pred_dist_neg_neu_pos": np.bincount(pred, minlength=3).tolist()}
    (OUT / "test_second_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[集成] {ens}")
    print(f"[分区] {zone_acc}")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
