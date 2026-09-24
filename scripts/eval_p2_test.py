"""P2 test 终评（一次性）：附件2 test 划分 727 条的全部冻结模型最终评估。

纪律：开发全程未触碰 test；本脚本为唯一一次 test 评估，运行后实验封版，
论文主表数字以本产物为准，不再迭代任何模型/超参。
协议（预注册）：5 个冻结模型（B0/B1/B2/B3/MRFN）× 3 seeds ×
  [clean + 掩码库 31 条 eval 条目 × K8 实例]，L1 指标 + D_S。
产物：runs/p2/test_final/test_final_results.json。
"""
from __future__ import annotations

import argparse
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
from train_p2_ladder import eval_masked, forward_model  # noqa: E402

from src.p2 import missing as M  # noqa: E402
from src.p2.data import CONTRACT, load_aligned  # noqa: E402
from src.p2.models import MODEL_REGISTRY, count_params  # noqa: E402
from src.p2.pipeline import load_stats, zscore_reset  # noqa: E402
from src.p2.text_mask import DEFAULT_MODEL_PATH, encode_text, load_frozen_bert  # noqa: E402

STATE = ROOT / "runs/p2/data/m1a_state.npz"
LIB = ROOT / "runs/p2/mask_library"
OUT = ROOT / "runs/p2/test_final"
MODS = ("text", "audio", "vision")
MODELS = ("B0", "B1", "B2", "B3", "MRFN")


def ckpt_path(name: str, seed: int) -> Path:
    if name == "B0":
        return ROOT / "runs/p2/b0" / f"seed{seed}" / "checkpoint.pt"
    if name == "MRFN":
        return ROOT / "runs/p2/mrfn" / f"MRFN_seed{seed}" / "checkpoint.pt"
    return ROOT / "runs/p2/ladder" / f"{name}_seed{seed}" / "checkpoint.pt"


def build_test(att, state, device):
    stats = load_stats()
    bert = load_frozen_bert(DEFAULT_MODEL_PATH, device=device)
    tb = np.asarray(att["test"]["text_bert"])
    text_f = encode_text(tb, bert, CONTRACT, batch_size=128)
    content = torch.from_numpy(state["test_content"]).bool().to(device)
    t = {"content": content,
         "y_cls": torch.from_numpy(state["test_cls"]).long().to(device),
         "y_reg": torch.from_numpy(state["test_rl"]).float().to(device),
         "text": torch.from_numpy(np.ascontiguousarray(text_f)).to(device)}
    for m in MODS:
        t[f"o_{m}"] = torch.from_numpy(state[f"test_{OKEY[m]}"]).bool().to(device)
    for mod in ("audio", "vision"):
        s = stats[mod]
        xn = zscore_reset(att["test"][mod], s["mean"], s["std"],
                          state[f"test_{OKEY[mod]}"].astype(bool), name=f"test.{mod}")
        t[mod] = torch.from_numpy(xn).to(device)
    va_np = {"y_cls": t["y_cls"].cpu().numpy(), "y_reg": t["y_reg"].cpu().numpy(),
             "text_bert": tb}
    return t, va_np, bert


OKEY = {"text": "o_text", "audio": "o_audio", "vision": "o_vision"}


@torch.no_grad()
def predict_clean(model, name, t, batch=256):
    model.eval()
    logits, regs = [], []
    for i in range(0, t["content"].shape[0], batch):
        idx = slice(i, i + batch)
        feats = {k: t[k][idx] for k in MODS}
        o = forward_model(model, name, feats, t["content"][idx],
                          {m: t[f"o_{m}"][idx] for m in MODS})
        logits.append(o["logits"]); regs.append(o["reg"])
    return torch.cat(logits), torch.cat(regs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()
    device = args.device
    OUT.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    att = load_aligned()
    state = np.load(STATE)
    t_te, va_np, bert = build_test(att, state, device)
    lib_index = json.loads((LIB / "library_index.json").read_text())
    entries = [e for e in lib_index["entries"]
               if e["split"] == "valid" and any(t_.startswith("eval") for t_ in e["tags"])]
    # 掩码库按 split 隔离：test 无库实例。终评网格取与 valid 相同的 31 个组合，
    # 用 missing.py 生成器以 test 自身 o/content 现场重生成 K=8 实例
    # （同 combo_idx 同参数 → 机制/率严格对齐；实例仅作用于 test 样本本身）。
    grid = M.library_grid()
    o_test = {"t": state["test_o_text"].astype(bool),
              "a": state["test_o_audio"].astype(bool),
              "v": state["test_o_vision"].astype(bool)}   # 生成器按单字母索引模态
    content_test = state["test_content"].astype(bool)
    pmf = lib_index["att3_pmf_frozen"]

    class ZDict(dict):
        @property
        def files(self):
            return list(self.keys())

    def regen_entry_z(e):
        combo = grid[e["combo_idx"]]
        inst = M.gen_combo_instances(combo, e["combo_idx"], content_test, o_test, pmf=pmf)
        z = ZDict()
        for k_i, b_map in enumerate(inst):
            for letter, b in b_map.items():
                z[f"b_{k_i}_{letter}"] = b.astype(np.uint8)
        return z

    print(f"device={device} | 冻结 commit={commit[:12]}", flush=True)

    results = {"created_at": datetime.now(timezone.utc).isoformat(),
               "frozen_commit": commit,
               "discipline": "附件2 test 唯一一次评估；此后实验封版",
               "protocol": {"models": list(MODELS), "seeds": [1, 2, 3],
                            "grid_entries": len(entries),
                            "grid_note": "掩码实例由 missing.py 生成器以 test 样本现场重生成"
                                         "（同组合参数与库一致，实例序号 0..7）"}}
    out_models = {}
    for name in MODELS:
        per_seed = []
        for seed in (1, 2, 3):
            model = MODEL_REGISTRY[name](dropout=0.3).to(device)
            model.load_state_dict(torch.load(ckpt_path(name, seed), weights_only=True))
            model.eval()
            logits, reg = predict_clean(model, name, t_te)
            clean = metrics(va_np["y_cls"], va_np["y_reg"],
                            logits.cpu().numpy(), reg.cpu().numpy())
            entries_out = {}
            for e in entries:
                z = regen_entry_z(e)
                key = f"{e['mechanism']}/{e['modalities']}/rate{e['rate']}/{e['position']}"
                entries_out[key] = eval_masked(model, name, t_te, va_np, e, z,
                                               clean["S"], bert, device)
            per_seed.append({"seed": seed, "clean": clean, "masked": entries_out})
            print(f"{name} seed{seed}: clean Acc={clean['acc']:.4f} S={clean['S']:.4f} "
                  f"masked_mean_S={np.mean([v['S'] for v in entries_out.values()]):.4f}",
                  flush=True)
        mean_d = {k: round(float(np.mean([r["masked"][k]["D_S"] for r in per_seed])), 6)
                  for k in per_seed[0]["masked"]}
        out_models[name] = {
            "param_count": count_params(MODEL_REGISTRY[name](dropout=0.3)),
            "clean": {k: {"mean": round(float(np.mean([r["clean"][k] for r in per_seed])), 6),
                          "std": round(float(np.std([r["clean"][k] for r in per_seed])), 6)}
                      for k in ("acc", "macro_f1", "weighted_f1", "mae", "pearson", "S")},
            "masked_D_S_mean": mean_d,
            "mean_D_S": round(float(np.mean(list(mean_d.values()))), 6),
            "masked_S_mean": {k: round(float(np.mean(
                [r["masked"][k]["S"] for r in per_seed])), 6) for k in per_seed[0]["masked"]},
        }
        print(f"[{name}] test: Acc={out_models[name]['clean']['acc']['mean']:.4f} "
              f"mean_D_S={out_models[name]['mean_D_S']:.4f}", flush=True)

    results["models"] = out_models
    (OUT / "test_final_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n产物:", OUT / "test_final_results.json")
    print("TEST FINAL DONE（此后实验封版）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
