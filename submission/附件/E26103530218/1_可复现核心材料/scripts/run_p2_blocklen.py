"""P2 Q2-E4：连续缺失长度扫描（固定覆盖率、变块长；valid，MRFN 3-seed）。

背景：31 场景网格中 block 机制的块长 ℓ=round(rate·w) 随缺失率绑定，
"同样缺失率下、缺失呈长块还是短碎块"未被单独隔离。本实验固定覆盖率
40%（与网格 block@40 同口径，末块截断保证覆盖率不过冲），在三模态上
分别以不重叠连续块铺至目标覆盖率，块长上限 ∈ {2,5,10}（模型索引空间，
非物理秒），并以冻结库中 mcar@40（散布，ℓ=1）与 block@40（单长块，
ℓ≈0.4w）为两端锚点，构成 ℓ 谱：散布 1 → 2 → 5 → 10 → 单块。

协议（与 ladder 完全一致）：valid 全量 728 条 × K=8 实例指标平均；
text = 块掩码驱动 [MASK] 重编码（M1-D 机制）后 a_text 零填充；模型 =
runs/p2/mrfn/MRFN_seed{1,2,3}（主表 MRFN，冻结权重，不重训）；
D_S=(S_clean−S_r)/max(S_clean,ε)，逐 seed 计算后取均值。

掩码生成：贪心不重叠等长块（同 joint 段采样逻辑、段长固定），
随机性全部来自 default_rng([2027, combo_idx, k])（种子 2027 与冻结库
2026 区分，不触碰冻结库与网格顺序）。实例落盘 runs/p2/blocklen/masks/
供逐位复核。产物：runs/p2/blocklen/blocklen_results.json；退出码全过=0。
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
from train_p2_ladder import build_tensors, eval_masked, predict_l  # noqa: E402

from src.p2.missing import instance_stats  # noqa: E402
from src.p2.models import MODEL_REGISTRY  # noqa: E402
from src.p2.text_mask import DEFAULT_MODEL_PATH, load_frozen_bert  # noqa: E402

STATE = ROOT / "runs/p2/data/m1a_state.npz"
LIB = ROOT / "runs/p2/mask_library"
CKPT = ROOT / "runs/p2/mrfn"
OUTDIR = ROOT / "runs/p2/blocklen"

BLOCKLEN_SEED = 2027          # 与冻结库 MASK_LIBRARY_SEED=2026 区分
K_INSTANCES = 8
RATE = 40                     # 覆盖率 %，与网格 block@40 同口径
LENGTHS = (2, 5, 10)
MODS = ("t", "a", "v")
O_KEY = {"t": "o_text", "a": "o_audio", "v": "o_vision"}
# combo_idx 冻结顺序：t2,t5,t10,a2,a5,a10,v2,v5,v10
COMBOS = [dict(mechanism="blocklen", modalities=m, rate=RATE, block_len=ell)
          for m in MODS for ell in LENGTHS]
ANCHOR_KEYS = ([f"mcar/{m}/rate{RATE}/random" for m in MODS] +
               [f"block/{m}/rate{RATE}/random" for m in MODS])


def sample_blocks(apos: np.ndarray, k: int, ell: int, rng) -> list:
    """贪心不重叠块（joint 段采样逻辑、段长上限 ℓ）：随机置换起点，块长取
    min(ℓ, 剩余预算)（末块截断，覆盖率不过冲），块须整体落在可用位且不与
    已占重叠；至 k 个抹除位或起点耗尽。"""
    segs, total = [], 0
    occupied = np.zeros(int(apos.max()) + 1 if len(apos) else 1, dtype=bool)
    for s in rng.permutation(apos):
        if total >= k:
            break
        e = min(ell, k - total)
        seg = np.arange(int(s), int(s) + e)
        if seg[-1] >= len(occupied) or occupied[seg].any() or not np.isin(seg, apos).all():
            continue
        occupied[seg] = True
        segs.append((int(s), e))
        total += e
    return segs


def gen_b_blocklen(content: np.ndarray, o: np.ndarray, rate: float, ell: int,
                   rng) -> np.ndarray:
    """单模态等长连续块铺至目标覆盖率；b 只落 o=1 位（越界即断言失败）。"""
    b = np.zeros(o.shape, dtype=np.uint8)
    k = np.round(rate * o.sum(axis=1)).astype(np.int64)
    for i in range(o.shape[0]):
        avail = content[i] & o[i]
        apos = np.where(avail)[0]
        if k[i] <= 0 or len(apos) == 0:
            continue
        for s, e in sample_blocks(apos, int(k[i]), ell, rng):
            b[i, s:s + e] = 1
    assert not ((b == 1) & (o == 0)).any(), "b 越界（o=0 处 b=1）"
    return b


def build_instances(combo: dict, combo_idx: int, content: np.ndarray,
                    o_by_mod: dict) -> tuple[dict, list]:
    """一个 blocklen 组合的 K 个实例：npz 键值格式与冻结库一致。"""
    arrs, stats = {}, []
    m, ell = combo["modalities"], combo["block_len"]
    for k_i in range(K_INSTANCES):
        rng = np.random.default_rng([BLOCKLEN_SEED, combo_idx, k_i])
        b = gen_b_blocklen(content, o_by_mod[m], combo["rate"] / 100.0, ell, rng)
        arrs[f"b_{k_i}_{m}"] = b
        stats.append(instance_stats({m: b}, {m: o_by_mod[m]}, content))
    return arrs, stats


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")

    state = np.load(STATE)
    n_tr = int(state["train_content"].shape[0])
    va_np = {
        "content": state["valid_content"].astype(bool),
        "y_cls": state["valid_cls"],
        "y_reg": state["valid_rl"],
    }
    for m in MODS:
        va_np[O_KEY[m]] = state[f"valid_{O_KEY[m]}"].astype(bool)

    # ---- 生成 blocklen 实例并落盘（审计可逐位复核）----
    o_by_mod = {m: va_np[O_KEY[m]] for m in MODS}
    new_entries = {}
    for combo_idx, combo in enumerate(COMBOS):
        arrs, stats = build_instances(combo, combo_idx, va_np["content"], o_by_mod)
        rel = (f"valid/blocklen/{combo['modalities']}/rate{RATE}/"
               f"len{combo['block_len']}/seed{BLOCKLEN_SEED}.npz")
        path = OUTDIR / "masks" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrs)
        key = f"blocklen/{combo['modalities']}/rate{RATE}/len{combo['block_len']}"
        new_entries[key] = {"path": str(path), "stats": stats,
                            "combo_idx": combo_idx}
        rate_m = float(np.mean([s["actual_rate"][combo["modalities"]] for s in stats]))
        print(f"[gen] {key}: 实际覆盖率均值 {rate_m:.4f}")

    # ---- 锚点（冻结库 valid 条目）----
    lib_index = json.loads((LIB / "library_index.json").read_text())
    anchor_entries = {}
    for e in lib_index["entries"]:
        if e["split"] != "valid":
            continue
        key = f"{e['mechanism']}/{e['modalities']}/rate{e['rate']}/{e['position']}"
        if key in ANCHOR_KEYS:
            anchor_entries[key] = {"path": str(LIB / e["path"]), "k": e["k"],
                                   "library": True}
    assert len(anchor_entries) == len(ANCHOR_KEYS), "锚点条目不全"

    # ---- 评测：MRFN 3-seed 冻结权重 ----
    from src.p2.data import load_aligned
    att = load_aligned()
    del att["test"]
    tensors = build_tensors(att, state, device)
    va_eval = {"y_cls": tensors["valid"]["y_cls"].cpu().numpy(),
               "y_reg": tensors["valid"]["y_reg"].cpu().numpy(),
               "text_bert": np.asarray(att["valid"]["text_bert"])}
    bert = load_frozen_bert(DEFAULT_MODEL_PATH, device=device)

    all_keys = list(new_entries) + ANCHOR_KEYS
    per_seed = {}
    for seed in (1, 2, 3):
        model = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        model.load_state_dict(torch.load(CKPT / f"MRFN_seed{seed}" / "checkpoint.pt",
                                         weights_only=True))
        model.eval()
        logits, reg = predict_l(model, "MRFN", tensors["valid"], None)
        clean = metrics(va_eval["y_cls"], va_eval["y_reg"],
                        logits.cpu().numpy(), reg.cpu().numpy())
        clean_S = clean["S"]
        rows = {}
        for key in all_keys:
            ent = new_entries.get(key) or anchor_entries[key]
            z = np.load(ent["path"])
            avg = eval_masked(model, "MRFN", tensors["valid"], va_eval,
                              {"k": K_INSTANCES}, z, clean_S, bert, device)
            rows[key] = avg
            print(f"[MRFN s{seed}] {key}: S={avg['S']:.4f} D_S={avg['D_S']:+.4f} "
                  f"acc={avg['acc']:.4f}")
        per_seed[seed] = {"clean": clean, "masked": rows}
        del model

    # ---- 汇总（seed 均值）----
    mean_table = {}
    for key in all_keys:
        ms = {k: float(np.mean([per_seed[s]["masked"][key][k] for s in (1, 2, 3)]))
              for k in ("acc", "macro_f1", "mae", "pearson", "S", "D_S")}
        m = key.split("/")[1]
        hist = None
        ent = new_entries.get(key)
        if ent:
            hist_sum: dict = {}
            for s in ent["stats"]:
                for l, c in s["run_len_hist"].items():
                    hist_sum[l] = hist_sum.get(l, 0) + c
            ms["actual_rate_mean"] = float(np.mean(
                [s["actual_rate"][m] for s in ent["stats"]]))
        else:
            lib_e = [e for e in lib_index["entries"]
                     if f"{e['mechanism']}/{e['modalities']}/rate{e['rate']}/{e['position']}" == key
                     and e["split"] == "valid"][0]
            hist_sum = lib_e.get("run_len_hist_mean", {})
            ms["actual_rate_mean"] = float(np.mean(
                [lib_e["actual_rate_mean"][x] for x in m]))
        tot = sum(hist_sum.values())
        ms["mean_run_len"] = (round(sum(int(l) * c for l, c in hist_sum.items()) / tot, 3)
                              if tot else 0.0)
        mean_table[key] = {k: round(v, 6) if isinstance(v, float) else v
                           for k, v in ms.items()}

    clean_mean = {k: round(float(np.mean([per_seed[s]["clean"][k] for s in (1, 2, 3)])), 6)
                  for k in ("acc", "macro_f1", "mae", "pearson", "S")}
    checks = [
        {"name": "blocklen_instances_saved", "ok": len(new_entries) == len(COMBOS),
         "detail": len(new_entries)},
        {"name": "coverage_close_to_target",
         "ok": all(abs(mean_table[k]["actual_rate_mean"] - RATE / 100) < 0.02
                   for k in mean_table if k.startswith("blocklen")),
         "detail": {k: mean_table[k]["actual_rate_mean"] for k in mean_table
                    if k.startswith("blocklen")}},
        {"name": "anchors_present", "ok": all(k in mean_table for k in ANCHOR_KEYS),
         "detail": len(ANCHOR_KEYS)},
        {"name": "run_len_monotone_2_5_10",
         "ok": True,  # 逐模态核对 len2<len5<len10 的平均段长；散布/单块锚点仅报告
         "detail": {m: {"len2/5/10": [mean_table[f"blocklen/{m}/rate{RATE}/len{e}"]["mean_run_len"]
                                       for e in LENGTHS],
                        "mcar1": mean_table[f"mcar/{m}/rate{RATE}/random"]["mean_run_len"],
                        "single_block": mean_table[f"block/{m}/rate{RATE}/random"]["mean_run_len"]}
                    for m in MODS}},
    ]
    for m in MODS:
        seq = [mean_table[f"blocklen/{m}/rate{RATE}/len{e}"]["mean_run_len"]
               for e in LENGTHS]
        checks[3]["ok"] = checks[3]["ok"] and all(
            seq[i] < seq[i + 1] + 1e-9 for i in range(len(seq) - 1))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "Q2-E4 连续缺失长度扫描（固定覆盖率 40%、变块长；连续语义位置长度）",
        "protocol": ("valid 728 × K=8 实例指标平均；MRFN 主表 3-seed 冻结权重；"
                     "D_S 逐 seed 计算后取均值；锚点 = 冻结库 mcar@40/block@40"),
        "mask_seed": BLOCKLEN_SEED,
        "coverage_rate": RATE,
        "lengths": LENGTHS,
        "clean_mean": clean_mean,
        "mean_table": mean_table,
        "per_seed": {str(s): {"clean": v["clean"],
                              "masked": v["masked"]} for s, v in per_seed.items()},
        "checks": checks,
        "all_checks_pass": all(c["ok"] for c in checks),
    }
    (OUTDIR / "blocklen_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in checks:
        print(("✅" if c["ok"] else "❌"), c["name"])
    print("ALL PASS" if summary["all_checks_pass"] else "FAILED")
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
