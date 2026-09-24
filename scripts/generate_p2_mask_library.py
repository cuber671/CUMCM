"""P2 M1-C 掩码库生成（真实数据）：从 m1a_state.npz 的 o 三态生成固定掩码库。

流程：
  1. 读 runs/p2/data/m1a_state.npz（o 三态 + ids），核对来源 sha 与当前附件2 一致；
  2. 实测附件3 联合缺失段长分布（预注册核对：605 content 位 / 131 联合零行）；
  3. 按预注册网格（missing.library_grid，47 组合 × K=8 实例 × 3 split）生成 b；
  4. 落盘 mask_library/<split>/<mech>/<mods>/rate<r>/<pos>/seed<seed>.npz
     + library_index.json + acceptance.json。
退出码：全部验收通过 = 0。
"""
from __future__ import annotations

import glob
import hashlib
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import ALIGNED_PKL, sha256_file
from src.p2 import missing as M

STATE = ROOT / "runs/p2/data/m1a_state.npz"
ATT3_DIR = ROOT / "data/附件3-模态缺失特征样本/对齐版本"
OUT = ROOT / "runs/p2/mask_library"
SPLITS = ("train", "valid", "test")
O_KEY = {"t": "o_text", "a": "o_audio", "v": "o_vision"}   # m1a_state.npz 键名


def load_split(state, split):
    return (state[f"{split}_content"].astype(bool),
            {m: state[f"{split}_{O_KEY[m]}"].astype(bool) for m in ("t", "a", "v")})


def compute_att3_joint_pmf(att3_dir: Path) -> dict:
    """附件3 联合缺失（audio/vision 同位零行）段长经验分布（模型索引空间）。"""
    files = sorted(glob.glob(str(att3_dir / "附件3_*.pkl")))
    runs, n_content, n_joint = [], 0, 0
    for f in files:
        d = pickle.load(open(f, "rb"))["test"]
        tb = np.asarray(d["text_bert"])
        am = (tb[:, 1, :] > 0.5).astype(int)
        T = am.shape[1]
        first0 = np.where(am.min(axis=1) == 1, T, np.argmax(am == 0, axis=1))
        sep = np.where(am[:, -1] == 1, T - 1, first0 - 1)
        pos = np.arange(T)[None, :]
        content = (am == 1) & (pos != 0) & (pos != sep[:, None])
        aud = np.abs(np.asarray(d["audio"])).max(axis=-1) == 0
        vis = np.abs(np.asarray(d["vision"])).max(axis=-1) == 0
        jz = content & aud & vis
        n_content += int(content.sum())
        n_joint += int(jz.sum())
        idx = np.where(jz[0])[0]
        if len(idx):
            for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                runs.append(len(seg))
    c4 = {ell: runs.count(ell) for ell in (1, 2, 3, 4)}
    tot = sum(c4.values())
    return {"n_files": len(files), "n_content_positions": n_content,
            "n_joint_zero": n_joint, "n_runs": len(runs),
            "raw_counts": c4, "runs_gt4": [r for r in runs if r > 4],
            "mean_seg_len": round(float(np.mean(runs)), 4),
            "pmf": [c4[ell] / tot for ell in (1, 2, 3, 4)]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = np.load(STATE, allow_pickle=False)
    cur_sha = sha256_file(ALIGNED_PKL)
    st_sha = str(state["source_sha256"])
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("✅" if ok else "❌"), name, "" if ok else detail)

    add("state_source_sha_matches_attachment2", st_sha == cur_sha,
        {"state": st_sha[:16], "current": cur_sha[:16]})

    pmf_info = compute_att3_joint_pmf(ATT3_DIR)
    add("att3_preregistered_counts",
        pmf_info["n_content_positions"] == M.ATT3_CONTENT
        and pmf_info["n_joint_zero"] == M.ATT3_JOINT_ZERO,
        {"got": [pmf_info["n_content_positions"], pmf_info["n_joint_zero"]],
         "expect": [M.ATT3_CONTENT, M.ATT3_JOINT_ZERO]})
    drift = float(np.abs(np.array(pmf_info["pmf"]) - np.array(M.ATT3_SEGMENT_PMF)).max())
    add("att3_pmf_matches_frozen", drift < 0.01, {"max_abs_diff": round(drift, 5)})
    print("附件3 pmf:", {l: round(p, 4) for l, p in zip((1, 2, 3, 4), pmf_info["pmf"])})

    grid = M.library_grid()
    index, failures = [], []
    t0 = datetime.now(timezone.utc).isoformat()
    for split in SPLITS:
        ids = state[f"{split}_ids"]
        content, o_by_mod = load_split(state, split)
        for ci, combo in enumerate(grid):
            mods = list(combo["modalities"])
            if set(mods) == {"t", "a", "v"}:
                failures.append(f"{split}#ci{ci}: 三模态全抹组合禁止")
                continue
            instances = M.gen_combo_instances(combo, ci, content, o_by_mod, pmf_info["pmf"])
            stats = [M.instance_stats(bm, o_by_mod, content) for bm in instances]
            path = M.combo_dir(OUT, split, combo)
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {"ids": ids}
            for k_i, bm in enumerate(instances):
                for m, b in bm.items():
                    arrays[f"b_{k_i}_{m}"] = b
            meta = {"mask_library_seed": M.MASK_LIBRARY_SEED, "combo_idx": ci, "split": split,
                    "mechanism": combo["mechanism"], "modalities": combo["modalities"],
                    "rate_requested": combo["rate"], "position": combo["position"],
                    "tags": combo["tags"], "k_instances": len(instances),
                    "ids_sample": [str(x) for x in ids[:3]],
                    "att3_pmf_used": [round(p, 6) for p in pmf_info["pmf"]],
                    "generated_at": t0}
            arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
            np.savez_compressed(path, **arrays)

            req = combo["rate"] / 100.0
            pooled = {m: round(float(np.mean([s["actual_rate"][m] for s in stats])), 6)
                      for m in mods}
            for m in mods:
                act = pooled[m]
                if combo["mechanism"] == "mcar" and abs(act - req) > 0.02:
                    failures.append(f"{split}/{combo}/seed.. {m}: mcar 实际率 {act} 偏离 {req}")
                if combo["mechanism"] in ("block", "joint") and act > req + 0.02:
                    failures.append(f"{split}/{combo} {m}: 实际率 {act} 超请求 {req}")
                if combo["mechanism"] == "full" and act < 1.0:
                    failures.append(f"{split}/{combo} {m}: full 实际率 {act} != 1.0")
            index.append({"split": split, "combo_idx": ci, "mechanism": combo["mechanism"],
                          "modalities": combo["modalities"], "rate": combo["rate"],
                          "position": combo["position"], "tags": combo["tags"],
                          "path": str(path.relative_to(OUT)), "k": len(instances),
                          "actual_rate_mean": pooled,
                          "run_len_hist_mean": _mean_hist(stats)})
    add("no_triple_modality_combo", not failures, failures[:5] or "none")

    # 全库逐位审计：b=1 ⟹ o=1；ids 一致；K=8
    n_files = n_bad_mask = n_bad_ids = n_bad_k = 0
    for split in SPLITS:
        o_by_mod = load_split(state, split)[1]
        ids = state[f"{split}_ids"]
        for ci, combo in enumerate(grid):
            p = M.combo_dir(OUT, split, combo)
            z = np.load(p, allow_pickle=False)
            mods = list(combo["modalities"])
            n_files += 1
            if len([k for k in z.files if k.startswith("b_0_")]) != len(mods):
                n_bad_k += 1
            if not (z["ids"] == ids).all():
                n_bad_ids += 1
            for key in z.files:
                if key.startswith("b_"):
                    m = key.rsplit("_", 1)[1]
                    if int(((z[key] == 1) & ~o_by_mod[m]).sum()):
                        n_bad_mask += 1
    add("library_audit_b_subset_o", n_bad_mask == 0, {"violations": n_bad_mask})
    add("library_audit_ids_aligned", n_bad_ids == 0, {"violations": n_bad_ids})
    add("library_audit_k_instances", n_bad_k == 0 and n_files == len(grid) * len(SPLITS),
        {"files": n_files, "expect": len(grid) * len(SPLITS), "bad_k": n_bad_k})

    # 决定论终验：全库重生成逐位比对（同 seed 必须逐位复现）
    n_mismatch = 0
    for split in SPLITS:
        content_r, o_r = load_split(state, split)
        for entry in index:
            if entry["split"] != split:
                continue
            combo_r = grid[entry["combo_idx"]]
            inst = M.gen_combo_instances(combo_r, entry["combo_idx"], content_r, o_r,
                                         pmf_info["pmf"])
            z = np.load(OUT / entry["path"], allow_pickle=False)
            for k_i, bm in enumerate(inst):
                for m, b in bm.items():
                    if not (b == z[f"b_{k_i}_{m}"]).all():
                        n_mismatch += 1
    add("library_reproducible_bitwise", n_mismatch == 0,
        {"mismatched_arrays": n_mismatch})

    index_doc = {"mask_library_seed": M.MASK_LIBRARY_SEED, "k_instances": M.K_INSTANCES,
                 "created_at": t0, "generator": "src/p2/missing.py",
                 "grid_size": len(grid), "splits": list(SPLITS),
                 "att3_pmf_measurement": pmf_info,
                 "att3_pmf_frozen": list(M.ATT3_SEGMENT_PMF),
                 "rate_denominator": "o_m=1 的自然可用位（2026-09-24 拍板）",
                 "entries": index}
    (OUT / "library_index.json").write_text(json.dumps(index_doc, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    acc = {"created_at": t0, "seed": M.MASK_LIBRARY_SEED, "checks": checks,
           "all_checks_pass": all(c["ok"] for c in checks)}
    (OUT / "acceptance.json").write_text(json.dumps(acc, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"\n库: {OUT}（{n_files} 文件）  索引: library_index.json")
    print("ALL PASS" if acc["all_checks_pass"] else "FAILED")
    return 0 if acc["all_checks_pass"] else 1


def _mean_hist(stats: list) -> dict:
    """K 实例的段长直方均值（保留小数）。"""
    agg = Counter()
    for s in stats:
        for k, v in s["run_len_hist"].items():
            agg[k] += v
    n = len(stats)
    return {k: round(v / n, 2) for k, v in sorted(agg.items(), key=lambda x: int(x[0]))}


if __name__ == "__main__":
    raise SystemExit(main())
