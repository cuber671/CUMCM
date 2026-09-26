"""P2 M6：附件3（模态局部缺失专项测试集，30 条无标签）全量推理与行为分析。

流程：
  1. 逐条读取附件3 对齐版（text_bert float32 → 整数性/值域断言 → int64，契约 §1.4）；
  2. 推导结构掩码与自然缺失（o = content & 行非零；联合缺失 = a、v 同位零行）；
  3. 文本现场重编码（frozen BERT）；audio/vision 用 train 统计量标准化；
  4. MRFN+（R8 BFT-lite）三种子概率/回归集成 → pred_polarity / pred_intensity / gates；
  5. text-only 对照（a、v 全不可用）→ 分歧分析（a/v 缺失越高应越趋同，门控旁证）；
  6. 置信度温度校准（T 在 valid 上 NLL 拟合，见 runs/p2/ensemble_check/）。
纪律（方案 §10）：无标签，不做任何精度声明，不反推标签。
产物：runs/p2/fj3/附件3_预测结果.csv + 附件3_行为分析.json。
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_p2_mrfn import load_stats  # noqa: E402  （训练器内的统计量工具）

from src.p2.data import CONTRACT  # noqa: E402
from src.p2.models import MODEL_REGISTRY  # noqa: E402
from src.p2.pipeline import zscore_reset  # noqa: E402
from src.p2.text_mask import (BertTextEncoder, DEFAULT_MODEL_PATH,  # noqa: E402
                              mask_text_tokens, to_int_token_ids)

ATT3 = ROOT / "data/附件3-模态缺失特征样本/对齐版本"
CKPT = ROOT / "runs/p2/bft/MRFN_seed{seed}/checkpoint.pt"
BERT_CKPT = ROOT / "runs/p2/bft/MRFN_seed{seed}/bert_checkpoint.pt"
OUT = ROOT / "runs/p2/fj3"
MODS = ("text", "audio", "vision")

# 方案 §1 预注册核对数
PREREG = {"n_files": 30, "n_content": 605, "n_joint_zero": 131, "max_joint_rate": 0.50}

# 温度标定系数（2026-09-24 valid NLL 拟合，MRFN+ 3-seed 集成口径）：
# ECE 0.0424→0.0347（−18%），argmax 逐位不变；作用于集成平均概率 p^(1/T) 重归一。
TEMP = 1.2142


def run_lengths(mask_row: np.ndarray) -> list:
    idx = np.where(mask_row)[0]
    if len(idx) == 0:
        return []
    return [len(s) for s in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(str(ATT3 / "附件3_*.pkl")))
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("✅" if ok else "❌"), name, detail if not ok else "")

    # ---- 1. 读取与逐样本统计 ----
    rows, stats = [], []
    n_content = n_joint = 0
    for f in files:
        d = pickle_load(f)["test"]
        sid = Path(f).stem
        tb = to_int_token_ids(np.asarray(d["text_bert"]))          # §1.4 断言
        am = tb[0, 1, :]
        first0 = int(np.where(am == 0)[0][0]) if (am == 0).any() else len(am)
        sep = len(am) - 1 if am[-1] == 1 else first0 - 1
        pos = np.arange(len(am))
        content = (am == 1) & (pos != 0) & (pos != sep)
        aud = np.asarray(d["audio"])[0]
        vis = np.asarray(d["vision"])[0]
        miss_a = content & (np.abs(aud).max(-1) == 0)
        miss_v = content & (np.abs(vis).max(-1) == 0)
        joint = miss_a & miss_v
        n_content += int(content.sum()); n_joint += int(joint.sum())

        def mstats(mmask):
            rl = run_lengths(mmask)
            return {"miss_rate": round(float(mmask.sum()) / max(int(content.sum()), 1), 6),
                    "mean_seg_len": round(float(np.mean(rl)), 4) if rl else 0.0,
                    "n_runs": len(rl)}
        sa, sv, sj = mstats(miss_a), mstats(miss_v), mstats(joint)
        stats.append({"sample_id": sid, "n_content": int(content.sum()),
                      "miss_text": 0.0, "miss_audio": sa, "miss_vision": sv,
                      "miss_joint": sj,
                      "any_joint": bool(joint.any())})
        rows.append({"sample_id": sid, "tb": tb, "audio": aud[None], "vision": vis[None],
                     "content": content[None], "miss_a": miss_a[None], "miss_v": miss_v[None],
                     "joint": joint[0]})

    add("file_count", len(files) == PREREG["n_files"], {"got": len(files)})
    add("preregistered_counts",
        n_content == PREREG["n_content"] and n_joint == PREREG["n_joint_zero"],
        {"content": n_content, "joint_zero": n_joint,
         "expect": [PREREG["n_content"], PREREG["n_joint_zero"]]})
    jrate = [x["miss_joint"]["miss_rate"] for x in stats]
    add("max_joint_rate", abs(max(jrate) - PREREG["max_joint_rate"]) < 1e-6,
        {"got": max(jrate), "expect": PREREG["max_joint_rate"]})
    n_any = sum(x["any_joint"] for x in stats)
    add("samples_with_joint_missing", n_any == 27, {"got": n_any, "expect": 27})

    # ---- 2. 组装 batch 输入 ----
    tb_all_np = np.concatenate([r["tb"] for r in rows], axis=0)          # (30,3,50) int64
    content_all = np.concatenate([r["content"] for r in rows], axis=0)
    o_a = content_all & ~(np.concatenate([r["miss_a"] for r in rows], 0))
    o_v = content_all & ~(np.concatenate([r["miss_v"] for r in rows], 0))
    stats_l = load_stats()
    aud_n = zscore_reset(np.concatenate([r["audio"] for r in rows], 0),
                         stats_l["audio"]["mean"], stats_l["audio"]["std"], o_a, "att3.audio")
    vis_n = zscore_reset(np.concatenate([r["vision"] for r in rows], 0),
                         stats_l["vision"]["mean"], stats_l["vision"]["std"], o_v, "att3.vision")

    def to_t(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    aud_t = to_t(aud_n)
    vis_t = to_t(vis_n)
    content_t = torch.from_numpy(content_all).to(device)
    tb_t = to_t(tb_all_np)
    avail = {"text": content_t,
             "audio": torch.from_numpy(o_a).to(device),
             "vision": torch.from_numpy(o_v).to(device)}

    # ---- 3. MRFN+（BFT）三种子集成推理：每 seed 用自己的微调 BERT 编码文本 ----
    prob_sum, reg_sum, gates_sum, n_models = None, None, None, 0
    per_seed_pred = []
    for seed in (1, 2, 3):
        be = BertTextEncoder(unfreeze_last=1).to(device)   # R8 = 解冻 1 层
        be.load_state_dict(torch.load(str(BERT_CKPT).format(seed=seed), weights_only=True))
        be.eval()
        model = MODEL_REGISTRY["MRFN"](dropout=0.3).to(device)
        model.load_state_dict(torch.load(str(CKPT).format(seed=seed), weights_only=True))
        model.eval()
        with torch.no_grad():
            text_f = be(tb_t).cpu().numpy()
        feats = {"text": to_t(text_f), "audio": aud_t, "vision": vis_t}
        with torch.no_grad():
            o = model(feats, content_t, avail)
        p = torch.softmax(o["logits"], dim=-1)
        prob_sum = p if prob_sum is None else prob_sum + p
        reg_sum = o["reg"] if reg_sum is None else reg_sum + o["reg"]
        gates_sum = o["gates"] if gates_sum is None else gates_sum + o["gates"]
        n_models += 1
        per_seed_pred.append({"seed": seed,
                              "polarity": o["logits"].argmax(1).cpu().tolist()})
    prob = (prob_sum / n_models).cpu().numpy()
    reg = (reg_sum / n_models).cpu().numpy()
    gates = (gates_sum / n_models).cpu().numpy()
    pred_polarity = prob.argmax(1)
    # ---- 温度校准（只动置信度，不动预测）：p_cal ∝ p^(1/T) 重归一 ----
    prob_cal = prob ** (1.0 / TEMP)
    prob_cal /= prob_cal.sum(1, keepdims=True)
    assert (prob_cal.argmax(1) == pred_polarity).all(), "校准改变 argmax，违反预注册"
    # 跨进程重推理保护：极性/强度必须与现有交付逐行一致，否则中止不写盘
    old_csv = OUT / "附件3_预测结果.csv"
    if old_csv.exists():
        import csv as _csv
        with open(old_csv, encoding="utf-8-sig") as f:
            old = list(_csv.DictReader(f))
        diff_pol = sum(int(r["pred_polarity"]) != int(p) for r, p in zip(old, pred_polarity))
        diff_int = sum(abs(float(r["pred_intensity"]) - float(x)) > 5e-4
                       for r, x in zip(old, reg))
        assert diff_pol == 0 and diff_int == 0, \
            f"重推理与现有交付不一致（极性 {diff_pol} 行/强度 {diff_int} 行），中止"
        print(f"✅ 重推理与现有交付逐行一致（30/30），仅重算置信度列")
    conf = prob_cal.max(1)

    # ---- 4. text-only 对照（a、v 全不可用）----
    avail_to = {"text": content_t,
                "audio": torch.zeros_like(content_t),
                "vision": torch.zeros_like(content_t)}
    with torch.no_grad():
        o_to = model({"text": to_t(text_f), "audio": aud_t, "vision": vis_t}, content_t, avail_to)
    to_pred = o_to["logits"].argmax(1).cpu().numpy()
    agree = (to_pred == pred_polarity).astype(int)
    av_miss = np.array([(x["miss_audio"]["miss_rate"] + x["miss_vision"]["miss_rate"]) / 2
                        for x in stats])

    # ---- 5. CSV + 行为分析 ----
    import csv
    csv_path = OUT / "附件3_预测结果.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "pred_polarity", "pred_intensity", "pred_confidence",
                    "n_content_positions",
                    "miss_rate_text", "miss_rate_audio", "miss_rate_vision",
                    "joint_miss_rate", "mean_seg_len_audio", "mean_seg_len_vision",
                    "mean_seg_len_joint",
                    "gate_text", "gate_audio", "gate_vision",
                    "text_only_pred_polarity", "text_only_agree"])
        for i, x in enumerate(stats):
            w.writerow([x["sample_id"], int(pred_polarity[i]), round(float(reg[i]), 4),
                        round(float(conf[i]), 4), x["n_content"],
                        x["miss_text"], x["miss_audio"]["miss_rate"],
                        x["miss_vision"]["miss_rate"], x["miss_joint"]["miss_rate"],
                        x["miss_audio"]["mean_seg_len"], x["miss_vision"]["mean_seg_len"],
                        x["miss_joint"]["mean_seg_len"],
                        round(float(gates[i, 0]), 4), round(float(gates[i, 1]), 4),
                        round(float(gates[i, 2]), 4),
                        int(to_pred[i]), int(agree[i])])

    from scipy.stats import spearmanr
    gate_av_corr = float(spearmanr(av_miss, gates[:, 0]).statistic)
    hi, lo = np.argsort(av_miss)[len(av_miss) // 2:], np.argsort(av_miss)[:len(av_miss) // 2]
    behavior = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "MRFN+（R8 BFT-lite：BERT 解冻 1 层 + B 分量 t×2 + S_select 选点，"
                 "五门槛全过采纳；3 seeds 概率/回归集成）",
        "calibration": {"temperature": TEMP, "fitted_on": "valid NLL（MRFN+ 集成口径）",
                        "ece": "0.0424→0.0347", "argmax_invariant": True},
        "n": len(files),
        "checks": checks, "all_checks_pass": all(c["ok"] for c in checks),
        "missing_overview": {
            "joint_zero_rows": n_joint, "content_rows": n_content,
            "joint_rate_mean": round(float(np.mean(jrate)), 4),
            "joint_rate_max": round(max(jrate), 4),
            "samples_with_joint_missing": n_any,
            "audio_miss_rate_mean": round(float(np.mean([x["miss_audio"]["miss_rate"]
                                                         for x in stats])), 4),
            "vision_miss_rate_mean": round(float(np.mean([x["miss_vision"]["miss_rate"]
                                                          for x in stats])), 4)},
        "gate_analysis": {
            "gate_text_mean": round(float(gates[:, 0].mean()), 4),
            "gate_audio_mean": round(float(gates[:, 1].mean()), 4),
            "gate_vision_mean": round(float(gates[:, 2].mean()), 4),
            "spearman(av缺失率, gate_text)": round(gate_av_corr, 6),
            "gate_text_hi_av_missing": round(float(gates[hi, 0].mean()), 4),
            "gate_text_lo_av_missing": round(float(gates[lo, 0].mean()), 4),
            "note": "a/v 缺失越高 g_text 越大 → 模型信任向文本重分配（门控行为旁证）"},
        "text_only_convergence": {
            "agree_rate_overall": round(float(agree.mean()), 4),
            "agree_rate_high_av_missing": round(float(agree[hi].mean()), 4),
            "agree_rate_low_av_missing": round(float(agree[lo].mean()), 4),
            "note": "a/v 缺失越高，全模态预测与 text-only 预测越趋同"},
        "hard_samples": {"rule": "a/v 平均缺失率降序，同率按预测置信度升序",
                         "samples": [stats[i]["sample_id"]
                                     for i in np.lexsort((conf, -av_miss))[:5]]},
        "per_seed_pred_agreement": {
            f"seed{s['seed']}": round(float(np.mean(np.array(s["polarity"]) == pred_polarity)), 4)
            for s in per_seed_pred},
        "discipline": "无标签测试集：不做精度声明、不反推标签（方案 §10）",
    }
    (OUT / "附件3_行为分析.json").write_text(json.dumps(behavior, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"\nCSV: {csv_path}")
    print(f"预测分布: {np.bincount(pred_polarity, minlength=3).tolist()} (neg/neu/pos)")
    print(f"gate_text 与 a/v 缺失率 Spearman: {gate_av_corr:.4f}")
    print("ALL PASS" if behavior["all_checks_pass"] else "PARTIAL")
    return 0 if behavior["all_checks_pass"] else 1


def pickle_load(f):
    import pickle
    return pickle.load(open(f, "rb"))


if __name__ == "__main__":
    raise SystemExit(main())
