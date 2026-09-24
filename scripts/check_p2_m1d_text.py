"""P2 M1-D 验收脚本：真实附件2/附件3 上的文本 [MASK] 重编码契约。

检查项：
  1. 保真：clean 重编码 vs 附件2 存储 text（32 条 train，逐样本最小位置余弦 ≥ 0.999）
  2. 替换：掩码库 train/mcar/t/rate40 实例驱动 [MASK] 替换——b 位恰为 103，
     attention/token_type/CLS/SEP/pad 逐位不变
  3. 冻结：requires_grad 全 False、eval、输出 requires_grad=False
  4. 掩码编码生效：b=1 位输出改变（"禁止事后置零"的存在性证明）
  5. 附件3 float32 token id：整数性+值域断言通过 → int64；篡改后必须拒绝
退出码：全部通过 = 0。
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import CONTRACT, load_aligned
from src.p2.text_mask import (DEFAULT_MODEL_PATH, MASK_ID, VOCAB_SIZE,
                              encode_text, load_frozen_bert, mask_text_tokens,
                              to_int_token_ids)

N_CHECK = 32
ATT3 = ROOT / "data/附件3-模态缺失特征样本/对齐版本/附件3_01.pkl"
OUT = ROOT / "runs/p2/text_encode"


def cos_min(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.astype(np.float64), b.astype(np.float64)
    c = (va * vb).sum(-1) / (np.linalg.norm(va, axis=-1) * np.linalg.norm(vb, axis=-1) + 1e-12)
    return float(c.min())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("✅" if ok else "❌"), name, detail if not ok else "")

    att = load_aligned()
    ids32 = att["train"]["text_bert"][:N_CHECK]
    stored_text = att["train"]["text"][:N_CHECK]

    model = load_frozen_bert(DEFAULT_MODEL_PATH, device="cpu")
    add("frozen_bert_flags", not model.training
        and not any(p.requires_grad for p in model.parameters()),
        {"training": model.training})

    clean = encode_text(ids32, model, CONTRACT)
    per_sample = [cos_min(clean[i], stored_text[i]) for i in range(N_CHECK)]
    worst = float(min(per_sample))
    add("clean_reencode_fidelity", worst >= 0.999,
        {"min_position_cos": round(worst, 6), "n": N_CHECK})

    lib = np.load(ROOT / "runs/p2/mask_library/train/mcar/t/rate40/random/seed2026.npz")
    b = lib["b_0_t"][:N_CHECK].astype(bool)
    masked_ids = mask_text_tokens(ids32, b, CONTRACT)
    from src.p2.data import derive_text_masks
    m = derive_text_masks(ids32, CONTRACT)
    ok_ids = bool((masked_ids[:, 0, :][b] == MASK_ID).all())
    ok_rest = bool((masked_ids[:, 1:, :] == ids32[:, 1:, :]).all()
                   and (masked_ids[:, 0, :][~b] == ids32[:, 0, :][~b]).all())
    add("mask_substitution_exact", ok_ids and ok_rest,
        {"b_positions": int(b.sum()), "ids_ok": ok_ids, "rest_ok": ok_rest,
         "b_outside_content": int((b & ~m.content).sum())})

    masked_h = encode_text(masked_ids, model, CONTRACT)
    diff_b = float(np.abs(masked_h[b] - clean[b]).max())
    diff_ctx = float(np.abs(masked_h[~b] - clean[~b]).mean())
    add("masked_encode_takes_effect", diff_b > 1e-6,
        {"max_abs_diff_at_b": round(diff_b, 6),
         "mean_abs_diff_context_note": round(diff_ctx, 8),
         "note": "上下文位输出也合法地改变（双向注意力），这正是禁止事后置零的原因"})

    # 附件3 float32 token id
    d3 = pickle.load(open(ATT3, "rb"))["test"]
    tb3_f32 = np.asarray(d3["text_bert"])
    add("att3_float32_tokens_pass_assertion",
        tb3_f32.dtype == np.float32 and to_int_token_ids(tb3_f32).dtype == np.int64,
        {"dtype": str(tb3_f32.dtype)})
    tampered = tb3_f32.copy()
    tampered[0, 0, 2] = np.round(tampered[0, 0, 2]) + 0.5
    try:
        to_int_token_ids(tampered)
        rejected = False
    except ValueError:
        rejected = True
    add("att3_tampered_rejected", rejected, {"rejected": rejected})

    rep = {"created_at": datetime.now(timezone.utc).isoformat(), "checks": checks,
           "all_checks_pass": all(c["ok"] for c in checks),
           "notes": ["P2 主线文本特征 = text_bert 现场重编码；附件2 存储 text 仅作保真对照",
                     "文本缺失位特征 = [MASK] 上下文编码（非零），可用性由 a_text=0 门控（M1-E 测试）"]}
    (OUT / "acceptance_m1d.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print("\nALL PASS" if rep["all_checks_pass"] else "FAILED")
    return 0 if rep["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
