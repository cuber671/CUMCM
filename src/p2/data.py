"""P2 M1-A 数据契约：附件2 aligned_50.pkl 的加载、结构校验、自然观测推导与 train-only 标准化。

契约依据：docs/问题二实施步骤.md §1（契约 v1.0，2026-09-24）+ M1-A 审阅纪要。
全部预注册，改动必须走讨论：

- 唯一训练/验证来源 = 附件2 aligned（audio 74 / vision 35）；附件3 只走推理入口。
- 自然观测代理：content 位且该模态整行严格 == 0（含 -0.0）→ o_m = 0，否则 1；
  **非 content 位（special/padding）o_m 一律 0**（2026-09-24 审阅拍板，与 P1
  observed_mask 语义统一，o_m 自带 content 限定，下游直接用 o_m 即可）。
  o_text = content_mask（附件2 文本 content 位无真实缺失；text 的 padding 行是
  [PAD] 的 BERT 编码、实测 100% 非零——"行非零"不构成观测证据，o_text 不由数值推导）。
- 标准化：clean train、content 且 o_m=1 的位置、逐维 z-score、float64、ddof=0；
  std < 1e-8 的常量维以 1.0 替代（防除零 NaN，2026-09-24 契约补充）；
  text 不做 z-score（P1 方案 §5 既定）。
- 缺失位语义：x[a=0] 无定义，x=0 仅为数值占位，模型任何路径不得读取（§1.3）。
- 数据事实（2026-09-24 实测）：audio/vision 源 dtype 为 float64，text 为 float32，
  text_bert 为 int64；非 content 位（special+padding）的 audio/vision 严格全零。

本模块只依赖 numpy；tokenizer 复核见 scripts/check_p2_m1a.py。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ALIGNED_PKL = ROOT / "data/附件2-数据集特征文件/aligned_50.pkl"

CLS_ID, SEP_ID = 101, 102
STD_EPS = 1e-8
CONTRACT_VERSION = "m1a.v1"

EXPECTED_FIELDS = ("id", "raw_text", "text_bert", "text", "audio", "vision",
                   "classification_labels", "regression_labels")


@dataclass(frozen=True)
class Contract:
    """数据契约（可参数化以便单测；默认值 = 附件2 实测契约，预注册）。"""

    counts: dict = field(default_factory=lambda: {"train": 3395, "valid": 728, "test": 727})
    d_audio: int = 74
    d_vision: int = 35
    d_text: int = 768
    t_grid: int = 50
    labels: tuple = (0, 1, 2)
    rl_range: tuple = (-3.0, 3.0)
    cls_counts_train: dict = field(default_factory=lambda: {0: 967, 1: 758, 2: 1670})


CONTRACT = Contract()


@dataclass
class Check:
    name: str
    ok: bool
    detail: dict


@dataclass
class TextMasks:
    content: np.ndarray   # (N,T) bool：真实语义 WordPiece
    special: np.ndarray   # (N,T) bool：CLS 与 SEP
    padding: np.ndarray   # (N,T) bool：attention=0
    sep_pos: np.ndarray   # (N,)   int64


@dataclass
class FeatureStats:
    mean: np.ndarray      # (D,) float64
    std: np.ndarray       # (D,) float64（已应用 std<eps→1.0）
    std_raw: np.ndarray
    n_positions: int
    clipped_dims: list


def load_aligned(path: Path = ALIGNED_PKL) -> dict:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def derive_text_masks(text_bert: np.ndarray, contract: Contract = CONTRACT) -> TextMasks:
    """由 text_bert 三行（input_ids/attention/token_type）推导 content/special/padding 分区。

    附件2 无显式 mask，分区完全由其自身结构推导，不依赖 tokenizer 或 P1 产物。
    违反结构假设（CLS@0、SEP@最后有效位、attention 单调、token_type 全零）即抛错。
    """
    text_bert = np.asarray(text_bert)
    T = contract.t_grid
    if text_bert.ndim != 3 or text_bert.shape[1:] != (3, T):
        raise ValueError(f"text_bert 应为 (N,3,{T})，实际 {text_bert.shape}")
    iid = text_bert[:, 0, :].astype(np.int64)
    am = text_bert[:, 1, :].astype(np.int64)
    n = iid.shape[0]

    if not np.all((am == 0) | (am == 1)):
        raise ValueError("attention_mask 取值超出 {0,1}")
    if not np.all(iid[:, 0] == CLS_ID):
        raise ValueError("text_bert 位置 0 并非全为 [CLS]=101")
    mono = np.all((np.cumsum(am == 0, axis=1) == 0) | (am == 0))
    if not mono:
        raise ValueError("attention_mask 非单调（前缀 1 后缀 0）")
    first0 = np.where(am.min(axis=1) == 1, T, np.argmax(am == 0, axis=1))
    sep_pos = np.where(am[:, -1] == 1, T - 1, first0 - 1).astype(np.int64)
    if not np.all(iid[np.arange(n), sep_pos] == SEP_ID):
        raise ValueError("SEP 不在 attention 最后有效位（违反附件2 既定结构）")
    if not np.all(text_bert[:, 2, :] == 0):
        raise ValueError("token_type_ids 非全零（单句假设破坏）")

    pos = np.arange(T)[None, :]
    special = (pos == 0) | (pos == sep_pos[:, None])
    padding = am == 0
    content = ~special & ~padding
    return TextMasks(content, special, padding, sep_pos)


def derive_observed(audio: np.ndarray, vision: np.ndarray,
                    content: np.ndarray) -> dict:
    """自然观测代理（契约 §1.2）：content 位且整行严格 == 0 → o=0，否则 1；
    非 content 位一律 0（与 P1 observed_mask 语义统一，o_m 自带 content 限定）。
    o_text = content（等价于"content 位 o_text≡1"，不由数值推导）。

    已知混淆（留痕，不在此处理）：音频真实静音词整行为零会被误判为缺失。
    """
    def _o(arr):
        zero_row = np.abs(arr).max(axis=-1) == 0  # 严格零（含 -0.0），不用阈值
        return content & ~zero_row

    return {
        "o_text": content.copy(),
        "o_audio": _o(audio),
        "o_vision": _o(vision),
    }


def compute_stats(arr: np.ndarray, include: np.ndarray, eps: float = STD_EPS) -> FeatureStats:
    """逐维 z-score 统计量：只统计 include mask 为 True 的位置，float64、ddof=0。

    std < eps（默认 1e-8）的常量维以 1.0 替代并记录（防除零 NaN）。
    """
    vals = np.asarray(arr)[include].astype(np.float64)
    if vals.shape[0] == 0:
        raise ValueError("纳入统计的位置数为 0")
    mean = vals.mean(axis=0)
    std_raw = vals.std(axis=0)  # ddof=0，预注册
    clipped = np.where(std_raw < eps)[0].tolist()
    std = std_raw.copy()
    std[clipped] = 1.0
    return FeatureStats(mean, std, std_raw, int(vals.shape[0]), clipped)


def validate_aligned(att: dict, contract: Contract = CONTRACT) -> list:
    """对附件2 加载结果执行全部结构校验，返回 Check 列表（脚本汇总入报告）。"""
    checks: list = []

    def add(name, fn):
        try:
            ok, detail = fn()
            ok = bool(ok)
        except (ValueError, IndexError, KeyError) as e:
            # 校验器必须报告违例而非崩溃：维度不一致抛 IndexError、缺失 split 抛 KeyError
            ok, detail = False, {"error": repr(e)}
        checks.append(Check(name, ok, detail))

    def _split_keys():
        got, want = set(att.keys()), set(contract.counts)
        return got == want, {"got": sorted(got), "expect": sorted(want)}

    def _split_counts():
        got = {sp: len(att[sp]["id"]) for sp in contract.counts}
        return got == dict(contract.counts), {"got": got, "expect": dict(contract.counts)}

    def _fields():
        bad = {sp: sorted(set(att[sp].keys()) - set(EXPECTED_FIELDS))
               for sp in att if set(att[sp].keys()) != set(EXPECTED_FIELDS)}
        return not bad, {"splits_with_wrong_fields": bad or "none"}

    def _shapes_dtypes():
        dt, exp_fail = {}, []
        for sp in contract.counts:
            n = len(att[sp]["id"])
            spec = {"audio": (n, contract.t_grid, contract.d_audio),
                    "vision": (n, contract.t_grid, contract.d_vision),
                    "text": (n, contract.t_grid, contract.d_text),
                    "text_bert": (n, 3, contract.t_grid)}
            for k, shape in spec.items():
                a = np.asarray(att[sp][k])
                dt[f"{sp}.{k}"] = str(a.dtype)
                if a.shape != shape:
                    exp_fail.append(f"{sp}.{k}{a.shape}!={shape}")
            if not np.issubdtype(np.asarray(att[sp]["text_bert"]).dtype, np.integer):
                exp_fail.append(f"{sp}.text_bert 非整型 dtype")
        return not exp_fail, {"violations": exp_fail or "none", "dtypes": dt}

    def _ids():
        seen = set()
        for sp in contract.counts:
            ids = list(att[sp]["id"])
            if len(set(ids)) != len(ids):
                return False, {"error": f"{sp} 内部 id 重复"}
            dup = seen.intersection(ids)
            if dup:
                return False, {"error": f"跨 split 重复 id", "examples": sorted(dup)[:5]}
            seen.update(ids)
        return True, {"total": len(seen), "unique": len(seen)}

    def _finite():
        bad = {}
        for sp in contract.counts:
            for k in ("audio", "vision", "text"):
                a = np.asarray(att[sp][k])
                n_nan, n_inf = int(np.isnan(a).sum()), int(np.isinf(a).sum())
                if n_nan or n_inf:
                    bad[f"{sp}.{k}"] = {"nan": n_nan, "inf": n_inf}
        return not bad, {"violations": bad or "none"}

    def _tb_structure():
        det = {}
        for sp in contract.counts:
            m = derive_text_masks(att[sp]["text_bert"], contract)  # 违例即抛 ValueError
            c = m.content.sum(axis=1)
            det[sp] = {"content_per_sample_min": int(c.min()), "max": int(c.max()),
                       "empty_content_samples": int((c == 0).sum()),
                       "full_length_samples": int((m.sep_pos == contract.t_grid - 1).sum())}
        return True, det

    def _av_noncontent_zero():
        bad = {}
        for sp in contract.counts:
            m = derive_text_masks(att[sp]["text_bert"], contract)
            nc = ~(m.content)
            for k in ("audio", "vision"):
                a = np.asarray(att[sp][k])
                n_bad = int((np.abs(a[nc]).max(axis=-1) > 0).sum())
                if n_bad:
                    bad[f"{sp}.{k}"] = n_bad
        return not bad, {"noncontent_nonzero_rows": bad or "none",
                         "note": "special+padding 位的 audio/vision 应严格全零（附件2 既定结构）"}

    def _field_lengths():
        bad = []
        for sp in contract.counts:
            n = len(att[sp]["id"])
            for k in ("raw_text", "classification_labels", "regression_labels"):
                ln = len(att[sp][k])
                if ln != n:
                    bad.append(f"{sp}.{k} 长度 {ln} != n={n}")
        return not bad, {"violations": bad or "none"}

    def _labels():
        det, bad = {}, []
        for sp in contract.counts:
            cl = np.asarray(att[sp]["classification_labels"], dtype=np.float64)
            rl = np.asarray(att[sp]["regression_labels"], dtype=np.float64)
            finite = bool(np.all(np.isfinite(cl)) and np.all(np.isfinite(rl)))
            fcl = cl[np.isfinite(cl)]
            u, c = np.unique(fcl, return_counts=True)
            frl = rl[np.isfinite(rl)]
            det[sp] = {"finite": finite,
                       "cls_counts": {int(a): int(b) for a, b in zip(u, c)},
                       "rl_min": float(frl.min()) if frl.size else None,
                       "rl_max": float(frl.max()) if frl.size else None}
            if not finite:
                bad.append(f"{sp}.labels 存在 NaN/Inf")
                continue
            if not set(u.tolist()) <= set(contract.labels):
                bad.append(f"{sp}.cls 出现契约外标签")
            lo, hi = contract.rl_range
            if float(rl.min()) < lo or float(rl.max()) > hi:
                bad.append(f"{sp}.rl 越界 [{rl.min():.3f},{rl.max():.3f}]")
        got = {int(k): v for k, v in det["train"]["cls_counts"].items()}
        if got != dict(contract.cls_counts_train):
            bad.append(f"train cls 计数 {got} != 契约 {dict(contract.cls_counts_train)}")
        return not bad, {"violations": bad or "none", "detail": det}

    def _raw_text():
        bad = [f"{sp}[{i}]" for sp in contract.counts
               for i, t in enumerate(att[sp]["raw_text"]) if not str(t).strip()]
        return not bad, {"empty": bad[:5] or "none"}

    add("split_keys_exact", _split_keys)
    add("split_counts", _split_counts)
    add("fields_exact", _fields)
    add("shapes_dtypes", _shapes_dtypes)
    add("field_lengths_aligned", _field_lengths)
    add("id_uniqueness", _ids)
    add("finite_arrays", _finite)
    add("text_bert_structure", _tb_structure)
    add("av_noncontent_zero", _av_noncontent_zero)
    add("labels_valid", _labels)
    add("raw_text_nonempty", _raw_text)
    return checks


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _summary_o(o: np.ndarray, content: np.ndarray) -> dict:
    c = content
    n = int(c.sum())
    n0 = int((~o[c]).sum())
    return {"n_content_positions": n, "n_o0": n0,
            "rate_o0": round(n0 / n, 6) if n else None}


def build_m1a(att: dict, outdir: Path, contract: Contract = CONTRACT,
              extra_checks: tuple = (), source_path: Path = ALIGNED_PKL) -> dict:
    """M1-A 产物：结构校验 + o 基础结构（非 content 位为 0）+ train-only 统计量 + 报告。

    source_path 必须显式对应 att 的实际来源（2026-09-24 审阅修正）：
    sha256 按该路径计算并写入报告与 m1a_state.npz，杜绝"产物来自其他对象
    却署官方 SHA"的冒名风险。
    校验失败时只写 m1a_report.json（含失败详情与 artifacts_written=false），
    不写任何 npz 产物，防止污染输入留下可被误消费的正式产物。

    产物（outdir 下，仅在全部检查通过时）：
      normalization_stats.npz —— mean/std/std_raw（canonical 与 reference 双口径）
      m1a_state.npz           —— 各 split 的 ids/三区 mask/o 三态/标签 + 来源指纹
      m1a_report.json         —— 全部校验、o 率、统计量元数据、来源 sha256
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path)
    src_sha = sha256_file(source_path)
    try:
        src_disp = str(source_path.resolve().relative_to(ROOT))
    except ValueError:
        src_disp = str(source_path)
    checks = list(validate_aligned(att, contract)) + list(extra_checks)
    all_ok = all(c.ok for c in checks)

    report = {
        "contract_version": CONTRACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": src_disp, "sha256": src_sha},
        "contract": {
            "counts": dict(contract.counts),
            "dims": {"audio": contract.d_audio, "vision": contract.d_vision, "text": contract.d_text},
            "t_grid": contract.t_grid,
            "labels": list(contract.labels), "rl_range": list(contract.rl_range),
            "cls_counts_train": {str(k): v for k, v in contract.cls_counts_train.items()},
            "std_eps": STD_EPS, "std_rule": "std<eps → 1.0（常量维防 NaN）",
            "stats_scope": "clean train, content 且 o_m=1 的位置, float64, ddof=0",
        },
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
        "all_checks_pass": all_ok,
    }

    if not all_ok:
        report["artifacts_written"] = False
        (outdir / "m1a_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return report

    per_split = {}
    for sp in contract.counts:
        m = derive_text_masks(att[sp]["text_bert"], contract)
        o = derive_observed(att[sp]["audio"], att[sp]["vision"], m.content)
        per_split[sp] = {
            "masks": m,
            "observed": o,
            "o_summary": {
                "audio": _summary_o(o["o_audio"], m.content),
                "vision": _summary_o(o["o_vision"], m.content),
                "text": {"n_content_positions": int(m.content.sum()), "n_o0": 0, "rate_o0": 0.0,
                         "note": "o_text=content_mask：content 位恒为 1，special/padding 位为 0"
                                 "（契约 §1.2，不由数值推导）"},
            },
        }

    # ---- train-only 统计量（canonical = content&o；reference = 全 content，仅供审阅对照）----
    tr_m = per_split["train"]["masks"]
    tr_o = per_split["train"]["observed"]
    stats, ref = {}, {}
    for mod, key, d in (("audio", "o_audio", contract.d_audio), ("vision", "o_vision", contract.d_vision)):
        stats[mod] = compute_stats(att["train"][mod], tr_m.content & tr_o[key])
        ref[mod] = compute_stats(att["train"][mod], tr_m.content)

    np.savez_compressed(
        outdir / "normalization_stats.npz",
        **{f"mean_{m}": stats[m].mean for m in stats},
        **{f"std_{m}": stats[m].std for m in stats},
        **{f"std_raw_{m}": stats[m].std_raw for m in stats},
        **{f"ref_mean_{m}": ref[m].mean for m in stats},
        **{f"ref_std_{m}": ref[m].std for m in stats},
        **{f"n_positions_{m}": np.int64(stats[m].n_positions) for m in stats},
    )

    state = {}
    for sp in contract.counts:
        m, o = per_split[sp]["masks"], per_split[sp]["observed"]
        state[f"{sp}_ids"] = np.array(list(att[sp]["id"]), dtype=object).astype(str)
        state[f"{sp}_sep_pos"] = m.sep_pos
        for name, arr in (("content", m.content), ("special", m.special), ("padding", m.padding),
                          ("o_text", o["o_text"]), ("o_audio", o["o_audio"]), ("o_vision", o["o_vision"])):
            state[f"{sp}_{name}"] = arr.astype(np.uint8)
        state[f"{sp}_cls"] = np.asarray(att[sp]["classification_labels"]).astype(np.int64)
        state[f"{sp}_rl"] = np.asarray(att[sp]["regression_labels"]).astype(np.float64)
    state["source_path"] = src_disp
    state["source_sha256"] = src_sha
    np.savez_compressed(outdir / "m1a_state.npz", **state)

    report["o_summary"] = {sp: per_split[sp]["o_summary"] for sp in per_split}
    report["stats_meta"] = {
        m: {"n_positions": stats[m].n_positions,
            "clipped_dims": stats[m].clipped_dims,
            "std_min_raw": round(float(stats[m].std_raw.min()), 8),
            "reference_all_content_n": ref[m].n_positions,
            "max_abs_mean_diff_vs_reference": round(
                float(np.abs(stats[m].mean - ref[m].mean).max()), 6),
            "max_abs_std_diff_vs_reference": round(
                float(np.abs(stats[m].std_raw - ref[m].std_raw).max()), 6)}
        for m in stats}
    report["notes"] = [
        "非 content 位 o_m=0（2026-09-24 审阅拍板）；content 位 b≡0、a≡o 为 M1-A 基础态，合成缺失 b 在 M1-C 引入（契约 §2）。",
        "附件2 text 的 padding 行 100% 非零（[PAD] 的 BERT 编码），任何 text 池化必须 content 掩码。",
        "归一化只覆盖 audio/vision；text 不做 z-score（P1 方案 §5）。",
    ]
    report["artifacts_written"] = True
    (outdir / "m1a_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report
