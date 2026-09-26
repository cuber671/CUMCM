"""P3 M1 数据契约：附件4（可解释专项测试集，20 条无标签）加载、校验与预注册口径。

契约（docs/问题三建模方案.md §1 + M1 预注册 2026-09-25）：
- 条数 20：对齐版本 01–20.pkl；目录内 videos/ 为渲染用视频子目录（非样本，曾致 21 项计数歧义）；
- schema：dict{raw_text, audio:(50,74), vision:(50,35), text:(50,768), text_bert:(3,50)}。
  text_bert 三行 = input_ids/attention/token_type，与附件2 同构仅少 batch 维（loader 统一补维）；
- 模型输入 = 对齐版本特征（附件2 同口径 74/35 维）；自然观测按 P2 契约 §1.2 推导（复用 src.p2.data）；
- 分词复现规则：tok(raw_text, truncation=True, max_length=50) 逐项复现 text_bert 有效段（20/20；
  #07/#18 恰好 50 长截断，官方规则保留 [SEP]，裸前缀截取会在位置 49 失配——据此固化复现规则）；
- 跨附件锚点：#03/#07/#08/#12/#15 与附件2 test 对应行四特征逐位相等（组织者复用同一样本），
  仅用于内部一致性核验与解释卡旁证，不改变“附件4 无标签、不做精度声明”纪律（方案 §9）；
- 已知数据观察（留痕不阻断）：#13/#16 未对齐 vision_lengths 字段失真（=1，实际非零行 17/40）；
  #13 对齐 vision 50 行全零 → vision 整模态自然缺失（唯一一条）；组织者对齐帧率非常数
  （vision_lengths/时长 ∈ [6.6,15]/s），物理时间一律以 P1 CTC 对齐为准，禁止帧序×常数帧率换算。

预注册数字（M6 交付前不得改动）：
  n_files=20, n_content=564, miss_audio=0, miss_vision=30, vision_missing_samples={13},
  att2_overlap={3,7,8,12,15}, tokenize_repro=20/20。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.p2.data import CONTRACT, Check, TextMasks, derive_observed, derive_text_masks

ROOT = Path(__file__).resolve().parents[2]
ATT4_ROOT = ROOT / "data/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件"
ALIGNED_DIR = ATT4_ROOT / "对齐版本"
UNALIGNED_DIR = ATT4_ROOT / "未对齐版本"
VIDEOS_DIR = UNALIGNED_DIR / "videos"
ALIGNED_VIDEOS_DIR = ALIGNED_DIR / "videos"

N_SAMPLES = 20
PREREG: dict[str, Any] = {
    "n_files": N_SAMPLES,
    "n_content": 564,
    "miss_audio": 0,
    "miss_vision": 30,
    "vision_missing_samples": [13],
    "att2_overlap": [3, 7, 8, 12, 15],
    "tokenize_repro": 20,
}

FIELD_SHAPES = {"audio": (50, 74), "vision": (50, 35), "text": (50, 768),
                "text_bert": (3, 50)}


@dataclass
class Att4Sample:
    """单条样本（batch 维已补齐，与附件2 split 数组同构，可直接走 P2 契约推导）。"""
    n: int                       # 1..20（文件名编号，即 sample_id）
    raw_text: str
    audio: np.ndarray            # (1,50,74)
    vision: np.ndarray           # (1,50,35)
    text: np.ndarray             # (1,50,768)
    text_bert: np.ndarray        # (1,3,50) float（整数性由 P2 契约保证）


def _as_batch3(arr: np.ndarray, name: str) -> np.ndarray:
    """附件4 无 batch 维 → 统一补为 (1,*shape)；已有则原样通过。"""
    arr = np.asarray(arr)
    want = (1,) + FIELD_SHAPES[name]
    if arr.shape == want:
        return arr
    if arr.shape == FIELD_SHAPES[name]:
        return arr[None]
    raise ValueError(f"{name} 应为 {want[1:]}（可带 batch 维），实际 {arr.shape}")


def load_att4(n: int | None = None) -> list[Att4Sample]:
    """读取对齐版本样本；n=None 读全量 20 条。schema 硬校验失败即抛错。"""
    ns = range(1, N_SAMPLES + 1) if n is None else [n]
    out = []
    for i in ns:
        d = pickle.load(open(ALIGNED_DIR / f"{i:02d}.pkl", "rb"))
        if set(d) != set(FIELD_SHAPES) | {"raw_text", "id"}:
            raise ValueError(f"#{i:02d} 字段集异常: {sorted(d)}")
        if str(d["id"]) != f"{i:02d}":
            raise ValueError(f"#{i:02d} id 字段 {d['id']!r} 与文件名不一致")
        out.append(Att4Sample(
            n=i, raw_text=str(d["raw_text"]),
            audio=_as_batch3(d["audio"], "audio"),
            vision=_as_batch3(d["vision"], "vision"),
            text=_as_batch3(d["text"], "text"),
            text_bert=_as_batch3(d["text_bert"], "text_bert")))
    return out


def att4_masks(s: Att4Sample) -> tuple[TextMasks, dict[str, np.ndarray]]:
    """结构分区 + 自然观测（复用 P2 契约推导；失败即附件4 破坏附件2 结构假设）。"""
    tm = derive_text_masks(s.text_bert.astype(np.float32))
    obs = derive_observed(s.audio, s.vision, tm.content)
    return tm, obs


def tokenize_repro_check(samples: list[Att4Sample]) -> Check:
    """分词复现规则核验：truncation=True, max_length=50 逐项复现 text_bert 有效段。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "cache/hf/bert-base-uncased"))
    ok, bad = 0, []
    for s in samples:
        ids = tok(s.raw_text, truncation=True, max_length=50)["input_ids"]
        ref = s.text_bert[0, 0].astype(np.int64)
        ref = ref[ref != 0].tolist()
        if ids == ref:
            ok += 1
        else:
            bad.append(s.n)
    return Check("tokenize_repro", ok == len(samples),
                 {"ok": ok, "total": len(samples), "bad": bad})


def att2_anchor_check(samples: list[Att4Sample]) -> Check:
    """跨附件锚点：预注册 5 条与附件2 test 四特征逐位相等。"""
    a2 = pickle.load(open(ROOT / "data/附件2-数据集特征文件/aligned_50.pkl", "rb"))
    texts = [str(t) for t in a2["test"]["raw_text"]]
    hits, mismatch = [], []
    for s in samples:
        if s.n not in PREREG["att2_overlap"]:
            continue
        j = texts.index(s.raw_text)
        equal = all(np.array_equal(
            getattr(s, f)[0] if getattr(s, f).ndim == 3 else getattr(s, f),
            np.asarray(a2["test"][f][j])) for f in FIELD_SHAPES)
        (hits if equal else mismatch).append((s.n, str(a2["test"]["id"][j])))
    return Check("att2_anchor", sorted(h[0] for h in hits) == PREREG["att2_overlap"]
                 and not mismatch,
                 {"hits": [h[0] for h in hits], "mismatch": mismatch,
                  "expect": PREREG["att2_overlap"]})


def prereg_summary(samples: list[Att4Sample]) -> dict[str, Any]:
    """预注册数字实测（content 总数 / 自然缺失 / vision 整模态缺失样本）。"""
    n_content = miss_a = miss_v = 0
    v_missing_samples = []
    for s in samples:
        tm, obs = att4_masks(s)
        c = tm.content[0]
        n_content += int(c.sum())
        miss_a += int((c & ~obs["o_audio"][0]).sum())
        miss_v += int((c & ~obs["o_vision"][0]).sum())
        if not obs["o_vision"][0].any():
            v_missing_samples.append(s.n)
    return {"n_files": len(samples), "n_content": n_content, "miss_audio": miss_a,
            "miss_vision": miss_v, "vision_missing_samples": sorted(v_missing_samples)}
