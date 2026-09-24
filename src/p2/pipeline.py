"""P2 输入管线（M2+）：z-score 标准化 + o 占位重置 + clean 文本缓存。

契约 §1.3：audio/vision 用 clean train 逐维统计量（runs/p2/data/normalization_stats.npz），
o=0 位重置 0（x 无定义、0 仅为占位，模型不读取由掩码保证）；text 不做 z-score
（text_mask 现场重编码）。M1-B 遗留项"标准化后所有有效值无 NaN/Inf"由 zscore_reset
内的断言闭合。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STATS_PATH = Path(__file__).resolve().parents[2] / "runs/p2/data/normalization_stats.npz"


def load_stats(path: Path = STATS_PATH) -> dict:
    """normalization_stats.npz → {"audio": {"mean","std"}, "vision": ...}（canonical 口径）。"""
    z = np.load(path)
    return {m: {"mean": z[f"mean_{m}"], "std": z[f"std_{m}"]} for m in ("audio", "vision")}


def zscore_reset(x: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 o: np.ndarray, name: str = "x") -> np.ndarray:
    """(N,T,D) 原始值 → 标准化 float32；o=0 位重置 0（占位语义）。

    o=1 位出现 NaN/Inf → 抛错（M1-B 遗留项的闭合点）；o=0 位不检查（永不读取）。
    """
    x = np.asarray(x, dtype=np.float64)
    out = (x - mean[None, None, :]) / std[None, None, :]
    o_b = np.asarray(o).astype(bool)
    out = np.where(o_b[..., None], out, 0.0)
    if not np.isfinite(out).all():
        n_bad = int((~np.isfinite(out)).sum())
        raise RuntimeError(f"{name} 标准化后在 o=1 位出现 NaN/Inf（{n_bad} 位）")
    return out.astype(np.float32)
