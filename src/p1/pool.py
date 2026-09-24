"""Audio/vision interval pooling and copying to the 50-token grid.

The module stops at the P1 pooling boundary: it consumes word intervals from
``align.py`` and a grid from ``grid.py``, but does not assemble the final pkl
schema. Audio frame centers follow the frozen ``(k + 0.5) * 10 ms`` rule;
vision uses the ``pts_time`` supplied by ``media.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .align import HOP_SECONDS

AUDIO_HOP_SECONDS = 0.01
AUDIO_DIM = 25
VISION_AU_COLUMNS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]
VISION_POSE_COLUMNS = ["Pitch", "Roll", "Yaw"]
VISION_COLUMNS = VISION_AU_COLUMNS + VISION_POSE_COLUMNS


def _interval(word: dict[str, Any]) -> tuple[float, float] | None:
    start, end = word.get("t_s"), word.get("t_e")
    if start is None or end is None:
        return None
    start, end = float(start), float(end)
    if not np.isfinite([start, end]).all() or not (start < end):
        return None
    return start, end


def _nanmean_rows(values: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Return mean, finite-cell frame ratio, and whether any row is observed."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros(values.shape[-1] if values.ndim == 2 else 0, dtype=np.float32), 0.0, False
    finite_rows = np.isfinite(values).any(axis=1)
    if not finite_rows.any():
        return np.zeros(values.shape[1], dtype=np.float32), 0.0, False
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.nanmean(values[finite_rows], axis=0)
    pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return pooled, float(finite_rows.mean()), True


def _quality(confidence: float, frame_ratio: float, observed: bool) -> float:
    if not observed:
        return 0.0
    return float(np.clip(float(confidence) * float(frame_ratio), 0.0, 1.0))


def pool_audio_lld(
    lld: pd.DataFrame | np.ndarray,
    words: Sequence[dict[str, Any]],
    frame_times: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pool 25-dimensional eGeMAPSv02 LLD rows into word intervals.

    ``lld`` may be the DataFrame returned by ``opensmile.Smile`` or a numeric
    matrix. A successful all-zero/silent frame remains observed; only missing
    or all-NaN rows become unobserved.
    """
    if isinstance(lld, pd.DataFrame):
        values = lld.to_numpy(dtype=np.float32, copy=True)
    else:
        values = np.asarray(lld, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != AUDIO_DIM:
        raise ValueError(f"audio LLD must have shape (N, 25), got {values.shape}")
    if frame_times is None:
        frame_times = (np.arange(values.shape[0], dtype=np.float64) + 0.5) * AUDIO_HOP_SECONDS
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if frame_times.shape != (values.shape[0],):
        raise ValueError("frame_times must have one center timestamp per LLD row")
    if not np.all(np.isfinite(frame_times)):
        raise ValueError("audio frame_times contain NaN or Inf")

    vectors, observed, quality, reasons = [], [], [], []
    for word in words:
        interval = _interval(word)
        if interval is None:
            vectors.append(np.zeros(AUDIO_DIM, dtype=np.float32))
            observed.append(False)
            quality.append(0.0)
            reasons.append("alignment_failed")
            continue
        start, end = interval
        selected = (frame_times >= start) & (frame_times < end)
        pooled, ratio, is_observed = _nanmean_rows(values[selected])
        vectors.append(pooled)
        observed.append(is_observed)
        quality.append(_quality(float(word.get("confidence", 1.0)), ratio, is_observed))
        reasons.append("observed" if is_observed else ("no_frame" if not selected.any() else "all_nan"))
    return {
        "vectors": np.asarray(vectors, dtype=np.float32),
        "observed_mask": np.asarray(observed, dtype=np.uint8),
        "quality": np.asarray(quality, dtype=np.float32),
        "missing_reason": reasons,
        "frame_times": frame_times,
        "feature_names": [f"lld_{index:02d}" for index in range(AUDIO_DIM)],
    }


def extract_audio_lld(wav_path: str | Path):
    """Run the frozen openSMILE eGeMAPSv02 LowLevelDescriptors extractor."""
    import opensmile

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    result = smile.process_file(str(wav_path))
    if result.shape[1] != AUDIO_DIM:
        raise RuntimeError(f"openSMILE returned {result.shape[1]} columns, expected 25")
    return result


def _vision_row(frame_result: Any) -> tuple[np.ndarray, bool]:
    if not isinstance(frame_result, pd.DataFrame) or frame_result.empty:
        return np.zeros(len(VISION_COLUMNS), dtype=np.float32), False
    available = [column for column in VISION_COLUMNS if column in frame_result.columns]
    if len(available) != len(VISION_COLUMNS):
        missing = sorted(set(VISION_COLUMNS) - set(available))
        raise RuntimeError(f"Py-Feat output missing feature columns: {missing}")
    # Detector emits one row per face. Use the first detected face; the dataset
    # is single-speaker and this keeps frame-to-time ownership deterministic.
    row = frame_result.iloc[0][VISION_COLUMNS].to_numpy(dtype=np.float32)
    observed = bool(np.isfinite(row).any())
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0), observed


def _sanitize_faces(faces: list, batch_images: list) -> list:
    """过滤病态人脸框（py-feat 0.6.1 实测坑，2026-09-23）。

    retinaface 偶发输出越界/负值/退化框，会使 py-feat 的
    ``extract_face_from_bbox`` 在畸形裁剪上触发 Rescale 死循环
    （单帧可空转 20+ 分钟）。规则：有限性 → 裁剪到帧内 →
    裁剪后边长 <16px 则丢弃该框；整帧无有效框 = 该帧无人脸。

    ⚠ 输出必须保持 **list[list[float]]**（py-feat 原生格式）：
    feat.utils.is_list_of_lists_empty 对元素做 any()/bool()，
    numpy 数组会抛 ValueError（首版补丁的教训，84 条样本翻车）。

    每帧仅保留**置信度最高的 1 张脸**（top-1 by confidence）：
    数据集为单人访谈，下游 _vision_row 本就只取第一张脸；
    且 py-feat 逐脸裁剪（Rescale）极慢——实测 4 脸/帧的视频
    达 5.3 秒/帧（-UuX1xuaiiE$_$1，2026-09-23），top-1 把成本
    压回 ~1.5 秒/帧。选择语义 = 置信度最高者（与 retinaface
    输出序一致），比"第一行"更确定。
    """
    MAX_FACES_PER_FRAME = 1
    cleaned = []
    for frame_faces, image in zip(faces, batch_images):
        if frame_faces is None or len(frame_faces) == 0:
            cleaned.append([])
            continue
        height, width = np.asarray(image).shape[:2]
        rows = np.atleast_2d(np.asarray(frame_faces, dtype=np.float64))
        candidates = []
        for row in rows:
            box = np.asarray(row[:-1], dtype=np.float64)
            if not np.isfinite(box).all():
                continue
            x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            candidates.append([float(x1), float(y1), float(x2), float(y2), float(row[-1])])
        candidates.sort(key=lambda r: r[-1], reverse=True)
        cleaned.append(candidates[:MAX_FACES_PER_FRAME])
    return cleaned


def detect_vision_frames(
    frames: Sequence[dict[str, Any]],
    detector: Any | None = None,
    *,
    device: str = "cpu",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run Py-Feat through ``compat_feat.get_detector`` and retain PTS rows."""
    if detector is None:
        from compat_feat import get_detector

        detector = get_detector(device=device)
    frame_list = list(frames)
    paths = [str(frame["path"]) for frame in frame_list]
    pts = np.asarray([float(frame["pts_time"]) for frame in frame_list], dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        raise ValueError("vision pts_time contains NaN or Inf")
    if not paths:
        return {
            "values": np.zeros((0, len(VISION_COLUMNS)), dtype=np.float32),
            "observed_mask": np.zeros(0, dtype=np.uint8),
            "pts_time": pts,
            "feature_names": list(VISION_COLUMNS),
        }
    # ``Detector.detect_image`` also runs the optional emotion model. P1 only
    # requires AU+pose, so use the same detector's public stages directly and
    # keep emotion inference out of the full 100-clip runner.
    from PIL import Image

    images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    values, observed = [], []
    for start in range(0, len(images), max(1, int(batch_size))):
        batch = torch.from_numpy(
            np.stack(images[start:start + max(1, int(batch_size))])
        ).permute(0, 3, 1, 2)
        faces = detector.detect_faces(batch)
        faces = _sanitize_faces(faces, images[start:start + max(1, int(batch_size))])
        landmarks = detector.detect_landmarks(batch, faces)
        poses_result = detector.detect_facepose(batch, landmarks)
        faces, poses = detector._match_faces_to_poses(
            faces, poses_result["faces"], poses_result["poses"]
        )
        aus = detector.detect_aus(batch, landmarks)
        for offset, frame_faces in enumerate(faces):
            frame_index = start + offset
            if not frame_faces or not landmarks[frame_index - start]:
                values.append(np.zeros(len(VISION_COLUMNS), dtype=np.float32))
                observed.append(False)
                continue
            pose_rows = poses[frame_index - start]
            au_rows = aus[frame_index - start]
            if len(pose_rows) == 0 or len(au_rows) == 0:
                values.append(np.zeros(len(VISION_COLUMNS), dtype=np.float32))
                observed.append(False)
                continue
            au_row = np.asarray(au_rows[0], dtype=np.float32).reshape(-1)
            pose_row = np.asarray(pose_rows[0], dtype=np.float32).reshape(-1)
            if au_row.size != len(VISION_AU_COLUMNS) or pose_row.size < len(VISION_POSE_COLUMNS):
                raise RuntimeError(
                    f"Py-Feat returned AU/pose dimensions {au_row.size}/{pose_row.size}, "
                    f"expected {len(VISION_AU_COLUMNS)}/{len(VISION_POSE_COLUMNS)}+"
                )
            row = np.concatenate([au_row, pose_row[:len(VISION_POSE_COLUMNS)]])
            is_observed = bool(np.isfinite(row).any())
            values.append(np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0))
            observed.append(is_observed)
    return {
        "values": np.asarray(values, dtype=np.float32),
        "observed_mask": np.asarray(observed, dtype=np.uint8),
        "pts_time": pts,
        "feature_names": list(VISION_COLUMNS),
    }


def pool_vision_frames(
    frame_features: dict[str, Any],
    words: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Pool detected AU/pose rows by PTS-centered half-open word intervals."""
    values = np.asarray(frame_features["values"], dtype=np.float32)
    observed_frames = np.asarray(frame_features["observed_mask"], dtype=bool)
    pts = np.asarray(frame_features["pts_time"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(VISION_COLUMNS):
        raise ValueError(f"vision features must have shape (N, {len(VISION_COLUMNS)}), got {values.shape}")
    if values.shape[0] != pts.size or observed_frames.shape != (pts.size,):
        raise ValueError("vision values, observed_mask and pts_time have inconsistent lengths")
    vectors, observed, quality, reasons = [], [], [], []
    for word in words:
        interval = _interval(word)
        if interval is None:
            vectors.append(np.zeros(len(VISION_COLUMNS), dtype=np.float32))
            observed.append(False)
            quality.append(0.0)
            reasons.append("alignment_failed")
            continue
        start, end = interval
        selected = (pts >= start) & (pts < end)
        valid = selected & observed_frames
        pooled, ratio, is_observed = _nanmean_rows(values[valid])
        vectors.append(pooled)
        observed.append(is_observed)
        quality.append(_quality(float(word.get("confidence", 1.0)), float(valid.sum() / max(1, selected.sum())), is_observed))
        if is_observed:
            reasons.append("observed")
        elif not selected.any():
            reasons.append("no_frame")
        else:
            reasons.append("no_face")
    return {
        "vectors": np.asarray(vectors, dtype=np.float32),
        "observed_mask": np.asarray(observed, dtype=np.uint8),
        "quality": np.asarray(quality, dtype=np.float32),
        "missing_reason": reasons,
        "feature_names": list(VISION_COLUMNS),
    }


def copy_word_features_to_grid(
    word_features: dict[str, Any],
    grid: dict[str, Any],
) -> dict[str, Any]:
    """Copy word-level vectors to mapped WordPiece/punctuation grid positions."""
    vectors = np.asarray(word_features["vectors"], dtype=np.float32)
    word_observed = np.asarray(word_features["observed_mask"], dtype=np.uint8)
    word_quality = np.asarray(word_features["quality"], dtype=np.float32)
    reasons = list(word_features["missing_reason"])
    if vectors.ndim != 2 or word_observed.shape != (vectors.shape[0],):
        raise ValueError("word feature arrays have inconsistent shapes")
    if word_quality.shape != word_observed.shape or len(reasons) != vectors.shape[0]:
        raise ValueError("word quality/reason arrays have inconsistent shapes")

    output = np.zeros((50, vectors.shape[1]), dtype=np.float32)
    observed = np.zeros(50, dtype=np.uint8)
    quality = np.zeros(50, dtype=np.float32)
    missing_reason: list[str | None] = [None] * 50
    for row in grid["wp_word_map"]:
        position = int(row["position"])
        word_id = row["word_id"]
        if word_id is None or position >= 50:
            continue
        word_id = int(word_id)
        if word_id >= vectors.shape[0]:
            missing_reason[position] = "alignment_failed"
            continue
        output[position] = vectors[word_id]
        observed[position] = word_observed[word_id]
        quality[position] = word_quality[word_id]
        missing_reason[position] = reasons[word_id]
    return {
        "values": output,
        "observed_mask": observed,
        "quality": quality,
        "missing_reason": missing_reason,
        "feature_names": list(word_features.get("feature_names", [])),
    }
