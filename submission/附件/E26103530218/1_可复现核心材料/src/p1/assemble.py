"""Final P1 sample schema, validation, and train-only normalization statistics."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = "p1.v2"
MISSING_REASON_ENUM = (
    "observed",
    "no_face",
    "no_frame",
    "all_nan",
    "alignment_failed",
    "synthetic_masked",
)
SCHEMA_SPEC = {
    "version": SCHEMA_VERSION,
    "arrays": {
        "text_bert": ["int64", [3, 50]],
        "text": ["float32", [50, 768]],
        "audio": ["float32", [50, 25]],
        "vision": ["float32", [50, 23]],
        "attention_mask": ["int64", [50]],
        "content_mask": ["uint8", [50]],
        "special_mask": ["uint8", [50]],
        "padding_mask": ["uint8", [50]],
        "observed_mask_text": ["uint8", [50]],
        "observed_mask_audio": ["uint8", [50]],
        "observed_mask_video": ["uint8", [50]],
        "quality_text": ["float32", [50]],
        "quality_audio": ["float32", [50]],
        "quality_video": ["float32", [50]],
        "alpha": ["float32", [50]],
    },
    "mapping_fields": [
        "position", "token_id", "token", "char_start", "char_end", "word_id",
        "word_text", "wp_count_in_word", "alpha", "t_s", "t_e", "confidence",
        "attach_type", "trunc_flag", "alignment_status", "failure_reason",
    ],
    "missing_reason_enum": list(MISSING_REASON_ENUM),
    # —— v2 新增契约（此前仅存在于 data_manifest，schema hash 未覆盖——审阅修订）——
    "vision_feature_columns": [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
        "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
        "AU25", "AU26", "AU28", "AU43", "Pitch", "Roll", "Yaw",
    ],
    "audio_feature_contract": {
        "extractor": "opensmile.eGeMAPSv02.LowLevelDescriptors",
        "dim": 25,
        "hop_seconds": 0.01,
        "frame_center": "(k+0.5)*hop_seconds",
        "column_order": "lld_00..lld_24（opensmile 输出列序）",
    },
    "pooling_contract": {
        "interval": "[t_s,t_e) 半开",
        "vision_frame_time": "pts_time（禁 fps 推算）",
        "aggregation": "区间内有效帧 nanmean",
        "face_policy": "每帧取置信度 top-1 人脸",
        "vision_model_stages": "face+landmark+pose+AU（emotion 显式跳过）",
    },
    "grid_contract": {
        "tokenizer": "bert-base-uncased（本地 cache/hf/bert-base-uncased，pin 86b5e093）",
        "rule": "[CLS]+前48非特殊片+[SEP]，头部截断",
    },
    "full_word_map_fields": [
        "word_id", "word_text", "char_start", "char_end", "wp_positions",
        "attach_positions", "retained_positions", "trunc_flag", "dropped",
    ],
}
SCHEMA_HASH = hashlib.sha256(
    json.dumps(SCHEMA_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _enrich_maps(grid: dict, words: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    by_id = {int(word["word_id"]): word for word in words}
    fields = ("t_s", "t_e", "confidence", "alignment_status", "failure_reason")
    wp_map = copy.deepcopy(grid["wp_word_map"])
    for row in wp_map:
        word = by_id.get(row["word_id"])
        for field in fields:
            row[field] = word.get(field) if word is not None else None
    full_map = copy.deepcopy(grid["full_word_map"])
    for row in full_map:
        word = by_id.get(int(row["word_id"]))
        for field in fields:
            row[field] = word.get(field) if word is not None else None
    return wp_map, full_map


def assemble_sample(
    *,
    sample_id: str,
    raw_text: str,
    grid: dict,
    alignment: dict,
    text: np.ndarray,
    audio_grid: dict,
    vision_grid: dict,
    label: float | None = None,
    annotation: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble and validate one immutable P1 sample dictionary."""
    content = np.asarray(grid["content_mask"], dtype=np.uint8)
    text_observed = content.copy()
    text_quality = content.astype(np.float32)
    text_reason = ["observed" if value else None for value in content]
    wp_map, full_map = _enrich_maps(grid, alignment["words"])
    sample = {
        "schema_version": SCHEMA_VERSION,
        "schema_hash": SCHEMA_HASH,
        "id": sample_id,
        "raw_text": raw_text,
        "normalized_text": grid["normalized_text"],
        "label": None if label is None else float(label),
        "annotation": annotation,
        "text_bert": np.stack([
            grid["input_ids"], grid["attention_mask"], grid["token_type_ids"]
        ]).astype(np.int64, copy=False),
        "text": np.asarray(text, dtype=np.float32),
        "audio": np.asarray(audio_grid["values"], dtype=np.float32),
        "vision": np.asarray(vision_grid["values"], dtype=np.float32),
        "attention_mask": np.asarray(grid["attention_mask"], dtype=np.int64),
        "content_mask": content,
        "special_mask": np.asarray(grid["special_mask"], dtype=np.uint8),
        "padding_mask": np.asarray(grid["padding_mask"], dtype=np.uint8),
        "observed_mask_text": text_observed,
        "observed_mask_audio": np.asarray(audio_grid["observed_mask"], dtype=np.uint8),
        "observed_mask_video": np.asarray(vision_grid["observed_mask"], dtype=np.uint8),
        "quality_text": text_quality,
        "quality_audio": np.asarray(audio_grid["quality"], dtype=np.float32),
        "quality_video": np.asarray(vision_grid["quality"], dtype=np.float32),
        "missing_reason_text": text_reason,
        "missing_reason_audio": list(audio_grid["missing_reason"]),
        "missing_reason_video": list(vision_grid["missing_reason"]),
        "alpha": np.asarray(grid["alpha"], dtype=np.float32),
        "wp_word_time_map": wp_map,
        "full_word_map": full_map,
        "alignment": alignment,
        "feature_names_audio": list(audio_grid.get("feature_names", [])),
        "feature_names_vision": list(vision_grid.get("feature_names", [])),
        "metadata": dict(metadata or {}),
    }
    validate_sample(sample)
    return sample


def validate_sample(sample: dict[str, Any]) -> None:
    errors = []
    for name, (dtype, shape) in SCHEMA_SPEC["arrays"].items():
        value = sample.get(name)
        if not isinstance(value, np.ndarray):
            errors.append(f"{name}: not ndarray")
            continue
        if list(value.shape) != shape:
            errors.append(f"{name}: shape {value.shape} != {tuple(shape)}")
        if value.dtype != np.dtype(dtype):
            errors.append(f"{name}: dtype {value.dtype} != {dtype}")
        if not np.isfinite(value).all():
            errors.append(f"{name}: contains NaN/Inf")

    content = sample["content_mask"]
    special = sample["special_mask"]
    padding = sample["padding_mask"]
    if not np.all(content + special + padding == 1):
        errors.append("content/special/padding do not form an exclusive partition")
    if not np.array_equal(sample["attention_mask"], 1 - padding.astype(np.int64)):
        errors.append("attention_mask does not match structural masks")
    for name in ("text", "audio", "video"):
        observed = sample[f"observed_mask_{name}"]
        if np.any(observed & ~content):
            errors.append(f"observed_mask_{name} is active outside content")
    if sample["schema_version"] != SCHEMA_VERSION or sample["schema_hash"] != SCHEMA_HASH:
        errors.append("schema version/hash mismatch")
    if len(sample["wp_word_time_map"]) != 50:
        errors.append("wp_word_time_map must contain 50 rows")
    for modality in ("text", "audio", "video"):
        reasons = sample[f"missing_reason_{modality}"]
        if len(reasons) != 50:
            errors.append(f"missing_reason_{modality} must contain 50 entries")
        unknown = {reason for reason in reasons if reason is not None} - set(MISSING_REASON_ENUM)
        if unknown:
            errors.append(f"missing_reason_{modality} has unknown values {sorted(unknown)}")
    if errors:
        raise ValueError("invalid P1 sample: " + "; ".join(errors))


def compute_normalization_stats(
    samples: Iterable[dict[str, Any]],
    train_ids: set[str],
) -> dict[str, Any]:
    """Compute audio/vision statistics from observed content rows of explicit train IDs."""
    if not train_ids:
        raise ValueError("train_ids must be explicit and non-empty")
    selected = [sample for sample in samples if sample["id"] in train_ids]
    missing_ids = train_ids - {sample["id"] for sample in selected}
    if missing_ids:
        raise ValueError(f"normalization train IDs missing from samples: {sorted(missing_ids)}")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_hash": SCHEMA_HASH,
        "train_ids": sorted(train_ids),
    }
    for modality in ("audio", "vision"):
        rows = []
        for sample in selected:
            mask = sample["content_mask"].astype(bool) & sample[f"observed_mask_{'video' if modality == 'vision' else modality}"].astype(bool)
            rows.append(sample[modality][mask])
        values = np.concatenate(rows, axis=0).astype(np.float64)
        if values.shape[0] == 0:
            raise ValueError(f"no observed {modality} rows in normalization train samples")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-8] = 1.0
        result[modality] = {
            "count": int(values.shape[0]),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }
    return result
