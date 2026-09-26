"""Resumable end-to-end runner for the 100 P1 source clips."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import pickle
import subprocess
import tempfile

import torch
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .align import ALIGNMENT_CONFIG, get_aligner
from .assemble import (
    MISSING_REASON_ENUM,
    SCHEMA_HASH,
    SCHEMA_SPEC,
    SCHEMA_VERSION,
    assemble_sample,
    compute_normalization_stats,
    validate_sample,
)
from .grid import build_grid, load_tokenizer
from .media import FF, extract_audio_16k, extract_frames_pts, probe_duration
from .pool import (
    VISION_COLUMNS,
    copy_word_features_to_grid,
    detect_vision_frames,
    extract_audio_lld,
    pool_audio_lld,
    pool_vision_frames,
)
from .text import TextEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "p1"
ATTACHMENT2_PATH = PROJECT_ROOT / "data" / "附件2-数据集特征文件" / "aligned_50.pkl"


def sample_id(video_id: str, clip_id: int) -> str:
    return f"{video_id}$_${int(clip_id)}"


def _safe_name(identifier: str) -> str:
    return identifier.replace("$_$", "__").replace("/", "_")


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        staged = Path(handle.name)
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    staged.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fps_from_pts(frames: list[dict]) -> tuple[float, bool]:
    pts = np.asarray([frame["pts_time"] for frame in frames], dtype=np.float64)
    deltas = np.diff(pts)
    deltas = deltas[deltas > 0]
    if not deltas.size:
        return 0.0, False
    median = float(np.median(deltas))
    variable = bool(np.max(np.abs(deltas - median)) > max(1e-6, median * 0.01))
    return float(1.0 / median), variable


def _coverage(sample: dict, modality: str) -> float:
    content = sample["content_mask"].astype(bool)
    observed = sample[f"observed_mask_{modality}"].astype(bool)
    return float(observed[content].mean()) if content.any() else 0.0


def _reason_counts(reasons: list[str | None]) -> str:
    return json.dumps(Counter(reason for reason in reasons if reason), sort_keys=True)


def _split_sets() -> tuple[set[str], set[str]]:
    with ATTACHMENT2_PATH.open("rb") as handle:
        attachment = pickle.load(handle)
    return set(attachment["train"]["id"]), set(attachment["test"]["id"])


def _tool_versions() -> dict[str, str]:
    packages = ("torch", "torchaudio", "transformers", "opensmile", "py-feat", "imageio-ffmpeg")
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    ffmpeg = subprocess.run([FF, "-version"], capture_output=True, text=True)
    result["ffmpeg"] = ffmpeg.stdout.splitlines()[0] if ffmpeg.stdout else "unknown"
    return result


class P1Runner:
    def __init__(
        self,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        *,
        device: str | None = None,
        vision_batch_size: int = 8,
    ) -> None:
        self.output_root = Path(output_root)
        self.sample_dir = self.output_root / "samples"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.vision_batch_size = vision_batch_size
        self.tokenizer = load_tokenizer()
        self.text_encoder = TextEncoder(device=device)
        self.aligner = get_aligner(device=device)
        self.attachment_train_ids, self.attachment_test_ids = _split_sets()

    def process_row(self, row: Any, input_root: Path = DEFAULT_INPUT_ROOT) -> tuple[dict, dict]:
        identifier = sample_id(str(row.video_id), int(row.clip_id))
        video = input_root / str(row.video_id) / f"{int(row.clip_id)}.mp4"
        with tempfile.TemporaryDirectory(prefix=f"p1-{_safe_name(identifier)}-") as temp_name:
            temp = Path(temp_name)
            wav = extract_audio_16k(video, temp / "audio.wav")
            frames = extract_frames_pts(video, temp / "frames")
            duration = probe_duration(video)
            grid = build_grid(str(row.text), self.tokenizer)
            alignment = self.aligner.align_file(wav, str(row.text))
            lld = extract_audio_lld(wav)
            audio_words = pool_audio_lld(lld, alignment["words"])
            audio_grid = copy_word_features_to_grid(audio_words, grid)
            vision_frames = detect_vision_frames(
                frames, device=str(self.aligner.device), batch_size=self.vision_batch_size
            )
            vision_words = pool_vision_frames(vision_frames, alignment["words"])
            vision_grid = copy_word_features_to_grid(vision_words, grid)
            text = self.text_encoder.encode(grid)
            fps, is_vfr = _fps_from_pts(frames)
            split = (
                "calibration" if identifier in self.attachment_train_ids
                else "holdout" if identifier in self.attachment_test_ids
                else "unassigned"
            )
            sample = assemble_sample(
                sample_id=identifier,
                raw_text=str(row.text),
                grid=grid,
                alignment=alignment,
                text=text,
                audio_grid=audio_grid,
                vision_grid=vision_grid,
                label=float(row.label),
                annotation=str(row.annotation),
                metadata={
                    "video_id": str(row.video_id),
                    "clip_id": int(row.clip_id),
                    "split": split,
                    "duration": duration,
                    "fps": fps,
                    "is_vfr": is_vfr,
                    "frame_count": len(frames),
                },
            )
        retained_words = [word for word in sample["full_word_map"] if not word["dropped"]]
        aligned_duration = sum(
            max(0.0, float(word["t_e"]) - float(word["t_s"]))
            for word in sample["full_word_map"]
            if word["t_s"] is not None and word["t_e"] is not None
        )
        retained_duration = sum(
            max(0.0, float(word["t_e"]) - float(word["t_s"]))
            for word in retained_words
            if word["t_s"] is not None and word["t_e"] is not None
        )
        manifest = {
            "id": identifier,
            "video_id": str(row.video_id),
            "clip_id": int(row.clip_id),
            "split": sample["metadata"]["split"],
            "status": "ok",
            "error": "",
            "schema_version": SCHEMA_VERSION,
            "schema_hash": SCHEMA_HASH,
            "duration": sample["metadata"]["duration"],
            "fps": sample["metadata"]["fps"],
            "is_vfr": sample["metadata"]["is_vfr"],
            "frame_count": sample["metadata"]["frame_count"],
            "text_len": len(sample["full_word_map"]),
            "wp_len": int(sample["content_mask"].sum()),
            "retained_word_ratio": len(retained_words) / max(1, len(sample["full_word_map"])),
            "retained_duration_ratio": retained_duration / max(1e-12, aligned_duration),
            "alignment_status": alignment["status"],
            "alignment_failure_rate": alignment["failure_rate"],
            "data_anomaly": alignment["status"] == "rollback",
            "audio_coverage": _coverage(sample, "audio"),
            "vision_coverage": _coverage(sample, "video"),
            "audio_missing_counts": _reason_counts(sample["missing_reason_audio"]),
            "vision_missing_counts": _reason_counts(sample["missing_reason_video"]),
            "label": sample["label"],
            "annotation": sample["annotation"],
        }
        return sample, manifest

    def run(
        self,
        input_root: str | Path = DEFAULT_INPUT_ROOT,
        *,
        limit: int | None = None,
        resume: bool = True,
    ) -> list[dict]:
        input_root = Path(input_root)
        labels = pd.read_excel(input_root / "label-100.xlsx", sheet_name="label")
        if limit is not None:
            # 审阅修订（2026-09-23）：防止 --limit 冒烟覆盖正式全量产物
            existing = list(self.sample_dir.glob("*.pkl"))
            if len(existing) > int(limit):
                raise RuntimeError(
                    f"--limit {limit} 将用少量样本重写 {self.output_root} 的全量汇总产物"
                    f"（现有 {len(existing)} 个样本 pkl）。冒烟测试请改用独立 output-root，"
                    "或先显式清空/移走现有产物。"
                )
            labels = labels.iloc[:limit]
        manifest_rows = []
        for index, row in enumerate(labels.itertuples(), 1):
            identifier = sample_id(str(row.video_id), int(row.clip_id))
            sample_path = self.sample_dir / f"{_safe_name(identifier)}.pkl"
            try:
                if resume and sample_path.is_file():
                    with sample_path.open("rb") as handle:
                        sample = pickle.load(handle)
                    validate_sample(sample)
                    _, manifest = self.process_manifest_only(sample, row)
                else:
                    sample, manifest = self.process_row(row, input_root)
                    _atomic_pickle(sample, sample_path)
                print(f"[{index}/{len(labels)}] {identifier} {manifest['alignment_status']}", flush=True)
            except Exception as error:  # keep batch alive and preserve the exact failure
                manifest = {
                    "id": identifier,
                    "video_id": str(row.video_id),
                    "clip_id": int(row.clip_id),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "schema_version": SCHEMA_VERSION,
                    "schema_hash": SCHEMA_HASH,
                }
                error_log = self.output_root / "errors.jsonl"
                with error_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
                print(f"[{index}/{len(labels)}] {identifier} ERROR {manifest['error']}", flush=True)
            manifest_rows.append(manifest)
            pd.DataFrame(manifest_rows).to_csv(self.output_root / "manifest.csv", index=False)
        self.finalize(manifest_rows)
        return manifest_rows

    def process_manifest_only(self, sample: dict, row: Any) -> tuple[dict, dict]:
        """Rebuild a manifest row from a resumed sample without expensive inference."""
        alignment = sample["alignment"]
        retained = [word for word in sample["full_word_map"] if not word["dropped"]]
        full_duration = sum((word["t_e"] or 0) - (word["t_s"] or 0) for word in sample["full_word_map"])
        retained_duration = sum((word["t_e"] or 0) - (word["t_s"] or 0) for word in retained)
        manifest = {
            "id": sample["id"], "video_id": str(row.video_id), "clip_id": int(row.clip_id),
            "split": sample["metadata"]["split"], "status": "ok", "error": "",
            "schema_version": SCHEMA_VERSION, "schema_hash": SCHEMA_HASH,
            "duration": sample["metadata"]["duration"], "fps": sample["metadata"]["fps"],
            "is_vfr": sample["metadata"]["is_vfr"], "frame_count": sample["metadata"]["frame_count"],
            "text_len": len(sample["full_word_map"]), "wp_len": int(sample["content_mask"].sum()),
            "retained_word_ratio": len(retained) / max(1, len(sample["full_word_map"])),
            "retained_duration_ratio": retained_duration / max(1e-12, full_duration),
            "alignment_status": alignment["status"], "alignment_failure_rate": alignment["failure_rate"],
            "data_anomaly": alignment["status"] == "rollback",
            "audio_coverage": _coverage(sample, "audio"), "vision_coverage": _coverage(sample, "video"),
            "audio_missing_counts": _reason_counts(sample["missing_reason_audio"]),
            "vision_missing_counts": _reason_counts(sample["missing_reason_video"]),
            "label": sample["label"], "annotation": sample["annotation"],
        }
        return sample, manifest

    def finalize(self, manifest_rows: list[dict]) -> None:
        good_ids = [row["id"] for row in manifest_rows if row.get("status") == "ok"]
        samples = []
        for identifier in good_ids:
            with (self.sample_dir / f"{_safe_name(identifier)}.pkl").open("rb") as handle:
                samples.append(pickle.load(handle))
        artifacts = {}
        train_ids = {sample["id"] for sample in samples} & self.attachment_train_ids
        if samples:
            features_path = self.output_root / "features.pkl"
            _atomic_pickle(
                {"schema_version": SCHEMA_VERSION, "schema_hash": SCHEMA_HASH, "samples": samples},
                features_path,
            )
            artifacts["features.pkl"] = _sha256(features_path)
        stats_path = None
        stats_error = None
        if train_ids:
            try:
                stats = compute_normalization_stats(samples, train_ids)
                stats_path = self.output_root / "normalization_stats.pkl"
                _atomic_pickle(stats, stats_path)
                artifacts["normalization_stats.pkl"] = _sha256(stats_path)
            except ValueError as error:
                stats_error = f"{type(error).__name__}: {error}"
        # 模型/特征契约（审阅修订 2026-09-23：方案 §7 要求 model revision 入 manifest）
        model_contract = {
            "text_encoder": {
                "model": "bert-base-uncased",
                "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
                "weights_path": "cache/hf/bert-base-uncased/model.safetensors",
                "weights_sha256": "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
                "output": "last_hidden_state",
            },
            "aligner": {
                "pipeline": "torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H",
                "checkpoint": "cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth",
                "checkpoint_sha256": _sha256(
                    Path(__file__).resolve().parents[2]
                    / "cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth"
                ),
            },
            "vision": {
                "extractor": "py-feat Detector（四段链 face+landmark+pose+AU，emotion 跳过）",
                "au_columns": VISION_COLUMNS,
                "face_policy": "top-1 by confidence",
                "sanitize": "有限性+帧内裁剪+<16px 丢弃",
            },
            "audio": {
                "extractor": "opensmile.Smile(eGeMAPSv02, LowLevelDescriptors)",
                "columns": "lld_00..lld_24（opensmile 输出列序）",
                "hop_seconds": 0.01,
            },
            "device": self.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "vision_batch_size": self.vision_batch_size,
        }
        data_manifest = {
            "schema_version": SCHEMA_VERSION,
            "schema_hash": SCHEMA_HASH,
            "schema": SCHEMA_SPEC,
            "missing_reason_enum": list(MISSING_REASON_ENUM),
            "sample_count": len(samples),
            "error_count": len(manifest_rows) - len(samples),
            "normalization_train_ids": sorted(train_ids),
            "normalization_error": stats_error,
            "alignment_config": ALIGNMENT_CONFIG,
            "model_contract": model_contract,
            "tool_versions": _tool_versions(),
            "artifacts": {
                "manifest.csv": _sha256(self.output_root / "manifest.csv"),
                **artifacts,
            },
        }
        (self.output_root / "data_manifest.json").write_text(
            json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
