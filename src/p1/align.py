"""wav2vec2 CTC 强制对齐（问题一 §2.2/§4）。

输出词区间统一为半开区间 ``[t_s, t_e)``。CTC 帧 ``k`` 到物理时间的
冻结换算式为 ``t = k * 320 / 16000 = k * 0.02s``。数字或其他词表外词不
进入 CTC target，之后使用相邻可信词边界按整帧线性插值。
"""
from __future__ import annotations

import copy
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .normalize import normalize_transcript

SAMPLE_RATE = 16_000
HOP_SAMPLES = 320
HOP_SECONDS = HOP_SAMPLES / SAMPLE_RATE
MIN_WORD_SECONDS = HOP_SECONDS
MAX_WORD_SECONDS = 3.0
LOW_CONFIDENCE = 0.5
REVIEW_FAILURE_RATE = 0.05
ROLLBACK_FAILURE_RATE = 0.20

ALIGNMENT_CONFIG = {
    "model": "torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H",
    "sample_rate": SAMPLE_RATE,
    "hop_samples": HOP_SAMPLES,
    "hop_seconds": HOP_SECONDS,
    "frame_to_time": "t_seconds = frame_index * 320 / 16000",
    "interval": "half-open [t_s, t_e)",
    "min_word_seconds": MIN_WORD_SECONDS,
    "max_word_seconds": MAX_WORD_SECONDS,
    "low_confidence": LOW_CONFIDENCE,
    "confidence": "mean posterior over CTC frames assigned to word characters",
    "review_failure_rate": REVIEW_FAILURE_RATE,
    "rollback_failure_rate": ROLLBACK_FAILURE_RATE,
    "numeric_policy": "exclude from CTC target; interpolate from trusted neighbours",
}


def _failure_reason(word: str, label_to_id: dict[str, int]) -> str | None:
    if any(char.isdigit() for char in word):
        return "numeric_oov"
    unsupported = sorted({char.upper() for char in word if char.upper() not in label_to_id})
    if unsupported:
        return "ctc_oov:" + "".join(unsupported)
    return None


def build_alignment_plan(
    raw_text: str,
    labels: tuple[str, ...] | list[str],
) -> tuple[list[dict[str, Any]], str]:
    """生成发音词记录和不含词表外词的 CTC target。"""
    if not isinstance(raw_text, str):
        raise TypeError(f"raw_text must be str, got {type(raw_text).__name__}")
    words = normalize_transcript(raw_text).split()
    if not words:
        raise ValueError("normalized transcript contains no pronounced words")

    label_to_id = {label: index for index, label in enumerate(labels)}
    records: list[dict[str, Any]] = []
    target_parts = []
    for word_id, word in enumerate(words):
        reason = _failure_reason(word, label_to_id)
        ctc_text = word.upper() if reason is None else None
        records.append({
            "word_id": word_id,
            "word_text": word,
            "ctc_text": ctc_text,
            "raw_start_frame": None,
            "raw_end_frame": None,
            "raw_t_s": None,
            "raw_t_e": None,
            "start_frame": None,
            "end_frame": None,
            "t_s": None,
            "t_e": None,
            "confidence": 0.0,
            "failure_reason": reason,
            "alignment_status": "pending" if reason is None else "excluded",
        })
        if ctc_text is not None:
            target_parts.append(ctc_text)
    return records, "|".join(target_parts)


def _weighted_confidence(spans: list[Any], frame_scores: torch.Tensor) -> float:
    posterior_parts = [
        frame_scores[int(span.start): int(span.end)].exp()
        for span in spans
        if int(span.end) > int(span.start)
    ]
    if not posterior_parts:
        return 0.0
    confidence = float(torch.cat(posterior_parts).mean().item())
    return float(min(1.0, max(0.0, confidence)))


def _interval_failure(
    start_frame: int,
    end_frame: int,
    duration: float,
    confidence: float,
) -> str | None:
    t_s = start_frame * HOP_SECONDS
    t_e = min(end_frame * HOP_SECONDS, duration)
    word_duration = t_e - t_s
    if start_frame <= 0 or t_s >= duration or t_e <= t_s:
        return "invalid_interval"
    if word_duration + 1e-9 < MIN_WORD_SECONDS or word_duration > MAX_WORD_SECONDS + 1e-9:
        return "invalid_duration"
    if end_frame * HOP_SECONDS > duration + HOP_SECONDS:
        return "out_of_bounds"
    if not math.isfinite(confidence) or confidence < LOW_CONFIDENCE:
        return "low_confidence"
    return None


def _assign_ctc_spans(
    records: list[dict[str, Any]],
    target: str,
    spans: list[Any],
    frame_scores: torch.Tensor,
    label_to_id: dict[str, int],
    duration: float,
) -> None:
    expected = [label_to_id[char] for char in target]
    actual = [int(span.token) for span in spans]
    if actual != expected:
        raise RuntimeError("merged CTC path does not match target sequence")

    cursor = 0
    aligned_records = [record for record in records if record["ctc_text"] is not None]
    for record_index, record in enumerate(aligned_records):
        char_count = len(record["ctc_text"])
        word_spans = spans[cursor: cursor + char_count]
        if len(word_spans) != char_count:
            raise RuntimeError(f"incomplete CTC spans for word {record['word_text']!r}")
        cursor += char_count
        if record_index < len(aligned_records) - 1:
            cursor += 1  # 跳过 target 中的 | 分词符

        start_frame = int(word_spans[0].start)
        end_frame = int(word_spans[-1].end)
        confidence = _weighted_confidence(word_spans, frame_scores)
        record.update({
            "raw_start_frame": start_frame,
            "raw_end_frame": end_frame,
            "raw_t_s": start_frame * HOP_SECONDS,
            "raw_t_e": min(end_frame * HOP_SECONDS, duration),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "t_s": start_frame * HOP_SECONDS,
            "t_e": min(end_frame * HOP_SECONDS, duration),
            "confidence": confidence,
        })
        reason = _interval_failure(start_frame, end_frame, duration, confidence)
        record["failure_reason"] = reason
        record["alignment_status"] = "aligned" if reason is None else "rejected"


def _interpolation_boundaries(left: int, right: int, count: int) -> list[int] | None:
    available = right - left
    if count <= 0 or available < count:
        return None
    boundaries = [left + (available * index) // count for index in range(count + 1)]
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        return None
    if any(b - a > round(MAX_WORD_SECONDS / HOP_SECONDS) for a, b in zip(boundaries, boundaries[1:])):
        return None
    return boundaries


def _clear_final_interval(record: dict[str, Any]) -> None:
    record.update({
        "start_frame": None,
        "end_frame": None,
        "t_s": None,
        "t_e": None,
    })


def interpolate_failed_words(
    records: list[dict[str, Any]],
    duration: float,
) -> list[dict[str, Any]]:
    """按相邻可信词边界插值失败词，保持原 confidence/failure_reason。"""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"duration must be positive and finite, got {duration}")
    result = copy.deepcopy(records)
    final_frame = max(1, int(math.floor((duration + 1e-9) / HOP_SECONDS)))
    trusted = [record["failure_reason"] is None for record in result]

    index = 0
    while index < len(result):
        if trusted[index]:
            index += 1
            continue
        run_start = index
        while index < len(result) and not trusted[index]:
            index += 1
        run_end = index

        left_frame = (
            int(result[run_start - 1]["end_frame"])
            if run_start > 0 and trusted[run_start - 1]
            else 1
        )
        right_frame = (
            int(result[run_end]["start_frame"])
            if run_end < len(result) and trusted[run_end]
            else final_frame
        )
        boundaries = _interpolation_boundaries(left_frame, right_frame, run_end - run_start)
        if boundaries is None:
            for record in result[run_start:run_end]:
                _clear_final_interval(record)
                record["alignment_status"] = "failed"
            continue

        for offset, record in enumerate(result[run_start:run_end]):
            start_frame, end_frame = boundaries[offset: offset + 2]
            t_s = start_frame * HOP_SECONDS
            t_e = min(end_frame * HOP_SECONDS, duration)
            if _interval_failure(start_frame, end_frame, duration, 1.0) is not None:
                _clear_final_interval(record)
                record["alignment_status"] = "failed"
                continue
            record.update({
                "start_frame": start_frame,
                "end_frame": end_frame,
                "t_s": t_s,
                "t_e": t_e,
                "alignment_status": "interpolated",
            })
    return result


def _summarize(records: list[dict[str, Any]], alignment_error: str | None) -> dict[str, Any]:
    failed = [record for record in records if record["failure_reason"] is not None]
    unresolved = [record for record in records if record["alignment_status"] == "failed"]
    failure_rate = len(failed) / len(records)
    rollback_required = failure_rate > ROLLBACK_FAILURE_RATE
    needs_review = failure_rate > REVIEW_FAILURE_RATE
    status = "rollback" if rollback_required else "review" if needs_review else "ok"
    return {
        "words": records,
        "word_count": len(records),
        "failure_count": len(failed),
        "failure_rate": failure_rate,
        "failed_word_ids": [record["word_id"] for record in failed],
        "low_confidence_word_ids": [
            record["word_id"]
            for record in failed
            if record["failure_reason"] == "low_confidence"
        ],
        "unsupported_word_ids": [
            record["word_id"]
            for record in failed
            if str(record["failure_reason"]).startswith(("numeric_oov", "ctc_oov:"))
        ],
        "unresolved_word_ids": [record["word_id"] for record in unresolved],
        "needs_review": needs_review,
        "rollback_required": rollback_required,
        "status": status,
        "alignment_error": alignment_error,
        "config": dict(ALIGNMENT_CONFIG),
    }


class ForcedAligner:
    """复用单个 wav2vec2 模型完成多条 16 kHz 单声道音频对齐。"""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.labels = self.bundle.get_labels()
        self.label_to_id = {label: index for index, label in enumerate(self.labels)}
        self.blank_id = self.label_to_id["-"]
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.bundle.get_model().to(self.device).eval()

    def align_waveform(
        self,
        waveform: np.ndarray | torch.Tensor,
        raw_text: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> dict[str, Any]:
        """对齐内存波形；输入必须是媒体阶段产出的 16 kHz mono。"""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sample_rate}")
        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        if tensor.ndim == 2 and 1 in tensor.shape:
            tensor = tensor.reshape(-1)
        if tensor.ndim != 1 or tensor.numel() == 0:
            raise ValueError(f"waveform must be non-empty mono audio, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("waveform contains NaN or Inf")
        duration = tensor.numel() / SAMPLE_RATE
        records, target = build_alignment_plan(raw_text, self.labels)
        alignment_error = None

        if target:
            try:
                targets = torch.tensor(
                    [[self.label_to_id[char] for char in target]],
                    dtype=torch.int32,
                    device=self.device,
                )
                with torch.inference_mode():
                    emissions, _ = self.model(tensor.unsqueeze(0).to(self.device))
                    path, scores = torchaudio.functional.forced_align(
                        emissions.log_softmax(dim=-1), targets, blank=self.blank_id
                    )
                spans = torchaudio.functional.merge_tokens(
                    path[0], scores[0], blank=self.blank_id
                )
                _assign_ctc_spans(
                    records, target, spans, scores[0], self.label_to_id, duration
                )
            except (RuntimeError, ValueError) as error:
                alignment_error = f"{type(error).__name__}: {error}"
                for record in records:
                    if record["ctc_text"] is not None:
                        record["failure_reason"] = "ctc_alignment_failed"
                        record["alignment_status"] = "rejected"

        records = interpolate_failed_words(records, duration)
        result = _summarize(records, alignment_error)
        result.update({
            "duration": duration,
            "sample_rate": SAMPLE_RATE,
            "num_samples": int(tensor.numel()),
            "ctc_target": target,
        })
        return result

    def align_file(self, wav_path: str | Path, raw_text: str) -> dict[str, Any]:
        """读取媒体阶段的 PCM WAV 并强制对齐。"""
        wav_path = Path(wav_path)
        if not wav_path.is_file():
            raise FileNotFoundError(f"alignment WAV does not exist: {wav_path}")
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        if audio.shape[1] != 1:
            raise ValueError(f"alignment WAV must be mono, got {audio.shape[1]} channels")
        result = self.align_waveform(audio[:, 0], raw_text, int(sample_rate))
        result["wav_path"] = str(wav_path)
        return result


@lru_cache(maxsize=None)
def get_aligner(device: str | None = None) -> ForcedAligner:
    """按 device 缓存模型，批量处理 100 条时避免重复加载权重。"""
    return ForcedAligner(device=device)
