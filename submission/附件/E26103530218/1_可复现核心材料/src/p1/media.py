"""媒体抽取（问题一）：16 kHz mono WAV 与保留源 PTS 的逐帧图像。"""
from __future__ import annotations

import math
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
_SHOWINFO_FRAME = re.compile(
    r"\b(?:n):\s*(?P<n>\d+)\s+"
    r"pts:\s*(?P<pts>-?\d+)\s+"
    r"pts_time:\s*(?P<pts_time>[-+\d.eE]+)"
)


def _source_path(src: str | Path) -> Path:
    path = Path(src)
    if not path.is_file():
        raise FileNotFoundError(f"media source does not exist: {path}")
    return path


def _error_tail(stderr: str, limit: int = 1200) -> str:
    return stderr[-limit:].strip()


def extract_audio_16k(src: str | Path, dst: str | Path) -> Path:
    """抽音频：单声道 16kHz pcm_s16le wav（对齐与 LLD 共用同一路径）。"""
    src = _source_path(src)
    dst = Path(dst)
    if dst.suffix.lower() != ".wav":
        raise ValueError(f"audio destination must use .wav suffix: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f".{dst.stem}.", suffix=".wav", dir=dst.parent, delete=False
    ) as handle:
        staged = Path(handle.name)
    try:
        result = subprocess.run(
            [
                FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(src), "-map", "0:a:0", "-vn", "-ac", "1",
                "-ar", "16000", "-c:a", "pcm_s16le", str(staged),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"audio extract failed: {src}\n{_error_tail(result.stderr)}"
            )
        with wave.open(str(staged), "rb") as wav:
            actual = (
                wav.getnchannels(), wav.getframerate(), wav.getsampwidth(),
                wav.getnframes(), wav.getcomptype(),
            )
        if actual[:3] != (1, 16000, 2) or actual[3] <= 0 or actual[4] != "NONE":
            raise RuntimeError(
                "invalid extracted WAV "
                f"(channels, rate, width, frames, compression)={actual}: {src}"
            )
        staged.replace(dst)
    finally:
        staged.unlink(missing_ok=True)
    return dst


def probe_duration(src: str | Path) -> float:
    """实测时长（秒）。注意与题目口径不一致，manifest 按此实测值记录。"""
    src = _source_path(src)
    result = subprocess.run(
        [FF, "-nostdin", "-hide_banner", "-i", str(src)],
        capture_output=True,
        text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", result.stderr)
    if not match:
        raise RuntimeError(f"duration probe failed: {src}\n{_error_tail(result.stderr)}")
    h, mn, s = match.groups()
    duration = int(h) * 3600 + int(mn) * 60 + float(s)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"invalid media duration {duration}: {src}")
    return duration


def extract_frames_pts(src: str | Path, dst_dir: str | Path) -> list[dict]:
    """逐帧导出 png 并记录 pts_time（fps 混用/VFR 安全，禁用 帧号/fps）。

    返回 ``[{frame_idx, pts, pts_time, path}]``。``frame_idx`` 仅表示解码顺序，
    物理时间只能读取 ``pts_time``，不得从序号和平均 fps 推算。
    """
    src = _source_path(src)
    dst_dir = Path(dst_dir)
    dst_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{dst_dir.name}.frames.", dir=dst_dir.parent
    ) as staged_name:
        staged_dir = Path(staged_name)
        result = subprocess.run(
            [
                FF, "-nostdin", "-hide_banner", "-loglevel", "info", "-y",
                "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", "showinfo", "-fps_mode", "passthrough",
                str(staged_dir / "frame_%06d.png"),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"frame extract failed: {src}\n{_error_tail(result.stderr)}"
            )

        timing = [
            (int(match["n"]), int(match["pts"]), float(match["pts_time"]))
            for line in result.stderr.splitlines()
            if "Parsed_showinfo" in line
            for match in [_SHOWINFO_FRAME.search(line)]
            if match is not None
        ]
        staged_frames = sorted(staged_dir.glob("frame_*.png"))
        if not timing or len(timing) != len(staged_frames):
            raise RuntimeError(
                "frame/PTS count mismatch "
                f"(frames={len(staged_frames)}, pts={len(timing)}): {src}"
            )
        expected_indices = list(range(len(timing)))
        decoded_indices = [item[0] for item in timing]
        if decoded_indices != expected_indices:
            raise RuntimeError(f"non-contiguous showinfo frame indices: {src}")
        pts_times = [item[2] for item in timing]
        if not all(math.isfinite(value) for value in pts_times):
            raise RuntimeError(f"non-finite frame PTS: {src}")
        if any(right < left for left, right in zip(pts_times, pts_times[1:])):
            raise RuntimeError(f"non-monotonic frame PTS: {src}")

        dst_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in dst_dir.glob("frame_*.png"):
            old_frame.unlink()
        records = []
        for frame_idx, (show_idx, pts, pts_time) in enumerate(timing):
            frame_path = dst_dir / staged_frames[frame_idx].name
            staged_frames[frame_idx].replace(frame_path)
            records.append({
                "frame_idx": show_idx,
                "pts": pts,
                "pts_time": pts_time,
                "path": str(frame_path),
            })
    return records
