"""媒体抽取（问题一）：imageio-ffmpeg 静态二进制，16kHz mono wav / 帧 pts。"""
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio_16k(src: str | Path, dst: str | Path) -> Path:
    """抽音频：单声道 16kHz pcm_s16le wav（对齐与 LLD 共用同一路径）。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [FF, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-acodec", "pcm_s16le", str(dst)],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"audio extract failed: {src}\n{r.stderr[-400:].decode(errors='ignore')}")
    return dst


def probe_duration(src: str | Path) -> float:
    """实测时长（秒）。注意与题目口径不一致，manifest 按此实测值记录。"""
    r = subprocess.run([FF, "-i", str(src)], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        raise RuntimeError(f"probe failed: {src}")
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def extract_frames_pts(src: str | Path, dst_dir: str | Path) -> list[dict]:
    """逐帧导出 png 并记录 pts_time（fps 混用/VFR 安全，禁用 帧号/fps）。

    返回 [{frame_idx, pts_time, path}]；showinfo 过滤 pts，帧文件名含序号。
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [FF, "-y", "-i", str(src), "-vf", "showinfo",
         "-vsync", "passthrough", str(dst_dir / "frame_%05d.png")],
        capture_output=True, text=True,
    )
    pts = [float(x) for x in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    frames = sorted(dst_dir.glob("frame_*.png"))
    return [
        {"frame_idx": i, "pts_time": p, "path": str(f)}
        for i, (p, f) in enumerate(zip(pts, frames))
    ]
