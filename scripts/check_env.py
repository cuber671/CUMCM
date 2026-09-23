#!/usr/bin/env python
"""S0 环境自检（问题一方案 §8 / 论文总表 S0 步骤）。

四项 smoke test 全绿 S0 才算完成：
  1. bert-base-uncased tokenizer + forward（768 维输出）
  2. wav2vec2 forced alignment 机械链路（随机波形上出区间）
  3. OpenSMILE eGeMAPSv02 LLD：1s 正弦 wav → 25 维帧级输出
  4. Py-Feat：附件1 真实帧 → 人脸检测 → AU/姿态（管线跑通即绿）

同时导出 environment-manifest.json（论文"工具版本+核心参数"的唯一出处）。
"""
import json
import platform
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "environment-manifest.json"

FEATURE_CONFIG = {
    "audio_sample_rate": 16000,
    "audio_hop_ms": 10,
    "vision_timestamp": "pts",
    "max_length": 50,
    "tokenizer": "bert-base-uncased",
    "truncation": "head: CLS+前48非特殊片+SEP@49",
}

results = []
manifest = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "feature_config": FEATURE_CONFIG,
    "models": {
        "bert": "bert-base-uncased",
        "forced_align": "torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H + functional.forced_align",
    },
}


def report(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    # ---- 基础链：torch / CUDA ----
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["cuda_runtime"] = torch.version.cuda
        manifest["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        report("torch import", True, torch.__version__)
        report(
            "CUDA/GPU",
            torch.cuda.is_available(),
            f"{torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}" if torch.cuda.is_available() else "CPU only",
        )
    except Exception as e:  # noqa: BLE001
        report("torch import", False, str(e))
        return finish()

    import torchaudio
    manifest["torchaudio"] = torchaudio.__version__

    # ---- smoke 1: BERT tokenizer + forward ----
    try:
        from transformers import BertModel, BertTokenizerFast
        import transformers
        manifest["transformers"] = transformers.__version__
        tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        enc = tok("environment smoke test", return_tensors="pt")
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        assert out.shape[-1] == 768 and out.shape[0] == 1
        report("smoke1 bert-base-uncased tokenizer+forward", True, f"last_hidden_state {tuple(out.shape)}")
    except Exception as e:  # noqa: BLE001
        report("smoke1 bert-base-uncased tokenizer+forward", False, str(e))

    # ---- smoke 2: forced alignment 机械链路 ----
    try:
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        w2v = bundle.get_model()
        labels = bundle.get_labels()
        with torch.no_grad():
            emission, _ = w2v(torch.randn(1, 16000))
        tokens = torch.tensor([[labels.index(c) for c in "cat"]])
        seg = torchaudio.functional.forced_align(emission, tokens)
        assert seg is not None and seg.shape[-1] == len("cat")
        report("smoke2 wav2vec2 forced_align", True, "1s 随机波形→词区间机械链路 OK")
    except Exception as e:  # noqa: BLE001
        report("smoke2 wav2vec2 forced_align", False, str(e))

    # ---- smoke 3: OpenSMILE 25 LLD ----
    try:
        import opensmile
        manifest["opensmile"] = opensmile.__version__
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        sr = 16000
        t = np.arange(sr) / sr
        wav = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((wav * 32767).astype(np.int16).tobytes())
            df = smile.process_file(f.name)
        assert df.shape[1] == 25, f"LLD 维度 {df.shape[1]} != 25"
        report("smoke3 opensmile eGeMAPSv02 LLD", True, f"25 维帧级输出，帧数 {df.shape[0]}")
    except Exception as e:  # noqa: BLE001
        report("smoke3 opensmile eGeMAPSv02 LLD", False, str(e))

    # ---- ffmpeg ----
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        ver = subprocess.run([ff, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        manifest["ffmpeg"] = ver
        report("ffmpeg (imageio-ffmpeg)", True, ver)
    except Exception as e:  # noqa: BLE001
        report("ffmpeg (imageio-ffmpeg)", False, str(e))

    # ---- smoke 4: Py-Feat 真实帧 ----
    try:
        from feat import Detector
        import feat as _f
        manifest["py_feat"] = getattr(_f, "__version__", "unknown")
        frame = extract_first_frame(ROOT / "data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条")
        assert frame is not None, "未取得测试帧"
        detector = Detector(device="cuda" if torch.cuda.is_available() else "cpu")
        out = detector.detect_image(frame)
        n = len(out) if hasattr(out, "__len__") else 0
        au_cols = [c for c in getattr(out, "columns", []) if c.startswith("AU")]
        report("smoke4 py-feat 人脸检测+AU", True, f"检出人脸 {n}，AU 列 {len(au_cols)}（管线绿即通过）")
    except Exception as e:  # noqa: BLE001
        report("smoke4 py-feat 人脸检测+AU", False, str(e))

    return finish()


def extract_first_frame(video_root: Path, n_clips: int = 2):
    """从附件1 任取片段抽首帧到临时目录，供 Py-Feat smoke。"""
    import imageio_ffmpeg
    from PIL import Image  # noqa: F401

    clips = sorted(video_root.rglob("*.mp4"))[:n_clips]
    if not clips:
        return None
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    for clip in clips:
        out = Path(tempfile.gettempdir()) / "s0_frame.png"
        r = subprocess.run(
            [ff, "-y", "-i", str(clip), "-frames:v", "1", str(out)],
            capture_output=True,
        )
        if r.returncode == 0 and out.exists():
            return str(out)
    return None


def finish() -> int:
    ok = all(r for _, r, _ in results)
    manifest["s0_smoke"] = [{"check": n, "pass": p, "detail": d[:300]} for n, p, d in results]
    manifest["s0_all_green"] = ok
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nS0 = {'ALL GREEN' if ok else 'NOT READY'}；manifest → {MANIFEST}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
