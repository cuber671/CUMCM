"""P3 M6-a：附件4 全 20 条视频 P1 对齐资产重生成（现场，附件4 与 P1 100 条无重合）。

链路（P1 同款）：mp4 → extract_audio_16k → wav2vec2 CTC 强制对齐 → 词时间表；
配合 build_grid 的 wp_word_map（M3 已验 20/20 对齐）构成 位置→原词→物理时间 映射。
产物：runs/p3/m6/alignment/{01..20}.json（words: id/text/t_s/t_e/status + duration）。
"""
from __future__ import annotations

import json
import sys
import tempfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p1.align import get_aligner  # noqa: E402
from src.p1.grid import build_grid  # noqa: E402
from src.p1.media import extract_audio_16k  # noqa: E402
from src.p3.data import VIDEOS_DIR, load_att4  # noqa: E402

OUT = ROOT / "runs/p3/m6/alignment"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    aligner = get_aligner()
    tmp = tempfile.mkdtemp()
    n_ok = 0
    for s in load_att4():
        wav = extract_audio_16k(VIDEOS_DIR / f"{s.n:02d}.mp4",
                                os.path.join(tmp, f"{s.n:02d}.wav"))
        out = aligner.align_file(str(wav), s.raw_text)
        g = build_grid(s.raw_text)
        words = out["words"]
        assert len(words) == len(g["full_word_map"]), \
            f"#{s.n:02d} 对齐词数 {len(words)} ≠ 网格词数 {len(g['full_word_map'])}"
        payload = {
            "sample_id": f"{s.n:02d}", "duration_s": round(out["duration"], 4),
            "failure_count": out["failure_count"],
            "needs_review": out["needs_review"],
            "words": [{"id": w["word_id"], "text": w["word_text"],
                       "t_s": round(w["t_s"], 4), "t_e": round(w["t_e"], 4),
                       "status": w["alignment_status"]} for w in words]}
        (OUT / f"{s.n:02d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        in_dur = all(w["t_e"] <= out["duration"] + 0.5 for w in words)
        ok = out["failure_count"] == 0 and in_dur
        n_ok += ok
        print(f"#{s.n:02d} {len(words)}词 失败{out['failure_count']} "
              f"时长{out['duration']:.1f}s {'✅' if ok else '⚠️ review'}", flush=True)
    print(f"\n{n_ok}/20 零失败且词时间在时长内")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
