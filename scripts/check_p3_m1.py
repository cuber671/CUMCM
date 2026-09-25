"""P3 M1 验收脚本：附件4 真实数据全量校验 + 预注册数字核验 + P1 对齐接口冒烟。

检查项（详见 docs/问题三实施步骤.md §M1）：
  1. 计数消歧：对齐版本 = 20 pkl + videos/（21 项目录条目中的第 21 项是子目录，非样本）；
  2. 逐条 schema/shape/dtype/有限性/id↔文件名；
  3. 结构分区与自然观测（复用 P2 契约，破坏即抛错）+ 预注册数字核验；
  4. 分词复现规则（truncation=True, max_length=50）20/20；
  5. 跨附件锚点：#03/07/08/12/15 与附件2 test 逐位相等；
  6. 视频资产：20 mp4（对齐/未对齐两处）、时长与 fps 记录、未对齐长度字段失真留痕；
  7. P1 对齐接口冒烟：#01 提取 wav → CTC 强制对齐 → 词时间表在时长内、零失败。
产物：runs/p3/m1/report.json。退出码：全过 = 0。
"""
from __future__ import annotations

import json
import sys
import tempfile
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import Check, sha256_file  # noqa: E402
from src.p3.data import (ALIGNED_DIR, ALIGNED_VIDEOS_DIR, FIELD_SHAPES,  # noqa: E402
                         PREREG, UNALIGNED_DIR, VIDEOS_DIR, att4_masks,
                         att2_anchor_check, load_att4, prereg_summary,
                         tokenize_repro_check)


def main() -> int:
    checks: list[Check] = []

    def add(name, ok, detail):
        checks.append(Check(name, bool(ok), detail))
        print(("✅" if ok else "❌"), name, "" if ok else detail)

    # ---- 1. 计数消歧 ----
    aligned_entries = sorted(p.name for p in ALIGNED_DIR.iterdir())
    pkls = [e for e in aligned_entries if e.endswith(".pkl")]
    others = [e for e in aligned_entries if not e.endswith(".pkl")]
    add("count_disambiguation",
        len(pkls) == PREREG["n_files"] and others == ["videos"],
        {"pkl": len(pkls), "non_pkl_entries": others,
         "note": "方案'20条'口径正确；21 项目录条目中 videos/ 为渲染视频子目录"})

    # ---- 2. 逐条 schema ----
    samples = load_att4()  # schema/id 失败即抛错（loader 内硬校验）
    n_bad = 0
    for s in samples:
        bad = [f for f in FIELD_SHAPES
               if getattr(s, f).shape != (1,) + FIELD_SHAPES[f]]
        finite = all(np.isfinite(getattr(s, f)).all()
                     for f in ("audio", "vision", "text", "text_bert"))
        if bad or not finite or len(s.raw_text) == 0:
            n_bad += 1
    add("schema_dtype_finite", n_bad == 0, {"bad_samples": n_bad})

    # ---- 3. 自然观测 + 预注册数字 ----
    got = prereg_summary(samples)
    prereg_ok = all(got[k] == PREREG[k] for k in got)
    add("prereg_numbers", prereg_ok, {"got": got, "expect": PREREG})

    # ---- 4. 分词复现 ----
    add_check = tokenize_repro_check(samples)
    checks.append(add_check)
    print(("✅" if add_check.ok else "❌"), add_check.name, add_check.detail)

    # ---- 5. 附件2 锚点 ----
    anc = att2_anchor_check(samples)
    checks.append(anc)
    print(("✅" if anc.ok else "❌"), anc.name, anc.detail)

    # ---- 6. 视频资产 ----
    import cv2
    vids, durs, length_anomalies = [], [], []
    for s in samples:
        mp4 = VIDEOS_DIR / f"{s.n:02d}.mp4"
        cap = cv2.VideoCapture(str(mp4))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        vids.append(mp4.exists())
        durs.append(round(fc / fps, 2) if fps else -1.0)
        un = __import__("pickle").load(open(UNALIGNED_DIR / f"{s.n:02d}.pkl", "rb"))
        v = np.asarray(un["vision"])
        nz = int((np.abs(v).max(-1) > 0).sum())
        if un["vision_lengths"] != nz:
            length_anomalies.append({"n": s.n, "length_field": int(un["vision_lengths"]),
                                     "actual_nonzero_rows": nz})
    add("videos", all(vids) and len(list(ALIGNED_VIDEOS_DIR.glob("*.mp4"))) == 20,
        {"missing": [s.n for s, ok in zip(samples, vids) if not ok],
         "durations": dict(zip([s.n for s in samples], durs)),
         "length_field_anomalies": length_anomalies,
         "note": "长度字段失真留痕（#13/#16）；物理时间以 P1 CTC 对齐为准"})

    # ---- 7. P1 对齐接口冒烟（#01）----
    try:
        from src.p1.align import get_aligner
        from src.p1.media import extract_audio_16k
        tmp = tempfile.mkdtemp()
        wav = extract_audio_16k(VIDEOS_DIR / "01.mp4", os.path.join(tmp, "01.wav"))
        out = get_aligner().align_file(str(wav), samples[0].raw_text)
        words = out["words"]
        in_dur = all(w["t_e"] <= out["duration"] + 0.5 for w in words)
        smoke_ok = (len(words) == 21 and out["failure_count"] == 0 and in_dur)
        add("p1_alignment_smoke_01", smoke_ok,
            {"n_words": len(words), "failures": out["failure_count"],
             "duration_s": round(out["duration"], 2),
             "first3": [(w["word_text"], round(w["t_s"], 2)) for w in words[:3]]})
    except Exception as e:  # 接口失败必须显式暴露
        add("p1_alignment_smoke_01", False, {"error": repr(e)})

    # ---- 落盘 ----
    out_dir = ROOT / "runs/p3/m1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"aligned_dir": str(ALIGNED_DIR),
                   "sha256_first_last": [sha256_file(ALIGNED_DIR / "01.pkl"),
                                         sha256_file(ALIGNED_DIR / "20.pkl")]},
        "prereg": PREREG,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
        "all_pass": all(c.ok for c in checks),
    }
    (out_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"\nreport → {out_dir / 'report.json'}")
    print("ALL PASS" if rep["all_pass"] else "FAILED")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
