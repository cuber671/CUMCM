#!/usr/bin/env python
"""Run the resumable P1 media-to-feature pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1.runner import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT, P1Runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--vision-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    runner = P1Runner(
        args.output_root,
        device=args.device,
        vision_batch_size=args.vision_batch_size,
    )
    rows = runner.run(args.input_root, limit=args.limit, resume=not args.no_resume)
    return 1 if any(row.get("status") == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
