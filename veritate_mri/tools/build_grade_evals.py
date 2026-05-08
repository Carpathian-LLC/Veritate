"""Build grade-level eval bins from hand-authored source .txt files.

Each band gets a fresh ~5 KB passage tuned to that reading level's prose
statistics (sentence length, clause density, vocabulary register). Sources
live in veritate_mri/grade_eval/sources/grade_<level>_source.txt; this
script encodes them as UTF-8 with normalized line endings and writes the
.bin files the probe reads.

Why fresh sources instead of Project-Gutenberg dumps: the previous bins were
the first 4 KB of Gutenberg works, which meant Pre-K and K were dominated by
"Produced by ... Distributed Proofreading Team ... [Illustration: ...]"
boilerplate, inflating their perplexity by an order of magnitude relative to
content the model could actually score. Fresh sources are clean from byte 0.

Usage:
    python veritate_mri/tools/build_grade_evals.py
"""

import sys
from pathlib import Path

LEVELS = ["prek", "k", "elem", "middle", "hs", "college", "phd"]
MIN_BYTES = 4096  # the probe reads exactly this many bytes per band


def main() -> int:
    here = Path(__file__).resolve().parent
    sources = here.parent / "grade_eval" / "sources"
    targets = here.parent / "grade_eval"
    if not sources.exists():
        print(f"sources directory missing: {sources}", file=sys.stderr)
        return 1
    short = []
    for level in LEVELS:
        src = sources / f"grade_{level}_source.txt"
        if not src.exists():
            print(f"  missing source: {src.name}")
            continue
        text = src.read_text(encoding="utf-8").lstrip("﻿").replace("\r\n", "\n")
        data = text.encode("utf-8")
        out = targets / f"grade_{level}_eval.bin"
        out.write_bytes(data)
        flag = ""
        if len(data) < MIN_BYTES:
            short.append(level)
            flag = "  <-- below 4096-byte probe window"
        print(f"  wrote: {out.name} ({len(data):,} bytes){flag}")
    if short:
        print(
            f"\nWARNING: {', '.join(short)} are below {MIN_BYTES} bytes; probe results will be partial.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
