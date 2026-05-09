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

This builder also runs a Flesch-Kincaid pass on every source to catch
calibration drift. If a band's FKGL is more than FK_TOLERANCE grades off its
target, the build flags it. The grade probe is only meaningful when each
band actually sits at its claimed reading level.

Usage:
    python veritate_mri/tools/build_grade_evals.py
"""

import re
import sys
from pathlib import Path

LEVELS = ["prek", "k", "elem", "middle", "hs", "college", "phd"]
MIN_BYTES = 8192  # the probe reads up to this many bytes per band; sources should meet or exceed this

# Target FKGL per band. prek doesn't really have an FKGL (kindergarten texts
# can score negative on Flesch-Kincaid because sentences are very short and
# words are mono-syllabic) -- so we accept anything <= 0 as on-target.
FK_TARGETS = {
    "prek":   -1.0,
    "k":       0.0,
    "elem":    4.0,
    "middle":  7.0,
    "hs":     11.0,
    "college": 14.0,
    "phd":    18.0,
}
FK_TOLERANCE = 2.0  # grade levels; bands outside this are flagged


def _syllables(word: str) -> int:
    w = word.lower().strip("'\".,!?;:()[]")
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    syl = len(re.findall(r"[aeiouy]+", w))
    return max(1, syl)


def fk_scores(text: str):
    """Return (FKGL, FRE, n_words, n_sents, ASL, ASW) or None if no text."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    sents = [s for s in sents if s.strip()]
    words = re.findall(r"[A-Za-z][A-Za-z']*", text)
    if not sents or not words:
        return None
    total_syl = sum(_syllables(w) for w in words)
    asl = len(words) / len(sents)
    asw = total_syl / len(words)
    fkgl = 0.39 * asl + 11.8 * asw - 15.59
    fre = 206.835 - 1.015 * asl - 84.6 * asw
    return fkgl, fre, len(words), len(sents), asl, asw


def main() -> int:
    here = Path(__file__).resolve().parent
    sources = here.parent / "grade_eval" / "sources"
    targets = here.parent / "grade_eval"
    if not sources.exists():
        print(f"sources directory missing: {sources}", file=sys.stderr)
        return 1
    short = []
    drift = []
    print(f"{'band':8s} {'bytes':>7s} {'FKGL':>6s} {'target':>7s} {'gap':>5s} {'words':>6s} {'sents':>6s} {'ASL':>5s} {'ASW':>5s}  status")
    print("-" * 88)
    for level in LEVELS:
        src = sources / f"grade_{level}_source.txt"
        if not src.exists():
            print(f"  missing source: {src.name}")
            continue
        text = src.read_text(encoding="utf-8").lstrip("﻿").replace("\r\n", "\n")
        data = text.encode("utf-8")
        out = targets / f"grade_{level}_eval.bin"
        out.write_bytes(data)

        size_flag = ""
        if len(data) < MIN_BYTES:
            short.append(level)
            size_flag = "  short"

        scored = fk_scores(text)
        if scored is None:
            print(f"{level:8s} {len(data):>7d} (no FK)")
            continue
        fkgl, fre, w, s, asl, asw = scored
        target = FK_TARGETS[level]
        # prek is special: anything <=0 is fine since FKGL bottoms at ~ -3
        if level == "prek":
            on_target = fkgl <= 0.5
        else:
            on_target = abs(fkgl - target) <= FK_TOLERANCE
        status = "ok" if on_target else f"DRIFT {fkgl - target:+.1f}"
        if not on_target:
            drift.append((level, fkgl, target))
        gap = fkgl - target
        print(f"{level:8s} {len(data):>7d} {fkgl:6.2f} {target:>7.1f} {gap:>+5.1f} {w:>6d} {s:>6d} {asl:>5.1f} {asw:>5.2f}  {status}{size_flag}")

    rc = 0
    if short:
        print(
            f"\nWARNING: {', '.join(short)} are below {MIN_BYTES} bytes; probe results will be partial.",
            file=sys.stderr,
        )
        rc = 2
    if drift:
        print(
            f"\nWARNING: {len(drift)} band(s) outside FK tolerance ({FK_TOLERANCE} grades): "
            + ", ".join(f"{lv} (FKGL {fk:.1f}, target {tg:.1f})" for lv, fk, tg in drift),
            file=sys.stderr,
        )
        rc = max(rc, 3)
    if rc == 0:
        print("\nall bands within FK tolerance.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
