# ------------------------------------------------------------------------------------
# veritate_mri/tools/build_comprehension_probe.py
# ------------------------------------------------------------------------------------
# Build per-band reading-comprehension probe items from the hand-authored
# grade_eval/sources/grade_<level>_source.txt files.
#
# The probe is a 4-way MCQ: given a passage prefix, the model must rank the
# correct content word higher than 3 distractors. Distractors are sampled from
# the SAME passage and filtered to similar byte length, so register/genre/style
# fluency cannot disambiguate -- only context-driven meaning can.
#
# Why this is a comprehension signal and not a fluency signal:
#   * All 4 candidates are register-matched (same passage, same genre).
#   * Slots sit mid-passage, so the model has to use long-range context, not
#     just sentence-initial bigrams.
#   * Distractors have similar byte length so length-normalization does not
#     reward arbitrary short or long candidates.
#   * Chance accuracy = 25% (4-way MCQ). Bigram-only models cluster around
#     30-35% (local cues). True context use shows up as 50%+.
#
# Output: veritate_mri/grade_eval/comprehension_<level>.json
#   {
#     "level":      "elem",
#     "source":     "grade_elem_source.txt",
#     "n_items":    30,
#     "seed":       17,
#     "items": [
#       {
#         "prefix_bytes": "...prose bytes...",     // utf-8 string, will be re-encoded
#         "correct":      "garden",
#         "distractors":  ["weeds", "dirt", "soil"],
#         "slot_offset":  872                       // byte offset of the slot in source
#       },
#       ...
#     ]
#   }
#
# Usage:
#   python veritate_mri/tools/build_comprehension_probe.py
# ------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

LEVELS = ["prek", "k", "elem", "middle", "hs", "college", "phd"]
ITEMS_PER_BAND = 30
SEED = 17
PREFIX_MAX_BYTES = 1024              # keep prefix bounded so even small-seq models score every item
MIN_PREFIX_BYTES = 64                # need real context, not just sentence start
MIN_WORD_LEN = 4                     # filter out function words by length
LENGTH_TOLERANCE = 2                 # distractor byte length within +/- this of correct
RECENT_WINDOW_BYTES = 30             # distractor cannot appear in the trailing N bytes of prefix

# Small function-word list. The byte-length filter above already drops most
# function words ("the", "and", "for"); this catches the longer ones that
# slip through (e.g. "would", "there", "after"). Keeping this short means we
# bias toward catching obvious function words rather than fighting POS edge cases.
STOPWORDS = {
    "about", "above", "across", "after", "again", "against", "along", "also",
    "although", "always", "another", "anyone", "anything", "around", "because",
    "before", "behind", "being", "below", "beside", "between", "could", "does",
    "doing", "down", "during", "each", "either", "enough", "even", "every",
    "everyone", "everything", "from", "further", "have", "having", "here",
    "however", "into", "just", "like", "many", "might", "more", "most", "much",
    "must", "near", "neither", "never", "next", "nobody", "none", "nothing",
    "often", "once", "only", "other", "ought", "over", "rather", "really",
    "same", "shall", "should", "since", "some", "someone", "something", "soon",
    "still", "such", "than", "that", "their", "them", "then", "there", "these",
    "they", "thing", "this", "those", "though", "through", "thus", "together",
    "toward", "under", "until", "upon", "very", "well", "were", "what", "when",
    "where", "whether", "which", "while", "with", "within", "without", "would",
    "your",
}

WORD_RE = re.compile(rb"[A-Za-z][A-Za-z']*")


def find_word_offsets(text_bytes: bytes) -> list[tuple[int, int, str]]:
    """Return list of (start_offset, end_offset, lowercase_word) for every word."""
    out = []
    for m in WORD_RE.finditer(text_bytes):
        out.append((m.start(), m.end(), m.group().decode("ascii").lower()))
    return out


def is_content_word(word: str) -> bool:
    return len(word) >= MIN_WORD_LEN and word not in STOPWORDS


def build_band(level: str, source_path: Path, rng: random.Random) -> dict | None:
    if not source_path.is_file():
        print(f"  [skip] {level}: missing {source_path}")
        return None

    text_bytes = source_path.read_bytes()
    # Strip the trailing newline-only padding some sources have so byte offsets
    # don't land in dead space.
    text_bytes = text_bytes.rstrip(b"\n\r ")

    words = find_word_offsets(text_bytes)
    content = [(s, e, w) for (s, e, w) in words if is_content_word(w)]

    if len(content) < ITEMS_PER_BAND * 2:
        print(f"  [skip] {level}: only {len(content)} content words, need >= {ITEMS_PER_BAND * 2}")
        return None

    # Stride through the passage so items aren't clustered at the start.
    # Skip the first MIN_PREFIX_BYTES of text (too little context) and the last
    # ~200 bytes (so distractors can be drawn from material the prefix hasn't
    # seen, AND there's room past the slot for the next item).
    eligible = [(s, e, w) for (s, e, w) in content
                if s >= MIN_PREFIX_BYTES and e <= len(text_bytes) - 200]
    if len(eligible) < ITEMS_PER_BAND:
        print(f"  [skip] {level}: only {len(eligible)} eligible slots after margins")
        return None

    # Evenly-spaced stride pick. rng-shuffles within each bucket so adjacent
    # words don't both get picked.
    stride = len(eligible) // ITEMS_PER_BAND
    slots = []
    for i in range(ITEMS_PER_BAND):
        bucket = eligible[i * stride : (i + 1) * stride]
        if not bucket:
            continue
        slots.append(rng.choice(bucket))

    # Build the per-passage content-word vocabulary for distractor sampling.
    # Sample distractors from the FULL content-word list of the passage, not
    # just the eligible-slots subset -- that gives more variety in distractor
    # surface forms.
    by_lower = {}
    for (s, e, w) in content:
        by_lower.setdefault(w, []).append((s, e))

    items = []
    for (s, e, w) in slots:
        word_len = e - s
        candidates = [other for other in by_lower
                      if other != w
                      and abs(len(other) - word_len) <= LENGTH_TOLERANCE]
        # Filter out distractors that appear in the immediate trailing context
        # of the prefix -- the model would just predict the word it just saw.
        prefix_end = s
        recent = text_bytes[max(0, prefix_end - RECENT_WINDOW_BYTES) : prefix_end].decode("utf-8", "ignore").lower()
        candidates = [d for d in candidates if d not in recent]
        if len(candidates) < 3:
            continue
        distractors = rng.sample(candidates, 3)

        prefix_bytes_full = text_bytes[:s]
        if len(prefix_bytes_full) > PREFIX_MAX_BYTES:
            prefix_bytes_full = prefix_bytes_full[-PREFIX_MAX_BYTES:]
        prefix_str = prefix_bytes_full.decode("utf-8", "ignore")

        # Re-extract the original-case word from source so casing is preserved
        # in the "correct" answer.
        correct_str = text_bytes[s:e].decode("utf-8", "ignore")

        items.append({
            "prefix_bytes": prefix_str,
            "correct":      correct_str,
            "distractors":  distractors,
            "slot_offset":  int(s),
        })

    if len(items) < ITEMS_PER_BAND // 2:
        print(f"  [skip] {level}: only {len(items)} items survived distractor filtering")
        return None

    return {
        "level":   level,
        "source":  source_path.name,
        "n_items": len(items),
        "seed":    SEED,
        "config": {
            "items_per_band":    ITEMS_PER_BAND,
            "prefix_max_bytes":  PREFIX_MAX_BYTES,
            "min_prefix_bytes":  MIN_PREFIX_BYTES,
            "min_word_len":      MIN_WORD_LEN,
            "length_tolerance":  LENGTH_TOLERANCE,
        },
        "items": items,
    }


HARD_MIN_CALLBACK_DISTANCE = 100     # correct word's prior occurrence must be at least this many bytes back
HARD_RECENT_BLOCK_BYTES   = 60       # correct word must NOT appear in trailing N bytes of prefix
HARD_SLOT_START_FRAC      = 0.30     # only pick slots from this fraction onward into the passage
HARD_ITEMS_PER_BAND       = 25       # slightly fewer than easy: callback-eligible slots are rarer


def build_band_hard(level: str, source_path: Path, rng: random.Random) -> dict | None:
    """Long-range entity reference with **syntactically-matched** distractors.

    Each item:
      * Correct word W appears EARLIER in the passage (>= 100 bytes back).
      * W does NOT appear in the immediate trailing 60 bytes of prefix.
      * Let P = the word immediately before W (in the source). All 3 distractors
        are content words observed in the passage as "P X" -- i.e. each distractor
        D appears at least once elsewhere in the passage immediately after the
        same preceding word P. This passage-internal co-occurrence acts as a
        syntactic compatibility filter without requiring a POS tagger: if a word
        was observed after "her" in this passage, it's grammatically compatible
        with "her ___". Distractors that don't pass this filter are excluded.
      * Additional length filter (+/- 2 bytes) so completion-length normalization
        cannot reward arbitrary short or long candidates.

    Why this is harder than the v2 hard probe: v2 allowed distractors with any
    POS (e.g. an adverb among nouns), so a strong-syntax model could win on POS
    alone. This v3 forces every candidate to pass the local syntactic frame,
    leaving only passage-level reference resolution as a disambiguator.
    """
    if not source_path.is_file():
        return None
    text_bytes = source_path.read_bytes().rstrip(b"\n\r ")
    words = find_word_offsets(text_bytes)
    if len(words) < 50:
        return None

    # Build a "what content words follow P?" index, where P is any word
    # immediately preceding a content word. Used to draw POS-matched distractors.
    # Keyed by lowercase P -> sorted-unique list of (followed_lowercase_word).
    followers_by_preceder: dict[str, set[str]] = {}
    content_set = set()
    for i in range(1, len(words)):
        ps, pe, pw = words[i - 1]
        cs, ce, cw = words[i]
        # Only register the follower if it's a content word and the boundary
        # between P and the follower is "word, then maybe space/punct, then word"
        # -- skip cases where there's another content word between them.
        if not is_content_word(cw):
            continue
        # Make sure the gap between P's end and follower's start contains only
        # whitespace/punctuation, no other word. find_word_offsets already gave
        # us contiguous word positions, so adjacency in the list IS adjacency
        # in word-stream; the byte gap may contain punctuation, which is fine.
        followers_by_preceder.setdefault(pw, set()).add(cw)
        content_set.add(cw)

    # Index of content-word occurrences for slot-finding + callback distance check.
    occurrences: dict[str, list[tuple[int, int]]] = {}
    for (s, e, w) in words:
        if w in content_set:
            occurrences.setdefault(w, []).append((s, e))

    n_bytes = len(text_bytes)
    slot_min_offset = int(n_bytes * HARD_SLOT_START_FRAC)

    # Eligible slots: word position i where words[i] is a content word that
    # has an earlier occurrence (callback) and meets recency + offset filters.
    # We need the preceding word words[i-1] for distractor lookup.
    callbacks = []
    for i in range(1, len(words)):
        ps, pe, pw = words[i - 1]
        cs, ce, cw = words[i]
        if cw not in occurrences:
            continue
        occs = occurrences[cw]
        first_start = occs[0][0]
        if cs <= first_start:
            continue                       # this IS the first occurrence
        if cs < slot_min_offset:
            continue
        if ce > n_bytes - 100:
            continue
        if cs - first_start < HARD_MIN_CALLBACK_DISTANCE:
            continue
        recent_window = text_bytes[max(0, cs - HARD_RECENT_BLOCK_BYTES) : cs].decode("utf-8", "ignore").lower()
        if re.search(rf"\b{re.escape(cw)}\b", recent_window):
            continue
        callbacks.append((cs, ce, cw, pw))

    if len(callbacks) < HARD_ITEMS_PER_BAND // 2:
        print(f"  [skip-hard] {level}: only {len(callbacks)} callback-eligible slots (need >= {HARD_ITEMS_PER_BAND // 2})")
        return None

    rng.shuffle(callbacks)
    items = []
    used_offsets = set()
    skipped_no_distractors = 0
    for (cs, ce, cw, pw) in callbacks:
        if len(items) >= HARD_ITEMS_PER_BAND:
            break
        if any(abs(cs - o) < 30 for o in used_offsets):
            continue

        # POS-matched distractor pool: words observed elsewhere in the passage
        # immediately after the same preceder pw, minus cw itself, length-matched,
        # and absent from the recent trailing context.
        pool = followers_by_preceder.get(pw, set())
        word_len = ce - cs
        recent_window = text_bytes[max(0, cs - HARD_RECENT_BLOCK_BYTES) : cs].decode("utf-8", "ignore").lower()
        candidates_pool = [
            other for other in pool
            if other != cw
            and abs(len(other) - word_len) <= LENGTH_TOLERANCE
            and not re.search(rf"\b{re.escape(other)}\b", recent_window)
        ]
        if len(candidates_pool) < 3:
            skipped_no_distractors += 1
            continue
        distractors = rng.sample(candidates_pool, 3)

        prefix_full = text_bytes[:cs]
        if len(prefix_full) > PREFIX_MAX_BYTES:
            prefix_full = prefix_full[-PREFIX_MAX_BYTES:]
        prefix_str = prefix_full.decode("utf-8", "ignore")
        correct_str = text_bytes[cs:ce].decode("utf-8", "ignore")

        items.append({
            "prefix_bytes":      prefix_str,
            "correct":           correct_str,
            "distractors":       distractors,
            "slot_offset":       int(cs),
            "callback_distance": int(cs - occurrences[cw][0][0]),
            "preceding_word":    pw,
        })
        used_offsets.add(cs)

    if len(items) < HARD_ITEMS_PER_BAND // 2:
        print(f"  [skip-hard] {level}: only {len(items)} items survived distractor filtering (skipped {skipped_no_distractors} for no POS-matched distractors)")
        return None

    return {
        "level":   level,
        "source":  source_path.name,
        "n_items": len(items),
        "seed":    SEED,
        "mode":    "hard_long_range_pos_matched",
        "config": {
            "items_per_band":           HARD_ITEMS_PER_BAND,
            "prefix_max_bytes":         PREFIX_MAX_BYTES,
            "min_callback_distance":    HARD_MIN_CALLBACK_DISTANCE,
            "recent_block_bytes":       HARD_RECENT_BLOCK_BYTES,
            "slot_start_fraction":      HARD_SLOT_START_FRAC,
            "length_tolerance":         LENGTH_TOLERANCE,
            "distractor_filter":        "same_preceding_word_in_passage",
        },
        "items": items,
    }


def main():
    here = Path(__file__).resolve().parent
    sources_dir = here.parent / "grade_eval" / "sources"
    out_dir = here.parent / "grade_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sources: {sources_dir}")
    print(f"output:  {out_dir}")
    print()

    print("=== easy (local context completion) ===")
    for level in LEVELS:
        source_path = sources_dir / f"grade_{level}_source.txt"
        rng = random.Random(SEED + LEVELS.index(level))
        band = build_band(level, source_path, rng)
        if band is None:
            continue
        out_path = out_dir / f"comprehension_{level}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(band, f, ensure_ascii=False, indent=1)
        print(f"  {level:8s}  {band['n_items']:3d} items  ->  {out_path.name}")

    print()
    print("=== hard (long-range entity reference) ===")
    for level in LEVELS:
        source_path = sources_dir / f"grade_{level}_source.txt"
        # Distinct seed offset so the hard-mode rng samples don't correlate with easy mode.
        rng = random.Random(SEED + 1000 + LEVELS.index(level))
        band = build_band_hard(level, source_path, rng)
        if band is None:
            continue
        out_path = out_dir / f"comprehension_hard_{level}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(band, f, ensure_ascii=False, indent=1)
        print(f"  {level:8s}  {band['n_items']:3d} items  ->  {out_path.name}")


if __name__ == "__main__":
    main()
