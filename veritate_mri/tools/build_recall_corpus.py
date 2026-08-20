# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - synthetic recall curriculum (IDEA 20, memory-training fuel): examples where a
#   codeword fact is stated once, buried under SPAN bytes of real filler drawn
#   from an existing corpus bin, then restated — the second statement is
#   predictable only from memory, so with state_carry training the byte loss
#   itself pays for cross-window retention. Spans default 1-6 windows so the
#   dependency cannot be satisfied inside one window. Revision variant: a third
#   of examples restate the fact with a NEW codeword mid-filler and the final
#   statement uses the newest value — rewarding in-place revision (the delta
#   rule's specialty). Facts use random noun/word pairs from disjoint pools so
#   nothing is memorizable into weights.
# - usage: python -m tools.build_recall_corpus [--n 4000] [--filler hansard_train]
#          [--min-span 1024] [--max-span 6144] [--stem recall_curr]
# veritate_mri/tools/build_recall_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from readers.paths import CORPUS_ROOT  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"
REVISE_FRAC = 0.33
VAL_FRAC = 0.05

# ------------------------------------------------------------------------------------
# Functions


def _word(rng, syllables=3):
    return "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS)
                   for _ in range(syllables))


def _example(rng, filler, min_span, max_span):
    noun, word = _word(rng), _word(rng)
    span = rng.randint(min_span, max_span)
    start = rng.randint(0, max(0, len(filler) - span - 1))
    body = filler[start:start + span]
    parts = [f"The codeword for {noun} is {word}. ".encode()]
    revised = rng.random() < REVISE_FRAC
    if revised:
        new_word = _word(rng)
        cut = span // 2
        parts += [body[:cut],
                  f" The codeword for {noun} is now {new_word}. ".encode(),
                  body[cut:]]
        word = new_word
    else:
        parts.append(body)
    parts.append(f" The codeword for {noun} is {word}.\n".encode())
    return b"".join(parts)


def build(n=4000, filler_stem="hansard_train", min_span=1024, max_span=6144,
          stem="recall_curr", seed=0, out_dir=None):
    out_dir = out_dir or CORPUS_ROOT
    filler_path = os.path.join(out_dir, f"{filler_stem}.bin")
    with open(filler_path, "rb") as f:
        filler = f.read()
    if len(filler) < max_span * 2:
        raise SystemExit(f"filler {filler_path} too small ({len(filler)}B)")
    rng = random.Random(seed)
    every = max(2, round(1 / VAL_FRAC))
    tp = os.path.join(out_dir, f"{stem}_train.bin")
    vp = os.path.join(out_dir, f"{stem}_val.bin")
    tb = vb = 0
    with open(tp + ".tmp", "wb") as ft, open(vp + ".tmp", "wb") as fv:
        for i in range(n):
            ex = _example(rng, filler, min_span, max_span)
            if i % every == 0:
                fv.write(ex)
                vb += len(ex)
            else:
                ft.write(ex)
                tb += len(ex)
    os.replace(tp + ".tmp", tp)
    os.replace(vp + ".tmp", vp)
    return tb, vb


def main():
    ap = argparse.ArgumentParser(description="Build the synthetic recall curriculum.")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--filler", default="hansard_train")
    ap.add_argument("--min-span", type=int, default=1024)
    ap.add_argument("--max-span", type=int, default=6144)
    ap.add_argument("--stem", default="recall_curr")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tb, vb = build(n=args.n, filler_stem=args.filler, min_span=args.min_span,
                   max_span=args.max_span, stem=args.stem, seed=args.seed)
    print(f"{args.stem}_train.bin {tb}B / {args.stem}_val.bin {vb}B")


if __name__ == "__main__":
    main()
