# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Build a unigram + bigram index over a corpus .bin file. Consumed by the
#   writing-health probe (checkpoint_probe.dump_writing_health) to compute PMI
#   of adjacent word pairs in model generations.
# - Owns the sidecar path (sidecar_path) and the build (write_index). The probe
#   builds a missing sidecar on demand through this module; --all pre-builds the
#   whole corpus library.
# - Output (.npz next to the corpus, <stem>_bigrams.npz):
#     vocab     unique tokens (N,) <Uk
#     uni_c     unigram counts (N,) int64
#     bi_keys   packed bigram keys uint64 = (i<<32)|j  (M,)
#     bi_c      bigram counts (M,) int64
#     n_tokens, n_bigrams, config
# - Usage:
#     python veritate_mri/tools/build_bigram_index.py --corpus base_v1
#     python veritate_mri/tools/build_bigram_index.py --all
# veritate_mri/tools/build_bigram_index.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_MRI_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _MRI_ROOT not in sys.path:
    sys.path.insert(0, _MRI_ROOT)

from readers import paths  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

WORD_RE = re.compile(rb"[a-z][a-z']*")
DEFAULT_TOP_UNI     = 150_000
DEFAULT_TOP_BIGRAMS = 500_000
CHUNK_BYTES         = 64 * 1024 * 1024   # 64 MB chunks; PG19 (10 GB) needs streaming
TRAIN_GLOB          = f"*{paths.CORPUS_TRAIN_SUFFIX}"
PG19_STEM           = "pg19"             # ~10 GB; opt-in only
BIN_SUFFIX          = ".bin"
SIDECAR_SUFFIX      = "_bigrams.npz"
SIDECAR_ALT_SUFFIX  = ".bigrams.npz"     # corpus files that are not .bin


# ------------------------------------------------------------------------------------
# Functions

def sidecar_path(corpus_path: str) -> str:
    """Where the bigram index for `corpus_path` lives. Single owner of the name
    (the writing-health probe reads the same path)."""
    if corpus_path.endswith(BIN_SUFFIX):
        return corpus_path[:-len(BIN_SUFFIX)] + SIDECAR_SUFFIX
    return corpus_path + SIDECAR_ALT_SUFFIX


def iter_words(corpus_path: str):
    """Yield lowercase byte-tokens from the corpus, streaming in chunks so we
    don't load the full file. Tokens at chunk boundaries are stitched by
    keeping a small overlap window."""
    overlap = 64
    with open(corpus_path, "rb") as f:
        carry = b""
        while True:
            buf = f.read(CHUNK_BYTES)
            if not buf:
                if carry:
                    for m in WORD_RE.finditer(carry.lower()):
                        yield m.group(0)
                break
            data = (carry + buf).lower()
            # find last whitespace boundary so we don't split a token
            end = max(0, len(data) - overlap)
            for m in WORD_RE.finditer(data, 0, end):
                yield m.group(0)
            carry = data[end:]


def build_index(corpus_path: str, top_uni: int, top_bigrams: int, max_bytes: int = 0):
    t0 = time.time()
    uni = Counter()
    bigrams = Counter()
    last_token = None
    n_tokens = 0
    n_bigrams = 0
    bytes_read = 0

    for tok in iter_words(corpus_path):
        n_tokens += 1
        bytes_read += len(tok) + 1  # rough
        uni[tok] += 1
        if last_token is not None:
            bigrams[(last_token, tok)] += 1
            n_bigrams += 1
        last_token = tok
        if max_bytes and bytes_read > max_bytes:
            break
        # Periodic memory cap: trim Counters when they balloon (PG19 case).
        # Stash the head, blow away the long tail of singletons, refill.
        if n_tokens % 5_000_000 == 0:
            now = time.time()
            print(f"  scanned {n_tokens:,} tokens ({n_bigrams:,} bigrams) "
                  f"in {now - t0:.1f}s; uni={len(uni):,} bi={len(bigrams):,}",
                  file=sys.stderr)
            if len(bigrams) > top_bigrams * 6:
                # shrink: keep top top_bigrams * 4 by current count
                keep = bigrams.most_common(top_bigrams * 4)
                bigrams = Counter(dict(keep))

    # Build vocab from top unigrams.
    uni_top = uni.most_common(top_uni)
    vocab = [tok.decode("utf-8", errors="replace") for tok, _ in uni_top]
    uni_c = np.array([c for _, c in uni_top], dtype=np.int64)
    tok2idx = {tok: i for i, (tok, _) in enumerate(uni_top)}

    # Filter bigrams to those whose both tokens are in vocab; keep top top_bigrams.
    filtered = []
    for (a, b), c in bigrams.most_common():
        ia = tok2idx.get(a)
        ib = tok2idx.get(b)
        if ia is None or ib is None:
            continue
        filtered.append(((ia, ib), c))
        if len(filtered) >= top_bigrams:
            break
    bi_keys = np.array([(ia << 32) | ib for (ia, ib), _ in filtered], dtype=np.uint64)
    bi_c    = np.array([c for _, c in filtered], dtype=np.int64)

    # Vocab as fixed-width unicode array.
    if vocab:
        max_w = max(len(v) for v in vocab)
        vocab_arr = np.array(vocab, dtype=f"<U{max(max_w, 1)}")
    else:
        vocab_arr = np.array([], dtype="<U1")

    elapsed = time.time() - t0
    config = {
        "corpus_path":    os.path.abspath(corpus_path),
        "corpus_bytes":   os.path.getsize(corpus_path),
        "max_bytes":      max_bytes,
        "top_uni":        top_uni,
        "top_bigrams":    top_bigrams,
        "n_tokens":       n_tokens,
        "n_bigrams":      n_bigrams,
        "vocab_size":     len(vocab),
        "bigrams_kept":   len(filtered),
        "build_time_s":   round(elapsed, 2),
        "tokens_per_sec": round(n_tokens / max(elapsed, 1e-3), 1),
    }
    return {
        "vocab":     vocab_arr,
        "uni_c":     uni_c,
        "bi_keys":   bi_keys,
        "bi_c":      bi_c,
        "n_tokens":  np.int64(n_tokens),
        "n_bigrams": np.int64(n_bigrams),
        "config":    np.array(json.dumps(config)),
    }, config


def write_index(corpus_path: str, top_uni: int = DEFAULT_TOP_UNI,
                top_bigrams: int = DEFAULT_TOP_BIGRAMS, max_bytes: int = 0) -> dict:
    print(f"\nbuilding index: {corpus_path}", file=sys.stderr)
    if max_bytes:
        print(f"  capped at {max_bytes:,} bytes", file=sys.stderr)
    payload, cfg = build_index(corpus_path, top_uni, top_bigrams, max_bytes)
    out_path = sidecar_path(corpus_path)
    np.savez_compressed(out_path, **payload)
    size = os.path.getsize(out_path)
    print(f"  wrote {os.path.basename(out_path)} ({size / 1024 / 1024:.2f} MB) in {cfg['build_time_s']}s",
          file=sys.stderr)
    print(f"  vocab={cfg['vocab_size']:,}  bigrams_kept={cfg['bigrams_kept']:,}  total_tokens={cfg['n_tokens']:,}",
          file=sys.stderr)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",      type=str, default=None,
                    help="corpus stem (e.g. base_v1) -- looks up trainers/corpus/<stem>_train.bin")
    ap.add_argument("--corpus-path", type=str, default=None,
                    help="explicit absolute path to a corpus .bin")
    ap.add_argument("--all",         action="store_true",
                    help="build indexes for every trainers/corpus/*_train.bin (skip pg19 unless --include-pg19)")
    ap.add_argument("--include-pg19", action="store_true",
                    help="also index pg19 (~10 GB; slow)")
    ap.add_argument("--top-uni",     type=int, default=DEFAULT_TOP_UNI)
    ap.add_argument("--top-bigrams", type=int, default=DEFAULT_TOP_BIGRAMS)
    ap.add_argument("--max-bytes",   type=int, default=0,
                    help="cap input bytes (debug)")
    args = ap.parse_args()

    targets: list[str] = []
    if args.corpus_path:
        targets.append(args.corpus_path)
    elif args.corpus:
        p = Path(paths.corpus_train_path(args.corpus))
        if not p.exists():
            print(f"corpus not found: {p}", file=sys.stderr)
            return 1
        targets.append(str(p))
    elif args.all:
        found = {}
        for root in paths.corpus_search_dirs():
            for cand in sorted(Path(root).glob(TRAIN_GLOB)):
                found.setdefault(cand.name, cand)
        for p in sorted(found.values(), key=lambda x: x.name):
            stem = p.stem.replace("_train", "")
            if stem == PG19_STEM and not args.include_pg19:
                print(f"  skipping {p.name} (pg19; pass --include-pg19 to index)", file=sys.stderr)
                continue
            targets.append(str(p))
    else:
        ap.print_help()
        return 1

    rc = 0
    for path in targets:
        try:
            write_index(path, args.top_uni, args.top_bigrams, args.max_bytes)
        except Exception as e:
            print(f"\nFAILED {path}: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
