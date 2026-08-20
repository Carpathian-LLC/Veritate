# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the "index its thoughts" step of sleep (IDEA 20 T3/E4): turns the experience
#   log (data/experience/*.jsonl — every exchange the model served) into corpus
#   bins the trainer can consolidate from. Each exchange is reconstructed as the
#   bytes the model actually lived: prompt as experienced (RAG injection and all)
#   + its own reply + a closing turn marker. Exact-duplicate exchanges dedupe;
#   sub-minimum replies drop. Replay mixing needs NO new machinery: sleep is a
#   normal dashboard launch with corpus "experience:0.75,<base>:0.25" — the
#   corpus mixer IS the rehearsal, and it draws only from the model's own past
#   (self-contained rule).
# - usage: python -m tools.build_experience_corpus [--days N] [--min-reply 8]
#          [--val-frac 0.05] [--stem experience]
# veritate_mri/tools/build_experience_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import base64
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from readers.paths import CORPUS_ROOT, EXPERIENCE_ROOT  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

CLOSE_MARKER = b"<|im_end|>\n"
MIN_REPLY_DEFAULT = 8
VAL_FRAC_DEFAULT = 0.05

# ------------------------------------------------------------------------------------
# Functions


def load_exchanges(days=None, min_reply=MIN_REPLY_DEFAULT):
    """Yield deduped exchange byte-blobs from the experience log, oldest first."""
    files = sorted(glob.glob(os.path.join(EXPERIENCE_ROOT, "*.jsonl")))
    if days:
        files = files[-days:]
    seen = set()
    for path in files:
        with open(path, encoding="ascii") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    prompt = base64.b64decode(rec["prompt_b64"])
                    output = base64.b64decode(rec["output_b64"])
                except (ValueError, KeyError):
                    continue   # a torn line must not kill the night's build
                if len(output) < min_reply:
                    continue
                blob = prompt + output
                if not blob.endswith(CLOSE_MARKER):
                    blob += CLOSE_MARKER
                h = hashlib.sha256(blob).digest()
                if h in seen:
                    continue
                seen.add(h)
                yield blob


def build(stem="experience", days=None, min_reply=MIN_REPLY_DEFAULT,
          val_frac=VAL_FRAC_DEFAULT, out_dir=None):
    """Write {stem}_train.bin / {stem}_val.bin. Returns (n_exchanges, train_b, val_b).
    Every val_frac-th exchange goes to val, so val stays a sample of the same
    days rather than only the newest."""
    out_dir = out_dir or CORPUS_ROOT
    os.makedirs(out_dir, exist_ok=True)
    every = max(2, round(1 / val_frac)) if val_frac > 0 else 0
    n = train_b = val_b = 0
    tp = os.path.join(out_dir, f"{stem}_train.bin")
    vp = os.path.join(out_dir, f"{stem}_val.bin")
    with open(tp + ".tmp", "wb") as ft, open(vp + ".tmp", "wb") as fv:
        for blob in load_exchanges(days=days, min_reply=min_reply):
            n += 1
            if every and n % every == 0:
                fv.write(blob)
                val_b += len(blob)
            else:
                ft.write(blob)
                train_b += len(blob)
    os.replace(tp + ".tmp", tp)
    os.replace(vp + ".tmp", vp)
    return n, train_b, val_b


def main():
    ap = argparse.ArgumentParser(description="Build consolidation bins from the experience log.")
    ap.add_argument("--stem", default="experience")
    ap.add_argument("--days", type=int, default=None, help="only the N most recent days")
    ap.add_argument("--min-reply", type=int, default=MIN_REPLY_DEFAULT)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC_DEFAULT)
    args = ap.parse_args()
    n, tb, vb = build(stem=args.stem, days=args.days,
                      min_reply=args.min_reply, val_frac=args.val_frac)
    print(f"{n} exchanges -> {args.stem}_train.bin {tb}B / {args.stem}_val.bin {vb}B")
    if n == 0:
        print("nothing to sleep on: the experience log is empty for the window")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
