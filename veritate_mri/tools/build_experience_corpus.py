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
# - optional extraction mode (--facts, default off): additionally mines declarative
#   facts from the same window (tools/extract_facts.py) and renders them through
#   build_fact_sft into {stem}_fact_sft_{train,val}.bin plus an auditable
#   {stem}_facts.json — raw-transcript sleep alone does not bind facts (failures.md
#   2026-08-21 m2); the sleep controller chooses the bin mix at launch.
# - usage: python -m tools.build_experience_corpus [--days N] [--min-reply 8]
#          [--val-frac 0.05] [--stem experience] [--facts] [--model NAME]
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
          val_frac=VAL_FRAC_DEFAULT, out_dir=None, min_val_bytes=0):
    """Write {stem}_train.bin / {stem}_val.bin. Returns (n_exchanges, train_b, val_b).
    The split runs on BYTES, not exchange count: val takes a blob whenever it holds
    less than val_frac of what has been written, so it stays a sample spread across
    the same days rather than only the newest. min_val_bytes raises a floor under
    val, filled from the oldest exchanges first -- a caller whose trainer draws a
    fixed contiguous window needs val to reach that window before the first blob
    would land there under val_frac alone."""
    out_dir = out_dir or CORPUS_ROOT
    os.makedirs(out_dir, exist_ok=True)
    n = train_b = val_b = 0
    tp = os.path.join(out_dir, f"{stem}_train.bin")
    vp = os.path.join(out_dir, f"{stem}_val.bin")
    with open(tp + ".tmp", "wb") as ft, open(vp + ".tmp", "wb") as fv:
        for blob in load_exchanges(days=days, min_reply=min_reply):
            n += 1
            to_val = val_frac > 0 and (val_b < min_val_bytes
                                       or val_b < val_frac * (train_b + val_b))
            if to_val:
                fv.write(blob)
                val_b += len(blob)
            else:
                ft.write(blob)
                train_b += len(blob)
    os.replace(tp + ".tmp", tp)
    os.replace(vp + ".tmp", vp)
    return n, train_b, val_b


def build_fact_bins(stem="experience", days=None, model=None, out_dir=None, per_fact=20, seed=0):
    """Extraction mode: mine declarative facts from the window and render them as
    fact-SFT bins ({stem}_fact_sft_{train,val}.bin) next to the raw bins, with the
    extracted facts persisted to {stem}_facts.json for audit. Returns
    (n_facts, train_b, val_b); (0, 0, 0) writes nothing."""
    from tools import build_fact_sft, extract_facts
    out_dir = out_dir or CORPUS_ROOT
    facts, _rejections = extract_facts.extract(extract_facts.load_records(days=days), model=model)
    if not facts:
        return 0, 0, 0
    os.makedirs(out_dir, exist_ok=True)
    facts_path = os.path.join(out_dir, f"{stem}_facts.json")
    with open(facts_path, "w") as f:
        json.dump(facts, f, indent=1)
    _nf, _ne, tb, vb = build_fact_sft.build(facts_path, stem=f"{stem}_fact_sft",
                                            per_fact=per_fact, seed=seed, out_dir=out_dir)
    return len(facts), tb, vb


def main():
    ap = argparse.ArgumentParser(description="Build consolidation bins from the experience log.")
    ap.add_argument("--stem", default="experience")
    ap.add_argument("--days", type=int, default=None, help="only the N most recent days")
    ap.add_argument("--min-reply", type=int, default=MIN_REPLY_DEFAULT)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC_DEFAULT)
    ap.add_argument("--facts", action="store_true", help="also extract facts and emit fact-SFT bins")
    ap.add_argument("--model", default=None, help="fact extraction: only records served by this model")
    args = ap.parse_args()
    n, tb, vb = build(stem=args.stem, days=args.days,
                      min_reply=args.min_reply, val_frac=args.val_frac)
    print(f"{n} exchanges -> {args.stem}_train.bin {tb}B / {args.stem}_val.bin {vb}B")
    if args.facts:
        nf, ftb, fvb = build_fact_bins(stem=args.stem, days=args.days, model=args.model)
        print(f"{nf} facts -> {args.stem}_fact_sft_train.bin {ftb}B / val {fvb}B")
    if n == 0:
        print("nothing to sleep on: the experience log is empty for the window")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
