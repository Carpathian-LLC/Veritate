# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Packages the four generated SFT JSONL families (idk_abstention, chitchat_greetings,
#   jokes, in_domain_qa) into a Veritate byte-level .bin using the same ChatML frame
#   as the shipped chat corpus (build_chat_corpus.py). Emits train.bin + val.bin +
#   LICENSE.md ready for Carpathian COS upload.
# - Family shares (approx): 3000 idk + 2000 chit-chat + 2000 jokes + 2000 in-domain
#   = 9000 pairs. Shuffle interleaves families so no long template runs land in one
#   window (failures.md 2026-07-06/08 lesson: narrow-template SFT interferes).
# - Deterministic: fixed seed makes the .bin sha256 stable across rebuilds.
# - Run:
#     python veritate_mri/tools/build_sft_idk_corpus.py \
#       --in-dir  C:/GitHub/Veritate/temp/sft_gen \
#       --out-dir C:/Users/malka/Desktop/veritate_sft_idk
# veritate_mri/tools/build_sft_idk_corpus.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import hashlib
import json
import os
import random
import sys

# ------------------------------------------------------------------------------------
# Constants

IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"
EOT      = "<|endoftext|>"

FAMILY_FILES = (
    "idk_abstention.jsonl",
    "chitchat_greetings.jsonl",
    "jokes.jsonl",
    "in_domain_qa.jsonl",
)

DEFAULT_SEED      = 20260715
DEFAULT_VAL_RATIO = 0.05
LICENSE_FILENAME  = "LICENSE.md"
TRAIN_FILENAME    = "sft_idk_train.bin"
VAL_FILENAME      = "sft_idk_val.bin"
MANIFEST_FILENAME = "manifest.json"
CORPUS_TRAIN_STEM = "sft_idk"

LICENSE_TEXT = """# Veritate SFT: IDK Abstention Corpus

For use with **Veritate** models only.

## What this teaches

This corpus teaches a small byte-level chat model to:

1. Honestly say **"I don't know that answer."** when the query is outside its trained knowledge (out-of-distribution facts, private data, future events, made-up entities, deep niche facts, opinions).
2. Handle casual chit-chat, greetings, empathy, and small talk without pretending to have facts.
3. Deliver short safe jokes (puns, dad jokes, knock-knock, observational).
4. Answer simple in-domain questions (arithmetic under 100, common word meanings, meta questions about itself) so the model does not collapse into always-abstaining.

## What this is NOT

- Not a general-knowledge corpus. This corpus deliberately has ZERO factual world knowledge (no capitals, dates, prices, historical events).
- Not a coding corpus.
- Not a math corpus beyond trivial arithmetic.
- Not licensed for training non-Veritate models.

## Format

ChatML byte-level:

```
<|im_start|>user
{user text}<|im_end|>
<|im_start|>assistant
{assistant text}<|im_end|>
<|endoftext|>
```

Each conversation is a single (user, assistant) turn separated by `<|endoftext|>`.

## Provenance

Generated 2026-07-15 by Sonnet subagents from prompt families designed against Veritate's failure/success ledgers (in-pretrain dosing at 2-8% per family; late-phase concentrated dosing destroys conversation-copy skill, see failures.md 2026-07-06 and 2026-07-08). Deterministic seed for reproducible sha256.

## Usage in Veritate

Mix into pretrain at 4-8% (NOT as a late-phase SFT phase; the ledger shows narrow templates cause interference at 15-25% doses). Pair with an inference-side abstention gate for the backstop case where the model still tries to answer confidently.
"""


# ------------------------------------------------------------------------------------
# Functions

def _load_jsonl(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}")
            u = obj.get("user")
            a = obj.get("assistant")
            if not isinstance(u, str) or not isinstance(a, str) or not u.strip() or not a.strip():
                raise ValueError(f"{path}:{line_no}: missing/empty user or assistant field")
            pairs.append((u, a))
    return pairs


def _format_pair(user, assistant):
    return (
        f"{IM_START}user\n{user}{IM_END}\n"
        f"{IM_START}assistant\n{assistant}{IM_END}\n"
        f"{EOT}\n"
    )


def _write_bin(pairs, out_path):
    buf = bytearray()
    for u, a in pairs:
        buf.extend(_format_pair(u, a).encode("utf-8"))
    with open(out_path, "wb") as f:
        f.write(buf)
    return len(buf), hashlib.sha256(buf).hexdigest()


def build(in_dir, out_dir, seed=DEFAULT_SEED, val_ratio=DEFAULT_VAL_RATIO,
          corpus_dir=None):
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"input dir not found: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)
    if corpus_dir:
        os.makedirs(corpus_dir, exist_ok=True)

    all_pairs = []
    family_counts = {}
    for fname in FAMILY_FILES:
        path = os.path.join(in_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing family file: {path}")
        pairs = _load_jsonl(path)
        family_counts[fname] = len(pairs)
        for u, a in pairs:
            all_pairs.append((u, a, fname))

    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    n_val = max(1, int(len(all_pairs) * val_ratio))
    val_pairs   = [(u, a) for (u, a, _) in all_pairs[:n_val]]
    train_pairs = [(u, a) for (u, a, _) in all_pairs[n_val:]]

    train_path = os.path.join(out_dir, TRAIN_FILENAME)
    val_path   = os.path.join(out_dir, VAL_FILENAME)
    lic_path   = os.path.join(out_dir, LICENSE_FILENAME)
    man_path   = os.path.join(out_dir, MANIFEST_FILENAME)

    train_bytes, train_sha = _write_bin(train_pairs, train_path)
    val_bytes,   val_sha   = _write_bin(val_pairs,   val_path)

    corpus_paths = None
    if corpus_dir:
        c_train = os.path.join(corpus_dir, f"{CORPUS_TRAIN_STEM}_train.bin")
        c_val   = os.path.join(corpus_dir, f"{CORPUS_TRAIN_STEM}_val.bin")
        _write_bin(train_pairs, c_train)
        _write_bin(val_pairs,   c_val)
        corpus_paths = {"train": c_train, "val": c_val, "stem": CORPUS_TRAIN_STEM}

    with open(lic_path, "w", encoding="utf-8") as f:
        f.write(LICENSE_TEXT)

    manifest = {
        "name": "veritate_sft_idk",
        "purpose": "Teach a small Veritate byte model honest abstention (I don't know that answer.) plus casual chit-chat, safe jokes, and simple in-domain Q/A.",
        "byte_template": "ChatML",
        "seed": seed,
        "val_ratio": val_ratio,
        "family_counts": family_counts,
        "total_pairs": sum(family_counts.values()),
        "train_pairs": len(train_pairs),
        "val_pairs":   len(val_pairs),
        "train_bytes": train_bytes,
        "val_bytes":   val_bytes,
        "train_sha256": train_sha,
        "val_sha256":   val_sha,
        "license": LICENSE_FILENAME,
        "usage": "Mix into pretrain at 4-8% (in-pretrain dosing, not late SFT; failures.md 2026-07-06/08).",
        "corpus_paths": corpus_paths,
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _parse_args():
    ap = argparse.ArgumentParser(description="Build Veritate SFT IDK corpus from JSONL families.")
    ap.add_argument("--in-dir",   default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp", "sft_gen")))
    ap.add_argument("--out-dir",  default=r"C:\Users\malka\Desktop\veritate_sft_idk")
    ap.add_argument("--corpus-dir", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "trainers", "corpus")),
        help="Also write sft_idk_train.bin + sft_idk_val.bin here so the trainer can resolve the stem. Set to empty to skip.")
    ap.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    ap.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    m = build(args.in_dir, args.out_dir, seed=args.seed, val_ratio=args.val_ratio,
              corpus_dir=(args.corpus_dir or None))
    print(json.dumps(m, indent=2))
