# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Generic ChatML SFT bin packer. Takes a directory of .jsonl families for a named
#   stem, emits <stem>_train.bin + <stem>_val.bin + LICENSE.md + manifest.json.
# - Byte-deterministic under fixed seed (regression: tests/mri/test_sft_builder.py).
# - Multi-turn friendly: a JSONL row may be {"user","assistant"} (single-turn) OR
#   {"turns":[{"role","content"},...]} (multi-turn). Both render via ChatML.
# - Sibling tool to build_sft_idk_corpus.py; that stays as the IDK-specific entry
#   point. Every other layered SFT (identity, grounded_read, multiturn, empathy,
#   engaged, instruct, prose) invokes THIS builder with a different stem + config.
# veritate_mri/tools/build_sft_corpus.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import hashlib
import json
import os
import random

# ------------------------------------------------------------------------------------
# Constants

IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"
EOT      = "<|endoftext|>"

LICENSE_FILENAME  = "LICENSE.md"
MANIFEST_FILENAME = "manifest.json"

# ------------------------------------------------------------------------------------
# Functions


def _load_jsonl(path):
    """Yield rows from a JSONL family file. Every row is a dict, either
    {"user","assistant"} single-turn or {"turns":[{"role","content"},...]}
    multi-turn. Empty/whitespace lines are skipped; malformed lines raise."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}")
            if "turns" in obj:
                turns = obj["turns"]
                if not isinstance(turns, list) or not turns:
                    raise ValueError(f"{path}:{line_no}: missing/empty turns")
                for t in turns:
                    if not (isinstance(t, dict) and t.get("role") in ("user", "assistant")
                            and isinstance(t.get("content"), str) and t["content"].strip()):
                        raise ValueError(f"{path}:{line_no}: invalid turn: {t!r}")
                out.append(turns)
            else:
                u = obj.get("user"); a = obj.get("assistant")
                if not (isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip()):
                    raise ValueError(f"{path}:{line_no}: missing/empty user or assistant")
                out.append([{"role": "user", "content": u},
                            {"role": "assistant", "content": a}])
    return out


def _format_conversation(turns):
    """Render a list of {"role","content"} turns as one ChatML conversation."""
    parts = []
    for t in turns:
        parts.append(f"{IM_START}{t['role']}\n{t['content']}{IM_END}\n")
    parts.append(f"{EOT}\n")
    return "".join(parts)


def _write_bin(convs, out_path):
    buf = bytearray()
    for c in convs:
        buf.extend(_format_conversation(c).encode("utf-8"))
    with open(out_path, "wb") as f:
        f.write(buf)
    return len(buf), hashlib.sha256(buf).hexdigest()


def build(stem, in_dir, out_dir, family_files, purpose, license_text,
          seed, val_ratio, corpus_dir=None):
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"input dir not found: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)
    if corpus_dir:
        os.makedirs(corpus_dir, exist_ok=True)

    all_convs = []
    family_counts = {}
    for fname in family_files:
        p = os.path.join(in_dir, fname)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing family file: {p}")
        convs = _load_jsonl(p)
        family_counts[fname] = len(convs)
        for c in convs:
            all_convs.append((c, fname))

    rng = random.Random(seed)
    rng.shuffle(all_convs)

    n_val = max(1, int(len(all_convs) * val_ratio))
    val_convs   = [c for (c, _) in all_convs[:n_val]]
    train_convs = [c for (c, _) in all_convs[n_val:]]

    train_name = f"{stem}_train.bin"
    val_name   = f"{stem}_val.bin"
    train_path = os.path.join(out_dir, train_name)
    val_path   = os.path.join(out_dir, val_name)
    lic_path   = os.path.join(out_dir, LICENSE_FILENAME)
    man_path   = os.path.join(out_dir, MANIFEST_FILENAME)

    train_bytes, train_sha = _write_bin(train_convs, train_path)
    val_bytes,   val_sha   = _write_bin(val_convs,   val_path)

    corpus_paths = None
    if corpus_dir:
        c_train = os.path.join(corpus_dir, train_name)
        c_val   = os.path.join(corpus_dir, val_name)
        _write_bin(train_convs, c_train)
        _write_bin(val_convs,   c_val)
        corpus_paths = {"train": c_train, "val": c_val, "stem": stem}

    with open(lic_path, "w", encoding="utf-8") as f:
        f.write(license_text)

    manifest = {
        "name": f"veritate_{stem}",
        "purpose": purpose,
        "byte_template": "ChatML",
        "seed": seed,
        "val_ratio": val_ratio,
        "family_counts": family_counts,
        "total_conversations": sum(family_counts.values()),
        "train_conversations": len(train_convs),
        "val_conversations":   len(val_convs),
        "train_bytes": train_bytes,
        "val_bytes":   val_bytes,
        "train_sha256": train_sha,
        "val_sha256":   val_sha,
        "license": LICENSE_FILENAME,
        "corpus_paths": corpus_paths,
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _parse_args():
    ap = argparse.ArgumentParser(description="Package JSONL families into a Veritate SFT ChatML bin.")
    ap.add_argument("--stem", required=True, help="Corpus stem (e.g. sft_identity_v1).")
    ap.add_argument("--in-dir", required=True, help="Directory with the family .jsonl files.")
    ap.add_argument("--out-dir", required=True, help="Output directory for bins + LICENSE + manifest.")
    ap.add_argument("--families", required=True, help="Comma-separated jsonl family filenames.")
    ap.add_argument("--purpose", default="Layered SFT corpus for Veritate byte-level chat.")
    ap.add_argument("--license-text", default="# Veritate SFT Corpus\n\nFor use with Veritate models only.\n")
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--corpus-dir", default="", help="Optional: also install bins here for the trainer's stem resolver.")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    families = tuple(x.strip() for x in a.families.split(",") if x.strip())
    m = build(a.stem, a.in_dir, a.out_dir, families, a.purpose, a.license_text,
              a.seed, a.val_ratio, corpus_dir=(a.corpus_dir or None))
    print(json.dumps(m, indent=2))
