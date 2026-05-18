# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Bridge from validated SFT-trace JSONL (Mintaka W13 output) to the flat
#   uint8 .bin format Veritate's trainer eats.
# - Each input record is a JSON object with at minimum a `trace_bytes` field
#   (either a list of ints 0..255, a base64 string, or a utf-8 string). We
#   accept any of those forms.
# - Output: one concatenated .bin under trainers/corpus/<stem>_train.bin.
#   Records are joined with a separator byte sequence (default: the literal
#   bytes "<|endoftext|>", used by the trainer's tokenizer-free byte
#   shuffler to know where a record ends).
# - This is the LAST mile from "we generated 110k traces" to "we can train
#   on them." Trivial code; the value is in the data, not the bridge.
# veritate_mri/tools/jsonl_to_bin.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import base64
import json
import os
import sys

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_SEPARATOR = b"<|endoftext|>"
DEFAULT_TRACE_KEY = "trace_bytes"
SUPPORTED_KEYS    = ("trace_bytes", "bytes", "text", "trace_text")
VALID_SPLITS      = ("train", "val", "test")
DEFAULT_VAL_FRAC  = 0.02


# ------------------------------------------------------------------------------------
# Functions

def _decode_record(rec, trace_key):
    """Convert one JSONL record's `trace_key` field into a raw bytes object.
    Accepts:
      - list[int]      (the byte-level training format)
      - base64 string  (prefixed 'b64:')
      - utf-8 string   (encode as utf-8)
    Returns bytes, or None if the field is missing / unparseable.
    """
    if trace_key not in rec:
        # Fallback: scan for any of the supported keys
        for k in SUPPORTED_KEYS:
            if k in rec:
                trace_key = k
                break
        else:
            return None
    v = rec[trace_key]
    if isinstance(v, list):
        try:
            return bytes(int(b) & 0xff for b in v)
        except (TypeError, ValueError):
            return None
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, str):
        if v.startswith("b64:"):
            try:
                return base64.b64decode(v[4:])
            except (ValueError, base64.binascii.Error):
                return None
        return v.encode("utf-8")
    return None


def jsonl_to_bin(jsonl_path, out_bin_path, separator=DEFAULT_SEPARATOR,
                 trace_key=DEFAULT_TRACE_KEY, val_split_ratio=0.0,
                 val_bin_path=None, max_records=None):
    """Concatenate JSONL traces into a flat .bin. Returns dict of stats.

    If val_split_ratio > 0, the trailing fraction of records goes to
    val_bin_path. Default 0 (no val split, train.bin only).
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)
    if val_split_ratio < 0 or val_split_ratio > 0.5:
        raise ValueError(f"val_split_ratio must be 0..0.5, got {val_split_ratio}")

    # First pass: read all records (memory-resident; OK for ~100k records,
    # ~500 bytes each = ~50 MB)
    records = []
    n_skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            b = _decode_record(rec, trace_key)
            if b is None:
                n_skipped += 1
                continue
            records.append(b)
            if max_records is not None and len(records) >= max_records:
                break

    if not records:
        raise RuntimeError(f"no usable records found in {jsonl_path} "
                           f"(check trace_key={trace_key!r}; skipped {n_skipped})")

    # Split
    n_val = int(round(len(records) * val_split_ratio))
    val_records = records[-n_val:] if n_val > 0 else []
    train_records = records[:len(records) - n_val] if n_val > 0 else records

    # Write train.bin
    os.makedirs(os.path.dirname(os.path.abspath(out_bin_path)) or ".", exist_ok=True)
    sep = bytes(separator)
    total_train_bytes = 0
    with open(out_bin_path, "wb") as f:
        for i, b in enumerate(train_records):
            f.write(b)
            total_train_bytes += len(b)
            if i + 1 < len(train_records):
                f.write(sep)
                total_train_bytes += len(sep)
    # Write val.bin if requested
    total_val_bytes = 0
    if val_records:
        if not val_bin_path:
            val_bin_path = out_bin_path.replace("_train.bin", "_val.bin")
            if val_bin_path == out_bin_path:
                val_bin_path = out_bin_path + ".val"
        with open(val_bin_path, "wb") as f:
            for i, b in enumerate(val_records):
                f.write(b)
                total_val_bytes += len(b)
                if i + 1 < len(val_records):
                    f.write(sep)
                    total_val_bytes += len(sep)

    return {
        "input":           jsonl_path,
        "train_bin":       out_bin_path,
        "val_bin":         val_bin_path if val_records else None,
        "n_records":       len(records),
        "n_train":         len(train_records),
        "n_val":           len(val_records),
        "n_skipped":       n_skipped,
        "train_bytes":     total_train_bytes,
        "val_bytes":       total_val_bytes,
        "separator_len":   len(sep),
    }


def main():
    ap = argparse.ArgumentParser(description="Concatenate SFT-trace JSONL into a flat byte-level .bin.")
    ap.add_argument("--input",        required=True, help="path to JSONL traces file")
    ap.add_argument("--output",       required=True, help="path to output _train.bin")
    ap.add_argument("--val-bin",      default=None,  help="optional output _val.bin (default derived from --output)")
    ap.add_argument("--val-ratio",    type=float, default=DEFAULT_VAL_FRAC,
                                       help=f"trailing fraction → val.bin (default {DEFAULT_VAL_FRAC})")
    ap.add_argument("--trace-key",    default=DEFAULT_TRACE_KEY,
                                       help=f"JSONL field containing the byte sequence (default '{DEFAULT_TRACE_KEY}'; auto-falls back to {SUPPORTED_KEYS})")
    ap.add_argument("--separator",    default=DEFAULT_SEPARATOR.decode("utf-8"),
                                       help="record separator (default '<|endoftext|>')")
    ap.add_argument("--max-records",  type=int, default=None,
                                       help="cap input record count (for testing)")
    args = ap.parse_args()

    sep = args.separator.encode("utf-8")
    stats = jsonl_to_bin(
        jsonl_path=args.input,
        out_bin_path=args.output,
        separator=sep,
        trace_key=args.trace_key,
        val_split_ratio=args.val_ratio,
        val_bin_path=args.val_bin,
        max_records=args.max_records,
    )

    print("\n=== JSONL → .bin concatenator ===")
    for k, v in stats.items():
        if isinstance(v, int) and v > 1_000_000:
            print(f"  {k:>14}: {v:,} bytes ({v / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {k:>14}: {v}")
    print()
    print(f"Ready for training. To use:")
    print(f"  cp {stats['train_bin']} trainers/corpus/tool_sft_train.bin")
    if stats.get("val_bin"):
        print(f"  cp {stats['val_bin']} trainers/corpus/tool_sft_val.bin")
    print(f"  # then train the 800M plugin with --corpus tool_sft")


if __name__ == "__main__":
    main()
