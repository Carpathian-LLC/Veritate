# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Assembles the single from-scratch PRETRAINING corpus for a 1B byte-level model.
#   Chinchilla-optimal at 20 tokens/param; byte-level means 1 byte = 1 token, so the
#   default target is 20 GB. Mixes the registered sources at DEFAULT_MIX ratios and
#   writes ONE train .bin plus ONE val .bin.
# - Sources resolve through the platform corpus library (training/sync/corpus_sync):
#   the catalog supplies format/size, install() does any download. "skills" and the
#   optional hand-authored extra corpus are local .bin paths, not catalog entries.
# - Every document goes through corpus_filters (clean -> quality -> pii -> dedup).
#   Chat sources are already ChatML-framed and pass through unchanged; nothing is
#   re-framed. Records are joined by the literal <|endoftext|> separator per
#   developer_documentation/corpus/framing.md.
# - Streaming end to end: sources are read in chunks, documents are emitted as they
#   are produced, output bytes are written incrementally. Nothing is held in RAM.
# - Sources are interleaved document by document (seeded weighted choice), so no long
#   single-source run dominates a training window. The trailing --val-ratio of the
#   assembled stream becomes the val bin.
# - --dry-run resolves and prints the plan (per-source budgets, what is installed,
#   what would download, disk needed) and writes nothing.
# - Run:
#     python veritate_mri/tools/build_base_corpus.py --dry-run
#     python veritate_mri/tools/build_base_corpus.py --target-gb 20
# veritate_mri/tools/build_base_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import random
import sys

HERE     = os.path.dirname(os.path.abspath(__file__))
MRI_ROOT = os.path.normpath(os.path.join(HERE, ".."))

sys.path.insert(0, MRI_ROOT)
sys.path.insert(0, HERE)

import corpus_filters  # noqa: E402

from readers import paths  # noqa: E402
from training.sync import corpus_sync  # noqa: E402
from training.sync import sync_common as sc  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_MIX = {
    "fineweb_edu":    0.27,
    "chat_5gb":       0.26,
    "pg19":           0.20,
    "openwebtext10g": 0.15,
    "skills":         0.06,
    "tinystories":    0.05,
    "wikitext103":    0.01,
}

SKILLS_SOURCE = "skills"
EXTRA_SOURCE  = "extra"

DEFAULT_TARGET_GB       = 20.0
DEFAULT_VAL_RATIO       = 0.005
DEFAULT_SEED            = 20260725
DEFAULT_EXTRA_FRACTION  = 0.001
MIX_SUM_TOLERANCE       = 1e-6

OUT_STEM           = "base_1b"
EXTRA_STEM         = "synthetic_v1"
MANIFEST_FILENAME  = "base_corpus_manifest.json"
MANIFEST_DIR       = os.path.join(paths.REPO_ROOT, "temp")

DEFAULT_OUT_TRAIN  = paths.corpus_train_path(OUT_STEM)
DEFAULT_OUT_VAL    = paths.corpus_val_path(OUT_STEM)
DEFAULT_SKILLS_BIN = paths.corpus_train_path(SKILLS_SOURCE)
DEFAULT_EXTRA_BIN  = paths.corpus_train_path(EXTRA_STEM)
DEFAULT_MANIFEST   = os.path.join(MANIFEST_DIR, MANIFEST_FILENAME)

EOT            = b"<|endoftext|>"
RECORD_SUFFIX  = EOT + b"\n"
HF_ROW_SEP     = corpus_sync.HF_ROW_SEP
HF_FORMAT      = "hf_dataset"

ENCODING       = "utf-8"
DECODE_ERRORS  = "replace"

BYTES_PER_GB          = 1024 ** 3
READ_CHUNK_BYTES      = 1024 * 1024
MIN_DOC_BYTES         = 4096
MAX_DOC_BYTES         = 8 * 1024 * 1024
PROGRESS_EVERY_BYTES  = 512 * 1024 * 1024

CLEAN_REJECT = "unclean"
PLAN_HEADER  = f"{'source':<16}{'budget GB':>12}{'available GB':>14}  state"

# ------------------------------------------------------------------------------------
# Functions

def _load_mix(mix_path):
    if not mix_path:
        return dict(DEFAULT_MIX)
    with open(mix_path, "r", encoding=ENCODING) as f:
        mix = json.load(f)
    total = sum(float(v) for v in mix.values())
    if abs(total - 1.0) > MIX_SUM_TOLERANCE:
        raise ValueError(f"mix fractions must sum to 1.0 (got {total})")
    return {k: float(v) for k, v in mix.items()}


def _new_stat():
    return {"docs_kept": 0, "dedup_drops": 0, "rejects": {}, "bytes_written": 0, "exhausted": False}


def _local_source(name, path, budget_bytes):
    size = os.path.getsize(path) if os.path.isfile(path) else 0
    return {
        "name": name, "path": path, "budget_bytes": budget_bytes,
        "separator": EOT, "min_doc_bytes": 0, "available_bytes": size,
        "installed": size > 0, "entry": None, "stat": _new_stat(),
    }


def _catalog_source(name, entry, budget_bytes):
    is_hf = (entry.get("format") == HF_FORMAT)
    return {
        "name": name,
        "path": paths.corpus_train_path(name),
        "budget_bytes": budget_bytes,
        "separator": HF_ROW_SEP if is_hf else EOT,
        "min_doc_bytes": MIN_DOC_BYTES if is_hf else 0,
        "available_bytes": entry.get("installed_size_train") or entry.get("size_train") or 0,
        "installed": bool(entry.get("installed_train")),
        "entry": entry, "stat": _new_stat(),
    }


def resolve_plan(mix, total_bytes, skills_bin, extra_bin, extra_fraction):
    by_stem = {c["stem"]: c for c in corpus_sync.catalog()["corpora"]}
    plan = []
    for name, fraction in mix.items():
        budget = int(total_bytes * fraction)
        if name == SKILLS_SOURCE:
            plan.append(_local_source(name, skills_bin, budget))
        else:
            plan.append(_catalog_source(name, by_stem[name], budget))
    if extra_fraction > 0:
        plan.append(_local_source(EXTRA_SOURCE, extra_bin, int(total_bytes * extra_fraction)))
    return plan


def print_plan(plan, total_bytes, val_ratio):
    print(PLAN_HEADER)
    to_download = 0
    for p in plan:
        if p["installed"]:
            state = f"installed: {p['path']}"
        elif p["entry"] is not None:
            state = "NEEDS DOWNLOAD"
            to_download += p["available_bytes"]
        else:
            state = f"MISSING local bin: {p['path']}"
        print(f"{p['name']:<16}{p['budget_bytes']/BYTES_PER_GB:>12.3f}"
              f"{p['available_bytes']/BYTES_PER_GB:>14.3f}  {state}")
        if p["available_bytes"] < p["budget_bytes"]:
            print(f"{'':<16}shortfall: source is smaller than its budget; it will exhaust early")
    budget_total = sum(p["budget_bytes"] for p in plan)
    print(f"\nassembled target : {budget_total/BYTES_PER_GB:.3f} GB "
          f"(train {(1.0 - val_ratio) * 100:.2f}% / val {val_ratio * 100:.2f}%)")
    print(f"target-gb        : {total_bytes/BYTES_PER_GB:.3f} GB")
    print(f"to download      : {to_download/BYTES_PER_GB:.3f} GB")
    print(f"disk needed      : {(budget_total + to_download)/BYTES_PER_GB:.3f} GB")


def install_missing(plan):
    for p in plan:
        if p["installed"]:
            continue
        if p["entry"] is None:
            raise FileNotFoundError(f"local source '{p['name']}' not found: {p['path']}")
        print(f"installing {p['name']} ...")
        res = corpus_sync.install(p["entry"])
        if not res.get("ok"):
            raise RuntimeError(f"install of '{p['name']}' failed: {res.get('error')}")
        p["available_bytes"] = os.path.getsize(p["path"])
        p["installed"] = True


def iter_documents(path, separator, min_doc_bytes):
    pending, pending_len, buf = [], 0, b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            parts = (buf + chunk).split(separator)
            buf = parts.pop()
            if len(buf) >= MAX_DOC_BYTES:
                parts.append(buf)
                buf = b""
            for part in parts:
                pending.append(part)
                pending_len += len(part)
                if pending_len >= min_doc_bytes:
                    yield separator.join(pending)
                    pending, pending_len = [], 0
    tail = pending + ([buf] if buf else [])
    if tail:
        yield separator.join(tail)


def filtered_documents(source, deduper):
    stat = source["stat"]
    for raw in iter_documents(source["path"], source["separator"], source["min_doc_bytes"]):
        text = corpus_filters.clean_document(raw.decode(ENCODING, errors=DECODE_ERRORS))
        if text is None:
            stat["rejects"][CLEAN_REJECT] = stat["rejects"].get(CLEAN_REJECT, 0) + 1
            continue
        reason = corpus_filters.quality_reject_reason(text)
        if reason:
            stat["rejects"][reason] = stat["rejects"].get(reason, 0) + 1
            continue
        text = corpus_filters.strip_pii(text)
        if deduper.is_duplicate(text):
            stat["dedup_drops"] += 1
            continue
        deduper.add(text)
        stat["docs_kept"] += 1
        yield text.encode(ENCODING)


def assemble(plan, out_train, out_val, val_ratio, seed):
    rng = random.Random(seed)
    deduper = corpus_filters.DocumentDeduper()
    streams = {p["name"]: filtered_documents(p, deduper) for p in plan}
    val_start = int(sum(p["budget_bytes"] for p in plan) * (1.0 - val_ratio))
    written = 0
    next_report = PROGRESS_EVERY_BYTES
    os.makedirs(os.path.dirname(os.path.abspath(out_train)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_val)), exist_ok=True)
    with open(out_train, "wb") as f_train, open(out_val, "wb") as f_val:
        while True:
            active = [p for p in plan
                      if not p["stat"]["exhausted"] and p["stat"]["bytes_written"] < p["budget_bytes"]]
            if not active:
                break
            source = rng.choices(active, weights=[p["budget_bytes"] for p in active])[0]
            doc = next(streams[source["name"]], None)
            if doc is None:
                source["stat"]["exhausted"] = True
                continue
            record = doc + RECORD_SUFFIX
            (f_train if written < val_start else f_val).write(record)
            written += len(record)
            source["stat"]["bytes_written"] += len(record)
            if written >= next_report:
                print(f"  {written/BYTES_PER_GB:.2f} GB assembled")
                next_report += PROGRESS_EVERY_BYTES
    return written


def build_manifest(plan, args, out_train, out_val, mix):
    return {
        "name": "veritate_base_1b",
        "purpose": "From-scratch byte-level pretraining corpus for a 1B model at 20 tokens/param.",
        "seed": args.seed,
        "target_gb": args.target_gb,
        "val_ratio": args.val_ratio,
        "mix": mix,
        "extra_fraction": args.extra_fraction,
        "sources": [{
            "name": p["name"],
            "path": p["path"],
            "budget_bytes": p["budget_bytes"],
            "bytes_written": p["stat"]["bytes_written"],
            "docs_kept": p["stat"]["docs_kept"],
            "dedup_drops": p["stat"]["dedup_drops"],
            "rejects": p["stat"]["rejects"],
            "exhausted": p["stat"]["exhausted"],
        } for p in plan],
        "train_path": out_train,
        "val_path": out_val,
        "train_bytes": os.path.getsize(out_train),
        "val_bytes": os.path.getsize(out_val),
        "train_sha256": sc.sha256_file(out_train),
        "val_sha256": sc.sha256_file(out_val),
        "docs_kept": sum(p["stat"]["docs_kept"] for p in plan),
        "dedup_drops": sum(p["stat"]["dedup_drops"] for p in plan),
        "rejects": sum(sum(p["stat"]["rejects"].values()) for p in plan),
    }


def build(args):
    mix = _load_mix(args.mix)
    total_bytes = int(args.target_gb * BYTES_PER_GB)
    plan = resolve_plan(mix, total_bytes, args.skills_bin, args.extra_bin, args.extra_fraction)
    print_plan(plan, total_bytes, args.val_ratio)
    if args.dry_run:
        print("\ndry run: nothing written")
        return None
    install_missing(plan)
    print("\nassembling ...")
    assemble(plan, args.out_train, args.out_val, args.val_ratio, args.seed)
    manifest = build_manifest(plan, args, args.out_train, args.out_val, mix)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w", encoding=ENCODING) as f:
        json.dump(manifest, f, indent=2)
    print(f"\ntrain: {args.out_train} ({manifest['train_bytes']/BYTES_PER_GB:.3f} GB)")
    print(f"val:   {args.out_val} ({manifest['val_bytes']/BYTES_PER_GB:.3f} GB)")
    print(f"manifest: {args.manifest}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble the 1B from-scratch pretraining corpus.")
    ap.add_argument("--out-train",      default=DEFAULT_OUT_TRAIN)
    ap.add_argument("--out-val",        default=DEFAULT_OUT_VAL)
    ap.add_argument("--manifest",       default=DEFAULT_MANIFEST)
    ap.add_argument("--target-gb",      type=float, default=DEFAULT_TARGET_GB)
    ap.add_argument("--val-ratio",      type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--seed",           type=int,   default=DEFAULT_SEED)
    ap.add_argument("--mix",            default="", help="json path overriding the default mix")
    ap.add_argument("--skills-bin",     default=DEFAULT_SKILLS_BIN)
    ap.add_argument("--extra-bin",      default=DEFAULT_EXTRA_BIN)
    ap.add_argument("--extra-fraction", type=float, default=DEFAULT_EXTRA_FRACTION)
    ap.add_argument("--dry-run",        action="store_true")
    build(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
