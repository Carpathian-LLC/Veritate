# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Collects pictures from anywhere on the machine into one named set under
#   data/images/<set>/, which is the single location the codec fit and the corpus
#   builder read. A photo library is spread across folders, export dumps and edits;
#   training wants one directory and no duplicates.
# - Content-addressed: the stored name carries a sha256 prefix, so ingesting the same
#   picture twice through two different paths stores it once, and re-running after
#   adding photos costs a stat per already-seen file instead of a re-read.
# - Hardlinks when the source is on the same volume, which is instant and costs no
#   disk; copies across volumes. Either way the original is never moved or altered.
# - Small pictures are rejected, not upscaled: a codec fitted on a thumbnail blurred
#   up to the training crop learns the blur. --min-edge is that floor.
# - Captions are optional and cost no architecture (a record is caption bytes then
#   code bytes). A <image>.txt sidecar is carried across; --caption-from-folder uses
#   the containing folder's name, which is what a photo library already encodes.
# - usage: .veritate_venv/bin/python -m tools.ingest_images <set> <dir> [<dir>...]
#          [--min-edge 512] [--copy] [--caption-from-folder] [--dry-run]
# veritate_mri/tools/ingest_images.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from readers import paths  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

# Kept in step with build_image_corpus.IMAGE_EXTS: a file this tool stores that the
# corpus builder will not read is a file that silently never trains.
IMAGE_EXTS     = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# Apple's camera format. Pillow cannot open it without the pillow-heif plugin, so
# these are counted and named rather than failing one by one halfway through a run.
HEIF_EXTS      = (".heic", ".heif")
CAPTION_SUFFIX = ".txt"
MANIFEST_NAME  = ".ingest.json"
HASH_PREFIX    = 10
STEM_MAX       = 40
MIN_EDGE_DEFAULT = 512
READ_CHUNK     = 1 << 20
STEM_CLEAN     = re.compile(r"[^A-Za-z0-9._-]+")
PROGRESS_EVERY = 25

# ------------------------------------------------------------------------------------
# Functions


def _heif_ready():
    """True when Pillow can open HEIC. The plugin registers itself on import."""
    try:
        import pillow_heif
    except ImportError:
        return False
    pillow_heif.register_heif_opener()
    return True


def _walk(sources):
    """Every candidate file under the source directories, sorted for determinism."""
    found = []
    for src in sources:
        if os.path.isfile(src):
            found.append(os.path.abspath(src))
            continue
        for root, _dirs, files in os.walk(src):
            for name in sorted(files):
                if name.startswith("."):
                    continue
                found.append(os.path.join(root, name))
    return sorted(set(found))


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(READ_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _probe(path, min_edge):
    """(sha, width, height) for a picture worth keeping, or (None, reason, None).

    Pillow's open is lazy, so the size check reads a header and not the pixels."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            width, height = img.size
    except UnidentifiedImageError:
        return None, "unreadable", None
    except OSError:
        return None, "unreadable", None
    if min(width, height) < min_edge:
        return None, "too_small", None
    return _digest(path), width, height


def _store_name(sha, source):
    stem = STEM_CLEAN.sub("_", os.path.splitext(os.path.basename(source))[0])[:STEM_MAX]
    ext = os.path.splitext(source)[1].lower()
    return sha[:HASH_PREFIX] + ("_" + stem if stem else "") + ext


def _place(source, target, copy):
    """Hardlink where the filesystem allows it, else copy. The source is never moved:
    a person's photo library is not this tool's to rearrange."""
    if not copy:
        try:
            os.link(source, target)
            return "link"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copy"


def _caption_for(source, from_folder):
    sidecar = os.path.splitext(source)[0] + CAPTION_SUFFIX
    if os.path.isfile(sidecar):
        with open(sidecar, "rb") as handle:
            text = handle.read().strip()
        if text:
            return text
    if from_folder:
        folder = os.path.basename(os.path.dirname(source)).strip()
        if folder:
            return folder.encode("utf-8", "replace")
    return b""


def _load_manifest(set_dir):
    path = os.path.join(set_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return {"stored": {}, "seen": {}}
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    doc.setdefault("stored", {})
    doc.setdefault("seen", {})
    return doc


def _save_manifest(set_dir, manifest):
    path = os.path.join(set_dir, MANIFEST_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _seen_key(path):
    """Cheap identity for an already-ingested source: a re-run stats, never re-reads."""
    st = os.stat(path)
    return str(int(st.st_size)) + ":" + str(int(st.st_mtime))


def ingest(set_name, sources, min_edge=MIN_EDGE_DEFAULT, copy=False,
           caption_from_folder=False, dry_run=False, workers=8, progress=None):
    """Collect every picture under `sources` into data/images/<set_name>/.

    Returns the report. Idempotent: running it again after adding photos ingests
    only what is new. `progress(done, total)` is called as pictures are read, because
    hashing a photo library takes minutes and a silent minute reads as a hang."""
    set_dir = paths.image_set_dir(set_name)
    if not dry_run:
        os.makedirs(set_dir, exist_ok=True)
    manifest = _load_manifest(set_dir) if os.path.isdir(set_dir) else {"stored": {}, "seen": {}}
    stored, seen = manifest["stored"], manifest["seen"]
    heif_ready = _heif_ready()

    candidates, skipped_ext, heif_blocked = [], 0, 0
    for path in _walk(sources):
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTS or (ext in HEIF_EXTS and heif_ready):
            candidates.append(path)
        elif ext in HEIF_EXTS:
            heif_blocked += 1
        elif ext != CAPTION_SUFFIX:
            skipped_ext += 1

    fresh = [p for p in candidates if seen.get(p) != _seen_key(p)]
    report = {"set": set_name, "dir": set_dir, "scanned": len(candidates),
              "already_ingested": len(candidates) - len(fresh), "added": 0,
              "duplicates": 0, "too_small": 0, "unreadable": 0, "linked": 0, "copied": 0,
              "captions": 0, "heif_blocked": heif_blocked, "other_files": skipped_ext,
              "total_in_set": len(stored)}

    probes = []
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        jobs = {pool.submit(_probe, path, min_edge): path for path in fresh}
        for k, job in enumerate(futures.as_completed(jobs), 1):
            probes.append((jobs[job], *job.result()))
            if progress is not None and (k % PROGRESS_EVERY == 0 or k == len(jobs)):
                progress(k, len(jobs))
    probes.sort(key=lambda row: row[0])

    for source, sha, width, height in probes:
        if sha is None:
            report["too_small" if width == "too_small" else "unreadable"] += 1
            continue
        seen[source] = _seen_key(source)
        if sha in stored:
            report["duplicates"] += 1
            continue
        name = _store_name(sha, source)
        report["added"] += 1
        stored[sha] = {"name": name, "source": source, "width": width, "height": height}
        if dry_run:
            continue
        how = _place(source, os.path.join(set_dir, name), copy)
        report["linked" if how == "link" else "copied"] += 1
        caption = _caption_for(source, caption_from_folder)
        if caption:
            with open(os.path.join(set_dir, os.path.splitext(name)[0] + CAPTION_SUFFIX),
                      "wb") as handle:
                handle.write(caption)
            report["captions"] += 1

    report["total_in_set"] = len(stored)
    if not dry_run:
        _save_manifest(set_dir, manifest)
    return report


def main():
    ap = argparse.ArgumentParser(description="Collect images into one named set.")
    ap.add_argument("set_name")
    ap.add_argument("sources", nargs="+", help="directories (or files) to collect from")
    ap.add_argument("--min-edge", type=int, default=MIN_EDGE_DEFAULT,
                    help="reject pictures whose short edge is below this; a thumbnail "
                         "upscaled to the training crop teaches the codec the blur")
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of hardlinking, even on the same volume")
    ap.add_argument("--caption-from-folder", action="store_true",
                    help="use the containing folder's name as the caption when no "
                         "<image>.txt sidecar exists")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rep = ingest(args.set_name, args.sources, min_edge=args.min_edge, copy=args.copy,
                 caption_from_folder=args.caption_from_folder, dry_run=args.dry_run,
                 workers=args.workers)
    print(f"{rep['scanned']} images scanned, {rep['already_ingested']} already ingested")
    print(f"added {rep['added']} ({rep['linked']} linked, {rep['copied']} copied, "
          f"{rep['captions']} captions), {rep['duplicates']} duplicates, "
          f"{rep['too_small']} below --min-edge, {rep['unreadable']} unreadable")
    if rep["heif_blocked"]:
        print(f"NOTE: {rep['heif_blocked']} HEIC/HEIF files skipped — Pillow cannot open "
              f"them here. Install the plugin to include them: "
              f".veritate_venv/bin/pip install pillow-heif")
    print(f"set now holds {rep['total_in_set']} images: {rep['dir']}")
    return 0 if rep["total_in_set"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
