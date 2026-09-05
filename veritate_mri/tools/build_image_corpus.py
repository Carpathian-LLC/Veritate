# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Encodes a directory of images into an ordinary byte corpus. Output is
#   data/corpus/<stem>_{train,val}.bin like every other builder: nothing downstream
#   knows these bytes came from pictures.
# - Record layout is `caption bytes + code block + <|endoftext|>`, the documented
#   separator. The code block is a FIXED length for a given codec and geometry, so the
#   last `image_code_bytes` bytes before each separator are the image and everything
#   before them is the caption. That is what lets the trainer find the code block
#   without a sidecar index or a marker that image bytes could collide with.
# - A caption is optional. With none the corpus is unconditional; with one, text
#   conditioning costs the byte model no architecture at all, because the caption is
#   simply the bytes that precede the image in the same stream.
# - build_streaming is the path the trainer takes. Frames the codec fit already decoded
#   come from the pixel cache; every other picture is decoded in a thread pool one
#   batch AHEAD of the codec, so the GPU never waits on Pillow and no picture is
#   decoded twice. A 140k-picture library never touches the disk as pixels.
# - Pillow is imported inside the loader so the packing path stays importable, and
#   testable, on an install that has not got it.
# veritate_mri/tools/build_image_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import concurrent.futures as futures
import json
import os
import sys

import numpy as np
import torch

from veritate_core.plugin import hardware, image_codec
from veritate_core.plugin.image_progress import StopRequested

# ------------------------------------------------------------------------------------
# Constants

RECORD_SEP     = b"<|endoftext|>"
IMAGE_EXTS     = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CAPTION_SUFFIX = ".txt"
VAL_EVERY      = 50
RGB_MAX        = 255.0
RGB            = 3
HASH_CHARS     = 8
ENCODE_BATCH   = 64
LOG_BATCHES    = 20

# ------------------------------------------------------------------------------------
# Functions


def _load_image(path, height, width):
    from PIL import Image

    with Image.open(path) as handle:
        img = handle.convert("RGB")
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        img = img.resize((max(width, int(src_w * scale + 0.5)), max(height, int(src_h * scale + 0.5))))
        left, top = (img.size[0] - width) // 2, (img.size[1] - height) // 2
        img = img.crop((left, top, left + width, top + height))
        flat = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return flat.reshape(height, width, 3).permute(2, 0, 1).float().div(RGB_MAX)


def pack_record(codec, image, caption=b""):
    """One corpus record: caption, then the fixed-length code block, then the separator."""
    with torch.no_grad():
        codes = codec.encode(image.unsqueeze(0))[0]
    block = codec.to_bytes(codes)
    if RECORD_SEP in block:
        raise ValueError("code block contains the record separator; re-encode this image")
    return caption + block + RECORD_SEP


def _sources(image_dir, limit):
    found = []
    for root, _dirs, files in os.walk(image_dir):
        for name in sorted(files):
            if name.lower().endswith(IMAGE_EXTS):
                found.append(os.path.join(root, name))
    found.sort()
    return found[:limit] if limit else found


def _caption_for(image_path, captions):
    if not captions:
        return b""
    sidecar = os.path.splitext(image_path)[0] + CAPTION_SUFFIX
    if not os.path.isfile(sidecar):
        return b""
    with open(sidecar, "rb") as handle:
        return handle.read().strip()


def build(paths, image_dir, codec_name, stem, height, width, out_dir=None,
          val_every=VAL_EVERY, captions=True, limit=0):
    """Encode every image under image_dir into <stem>_{train,val}.bin.

    Returns the report the caller needs to configure the run, including
    `image_code_bytes`: the trainer's masked objective needs that number to know
    which bytes of a record are the image."""
    codec = image_codec.load(paths.codec_path(codec_name))
    code_bytes = codec.code_bytes(height, width)
    out_dir = out_dir or paths.corpus_dir()
    os.makedirs(out_dir, exist_ok=True)
    sources = _sources(image_dir, limit)
    if not sources:
        raise ValueError("no images under " + str(image_dir))

    train_path = os.path.join(out_dir, stem + "_train.bin")
    val_path   = os.path.join(out_dir, stem + "_val.bin")
    written = {"train": 0, "val": 0}
    counts  = {"train": 0, "val": 0}
    with open(train_path + ".tmp", "wb") as ft, open(val_path + ".tmp", "wb") as fv:
        for i, src in enumerate(sources):
            record = pack_record(codec, _load_image(src, height, width), _caption_for(src, captions))
            split = "val" if i % val_every == 0 else "train"
            (fv if split == "val" else ft).write(record)
            written[split] += len(record)
            counts[split] += 1
    os.replace(train_path + ".tmp", train_path)
    os.replace(val_path + ".tmp", val_path)
    return {
        "stem": stem, "codec": codec_name, "images": len(sources),
        "height": height, "width": width,
        "image_code_bytes": code_bytes, "planes": codec.planes, "patch": codec.patch,
        "train_records": counts["train"], "val_records": counts["val"],
        "train_bytes": written["train"], "val_bytes": written["val"],
    }


def _cached_val(name, i, val_every):
    """Val membership from the content hash in the filename, so a picture keeps its
    side of the split as the set grows. fit_image_codec splits the same way, so the
    codec is never fitted on a picture the corpus holds out."""
    try:
        return int(name[:HASH_CHARS], 16) % val_every == 0
    except ValueError:
        return i % val_every == 0


def _open_cache(paths, set_name, height, width):
    """(memmap, {name: row}) for the pixel cache at this geometry, or (None, {})."""
    cache_path = paths.image_cache_path(set_name, height, width)
    index_path = paths.image_cache_index_path(set_name, height, width)
    if not (os.path.isfile(cache_path) and os.path.isfile(index_path)):
        return None, {}
    with open(index_path, encoding="utf-8") as handle:
        names = json.load(handle)["names"]
    cache = np.memmap(cache_path, dtype=np.uint8, mode="r",
                      shape=(len(names), height, width, RGB))
    return cache, {n: i for i, n in enumerate(names)}


def build_streaming(paths, set_name, codec_name, stem, height, width, out_dir=None,
                    val_every=VAL_EVERY, captions=True, device="auto", verbose=True,
                    workers=None, progress=None, should_stop=None, out_scale=1):
    """The corpus for a whole set, at the speed of the codec rather than of Pillow.

    Frames in the pixel cache (the sample the codec was fitted on) are read from it;
    the rest are decoded by a thread pool that always works one batch ahead of the
    encode, so decode and GPU overlap. `progress(done, total)` after every batch;
    `should_stop()` true raises StopRequested and removes the half-written bins.
    With `out_scale` > 1 the cache holds the pictures at out_scale x the frame (what the
    decoder learned from), so every picture is decoded at the frame here instead: one
    decode path for the whole corpus, at the cost of re-decoding the fitted sample."""
    from tools import fit_image_codec

    set_dir = paths.image_set_dir(set_name)
    if not os.path.isdir(set_dir):
        raise ValueError("no image set at " + set_dir)
    names = fit_image_codec._set_images(set_dir)
    if not names:
        raise ValueError("no images in " + set_dir)
    cache, row_of = _open_cache(paths, set_name, height, width) if int(out_scale) == 1 else (None, {})

    codec = image_codec.load(paths.codec_path(codec_name))
    dev = hardware.pick_device(device)
    codec = codec.to(dev).eval()
    code_bytes = codec.code_bytes(height, width)
    out_dir = out_dir or paths.corpus_dir()
    os.makedirs(out_dir, exist_ok=True)
    workers = workers or fit_image_codec.DECODE_WORKERS
    n_cached = sum(1 for n in names if n in row_of)
    if verbose:
        print(f"corpus: {len(names)} pictures, {n_cached} from the cache, "
              f"{len(names) - n_cached} to decode, encoding on {dev}", flush=True)

    def submit(pool, chunk):
        out = []
        for n in chunk:
            if n in row_of:
                out.append((n, row_of[n]))
            else:
                out.append((n, pool.submit(fit_image_codec._decode,
                                           (os.path.join(set_dir, n), height, width))))
        return out

    chunks = [names[i:i + ENCODE_BATCH] for i in range(0, len(names), ENCODE_BATCH)]
    train_path = os.path.join(out_dir, stem + "_train.bin")
    val_path   = os.path.join(out_dir, stem + "_val.bin")
    written = {"train": 0, "val": 0}
    counts  = {"train": 0, "val": 0}
    done = 0
    try:
        with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool, \
                open(train_path + ".tmp", "wb") as ft, open(val_path + ".tmp", "wb") as fv:
            pending = submit(pool, chunks[0])
            for i, _chunk in enumerate(chunks):
                current = pending
                pending = submit(pool, chunks[i + 1]) if i + 1 < len(chunks) else None
                frames = np.stack([np.array(cache[src]) if isinstance(src, int) else src.result()
                                   for _n, src in current])
                images = torch.from_numpy(frames).to(dev).permute(0, 3, 1, 2).float().div_(RGB_MAX)
                with torch.no_grad():
                    codes = codec.encode(images).to("cpu")
                for k, (name, _src) in enumerate(current):
                    block = codec.to_bytes(codes[k])
                    if RECORD_SEP in block:
                        raise ValueError("code block contains the record separator: " + name)
                    caption = _caption_for(os.path.join(set_dir, name), captions)
                    record = caption + block + RECORD_SEP
                    split = "val" if _cached_val(name, done + k, val_every) else "train"
                    (fv if split == "val" else ft).write(record)
                    written[split] += len(record)
                    counts[split] += 1
                done += len(current)
                if progress:
                    progress(done, len(names))
                if should_stop and should_stop():
                    raise StopRequested("stopped while encoding the corpus")
                if verbose and (i + 1) % LOG_BATCHES == 0:
                    print("  encoded " + str(done) + "/" + str(len(names)), flush=True)
    except BaseException:
        for p in (train_path, val_path):
            try:
                os.remove(p + ".tmp")
            except OSError:
                pass
        raise
    os.replace(train_path + ".tmp", train_path)
    os.replace(val_path + ".tmp", val_path)
    return {
        "stem": stem, "codec": codec_name, "images": len(names),
        "height": height, "width": width,
        "image_code_bytes": code_bytes, "planes": codec.planes, "patch": codec.patch,
        "train_records": counts["train"], "val_records": counts["val"],
        "train_bytes": written["train"], "val_bytes": written["val"],
        "from_cache": n_cached, "decoded": len(names) - n_cached,
    }


# The older name. Same corpus; the streaming build simply no longer requires every
# frame to be in the cache first.
build_from_cache = build_streaming


def main():
    ap = argparse.ArgumentParser(description="Encode images into a training corpus.")
    ap.add_argument("stem", help="corpus stem: writes <stem>_train.bin / <stem>_val.bin")
    ap.add_argument("codec_name")
    ap.add_argument("--set", dest="set_name", default="",
                    help="named image set (data/images/<set>): cached frames are reused, "
                         "the rest are decoded one batch ahead of the codec")
    ap.add_argument("--image-dir", default="", help="a directory of images instead")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--val-every", type=int, default=VAL_EVERY)
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if bool(args.set_name) == bool(args.image_dir):
        ap.error("pass exactly one of --set or --image-dir")

    from readers import paths as paths_mod
    if args.set_name:
        rep = build_streaming(paths_mod, args.set_name, args.codec_name, args.stem,
                              args.height, args.width, val_every=args.val_every,
                              captions=not args.no_captions, device=args.device)
    else:
        rep = build(paths_mod, args.image_dir, args.codec_name, args.stem,
                    args.height, args.width, val_every=args.val_every,
                    captions=not args.no_captions, limit=args.limit)
    print(f"{rep['images']} images -> {rep['train_records']} train / "
          f"{rep['val_records']} val records")
    print(f"{rep['stem']}_train.bin {rep['train_bytes']}B / val {rep['val_bytes']}B")
    print(f"image_code_bytes: {rep['image_code_bytes']}  "
          f"(the run needs this, and seq >= it)")
    return 0 if rep["images"] else 1


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))
    raise SystemExit(main())
