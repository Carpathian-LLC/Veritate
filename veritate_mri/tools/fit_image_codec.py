# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - IDEA 24 F1: fit the image <-> bytes codec on real pictures. This is codec fitting,
#   not a model run — no checkpoint, no config.json, no dashboard run record — which is
#   why it is a tool and not a trainer launch (rule 13 governs the latter).
# - The cost of a fit over personal photos is Pillow, not arithmetic. Decoding is paid
#   once per (set, geometry) into a flat uint8 memmap and never again: every later
#   epoch, every later fit, and the corpus build all read that cache. Adding photos
#   appends; already-decoded frames are never decoded twice.
# - A codec is fitted on a SAMPLE of the set (FIT_SAMPLE_MAX pictures, the first that
#   many content-hash-ordered names, which is a random draw). A codec does not get
#   better past ~10k pictures, and a 140k-picture library would otherwise mean a 40 GB
#   cache and hours of epochs before the model sees a single step. The corpus build
#   (build_image_corpus) streams the rest of the library past the fitted codec.
# - JPEGs are decoded at a reduced DCT scale (Image.draft) when the target frame is
#   much smaller than the photo, which is most of the wall clock of this stage on a
#   phone library: a 12 MP photo going to 320x320 decodes at 1/4 scale. EXIF orientation
#   is applied, so portrait shots are upright. Both are pure CPU work: this stage is
#   the one that is not on the GPU, by nature of what JPEG decoding is.
# - The cache is refused before it is started when it would not fit on the disk.
#   A 1920x1080 frame over 140k pictures is 869 GB; that must be an error message, not
#   a full disk an hour later.
# - Frames stay uint8 until they are on the device, so the host->device copy carries a
#   quarter of the bytes and the divide runs where the arithmetic is.
# - Held-out PSNR is reported on pictures the fit never sees, split by the content hash
#   in the filename so an image keeps its side of the split for the life of the set.
#   F1's falsifier is LPIPS <= 0.15 under 2,048 codes (ideas.md); PSNR is the cheap
#   proxy this tool can compute with no extra dependency, and it is NOT that number.
# - usage: .veritate_venv/bin/python -m tools.fit_image_codec <set> <codec-name>
#          [--height 320] [--width 320] [--planes 4] [--patch 20] [--epochs 8]
#          [--batch-size 32] [--lr 3e-4] [--device auto] [--resume] [--limit 16384]
# veritate_mri/tools/fit_image_codec.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from readers import paths  # noqa: E402

from veritate_core.plugin import hardware, image_codec  # noqa: E402
from veritate_core.plugin.image_progress import StopRequested  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

IMAGE_EXTS     = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
RGB            = 3
RGB_MAX        = 255.0
VAL_EVERY      = 50            # 1 image in 50 is held out, matching build_image_corpus
HASH_CHARS     = 8             # of the content-addressed filename, for the stable split
LOG_EVERY      = 20
PROGRESS_EVERY = 100           # decoded frames between progress callbacks
DECODE_WORKERS = max(4, min(16, os.cpu_count() or 4))
FIT_SAMPLE_MAX = 16384         # pictures a codec is fitted on; 0 = the whole set
DISK_HEADROOM  = 0.85          # a cache may use this share of the free disk, no more
GB             = 1e9

# ------------------------------------------------------------------------------------
# Functions


def _set_images(set_dir, limit=0):
    names = sorted(n for n in os.listdir(set_dir)
                   if not n.startswith(".") and n.lower().endswith(IMAGE_EXTS))
    return names[:limit] if limit else names


def _decode(args):
    """One picture to a center-cropped HxWx3 uint8 array. Cover-scale then crop, so
    nothing is stretched and no edge is padded. JPEGs decode at the coarsest DCT scale
    that still covers the frame; orientation comes from EXIF."""
    path, height, width = args
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        src_w, src_h = handle.size
        scale = max(width / src_w, height / src_h)
        handle.draft("RGB", (max(width, int(src_w * scale + 0.5)),
                             max(height, int(src_h * scale + 0.5))))
        img = ImageOps.exif_transpose(handle).convert("RGB")
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        img = img.resize((max(width, int(src_w * scale + 0.5)),
                          max(height, int(src_h * scale + 0.5))))
        left, top = (img.size[0] - width) // 2, (img.size[1] - height) // 2
        img = img.crop((left, top, left + width, top + height))
        return np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(height, width, RGB)


def check_disk(directory, need_bytes):
    """Refuse a cache that would not fit. Returns the free bytes for the log line."""
    os.makedirs(directory, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if need_bytes > free * DISK_HEADROOM:
        raise ValueError(
            f"the decoded picture cache would need {need_bytes / GB:.1f} GB but only "
            f"{free / GB:.1f} GB is free on this disk; pick a smaller picture size or "
            f"fewer pictures")
    return free


def build_cache(set_name, height, width, limit=0, workers=DECODE_WORKERS, verbose=True,
                progress=None, should_stop=None):
    """Decode every picture in the set (or the first `limit`) once into a flat uint8
    memmap.

    Append-only: frames already in the cache are copied forward, never re-decoded, so
    adding photos to a set costs only the new photos. `progress(done, total)` is called
    every PROGRESS_EVERY frames; `should_stop()` true raises StopRequested and leaves
    the old cache intact. Returns (cache_path, names)."""
    set_dir = paths.image_set_dir(set_name)
    if not os.path.isdir(set_dir):
        raise ValueError("no image set at " + set_dir + " — run tools.ingest_images first")
    names = _set_images(set_dir, limit)
    if not names:
        raise ValueError("no images in " + set_dir)

    cache_path = paths.image_cache_path(set_name, height, width)
    index_path = paths.image_cache_index_path(set_name, height, width)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    frame_bytes = height * width * RGB

    have = []
    if os.path.isfile(index_path) and os.path.isfile(cache_path):
        with open(index_path, encoding="utf-8") as handle:
            have = json.load(handle).get("names") or []
    wanted = set(names)
    keep = [n for n in have if n in wanted]
    kept = set(keep)
    fresh = [n for n in names if n not in kept]
    order = keep + fresh
    if verbose:
        print(f"cache: {len(keep)} frames reused, {len(fresh)} to decode "
              f"({len(order)} x {height}x{width} = {len(order) * frame_bytes / GB:.2f} GB)",
              flush=True)
    if not fresh and len(keep) == len(have) == len(order):
        return cache_path, order
    check_disk(os.path.dirname(cache_path), len(order) * frame_bytes)

    tmp = cache_path + ".tmp"
    out = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(len(order), height, width, RGB))
    try:
        if keep:
            old = np.memmap(cache_path, dtype=np.uint8, mode="r",
                            shape=(len(have), height, width, RGB))
            row_of = {n: i for i, n in enumerate(have)}
            for i, name in enumerate(keep):
                out[i] = old[row_of[name]]
            del old
        started = time.perf_counter()
        if progress:
            progress(0, len(fresh))
        with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            jobs = ((os.path.join(set_dir, n), height, width) for n in fresh)
            for k, frame in enumerate(pool.map(_decode, jobs)):
                out[len(keep) + k] = frame
                if (k + 1) % PROGRESS_EVERY == 0 or k + 1 == len(fresh):
                    if progress:
                        progress(k + 1, len(fresh))
                    if should_stop and should_stop():
                        raise StopRequested("stopped while decoding pictures")
                if verbose and (k + 1) % 200 == 0:
                    rate = (k + 1) / (time.perf_counter() - started)
                    print(f"  decoded {k + 1}/{len(fresh)}  {rate:.0f} img/s", flush=True)
        out.flush()
    except BaseException:
        del out
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    del out
    os.replace(tmp, cache_path)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump({"set": set_name, "height": height, "width": width, "names": order},
                  handle, indent=1)
    return cache_path, order


def _split(names):
    """Val membership from the content hash in the filename, so a picture keeps its
    side of the split however the set later grows. Positional splits do not."""
    val = []
    for i, name in enumerate(names):
        try:
            val.append(int(name[:HASH_CHARS], 16) % VAL_EVERY == 0)
        except ValueError:
            val.append(i % VAL_EVERY == 0)
    mask = np.array(val, dtype=bool)
    if not mask.any():
        mask[0] = True
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def _to_device(frames, device):
    """uint8 HWC on the host -> float CHW in [0, 1] on the device. The cast happens
    after the copy, so the bus carries a quarter of the bytes."""
    batch = torch.from_numpy(np.array(frames)).to(device, non_blocking=True)
    return batch.permute(0, 3, 1, 2).float().div_(RGB_MAX)


def evaluate(codec, cache, rows, device, batch_size):
    """Held-out L1 and PSNR. No grad, no optimizer, pictures the fit never saw."""
    codec.eval()
    total_l1, total_mse, seen = 0.0, 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            idx = rows[start:start + batch_size]
            images = _to_device(cache[idx], device)
            recon, parts = codec(images)
            target = images.permute(0, 2, 3, 1)
            n = len(idx)
            total_l1 += float(parts["recon"]) * n
            total_mse += float(torch.mean((recon - target) ** 2)) * n
            seen += n
    codec.train()
    mse = total_mse / max(1, seen)
    psnr = float("inf") if mse <= 0 else 10.0 * float(np.log10(1.0 / mse))
    return {"l1": total_l1 / max(1, seen), "psnr": psnr, "images": seen}


def fit(set_name, codec_name, height=320, width=320, planes=4, patch=20, latent_dim=32,
        dec_hidden=64, epochs=8, batch_size=32, lr=3e-4, device="auto", resume=False,
        limit=FIT_SAMPLE_MAX, seed=0, verbose=True, progress=None, should_stop=None,
        out_scale=image_codec.DEFAULT_OUT_SCALE):
    """Fit the codec on (a sample of) a set and save it. Returns the report.

    `progress(stage, done, total, **facts)` is called with stage "decode" while the
    cache fills and "codec" during the epochs (facts: epoch, epochs, loss, recon).
    `out_scale` > 1 caches and scores the pictures at out_scale x the frame: the encoder
    reads them pooled down to the frame, the decoder learns to paint the full size."""
    out_scale = int(out_scale)
    out_h, out_w = height * out_scale, width * out_scale
    decode_cb = (lambda d, t: progress("decode", d, t)) if progress else None
    cache_path, names = build_cache(set_name, out_h, out_w, limit=limit, verbose=verbose,
                                    progress=decode_cb, should_stop=should_stop)
    cache = np.memmap(cache_path, dtype=np.uint8, mode="r",
                      shape=(len(names), out_h, out_w, RGB))
    train_rows, val_rows = _split(names)

    dev = hardware.pick_device(device)
    out_path = paths.codec_path(codec_name)
    if resume and os.path.isfile(out_path):
        codec = image_codec.load(out_path)
        if codec.patch != patch or codec.planes != planes or codec.out_scale != out_scale:
            raise ValueError("resumed codec geometry differs from the flags; a corpus "
                             "built under one codec is unreadable by another")
    else:
        codec = image_codec.ImageCodec(planes=planes, latent_dim=latent_dim, patch=patch,
                                       dec_hidden=dec_hidden, out_scale=out_scale)
    codec = codec.to(dev)
    codec.train()
    opt = torch.optim.AdamW(codec.parameters(), lr=lr)
    code_bytes = codec.code_bytes(height, width)
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"device: {dev}   images: {len(names)} "
              f"({len(train_rows)} train / {len(val_rows)} val)", flush=True)
        print(f"geometry: {height}x{width}  patch {patch}  planes {planes}  "
              f"-> {code_bytes} code bytes/image "
              f"({height * width * RGB / code_bytes:.0f}x compression)"
              + (f"  decoded at {out_h}x{out_w} ({out_scale}x)" if out_scale > 1 else ""), flush=True)

    steps_per_epoch = max(1, len(train_rows) // batch_size)
    total_images = epochs * steps_per_epoch * batch_size
    history, started, images_seen = [], time.perf_counter(), 0
    for epoch in range(epochs):
        order = rng.permutation(train_rows)
        for step in range(steps_per_epoch):
            idx = np.sort(order[step * batch_size:(step + 1) * batch_size])
            if not len(idx):
                continue
            losses = codec.fit_step(_to_device(cache[idx], dev), opt)
            images_seen += len(idx)
            if (step + 1) % LOG_EVERY == 0 or step + 1 == steps_per_epoch:
                rate = images_seen / (time.perf_counter() - started)
                if verbose:
                    print(f"  epoch {epoch + 1}/{epochs} step {step + 1}/{steps_per_epoch}  "
                          f"loss {losses['loss']:.4f}  recon {losses['recon']:.4f}  "
                          f"{rate:.1f} img/s", flush=True)
                if progress:
                    progress("codec", images_seen, total_images, epoch=epoch + 1, epochs=epochs,
                             loss=round(float(losses["loss"]), 5),
                             recon=round(float(losses["recon"]), 5), img_per_s=round(rate, 1))
                if should_stop and should_stop():
                    raise StopRequested("stopped while fitting the codec")
        val = evaluate(codec, cache, val_rows, dev, batch_size)
        history.append({"epoch": epoch + 1, **val})
        if verbose:
            print(f"epoch {epoch + 1}: held-out L1 {val['l1']:.4f}  "
                  f"PSNR {val['psnr']:.2f} dB  ({val['images']} images)", flush=True)
        if progress:
            progress("codec", images_seen, total_images, epoch=epoch + 1, epochs=epochs,
                     psnr=round(val["psnr"], 2), l1=round(val["l1"], 5))
        image_codec.save(codec.to("cpu"), out_path)
        codec = codec.to(dev)

    elapsed = time.perf_counter() - started
    return {"codec": codec_name, "path": out_path, "set": set_name,
            "images": len(names), "train": len(train_rows), "val": len(val_rows),
            "height": height, "width": width, "patch": patch, "planes": planes,
            "out_scale": out_scale, "out_height": out_h, "out_width": out_w,
            "image_code_bytes": code_bytes, "device": dev, "epochs": epochs,
            "seconds": round(elapsed, 1), "images_per_s": round(images_seen / elapsed, 1),
            "history": history}


def main():
    ap = argparse.ArgumentParser(description="Fit the image codec on a set of pictures.")
    ap.add_argument("set_name")
    ap.add_argument("codec_name")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--planes", type=int, default=image_codec.DEFAULT_PLANES)
    ap.add_argument("--patch", type=int, default=image_codec.DEFAULT_PATCH)
    ap.add_argument("--latent-dim", type=int, default=image_codec.DEFAULT_LATENT_DIM)
    ap.add_argument("--dec-hidden", type=int, default=image_codec.DEFAULT_DEC_HIDDEN)
    ap.add_argument("--out-scale", type=int, default=image_codec.DEFAULT_OUT_SCALE,
                    choices=image_codec.OUT_SCALES,
                    help="decode at this many x the frame (the decoder learns the extra pixels)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="continue an existing codec")
    ap.add_argument("--limit", type=int, default=FIT_SAMPLE_MAX,
                    help="pictures to fit on (a hash-ordered sample); 0 = the whole set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-only", action="store_true",
                    help="decode into the cache and stop, fitting nothing")
    args = ap.parse_args()

    if args.cache_only:
        path, names = build_cache(args.set_name, args.height, args.width, limit=args.limit)
        print(f"{len(names)} frames cached: {path}")
        return 0

    rep = fit(args.set_name, args.codec_name, height=args.height, width=args.width,
              planes=args.planes, patch=args.patch, latent_dim=args.latent_dim,
              dec_hidden=args.dec_hidden, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr, device=args.device, resume=args.resume, limit=args.limit,
              seed=args.seed, out_scale=args.out_scale)
    best = min(rep["history"], key=lambda h: h["l1"]) if rep["history"] else {}
    print(f"saved {rep['path']}")
    print(f"{rep['images']} images, {rep['epochs']} epochs in {rep['seconds']}s "
          f"({rep['images_per_s']} img/s on {rep['device']})")
    print(f"best held-out L1 {best.get('l1', float('nan')):.4f}  "
          f"PSNR {best.get('psnr', float('nan')):.2f} dB")
    print(f"image_code_bytes: {rep['image_code_bytes']}  "
          f"(the trainer needs this, and seq >= it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
