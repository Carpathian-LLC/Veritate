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
# - Pillow is imported inside the loader so the packing path stays importable, and
#   testable, on an install that has not got it.
# veritate_mri/tools/build_image_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import torch

from veritate_core.plugin import image_codec

# ------------------------------------------------------------------------------------
# Constants

RECORD_SEP     = b"<|endoftext|>"
IMAGE_EXTS     = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CAPTION_SUFFIX = ".txt"
VAL_EVERY      = 50
RGB_MAX        = 255.0

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
