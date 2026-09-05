# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - What the Training tab needs to know about pictures on this machine: the named
#   image sets under data/images/ and the fitted codecs under data/codecs/. Read-only,
#   no torch: the dashboard polls discovery every few seconds and a codec's config is
#   not worth a torch.load on each poll.
# - Counting is a directory listing, so a set of 100k photos costs a listdir, not a
#   walk. Sets are flat by construction (ingest_images writes them flat).
# veritate_mri/readers/images.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from . import paths

# ------------------------------------------------------------------------------------
# Constants

# Kept in step with tools/ingest_images.IMAGE_EXTS and build_image_corpus.IMAGE_EXTS.
IMAGE_EXTS     = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CAPTION_SUFFIX = ".txt"

# ------------------------------------------------------------------------------------
# Functions


def list_sets():
    """[{name, images, captions}] for every set under data/images/, name-sorted."""
    root = paths.IMAGES_ROOT
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        set_dir = os.path.join(root, name)
        if name.startswith(".") or not os.path.isdir(set_dir):
            continue
        images = captions = 0
        for entry in os.listdir(set_dir):
            if entry.startswith("."):
                continue
            low = entry.lower()
            if low.endswith(IMAGE_EXTS):
                images += 1
            elif low.endswith(CAPTION_SUFFIX):
                captions += 1
        out.append({"name": name, "images": images, "captions": captions})
    return out


def list_codecs():
    """[{name}] for every fitted codec under data/codecs/, name-sorted."""
    root = paths.CODEC_ROOT
    if not os.path.isdir(root):
        return []
    suffix = paths.CODEC_SUFFIX
    return [{"name": f[:-len(suffix)]} for f in sorted(os.listdir(root))
            if f.endswith(suffix) and not f.startswith(".")]
