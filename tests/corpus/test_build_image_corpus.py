# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The record layout the masked objective depends on: caption, then a fixed-length
#   code block, then the documented separator. Because the block length is fixed, the
#   image is the last image_code_bytes bytes before each separator and no marker that
#   image bytes could collide with is needed inside a record.
# - Packing only. Reading image FILES needs Pillow; that boundary is not exercised here.
# tests/corpus/test_build_image_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import torch
from tools import build_image_corpus

from veritate_core.plugin import image_codec

# ------------------------------------------------------------------------------------
# Constants

H, W    = 40, 60
CAPTION = b"a red square"

# ------------------------------------------------------------------------------------
# Functions


def _codec():
    torch.manual_seed(0)
    return image_codec.ImageCodec(planes=2, latent_dim=8, patch=20, dec_hidden=32)


def test_a_record_is_caption_then_a_fixed_code_block_then_the_separator():
    """Record layout is what lets the trainer locate an image without a sidecar."""
    codec = _codec()
    record = build_image_corpus.pack_record(codec, torch.rand(3, H, W), CAPTION)
    assert record.endswith(build_image_corpus.RECORD_SEP)
    assert len(record) == len(CAPTION) + codec.code_bytes(H, W) + len(build_image_corpus.RECORD_SEP)


def test_the_code_block_recovers_the_encoded_image():
    """The last image_code_bytes before the separator must rebuild the codes exactly."""
    codec = _codec()
    image = torch.rand(3, H, W)
    record = build_image_corpus.pack_record(codec, image, CAPTION)
    n = codec.code_bytes(H, W)
    block = record[-(n + len(build_image_corpus.RECORD_SEP)):-len(build_image_corpus.RECORD_SEP)]
    with torch.no_grad():
        assert torch.equal(codec.from_bytes(block, H, W), codec.encode(image.unsqueeze(0))[0])


def test_a_record_without_a_caption_is_just_the_image():
    """Unconditional corpora are the same layout with an empty prefix."""
    codec = _codec()
    record = build_image_corpus.pack_record(codec, torch.rand(3, H, W))
    assert len(record) == codec.code_bytes(H, W) + len(build_image_corpus.RECORD_SEP)


def test_the_streaming_build_reads_cached_frames_and_decodes_the_rest(tmp_path):
    """The trainer's path: the codec was fitted on a sample (in the cache); the corpus
    must hold every picture, with cached and freshly decoded frames encoding alike."""
    import json
    import types

    import numpy as np
    from PIL import Image
    from tools import fit_image_codec

    images = tmp_path / "images" / "set"
    images.mkdir(parents=True)
    names = [f"{i:08x}_p.png" for i in range(5)]
    for i, n in enumerate(names):
        Image.new("RGB", (H * 2, W * 2), (40 * i, 10, 200 - 30 * i)).save(str(images / n))
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fake = types.SimpleNamespace(
        image_set_dir=lambda name: str(tmp_path / "images" / name),
        image_cache_path=lambda name, h, w: str(cache_dir / f"{name}_{h}x{w}.u8"),
        image_cache_index_path=lambda name, h, w: str(cache_dir / f"{name}_{h}x{w}.index.json"),
        codec_path=lambda name: str(tmp_path / f"{name}.codec.pt"),
        corpus_dir=lambda: str(tmp_path / "corpus"),
    )
    codec = _codec()
    image_codec.save(codec, fake.codec_path("c"))
    # cache holds the first two pictures only
    frames = np.stack([fit_image_codec._decode((str(images / n), H, W)) for n in names[:2]])
    mm = np.memmap(fake.image_cache_path("set", H, W), dtype=np.uint8, mode="w+", shape=frames.shape)
    mm[:] = frames
    mm.flush()
    del mm
    with open(fake.image_cache_index_path("set", H, W), "w", encoding="utf-8") as handle:
        json.dump({"names": names[:2]}, handle)

    seen = []
    rep = build_image_corpus.build_streaming(fake, "set", "c", "img", H, W, val_every=2,
                                             device="cpu", verbose=False,
                                             progress=lambda d, t: seen.append((d, t)))
    assert rep["images"] == 5 and rep["from_cache"] == 2 and rep["decoded"] == 3
    assert rep["train_records"] + rep["val_records"] == 5
    assert seen[-1] == (5, 5)
    data = (tmp_path / "corpus" / "img_train.bin").read_bytes() + (tmp_path / "corpus" / "img_val.bin").read_bytes()
    assert data.count(build_image_corpus.RECORD_SEP) == 5
    # a freshly decoded picture encodes exactly as it would have from the cache
    frame = fit_image_codec._decode((str(images / names[4]), H, W))
    image = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float().div(255.0)
    assert build_image_corpus.pack_record(codec, image) in data

    # with an output scale the cache holds the pictures at 2x, so every picture is decoded
    # at the frame instead: the same corpus, none of it from the cache
    rep2 = build_image_corpus.build_streaming(fake, "set", "c", "img2", H, W, val_every=2,
                                              device="cpu", verbose=False, out_scale=2)
    assert rep2["images"] == 5 and rep2["from_cache"] == 0 and rep2["decoded"] == 5
    train2 = (tmp_path / "corpus" / "img2_train.bin").read_bytes()
    assert train2.count(build_image_corpus.RECORD_SEP) == rep2["train_records"]
