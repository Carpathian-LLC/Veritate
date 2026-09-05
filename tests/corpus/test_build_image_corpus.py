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
