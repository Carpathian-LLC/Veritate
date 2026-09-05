# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The image <-> bytes contract. Pins: an encoded image is exactly one byte per code,
#   the byte string round-trips, the mask byte is never a real code (the masked
#   objective needs one value that cannot appear in data), and fitting actually
#   reduces reconstruction error.
# - Tiny geometry throughout; no image files, no Pillow, no disk beyond tmp_path.
# tests/plugin_contract/test_image_codec.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
import torch

from veritate_core.plugin import image_codec

# ------------------------------------------------------------------------------------
# Constants

H, W  = 40, 60
PATCH = 20
KW    = {"planes": 2, "latent_dim": 8, "patch": PATCH, "dec_hidden": 32}

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def codec():
    torch.manual_seed(0)
    return image_codec.ImageCodec(**KW)


def test_one_code_is_exactly_one_byte(codec):
    """Grid cells times planes is the byte length, so a 256-vocab model reads it raw."""
    assert codec.code_bytes(H, W) == KW["planes"] * (H // PATCH) * (W // PATCH)
    codes = codec.encode(torch.rand(1, 3, H, W))
    assert len(codec.to_bytes(codes[0])) == codec.code_bytes(H, W)


def test_a_prefix_of_the_planes_decodes_to_a_coarser_picture(codec):
    """Residual planes are plane-major: rendering the first k is the picture the higher
    planes refine, so a probe can show what each plane adds."""
    codes = codec.encode(torch.rand(1, 3, H, W))[0]
    coarse = codec.decode(codes, planes=1)
    full = codec.decode(codes)
    assert coarse.shape == full.shape == (H, W, 3) and coarse.dtype == torch.uint8
    assert not torch.equal(coarse, full)
    assert torch.equal(codec.decode(codes, planes=KW["planes"]), full)
    assert torch.equal(codec.decode(codes, planes=99), full)          # clamps to what exists


def test_an_output_scale_paints_more_pixels_per_cell_from_the_same_bytes(tmp_path):
    """out_scale=2: the encoder still reads the frame at `patch` px per cell (same code
    bytes, same sequence), the decoder renders 2x the frame; the scale survives save/load."""
    torch.manual_seed(0)
    big = image_codec.ImageCodec(**KW, out_scale=2)
    assert big.output_size(H, W) == (2 * H, 2 * W)
    assert big.code_bytes(H, W) == KW["planes"] * (H // PATCH) * (W // PATCH)
    frames = torch.rand(2, 3, 2 * H, 2 * W)                    # pictures at the output size
    recon, parts = big(frames)
    assert recon.shape == (2, 2 * H, 2 * W, 3)
    assert parts["codes"].shape == (2, KW["planes"], H // PATCH, W // PATCH)
    codes = big.encode(torch.rand(1, 3, H, W))[0]              # the frame the model reasons about
    assert big.decode(codes).shape == (2 * H, 2 * W, 3)
    path = str(tmp_path / "big.codec.pt")
    image_codec.save(big, path)
    again = image_codec.load(path)
    assert again.out_scale == 2 and again.decode(codes).shape == (2 * H, 2 * W, 3)
    assert image_codec.load(path).config["out_scale"] == 2
    with pytest.raises(ValueError, match="out_scale"):
        image_codec.ImageCodec(**KW, out_scale=5)


def test_the_byte_string_round_trips(codec):
    """Bytes out of a corpus rebuild the exact codes that were encoded."""
    codes = codec.encode(torch.rand(1, 3, H, W))[0]
    assert torch.equal(codec.from_bytes(codec.to_bytes(codes), H, W), codes)


def test_no_code_can_collide_with_the_mask_byte(codec):
    """The masked objective needs a value that never appears in data."""
    codes = codec.encode(torch.rand(8, 3, H, W))
    assert int(codes.max()) < image_codec.MASK_BYTE


def test_decoding_returns_a_uint8_frame(codec):
    """Decode produces displayable bytes, not a float feature map."""
    frame = codec.decode(codec.encode(torch.rand(1, 3, H, W))[0])
    assert frame.shape == (H, W, 3)
    assert frame.dtype == torch.uint8


def test_fitting_reduces_reconstruction_error(codec):
    """A codec that cannot fit a smooth image cannot encode a real one."""
    target = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(2, 3, H, W).contiguous()
    opt = torch.optim.AdamW(codec.parameters(), lr=5e-3)
    first = codec.fit_step(target, opt)["recon"]
    for _ in range(60):
        last = codec.fit_step(target, opt)["recon"]
    assert last < first * 0.5


def test_a_saved_codec_reloads_to_the_same_codes(codec, tmp_path):
    """A corpus is unreadable without the codec that wrote it, so reload must be exact."""
    images = torch.rand(2, 3, H, W)
    before = codec.encode(images)
    path = image_codec.save(codec, str(tmp_path / "t.codec.pt"))
    assert torch.equal(image_codec.load(path).encode(images), before)


def test_a_frame_the_patch_grid_cannot_tile_is_refused(codec):
    """Geometry is a caller error, not a silently cropped image."""
    with pytest.raises(ValueError, match="does not divide"):
        codec.code_bytes(H + 1, W)
