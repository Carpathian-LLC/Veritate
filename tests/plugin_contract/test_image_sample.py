# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The generation contract at random weights: the window is laid out as training saw
#   it, the mask byte never reaches a picture, kept cells survive a fill, a seed
#   replays, and every mode ends in a PNG at the model's frame.
# tests/plugin_contract/test_image_sample.py
# ------------------------------------------------------------------------------------
# Imports:

import io

import numpy as np
import pytest
import torch
from PIL import Image

from veritate_core.model import Veritate
from veritate_core.plugin import image_codec, image_grid, image_sample

# ------------------------------------------------------------------------------------
# Constants

H = W = 40
PATCH, PLANES = 20, 2
CODE_BYTES = PLANES * (H // PATCH) * (W // PATCH)   # 8
SEQ = 64

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def rig():
    torch.manual_seed(0)
    model = Veritate(256, 32, 1, 64, 2, SEQ, causal=False).eval()
    codec = image_codec.ImageCodec(planes=PLANES, latent_dim=8, patch=PATCH, dec_hidden=32).eval()
    geometry = {"height": H, "width": W, "seq": SEQ, "code_bytes": CODE_BYTES, "patch": PATCH, "planes": PLANES}
    return model, codec, geometry


def _png(color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (90, 60), color).save(buf, format="PNG")
    return buf.getvalue()


def test_the_window_is_laid_out_as_training_saw_it():
    """pad, separator, caption, then the image slice, all masked when there are no codes."""
    win = image_sample.build_window(SEQ, CODE_BYTES, caption=b"a cat")
    first = SEQ - CODE_BYTES
    prefix = image_grid.RECORD_SEP + b"a cat"
    assert bytes(win[first - len(prefix):first].tolist()) == prefix
    assert (win[:first - len(prefix)] == image_grid.PAD_BYTE).all()
    assert (win[first:] == image_codec.MASK_BYTE).all()


def test_kept_codes_stay_and_the_rest_are_masked():
    codes = np.arange(CODE_BYTES, dtype=np.uint8)
    keep = np.array([True, False] * (CODE_BYTES // 2))
    win = image_sample.build_window(SEQ, CODE_BYTES, codes=codes, keep=keep)
    image = win[SEQ - CODE_BYTES:]
    assert (image[keep] == codes[keep]).all()
    assert (image[~keep] == image_codec.MASK_BYTE).all()


def test_fill_never_emits_the_mask_byte_and_holds_kept_cells(rig):
    model, _codec, _g = rig
    codes = np.full(CODE_BYTES, 7, dtype=np.uint8)
    keep = np.zeros(CODE_BYTES, dtype=bool)
    keep[:2] = True
    win = image_sample.build_window(SEQ, CODE_BYTES, b"x", codes, keep)
    out = image_sample.fill(model, win, SEQ - CODE_BYTES, keep, passes=3, seed=1)
    assert out.shape == (CODE_BYTES,) and out.dtype == np.uint8
    assert int(out.max()) < image_codec.MASK_BYTE
    assert (out[:2] == 7).all()


def test_a_seed_replays_and_one_pass_completes(rig):
    model, _codec, _g = rig
    win = image_sample.build_window(SEQ, CODE_BYTES)
    a = image_sample.fill(model, win, SEQ - CODE_BYTES, passes=4, seed=5)
    b = image_sample.fill(model, win, SEQ - CODE_BYTES, passes=4, seed=5)
    assert np.array_equal(a, b)
    one = image_sample.fill(model, win, SEQ - CODE_BYTES, passes=1, seed=5)
    assert int(one.max()) < image_codec.MASK_BYTE


def test_greedy_decode_is_deterministic_without_a_seed(rig):
    model, _codec, _g = rig
    win = image_sample.build_window(SEQ, CODE_BYTES)
    a = image_sample.fill(model, win, SEQ - CODE_BYTES, passes=2, temperature=0.0, seed=1)
    b = image_sample.fill(model, win, SEQ - CODE_BYTES, passes=2, temperature=0.0, seed=99)
    assert np.array_equal(a, b)


def test_cell_masks_cover_every_plane():
    cells = image_sample.rect_cells(2, 2, (0.0, 0.0, 0.5, 0.5))
    assert cells.tolist() == [[True, False], [False, False]]
    pos = image_sample.cells_to_positions(cells, PLANES)
    assert pos.shape == (CODE_BYTES,) and pos.sum() == PLANES
    assert image_sample.inner_cells(4, 4, 0.5).sum() == 4
    assert image_sample.inner_cells(4, 4, 1.0).all()


@pytest.mark.parametrize("mode", image_sample.MODES)
def test_every_mode_ends_in_a_png_at_the_frame(rig, mode):
    model, codec, g = rig
    png, info = image_sample.generate(model, codec, g, mode=mode, caption=b"a red wall",
                                      source=_png() if mode in image_sample.SOURCE_MODES else None,
                                      rect=(0.0, 0.0, 0.5, 1.0), passes=2, seed=0)
    image = Image.open(io.BytesIO(png))
    assert image.size == (W, H)
    assert info["mode"] == mode
    assert 0 < info["regenerated"] <= CODE_BYTES
    if mode == "unconditional":
        assert info["caption_bytes"] == 0


def test_source_modes_refuse_to_run_without_a_source(rig):
    model, codec, g = rig
    with pytest.raises(ValueError, match="needs a source"):
        image_sample.generate(model, codec, g, mode="variation")
