# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The image probe at random weights: every file lands, the metrics carry the numbers
#   the Models tab plots, per-plane accuracy has one entry per plane, attention maps
#   come out per layer, and read() returns steps oldest first.
# tests/plugin_contract/test_image_probe.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import numpy as np
import pytest
import torch
from PIL import Image
from readers import paths

from veritate_core.model import Veritate
from veritate_core.plugin import image_codec, image_grid, image_probe

# ------------------------------------------------------------------------------------
# Constants

H = W = 40
PATCH, PLANES, SEQ, LAYERS = 20, 2, 64, 2
CODE_BYTES = PLANES * (H // PATCH) * (W // PATCH)

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path / "models"))
    torch.manual_seed(0)
    model = Veritate(256, 32, LAYERS, 64, 2, SEQ, causal=False).eval()
    codec = image_codec.ImageCodec(planes=PLANES, latent_dim=8, patch=PATCH, dec_hidden=32).eval()
    rng = np.random.RandomState(0)
    val = tmp_path / "v_val.bin"
    with open(val, "wb") as handle:
        for i in range(6):
            codes = bytes(rng.randint(0, image_codec.MASK_BYTE, size=CODE_BYTES, dtype=np.uint8).tolist())
            handle.write(b"cap %d" % i + codes + image_grid.RECORD_SEP)
    geometry = {"height": H, "width": W, "seq": SEQ, "code_bytes": CODE_BYTES}
    return model, codec, geometry, str(val)


def test_dump_writes_every_file_and_the_numbers_the_tab_plots(rig):
    model, codec, g, val = rig
    m = image_probe.dump(model, codec, g, "pix", 10, val, "cpu")
    d = os.path.join(paths.hook_step_dir("pix", 10), image_probe.IMAGE_DIR)
    for f in image_probe.FILES:
        assert os.path.isfile(os.path.join(d, f)), f
    assert len(m["fill_accuracy_per_plane"]) == PLANES
    assert 0.0 <= m["fill_accuracy"] <= 1.0
    assert set(m["loss_by_hidden_fraction"]) == {"0.25", "0.5", "0.75", "1.0"}
    assert len(m["attention_entropy_per_layer"]) == LAYERS
    assert all(0.0 <= e <= 1.0 + 1e-6 for e in m["attention_entropy_per_layer"])
    assert 1 <= m["codes_used"] <= image_codec.CODEBOOK_ENTRIES
    samples = Image.open(os.path.join(d, "samples.png"))
    assert samples.size[0] > samples.size[1]          # one row of N_SAMPLES (+ caption samples)
    fill = Image.open(os.path.join(d, "fill.png"))
    assert fill.size[0] == 3 * image_probe.THUMB + 4 * image_probe.GAP   # original | masked | filled


def test_read_returns_steps_oldest_first_with_their_files(rig):
    model, codec, g, val = rig
    for step in (30, 10, 20):
        image_probe.dump(model, codec, g, "pix", step, val, "cpu")
    rows = image_probe.read("pix")
    assert [r["step"] for r in rows] == [10, 20, 30]
    assert "samples.png" in rows[0]["files"]


def test_dump_without_a_val_bin_still_writes_samples(rig):
    model, codec, g, _val = rig
    m = image_probe.dump(model, codec, g, "pix", 5, None, "cpu")
    d = os.path.join(paths.hook_step_dir("pix", 5), image_probe.IMAGE_DIR)
    assert os.path.isfile(os.path.join(d, "samples.png"))
    assert not os.path.isfile(os.path.join(d, "fill.png"))
    assert m["val_records"] == 0
