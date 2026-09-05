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
    m = image_probe.dump(model, codec, g, "pix", 10, val, "cpu", train_path=val)
    d = os.path.join(paths.hook_step_dir("pix", 10), image_probe.IMAGE_DIR)
    for f in image_probe.FILES:
        assert os.path.isfile(os.path.join(d, f)), f
    assert len(m["fill_accuracy_per_plane"]) == PLANES
    assert 0.0 <= m["fill_accuracy"] <= 1.0
    assert set(m["loss_by_hidden_fraction"]) == {"0.25", "0.5", "0.75", "1.0"}
    assert len(m["attention_entropy_per_layer"]) == LAYERS
    assert all(0.0 <= e <= 1.0 + 1e-6 for e in m["attention_entropy_per_layer"])
    assert len(m["attention_entropy_per_head"]) == LAYERS and len(m["attention_entropy_per_head"][0]) == 2
    assert 1 <= m["codes_used"] <= image_codec.CODEBOOK_ENTRIES
    # through the layers: one picture per block, agreement with the final layer is 1 at the top
    assert len(m["lens_agreement_per_layer"]) == LAYERS == len(m["residual_norm_per_layer"])
    assert abs(m["lens_agreement_per_layer"][-1] - 1.0) < 1e-6
    assert m["commit_layer"] in (None, *range(1, LAYERS + 1))
    layers = Image.open(os.path.join(d, "layers.png"))
    assert layers.size[0] == LAYERS * image_probe.THUMB + (LAYERS + 1) * image_probe.GAP
    # how a picture forms: one tile per decode pass, committed counts sum to the whole image
    assert sum(m["pass_committed"]) == CODE_BYTES
    passes = Image.open(os.path.join(d, "passes.png"))
    n_pass = len(m["pass_committed"])
    assert passes.size[0] == n_pass * image_probe.THUMB + (n_pass + 1) * image_probe.GAP
    # how sure, and is it earned
    assert 0.0 <= m["mean_confidence"] <= 1.0 and 0.0 <= m["expected_calibration_error"] <= 1.0
    assert len(m["calibration"]) == len(image_probe.CALIBRATION_BINS) - 1
    assert sum(b["n"] for b in m["calibration"]) > 0
    # copying or inventing: a nearest training picture per sample, novelty in [0, 1]
    assert len(m["novelty_per_sample"]) == image_probe.N_SAMPLES
    assert all(0.0 <= v <= 1.0 for v in m["novelty_per_sample"])
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


def test_a_sample_that_copies_a_training_picture_has_zero_novelty(rig):
    """The novelty gauge: a sample identical to a training record differs in no cell."""
    _model, _codec, _g, val = rig
    arr = np.memmap(val, dtype=np.uint8, mode="r")
    ends = image_grid.code_block_ends(val)
    copy = np.asarray(arr[int(ends[0]) - CODE_BYTES:int(ends[0])])
    fresh = np.full(CODE_BYTES, 7, dtype=np.uint8)
    near = image_probe.novelty_probe(val, [copy, fresh], CODE_BYTES, 0)
    assert near[0]["novelty"] == 0.0
    assert np.array_equal(near[0]["nearest"], copy)
    assert 0.0 < near[1]["novelty"] <= 1.0


def test_the_probe_records_how_the_picture_formed_and_what_formed_first(rig):
    """Formation order (which pass decided each cell), the coarse-to-fine render, how far
    attention reaches, and the detail / colour gauges the 'what forms first' chart plots."""
    model, codec, g, val = rig
    m = image_probe.dump(model, codec, g, "pix", 7, val, "cpu", train_path=val)
    d = os.path.join(paths.hook_step_dir("pix", 7), image_probe.IMAGE_DIR)
    gh, gw = H // PATCH, W // PATCH
    n_pass = m["formation_passes"]
    assert n_pass == len(m["pass_committed"])
    assert len(m["commit_pass_map"]) == gh * gw
    assert all(1 <= p <= n_pass for p in m["commit_pass_map"])
    assert len(m["commit_pass_per_plane"]) == PLANES
    assert all(1.0 <= p <= n_pass for p in m["commit_pass_per_plane"])
    assert Image.open(os.path.join(d, "formation.png")).size == (2 * image_probe.THUMB, 2 * image_probe.THUMB)
    planes = Image.open(os.path.join(d, "planes.png"))
    assert planes.size[0] == PLANES * image_probe.THUMB + (PLANES + 1) * image_probe.GAP
    assert len(m["attention_distance_per_layer"]) == LAYERS
    assert all(0.0 <= v <= ((gh ** 2 + gw ** 2) ** 0.5) for v in m["attention_distance_per_layer"])
    assert m["sample_sharpness"] >= 0.0 and m["heldout_sharpness"] >= 0.0
    assert m["detail_ratio"] is None or m["detail_ratio"] >= 0.0
    assert 0.0 <= m["colour_match"] <= 1.0


def test_the_probe_paints_hidden_cells_at_the_codecs_output_scale(rig):
    """A 2x codec decodes 2x frames; the grey cells and the tiles must still line up."""
    model, _codec, g, val = rig
    torch.manual_seed(0)
    big = image_codec.ImageCodec(planes=PLANES, latent_dim=8, patch=PATCH, dec_hidden=32, out_scale=2).eval()
    m = image_probe.dump(model, big, g, "pix2", 3, val, "cpu", train_path=val)
    d = os.path.join(paths.hook_step_dir("pix2", 3), image_probe.IMAGE_DIR)
    assert m["fill_accuracy"] >= 0.0
    assert Image.open(os.path.join(d, "fill.png")).size[0] == 3 * image_probe.THUMB + 4 * image_probe.GAP
    assert m["thumb"] == image_probe.THUMB and m["gap"] == image_probe.GAP


def test_formation_order_reads_the_pass_each_cell_was_decided_in():
    trace = [{"pass": 1, "unknown": np.array([True, False, True, True])},
             {"pass": 2, "unknown": np.array([True, False, False, True])},
             {"pass": 3, "unknown": np.array([False, False, False, False])}]
    assert image_probe._formation_order(trace, 4).tolist() == [3, 1, 2, 3]


def test_the_probe_stores_the_sample_codes_and_maps_the_tab_derives_churn_from(rig):
    import base64
    model, codec, g, val = rig
    m = image_probe.dump(model, codec, g, "pix", 5, val, "cpu", train_path=val)
    codes = np.frombuffer(base64.b64decode(m["sample_codes_b64"]), dtype=np.uint8)
    assert codes.size == image_probe.N_SAMPLES * CODE_BYTES
    assert m["grid"] == [H // PATCH, W // PATCH]
    assert len(m["loss_map"]) == len(m["confidence_map"]) == (H // PATCH) * (W // PATCH)
