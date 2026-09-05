# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The canonical image trainer, end to end and in-process: a set of pictures goes in,
#   and out come a fitted codec, a corpus with its sidecar, config.json stamped image,
#   train.csv rows and a checkpoint. Every root is redirected into tmp_path, so the
#   real data/ and models/ trees are never touched.
# - The guard rails are pinned separately: geometry that does not tile, a seq smaller
#   than the image, a corpus that is current is not rebuilt.
# tests/training/test_image_trainer.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

import pytest
from PIL import Image
from readers import paths
from readers import trainers as trainers_reader
from training import image_trainer

from veritate_core.plugin import hardware

# ------------------------------------------------------------------------------------
# Constants

H = W = 40
TINY = {"layers": 1, "hidden": 32, "ffn": 64, "heads": 2, "params": 10000}
ARGV = ["image_trainer.py", "--name", "smoke", "--description", "image trainer test",
        "--image_set", "set", "--size", "tiny", "--height", str(H), "--width", str(W),
        "--planes", "2", "--patch", "20", "--caption_bytes", "24", "--seq", "0",
        "--total_steps", "4", "--batch_size", "2", "--codec_epochs", "1",
        "--codec_batch_size", "4", "--ckpt_every", "2", "--eval_every", "2",
        "--eval_iters", "1", "--log_every", "1", "--precision", "fp32",
        "--optimizer", "adamw", "--warmup_steps", "1"]

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Every on-disk root under tmp_path, a tiny size, and the CPU."""
    for root in ("IMAGES_ROOT", "IMAGE_CACHE_ROOT", "CODEC_ROOT", "CORPUS_ROOT", "MODELS_ROOT"):
        monkeypatch.setattr(paths, root, str(tmp_path / root.lower()))
    manifest = dict(trainers_reader.IMAGE_TRAINER_MANIFEST)
    manifest["sizes"] = {"tiny": TINY}
    manifest["defaults"] = dict(manifest["defaults"], size="tiny")
    monkeypatch.setattr(trainers_reader, "IMAGE_TRAINER_MANIFEST", manifest)
    monkeypatch.setenv(hardware.DEVICE_ENV, "cpu")
    set_dir = os.path.join(paths.IMAGES_ROOT, "set")
    os.makedirs(set_dir)
    for i in range(24):
        Image.new("RGB", (H * 2, H * 2), (10 * i, 255 - 10 * i, 80)).save(
            os.path.join(set_dir, f"{i:08x}_pic.png"))
    return tmp_path


def _run(monkeypatch, argv=ARGV):
    monkeypatch.setattr(sys, "argv", list(argv))
    image_trainer.run(plugin_id=trainers_reader.IMAGE_TRAINER_ID)


def test_one_launch_produces_codec_corpus_config_and_checkpoint(home, monkeypatch):
    """The whole point: pictures in, a trained image model out, nothing else to run."""
    _run(monkeypatch)
    name = "smoke_tiny"
    assert os.path.isfile(paths.codec_path(name + "_codec"))
    assert os.path.isfile(paths.corpus_train_path(name + "_img"))
    assert os.path.isfile(paths.image_corpus_meta_path(name + "_img"))
    with open(paths.config_path(name), encoding="utf-8") as handle:
        cfg = json.load(handle)
    ta = cfg["training_args"]
    assert cfg["training"] == "image"
    assert ta["trunk"] == "dense"
    assert ta["objective"] == "masked_grid"
    assert ta["image_code_bytes"] == 2 * (H // 20) * (W // 20)
    assert ta["seq"] == 64                      # 8 image + 24 caption bytes, rounded to 64
    assert ta["codec"] == name + "_codec"
    assert ta["corpus"] == name + "_img"
    assert os.path.isfile(paths.checkpoint_path(name, 4))
    with open(paths.train_csv_path(name), encoding="utf-8") as handle:
        rows = handle.read().strip().splitlines()
    assert any(",train," in r for r in rows[1:])
    assert any(",val," in r for r in rows[1:])


def test_a_current_corpus_is_not_rebuilt(home, monkeypatch):
    """Re-launching on the same set must not spend the encode again."""
    _run(monkeypatch)
    stem = "smoke_tiny_img"
    before = os.path.getmtime(paths.corpus_train_path(stem))
    argv = [a if a != "smoke" else "smoke2" for a in ARGV]
    argv += ["--codec", "smoke_tiny_codec", "--corpus", stem]
    _run(monkeypatch, argv)
    assert os.path.getmtime(paths.corpus_train_path(stem)) == before


def test_a_grown_set_rebuilds_the_corpus(home, monkeypatch):
    """New pictures must reach the corpus; the sidecar's image count is the tell."""
    _run(monkeypatch)
    stem = "smoke_tiny_img"
    Image.new("RGB", (H * 2, H * 2), (1, 2, 3)).save(
        os.path.join(paths.IMAGES_ROOT, "set", "ffffffff_new.png"))
    argv = [a if a != "smoke" else "smoke3" for a in ARGV]
    argv += ["--codec", "smoke_tiny_codec", "--corpus", stem]
    _run(monkeypatch, argv)
    with open(paths.image_corpus_meta_path(stem), encoding="utf-8") as handle:
        assert json.load(handle)["images"] == 25


def test_seq_is_derived_from_the_geometry_and_never_smaller_than_the_image():
    assert image_trainer.resolve_seq(0, 1024, 128) == 1152
    assert image_trainer.resolve_seq(0, 1000, 0) == 1024
    assert image_trainer.resolve_seq(2048, 1024, 128) == 2048
    with pytest.raises(ValueError, match="smaller than image_code_bytes"):
        image_trainer.resolve_seq(512, 1024, 0)


def test_geometry_the_patch_cannot_tile_is_refused():
    assert image_trainer.check_geometry(320, 320, 20, 4) == 1024
    with pytest.raises(ValueError, match="does not divide"):
        image_trainer.check_geometry(330, 320, 20, 4)


def test_a_missing_image_set_is_a_clear_error(home, monkeypatch):
    argv = [a if a != "set" else "nope" for a in ARGV]
    with pytest.raises(ValueError, match="no image set"):
        _run(monkeypatch, argv)


def test_every_images_flow_field_is_a_flag_the_image_trainer_parses():
    """The Images form sends TRAINER_SCHEMA.image; every field must be a manifest
    default (parsed by type) or a reserved flag, or the Training tab's start button
    would refuse the launch. Mirror of the text trainer's guard, scoped to this flow."""
    import re

    import veritate_trainer as vt

    from tests.conftest import REPO_ROOT
    with open(os.path.join(REPO_ROOT, "veritate_mri", "web", "index.js"), encoding="utf-8") as f:
        src = f.read()
    seg = src[src.find("TRAINER_SCHEMA"):]
    seg = seg[seg.find("  image: ["):]
    seg = seg[:seg.find("\n  ],")]
    fields = {m.group(1) for m in re.finditer(r'\{\s*name:\s*"([a-z0-9_]+)"', seg)}
    assert "image_set" in fields, "TRAINER_SCHEMA.image scrape found nothing; the JS moved"
    accepted = (set(trainers_reader.IMAGE_TRAINER_DEFAULTS) | set(vt.RESERVED_STRING_FLAGS)
                | set(vt.RESERVED_BOOL_FLAGS) | set(vt.RESERVED_STR_FLAGS)
                | set(vt.RESERVED_FLOAT_FLAGS) | set(vt.RESERVED_INT_FLAGS))
    unhandled = sorted(f for f in fields if f not in accepted)
    assert not unhandled, f"Images-flow fields the image trainer would reject: {unhandled}"



def test_progress_is_written_through_every_stage(home, monkeypatch):
    """The Training tab reads this file from the first second; a run must leave it in
    the done state with each stage accounted for."""
    from veritate_core.plugin import image_progress
    _run(monkeypatch)
    prog = image_progress.read(paths.model_dir("smoke_tiny"))
    assert prog["state"] == "done"
    assert prog["device"] == "cpu"
    assert {k: v["state"] for k, v in prog["stages"].items()} == {
        "decode": "done", "codec": "done", "encode": "done", "train": "done"}
    assert prog["stages"]["train"]["done"] == prog["stages"]["train"]["total"] == 4
    assert prog["notes"]["last_checkpoint_step"] == 4


def test_a_stop_request_saves_a_checkpoint_and_ends_the_run_cleanly(home, monkeypatch):
    """Stop is the dashboard's SIGTERM. The step in flight is saved, the run exits
    without an error and progress.json says stopped, so nothing trained is lost."""
    from veritate_core.plugin import image_grid, image_progress
    real = image_grid.masked_step
    calls = {"n": 0}

    def stop_on_third(*a, **k):
        if k.get("backward"):                    # training steps only; eval uses it too
            calls["n"] += 1
            if calls["n"] == 3:
                image_trainer.request_stop()
        return real(*a, **k)
    monkeypatch.setattr(image_grid, "masked_step", stop_on_third)
    _run(monkeypatch)                            # returns, no exception
    name = "smoke_tiny"
    assert os.path.isfile(paths.checkpoint_path(name, 3))
    assert not os.path.isfile(paths.checkpoint_path(name, 4))
    prog = image_progress.read(paths.model_dir(name))
    assert prog["state"] == "stopped"
    assert prog["notes"]["last_checkpoint_step"] == 3


def test_a_frame_above_the_limit_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="above the 1024 px limit"):
        image_trainer.check_geometry(1920, 1080, 20, 4)


def test_a_failed_stage_is_recorded_in_progress(home, monkeypatch):
    from veritate_core.plugin import image_progress
    argv = [a if a != "set" else "nope" for a in ARGV]
    with pytest.raises(ValueError, match="no image set"):
        _run(monkeypatch, argv)
    prog = image_progress.read(paths.model_dir("smoke_tiny"))
    assert prog["state"] == "failed"
    assert "no image set" in prog["message"]
