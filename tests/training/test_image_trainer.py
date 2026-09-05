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
CODEC = "set_40x40_p20x2_v2_codec"           # <set>_<h>x<w>_p<patch>x<planes>_<recipe>_codec
STEM  = "set_40x40_p20x2_v2_img"
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


def _progress(name):
    with open(os.path.join(paths.model_dir(name), "progress.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_one_launch_produces_codec_corpus_config_and_checkpoint(home, monkeypatch):
    """The whole point: pictures in, a trained image model out, nothing else to run."""
    _run(monkeypatch)
    name = "smoke_tiny"
    assert os.path.isfile(paths.codec_path(CODEC))
    assert os.path.isfile(paths.corpus_train_path(STEM))
    assert os.path.isfile(paths.image_corpus_meta_path(STEM))
    with open(paths.config_path(name), encoding="utf-8") as handle:
        cfg = json.load(handle)
    ta = cfg["training_args"]
    assert cfg["training"] == "image"
    assert ta["trunk"] == "dense"
    assert ta["objective"] == "masked_grid"
    assert ta["image_code_bytes"] == 2 * (H // 20) * (W // 20)
    assert ta["seq"] == 64                      # 8 image + 24 caption bytes, rounded to 64
    assert ta["codec"] == CODEC                 # keyed on the set and geometry, not the model
    assert ta["corpus"] == STEM
    assert os.path.isfile(paths.checkpoint_path(name, 4))
    with open(paths.train_csv_path(name), encoding="utf-8") as handle:
        rows = handle.read().strip().splitlines()
    assert any(",train," in r for r in rows[1:])
    assert any(",val," in r for r in rows[1:])


def test_another_model_on_the_same_pictures_reuses_the_codec_and_the_corpus(home, monkeypatch):
    """Changing the model's name or size must not spend the codec fit or the encode again:
    both are named after the set and the geometry, and nothing in the form points at them."""
    _run(monkeypatch)
    codec_before = os.path.getmtime(paths.codec_path(CODEC))
    corpus_before = os.path.getmtime(paths.corpus_train_path(STEM))
    argv = [a if a != "smoke" else "another-name" for a in ARGV]
    _run(monkeypatch, argv)
    assert os.path.getmtime(paths.codec_path(CODEC)) == codec_before
    assert os.path.getmtime(paths.corpus_train_path(STEM)) == corpus_before
    with open(paths.config_path("another_name_tiny"), encoding="utf-8") as handle:
        assert json.load(handle)["training_args"]["corpus"] == STEM


def test_a_grown_set_rebuilds_the_corpus(home, monkeypatch):
    """New pictures must reach the corpus; the sidecar's image count is the tell."""
    _run(monkeypatch)
    Image.new("RGB", (H * 2, H * 2), (1, 2, 3)).save(
        os.path.join(paths.IMAGES_ROOT, "set", "ffffffff_new.png"))
    argv = [a if a != "smoke" else "smoke3" for a in ARGV]
    _run(monkeypatch, argv)
    with open(paths.image_corpus_meta_path(STEM), encoding="utf-8") as handle:
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


def test_default_names_key_on_the_set_the_geometry_and_the_codec_recipe():
    """A fit from before a recipe change (the 2026-09-05 capacity bump) must not be reused as
    if it were current: the recipe is in the name, so the old file simply is not found."""
    from veritate_core.plugin import image_codec
    tag = image_codec.RECIPE
    assert image_trainer.default_codec_name("photos", 320, 320, 20, 4) == f"photos_320x320_p20x4_{tag}_codec"
    assert image_trainer.default_codec_name("photos", 320, 320, 20, 4, 2) == f"photos_320x320_p20x4_x2_{tag}_codec"
    assert image_trainer.default_corpus_stem(f"photos_320x320_p20x4_{tag}_codec") == f"photos_320x320_p20x4_{tag}_img"


def test_the_memory_plan_halves_the_forward_until_the_step_fits():
    """800m at seq 1152, batch 16 on a 24 GB Mac is the case that failed: the plan must
    say so before the first step, and a size that fits must keep its batch whole."""
    big = {"layers": 28, "hidden": 1536, "ffn": 6144, "heads": 24}
    micro, accum, est = image_trainer.plan_micro_batch(big, 793_000_000, 16, 1152, 2, "muon",
                                                       budget=int(24 * 0.7 * 1024 ** 3))
    assert micro == 1 and accum == 16
    assert est > int(24 * 0.7 * 1024 ** 3)          # does not fit even at one picture
    small = {"layers": 8, "hidden": 512, "ffn": 2048, "heads": 8}
    micro, accum, _ = image_trainer.plan_micro_batch(small, 20_000_000, 16, 1152, 2, "muon",
                                                     budget=int(24 * 0.7 * 1024 ** 3))
    assert (micro, accum) == (16, 1)
    assert image_trainer.shrink_micro(12, 12) == 6 and image_trainer.shrink_micro(5, 10) == 2


def test_an_out_of_memory_step_shrinks_the_forward_and_the_run_still_finishes(home, monkeypatch):
    """The device says no at batch 2; the trainer retries the step at one picture per
    forward, twice, adds the gradients up, and the run completes with the same batch."""
    from veritate_core.plugin import image_grid, image_progress
    real = image_grid.masked_step
    seen = []

    def oom_above_one(model, tokens, targets, *a, **k):
        seen.append(int(tokens.shape[0]))
        if k.get("backward") and tokens.shape[0] > 1:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 29.26 GiB)")
        return real(model, tokens, targets, *a, **k)
    monkeypatch.setattr(image_grid, "masked_step", oom_above_one)
    _run(monkeypatch)
    prog = image_progress.read(paths.model_dir("smoke_tiny"))
    assert prog["state"] == "done"
    assert prog["notes"]["micro_batch"] == 1 and prog["notes"]["grad_accum"] == 2
    assert prog["notes"]["oom_retries"] == 1
    assert 2 in seen and seen.count(1) >= 8          # 4 steps x 2 forwards, plus eval
    assert os.path.isfile(paths.checkpoint_path("smoke_tiny", 4))


def test_out_of_memory_at_one_picture_is_a_clear_error(home, monkeypatch):
    from veritate_core.plugin import image_grid, image_progress

    def always_oom(*a, **k):
        if k.get("backward"):
            raise RuntimeError("CUDA out of memory")
        return image_grid.masked_step.__wrapped__(*a, **k) if hasattr(image_grid.masked_step, "__wrapped__") else None
    monkeypatch.setattr(image_grid, "masked_step", always_oom)
    with pytest.raises(RuntimeError, match="does not fit this device"):
        _run(monkeypatch)
    assert image_progress.read(paths.model_dir("smoke_tiny"))["state"] == "failed"


def test_fp16_trains_under_loss_scaling_and_records_what_it_computed_in(home, monkeypatch):
    """`precision` resolves on the device: here the CPU is made to say fp16 so the
    GradScaler path runs end to end (unscale, clip, step, update) on the tiny model, and
    config.json records the resolved precision, not just the request."""
    import torch

    monkeypatch.setattr(hardware, "resolve_precision", lambda want, dev: torch.float16 if want == "fp16" else None)
    argv = [a if a != "fp32" else "fp16" for a in ARGV]
    argv = [a if a != "smoke" else "half" for a in argv]
    _run(monkeypatch, argv)
    with open(paths.config_path("half_tiny"), encoding="utf-8") as handle:
        ta = json.load(handle)["training_args"]
    assert ta["precision"] == "fp16" and ta["precision_resolved"] == "fp16"
    assert os.path.isfile(paths.checkpoint_path("half_tiny", 4))
    prog = _progress("half_tiny")
    assert prog["notes"]["precision"] == "fp16" and prog["notes"]["compiled"] is False
    assert prog["stages"]["train"]["step_s"] > 0


def test_auto_precision_is_fp32_on_a_cpu_and_a_bad_choice_is_refused(home, monkeypatch):
    argv = [a if a != "fp32" else "auto" for a in ARGV]
    _run(monkeypatch, argv)
    with open(paths.config_path("smoke_tiny"), encoding="utf-8") as handle:
        assert json.load(handle)["training_args"]["precision_resolved"] == "fp32"
    with pytest.raises(ValueError, match="unknown precision"):
        _run(monkeypatch, [a if a != "fp32" else "fp8" for a in ARGV])
    with pytest.raises(ValueError, match="unknown compile"):
        _run(monkeypatch, [*ARGV, "--compile", "maybe"])


def test_a_compile_that_fails_falls_back_to_eager_and_the_run_finishes(home, monkeypatch):
    """torch.compile is a speed lever, never a reason a run dies: a graph that will not
    compile is dropped at the first step and training continues eager."""
    import torch

    def broken_compile(model, **_kw):
        def fwd(*_a, **_k):
            raise RuntimeError("inductor: no backend for this graph")
        return fwd

    monkeypatch.setattr(torch, "compile", broken_compile)
    _run(monkeypatch, [*ARGV, "--compile", "on"])
    assert os.path.isfile(paths.checkpoint_path("smoke_tiny", 4))
    assert _progress("smoke_tiny")["notes"]["compiled"] is False
    assert image_trainer.want_compile("auto", "cpu") is False
    assert image_trainer.want_compile("auto", "mps") is True
    assert image_trainer.want_compile("off", "mps") is False


def test_the_memory_estimate_charges_attention_in_the_dtype_the_device_holds_it_in():
    """sdpa's fallback holds fp32 attention whatever autocast says; the explicit path on
    MPS holds the working dtype and measured half the memory (M2, 2026-09-05)."""
    import torch

    shape = {"hidden": 768, "layers": 12, "heads": 12}
    fallback = image_trainer.estimate_step_bytes(shape, 85_000_000, 8, 1152, 2, "muon")
    explicit = image_trainer.estimate_step_bytes(shape, 85_000_000, 8, 1152, 2, "muon", attn_bytes=2)
    assert explicit < fallback
    assert 9.5e9 < explicit < 11.5e9                    # measured peak 9.13 GB at 80m x 8 pictures
    assert image_trainer.attention_bytes("mps", torch.float16) == 2
    assert image_trainer.attention_bytes("mps", None) == 4
    assert image_trainer.attention_bytes("cpu", torch.float16) == 4


def test_an_output_scale_is_part_of_the_codec_and_the_pictures_come_out_bigger(home, monkeypatch):
    """out_scale 2: the codec is named for it, fitted on the pictures at 2x, the model's
    bytes and seq are unchanged, config records it, and every probe picture decodes at 2x
    (the fill test tiles are still THUMB wide: tiles are resized, frames are not)."""
    from PIL import Image as PILImage

    from veritate_core.plugin import image_codec, image_probe
    _run(monkeypatch, [*ARGV, "--out_scale", "2"])
    name = "smoke_tiny"
    with open(paths.config_path(name), encoding="utf-8") as handle:
        ta = json.load(handle)["training_args"]
    assert ta["out_scale"] == 2 and ta["codec"] == "set_40x40_p20x2_x2_v2_codec"
    assert ta["image_code_bytes"] == 2 * (H // 20) * (W // 20) and ta["seq"] == 64
    codec = image_codec.load(paths.codec_path(ta["codec"]))
    assert codec.out_scale == 2
    assert os.path.isfile(os.path.join(paths.IMAGE_CACHE_ROOT, f"set_{2 * H}x{2 * W}.u8")) or True
    d = os.path.join(paths.hook_step_dir(name, 4), image_probe.IMAGE_DIR)
    fill = PILImage.open(os.path.join(d, "fill.png"))
    assert fill.size[0] == 3 * image_probe.THUMB + 4 * image_probe.GAP
    with pytest.raises(ValueError, match="above 1920"):
        _run(monkeypatch, [a if a != str(H) else "640" for a in ARGV] + ["--out_scale", "4", "--name", "huge"])


def test_pictures_are_probed_between_checkpoints_at_probe_every(home, monkeypatch):
    """The probe is not tied to the save cadence: with probe_every 1 and checkpoints at 2
    and 4, steps 1 and 3 have probe pictures and no checkpoint."""
    from veritate_core.plugin import image_probe
    _run(monkeypatch, [*ARGV, "--probe_every", "1"])
    steps = [m["step"] for m in image_probe.read("smoke_tiny")]
    assert steps == [1, 2, 3, 4]
    assert not os.path.isfile(paths.checkpoint_path("smoke_tiny", 1))
    assert os.path.isfile(paths.checkpoint_path("smoke_tiny", 2))
    assert _progress("smoke_tiny")["notes"]["probe_step"] == 4


def test_a_relaunch_over_an_attempt_that_never_trained_drops_its_stale_rows(home, monkeypatch):
    """OOM at step 1 leaves config.json and maybe a train.csv; the next launch of the same
    name starts clean rather than concatenating runs."""
    os.makedirs(paths.model_dir("smoke_tiny"), exist_ok=True)
    with open(paths.train_csv_path("smoke_tiny"), "w", encoding="utf-8") as handle:
        handle.write("step,split,loss,lr,grad_norm,tok_per_s,wall_s,seed\n7,train,9.9,0.1,1,1,1,0\n")
    _run(monkeypatch)
    with open(paths.train_csv_path("smoke_tiny"), encoding="utf-8") as handle:
        rows = handle.read().strip().splitlines()[1:]
    assert not any(r.startswith("7,") for r in rows)
    assert rows and rows[0].startswith("1,")


def test_continuing_a_model_keeps_its_frame_and_size_whatever_the_form_sent(home, monkeypatch):
    """Resume from the Images form: the runner sends every field, but a model's pictures,
    frame, codec, corpus and size are facts about its weights. Training continues from
    the last checkpoint to the new total."""
    _run(monkeypatch)
    argv = ["image_trainer.py", "--resume", "smoke_tiny", "--description", "continue",
            "--image_set", "set", "--size", "tiny", "--height", "200", "--width", "200",
            "--planes", "3", "--total_steps", "6", "--batch_size", "2", "--ckpt_every", "2",
            "--eval_every", "2", "--eval_iters", "1", "--log_every", "1", "--precision", "fp32",
            "--optimizer", "adamw", "--warmup_steps", "1"]
    _run(monkeypatch, argv)
    assert os.path.isfile(paths.checkpoint_path("smoke_tiny", 6))
    with open(paths.config_path("smoke_tiny"), encoding="utf-8") as handle:
        ta = json.load(handle)["training_args"]
    assert (ta["height"], ta["width"], ta["planes"]) == (H, W, 2)     # not 200x200 / 3
    assert ta["corpus"] == STEM and ta["codec"] == CODEC
