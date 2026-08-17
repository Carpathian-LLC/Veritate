# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the trainer used to build every run from `size_presets[args.size]`, so a resume
#   trusted a name for its architecture rather than the weights it was about to load.
# - reading the weights is only half of it. A checkpoint records the blocks a trunk
#   BUILT, not the `layers` it was built from, and VeritatePatched wraps its `layers`
#   global blocks in N_LOCAL_ENC + N_LOCAL_DEC local ones. wren_base is the case that
#   proved this: --size 200m (layers=16), 20 block entries on disk. Counting entries
#   built a 24-block model, and load_state_dict(strict=False) loaded it without a
#   word, leaving 54,649,152 parameters random.
# - these pin both halves: resume reads shape from the weights, the count is adjusted
#   for the trunk, fresh runs still read the preset, and a real disagreement resolves
#   toward the weights, loudly.
# tests/training/test_trainer_resume_shape.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib.util
import os
import sys
import types

import pytest
import torch

# ------------------------------------------------------------------------------------
# Constants

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINER = os.path.join(REPO, "veritate_mri", "training", "veritate_trainer.py")
PRESET_200M = {"layers": 16, "hidden": 1024, "ffn": 4096, "heads": 16, "params": 202_000_000}

# ------------------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="module")
def trainer():
    if os.path.join(REPO, "veritate_mri") not in sys.path:
        sys.path.insert(0, os.path.join(REPO, "veritate_mri"))
    spec = importlib.util.spec_from_file_location("veritate_trainer_shape_test", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_run(tmp_path, trainer, monkeypatch):
    """A models/<name>/checkpoints/step_N.pt holding a 20-layer state dict."""
    def _make(layers=20, hidden=1024, ffn=4096, step=140000, name="faker"):
        ckpt_dir = tmp_path / name / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sd = {"tok_emb.weight": torch.zeros(256, hidden)}
        for i in range(layers):
            sd["blocks.%d.ff.up.weight" % i] = torch.zeros(ffn, hidden)
            sd["blocks.%d.ff.down.weight" % i] = torch.zeros(hidden, ffn)
        torch.save({"model": sd, "step": step}, ckpt_dir / ("step_%d.pt" % step))
        monkeypatch.setattr(trainer.paths, "checkpoints_dir",
                            lambda n, _d=str(tmp_path): os.path.join(_d, n, "checkpoints"))
        return name
    return _make


# ------------------------------------------------------------------------------------
# Tests


def test_reads_layer_count_from_weights(trainer, fake_run):
    name = fake_run(layers=20)
    assert trainer.shape_from_checkpoint(name)["layers"] == 20


def test_reads_ffn_width_from_up_projection(trainer, fake_run):
    name = fake_run(layers=4, ffn=2048)
    assert trainer.shape_from_checkpoint(name)["ffn"] == 2048


def test_resume_overrides_a_wrong_preset(trainer, fake_run):
    """Dense trunk: preset says 16, the weights hold 20 blocks, the weights win."""
    name = fake_run(layers=20)
    args = types.SimpleNamespace(size="200m", resume=name)
    assert trainer.shape_for_run(args, {"200m": PRESET_200M})["layers"] == 20


def test_fresh_run_still_uses_the_preset(trainer):
    args = types.SimpleNamespace(size="200m", resume="")
    assert trainer.shape_for_run(args, {"200m": PRESET_200M})["layers"] == 16


def test_preset_is_not_mutated_by_a_resume(trainer, fake_run):
    """shape_for_run must copy; a mutated preset would poison later lookups."""
    name = fake_run(layers=20)
    presets = {"200m": dict(PRESET_200M)}
    trainer.shape_for_run(types.SimpleNamespace(size="200m", resume=name), presets)
    assert presets["200m"]["layers"] == 16


def test_mismatch_is_announced(trainer, fake_run, capsys):
    """Silent correction would hide a broken config.json; say it out loud."""
    name = fake_run(layers=20)
    trainer.shape_for_run(types.SimpleNamespace(size="200m", resume=name),
                          {"200m": dict(PRESET_200M)})
    out = capsys.readouterr().out
    assert "layers=16" in out and "layers=20" in out


def test_heads_falls_back_to_preset(trainer, fake_run):
    """qkv is packed, so head count is not recoverable from weights. Keep the preset."""
    name = fake_run(layers=20)
    shape = trainer.shape_for_run(types.SimpleNamespace(size="200m", resume=name),
                                  {"200m": dict(PRESET_200M)})
    assert shape["heads"] == 16


def test_agreeing_preset_is_left_alone(trainer, fake_run, capsys):
    name = fake_run(layers=16)
    shape = trainer.shape_for_run(types.SimpleNamespace(size="200m", resume=name),
                                  {"200m": dict(PRESET_200M)})
    assert shape["layers"] == 16
    assert "using the weights" not in capsys.readouterr().out


# ------------------------------------------------------------------------------------
# Explicit shape flags


def test_matching_shape_flag_is_accepted(trainer, fake_run):
    """--layers 20 against 20 layers of weights is redundant, not wrong."""
    name = fake_run(layers=20)
    args = types.SimpleNamespace(size="200m", resume=name, layers=20)
    shape = trainer.shape_for_run(args, {"200m": dict(PRESET_200M)},
                                  argv=["--layers", "20"])
    assert shape["layers"] == 20


def test_conflicting_shape_flag_refuses_to_start(trainer, fake_run):
    """Shape cannot be set from the CLI, so a disagreement must not be ignored."""
    name = fake_run(layers=20)
    args = types.SimpleNamespace(size="200m", resume=name, layers=24)
    with pytest.raises(SystemExit) as e:
        trainer.shape_for_run(args, {"200m": dict(PRESET_200M)},
                              argv=["--layers", "24"])
    assert "--layers" in str(e.value) and "24" in str(e.value)


def test_conflicting_shape_flag_on_a_fresh_run_refuses(trainer):
    args = types.SimpleNamespace(size="200m", resume="", hidden=2048)
    with pytest.raises(SystemExit) as e:
        trainer.shape_for_run(args, {"200m": dict(PRESET_200M)},
                              argv=["--hidden", "2048"])
    assert "--hidden" in str(e.value)


def test_shape_flag_not_on_argv_is_ignored(trainer, fake_run):
    """apply_resume_overrides can set these from a stale config.json, and a stale
    number must not block a resume. Only an operator typing the flag counts."""
    name = fake_run(layers=20)
    args = types.SimpleNamespace(size="200m", resume=name, layers=16)
    assert trainer.shape_for_run(args, {"200m": dict(PRESET_200M)},
                                 argv=[])["layers"] == 20


def test_zero_means_unsupplied(trainer, fake_run):
    """argparse default is 0; that must never be read as 'the user wants 0'."""
    name = fake_run(layers=20)
    args = types.SimpleNamespace(size="200m", resume=name, layers=0)
    assert trainer.shape_for_run(args, {"200m": dict(PRESET_200M)},
                                 argv=["--layers", "0"])["layers"] == 20


# ------------------------------------------------------------------------------------
# Trunk block overhead


def test_patched_trunk_subtracts_its_local_blocks(trainer, fake_run):
    """20 block entries on a hybrid checkpoint is layers=16: 2 enc + 16 global + 2 dec."""
    name = fake_run(layers=20)
    assert trainer.shape_from_checkpoint(name, "hybrid")["layers"] == 16


def test_dense_trunk_counts_blocks_directly(trainer, fake_run):
    name = fake_run(layers=20)
    assert trainer.shape_from_checkpoint(name, "dense")["layers"] == 20


def test_default_trunk_is_dense(trainer, fake_run):
    name = fake_run(layers=20)
    assert trainer.shape_from_checkpoint(name)["layers"] == 20


@pytest.mark.parametrize("trunk", sorted(
    {"patched", "hybrid", "hybrid_moe", "hybrid_monarch",
     "hybrid_pkm", "hybrid_pkm_fire", "looped"}))
def test_every_patched_trunk_gets_the_overhead(trainer, fake_run, trunk):
    """All of them go through VeritatePatched, so all wrap their global blocks."""
    name = fake_run(layers=20)
    assert trainer.shape_from_checkpoint(name, trunk)["layers"] == 16


def test_overhead_matches_the_model_module(trainer):
    """Read from model_patched, never hardcoded, so N_LOCAL_* cannot drift."""
    from veritate_core import model_patched as mp
    assert trainer.trunk_block_overhead("hybrid") == mp.N_LOCAL_ENC + mp.N_LOCAL_DEC
    assert trainer.trunk_block_overhead("dense") == 0


def test_wren_base_preset_and_weights_agree_on_hybrid(trainer, fake_run, capsys):
    """The real regression, end to end: --size 200m resuming a 20-block hybrid
    checkpoint must resolve to 16 and say nothing, because nothing is wrong."""
    name = fake_run(layers=20)
    args = types.SimpleNamespace(size="200m", resume=name, trunk="hybrid")
    shape = trainer.shape_for_run(args, {"200m": dict(PRESET_200M)}, argv=[])
    assert shape["layers"] == 16
    assert "using the weights" not in capsys.readouterr().out


def test_too_few_blocks_for_the_trunk_is_an_error(trainer, fake_run):
    """Never return a nonsense layer count; a 3-block hybrid cannot exist."""
    name = fake_run(layers=3)
    with pytest.raises(RuntimeError, match="too few"):
        trainer.shape_from_checkpoint(name, "hybrid")
