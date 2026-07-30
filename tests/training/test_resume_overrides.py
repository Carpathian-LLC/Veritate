# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - apply_resume_overrides restores training_args from a resumed config.json onto a
#   fresh argparse.Namespace, unless the caller already set that flag on argv.
# - it reads cfg["training_args"], which older flat config.json files never had (the
#   shape/lr fields lived at the top level instead). Resuming one of those configs
#   silently restores nothing, so --size must be passed explicitly on a continue.
#   These tests pin that trap alongside the restore/CLI-wins paths.
# tests/training/test_resume_overrides.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import sys

import pytest

TRAINER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "veritate_mri", "training")
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)

import veritate_trainer as vt

# ------------------------------------------------------------------------------------
# Helpers


def _cfg(tmp_path, monkeypatch, contents):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(contents), encoding="utf-8")
    monkeypatch.setattr(vt.paths, "config_path", lambda name: str(cfg_path))


# ------------------------------------------------------------------------------------
# Flat old-style config: no training_args key


def test_flat_old_style_config_does_not_restore_model_shape(tmp_path, monkeypatch):
    """A config.json with no training_args key leaves args untouched (--size trap)."""
    _cfg(tmp_path, monkeypatch, {"hidden": 999, "layers": 24, "size": "1b"})
    args = argparse.Namespace(resume="some_model", hidden=64, layers=4, size="50m")
    vt.apply_resume_overrides(args, argv=[])
    assert args.hidden == 64
    assert args.layers == 4
    assert args.size == "50m"


# ------------------------------------------------------------------------------------
# Current nested config: training_args key present


def test_nested_current_config_restores_training_args(tmp_path, monkeypatch):
    """A config.json with a training_args block restores those values onto args."""
    _cfg(tmp_path, monkeypatch, {"training_args": {"hidden": 999, "layers": 24, "base_lr": 0.0005}})
    args = argparse.Namespace(resume="some_model", hidden=64, layers=4, base_lr=0.001)
    vt.apply_resume_overrides(args, argv=[])
    assert args.hidden == 999
    assert args.layers == 24
    assert args.base_lr == 0.0005


# ------------------------------------------------------------------------------------
# Explicit CLI flag wins over the resumed value


def test_explicit_cli_flag_wins_over_resumed_value(tmp_path, monkeypatch):
    """A flag present on argv is left alone even though training_args has a value for it."""
    _cfg(tmp_path, monkeypatch, {"training_args": {"hidden": 999}})
    args = argparse.Namespace(resume="some_model", hidden=777)
    vt.apply_resume_overrides(args, argv=["--hidden", "777"])
    assert args.hidden == 777


# ------------------------------------------------------------------------------------
# Missing config.json


def test_missing_config_raises_file_not_found(tmp_path, monkeypatch):
    """Resuming a target with no config.json on disk raises FileNotFoundError."""
    monkeypatch.setattr(vt.paths, "config_path", lambda name: str(tmp_path / "absent" / "config.json"))
    args = argparse.Namespace(resume="ghost_model")
    with pytest.raises(FileNotFoundError):
        vt.apply_resume_overrides(args, argv=[])
