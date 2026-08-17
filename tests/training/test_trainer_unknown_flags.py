# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_trainer.parse_args used to call parse_known_args and discard the
#   remainder, so a typo or a flag carried over from a retired trainer changed
#   nothing, raised nothing, and logged nothing. These tests pin the replacement:
#   dashboard schema flags stay ignorable, everything else is fatal.
# tests/training/test_trainer_unknown_flags.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib.util
import os
import sys

import pytest

# ------------------------------------------------------------------------------------
# Constants

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINER = os.path.join(REPO, "veritate_mri", "training", "veritate_trainer.py")
MANIFEST = {"description": "test", "defaults": {"batch_size": 8, "seq": 256}}

# ------------------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="module")
def trainer():
    """Import veritate_trainer as a module without running its __main__ path."""
    if os.path.join(REPO, "veritate_mri") not in sys.path:
        sys.path.insert(0, os.path.join(REPO, "veritate_mri"))
    spec = importlib.util.spec_from_file_location("veritate_trainer_under_test", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse(trainer, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["veritate_trainer.py"] + argv)
    return trainer.parse_args(MANIFEST)


# ------------------------------------------------------------------------------------
# Tests


def test_known_flags_parse(trainer, monkeypatch):
    """Flags the trainer implements are parsed, not rejected."""
    args = _parse(trainer, ["--batch_size", "16", "--seq", "512"], monkeypatch)
    assert args.batch_size == 16
    assert args.seq == 512


def test_typo_refuses_to_start(trainer, monkeypatch):
    """A misspelled flag is fatal. Silently dropping it would train the wrong shape."""
    with pytest.raises(SystemExit) as e:
        _parse(trainer, ["--batch_size", "16", "--btach_size", "32"], monkeypatch)
    assert "--btach_size" in str(e.value)


def test_retired_trainer_flag_refuses_to_start(trainer, monkeypatch):
    """The concrete regression: a flag that only the deleted vanilla_trainer took."""
    with pytest.raises(SystemExit) as e:
        _parse(trainer, ["--seq", "512", "--bptt_chunks", "4"], monkeypatch)
    assert "--bptt_chunks" in str(e.value)


@pytest.mark.parametrize("flag", ["quant_mode", "n_experts", "router_topk", "recipe"])
def test_dashboard_schema_flags_stay_ignorable(trainer, monkeypatch, flag):
    """The dashboard renders the whole schema for every plugin; those must not be fatal."""
    args = _parse(trainer, ["--batch_size", "16", "--" + flag, "x"], monkeypatch)
    assert args.batch_size == 16


def test_negated_boolean_form_is_not_unknown(trainer, monkeypatch):
    """BooleanOptionalAction emits --no-<flag>; the check must strip that prefix."""
    _parse(trainer, ["--no-qat_enabled"], monkeypatch)


def test_error_names_every_offending_flag(trainer, monkeypatch):
    """Report all of them at once rather than one per failed launch."""
    with pytest.raises(SystemExit) as e:
        _parse(trainer, ["--nope_one", "1", "--nope_two", "2"], monkeypatch)
    msg = str(e.value)
    assert "--nope_one" in msg and "--nope_two" in msg


def test_value_of_ignored_flag_is_not_read_as_a_flag(trainer, monkeypatch):
    """`--quant_mode int8` leaves a bare `int8` in the remainder; it is a value."""
    args = _parse(trainer, ["--quant_mode", "int8", "--seq", "512"], monkeypatch)
    assert args.seq == 512


def test_every_dashboard_schema_field_is_handled(trainer):
    """The whole point of the allowlist: the dashboard renders TRAINER_SCHEMA for
    every plugin, so any field it can send must either be a flag this trainer
    parses or be explicitly ignorable. A new schema field that is neither would
    make the Training tab refuse to start, and this is the only place that
    catches it."""
    import re

    from readers import trainers as reader
    src = open(os.path.join(REPO, "veritate_mri", "web", "index.js"),
               encoding="utf-8").read()
    seg = src[src.find("TRAINER_SCHEMA"):][:40000]
    fields = {m.group(1) for m in re.finditer(r'\{\s*name:\s*"([a-z0-9_]+)"', seg)}
    assert "base_lr" in fields, "TRAINER_SCHEMA scrape found nothing; the JS moved"
    accepted = (set((reader.NATIVE_TRAINER_MANIFEST.get("defaults") or {}))
                | set(trainer.RESERVED_STRING_FLAGS) | set(trainer.RESERVED_BOOL_FLAGS)
                | set(trainer.RESERVED_STR_FLAGS) | set(trainer.RESERVED_FLOAT_FLAGS)
                | set(trainer.RESERVED_INT_FLAGS) | set(trainer.SHAPE_OVERRIDE_FLAGS)
                | {"bench"})
    unhandled = sorted(f for f in fields
                       if f not in accepted and f not in trainer.SCHEMA_IGNORED_FLAGS)
    assert not unhandled, "dashboard fields the trainer would reject: %s" % unhandled


def test_model_type_is_not_fatal(trainer, monkeypatch):
    """The runner passes it as an env var AND leaves it on argv."""
    args = _parse(trainer, ["--model_type", "language", "--seq", "512"], monkeypatch)
    assert args.seq == 512
