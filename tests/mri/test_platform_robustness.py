# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - platform-robustness units: hardware.resolve_precision downgrades bf16 on
#   machines without bf16 acceleration, and _build_argv emits --no-<flag> for
#   negatable bools so auto-optimize can turn QAT/act-ckpt off from the dashboard.
# tests/mri/test_platform_robustness.py
# ------------------------------------------------------------------------------------
# Imports:


import torch
from training import trainer_runner

from veritate_core.plugin import hardware

# ------------------------------------------------------------------------------------
# Functions

def test_bf16_supported_cpu_is_false():
    """CPU reports no bf16 acceleration (emulated bf16 is slower than fp32)."""
    assert hardware.bf16_supported("cpu") is False


def test_bf16_supported_mps_is_true():
    """MPS supports bf16 autocast."""
    assert hardware.bf16_supported("mps") is True


def test_resolve_precision_downgrades_bf16_on_cpu():
    """bf16 requested on a CPU resolves to fp32 (None autocast dtype)."""
    assert hardware.resolve_precision("bf16", "cpu") is None


def test_resolve_precision_keeps_bf16_on_mps():
    """bf16 requested on MPS stays bf16."""
    assert hardware.resolve_precision("bf16", "mps") is torch.bfloat16


def test_resolve_precision_fp32_is_none_everywhere():
    """fp32 always resolves to None (no autocast) regardless of device."""
    assert hardware.resolve_precision("fp32", "cpu") is None
    assert hardware.resolve_precision("fp32", "mps") is None


def test_build_argv_true_bool_emits_flag():
    """A True bool emits --flag."""
    argv = trainer_runner._build_argv({"path": "/t.py"}, {"qat_enabled": True})
    assert "--qat_enabled" in argv and "--no-qat_enabled" not in argv


def test_build_argv_negatable_false_emits_no_flag():
    """A False value on a negatable bool emits --no-<flag> so a manifest default
    of true can be overridden from the dashboard."""
    argv = trainer_runner._build_argv({"path": "/t.py"}, {"qat_enabled": False, "use_act_ckpt": False})
    assert "--no-qat_enabled" in argv
    assert "--no-use_act_ckpt" in argv


def test_build_argv_non_negatable_false_is_omitted():
    """A False value on a non-negatable bool emits nothing (unchanged behavior)."""
    argv = trainer_runner._build_argv({"path": "/t.py"}, {"some_plain_flag": False})
    assert "--some_plain_flag" not in argv
    assert "--no-some_plain_flag" not in argv


def test_build_argv_scalar_pairs():
    """Non-bool values emit --key value pairs; empty/None are skipped."""
    argv = trainer_runner._build_argv({"path": "/t.py"}, {"seq": 512, "name": "", "x": None})
    assert argv[-2:] == ["--seq", "512"]
    assert "--name" not in argv and "--x" not in argv
