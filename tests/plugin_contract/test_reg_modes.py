# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the regularization modes: l1 (default) and group (group-lasso
#   for prunable structured sparsity), the FFN/Veritate reg_mode wiring, and the
#   group-lasso core plugin.
# tests/plugin_contract/test_reg_modes.py
# ------------------------------------------------------------------------------------
# Imports

import pytest
import torch

from veritate_core import model as m
from veritate_core import core_plugins as cp


# ------------------------------------------------------------------------------------
# Functions

def test_group_penalty_zero_when_all_units_silent():
    """group_penalty is ~0 when every unit is off, large when units fire."""
    silent = float(m.group_penalty(torch.zeros(2, 8, 16)))
    firing = float(m.group_penalty(torch.ones(2, 8, 16)))
    assert silent < 1e-2 and firing > 1.0 and firing > 100 * silent


def test_group_penalty_counts_active_units():
    """group_penalty grows with the number of active units (structured)."""
    one = torch.zeros(2, 8, 16); one[..., 0] = 1.0
    four = torch.zeros(2, 8, 16); four[..., :4] = 1.0
    assert float(m.group_penalty(four)) > float(m.group_penalty(one))


def test_ffn_group_mode_captures_penalty():
    """FFN(reg_mode=group) stores the group penalty in _last_l1."""
    f = m.FFN(32, 64, activation="relu", capture_l1=True, reg_mode="group")
    f(torch.randn(2, 8, 32))
    assert f._last_l1 is not None and float(f._last_l1) >= 0.0


def test_post_l1_sum_sums_group_over_blocks():
    """Veritate(reg_mode=group) sums per-block group penalties."""
    net = m.Veritate(vocab=256, hidden=64, layers=3, ffn=64, heads=4, seq=32,
                     activation="relu", capture_l1=True, reg_mode="group")
    net(torch.randint(0, 256, (2, 16)))
    assert float(net.post_l1_sum()) >= 0.0


def test_reg_mode_defaults_to_l1():
    """Default reg_mode is l1 so existing callers are unchanged."""
    net = m.Veritate(vocab=256, hidden=64, layers=2, ffn=64, heads=4, seq=32)
    assert net.reg_mode == "l1"


def test_invalid_reg_mode_raises():
    """Unknown reg_mode raises ValueError at construction."""
    with pytest.raises(ValueError):
        m.FFN(32, 64, reg_mode="bogus")


def test_group_plugin_in_regularizer_group():
    """neuron_prune_group shares the regularizer group with l1 (mutually exclusive)."""
    by_id = {p["id"]: p for p in cp.REGISTRY}
    assert by_id["neuron_prune_group"]["group"] == cp.GROUP_REGULARIZER
    assert cp.conflicts(["neuron_prune_group", "l1_sparsity_light"])


def test_group_plugin_sets_group_args():
    """Selecting neuron_prune_group injects reg_mode=group with a positive l1_lambda."""
    args = cp.args_for_selection(["neuron_prune_group"])
    assert args["reg_mode"] == "group"
    assert args["l1_lambda"] > 0.0
