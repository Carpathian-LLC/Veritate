# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the load-balance regularization mode: model.balance_penalty,
#   FFN/Veritate reg_mode wiring, and the neuron_balance core plugin.
# tests/plugin_contract/test_reg_balance.py
# ------------------------------------------------------------------------------------
# Imports

import math

import pytest
import torch

from veritate_core import model as m
from veritate_core import core_plugins as cp


# ------------------------------------------------------------------------------------
# Functions

def test_balance_penalty_zero_on_uniform_load():
    """balance_penalty is ~0 when every unit carries equal load."""
    out = m.balance_penalty(torch.ones(2, 8, 64))
    assert abs(float(out)) < 1e-4


def test_balance_penalty_positive_on_skewed_load():
    """balance_penalty approaches log(ffn) when one unit carries all load."""
    skew = torch.zeros(2, 8, 64)
    skew[..., 0] = 10.0
    assert float(m.balance_penalty(skew)) > 0.9 * math.log(64)


def test_ffn_captures_balance_when_reg_mode_balance():
    """FFN(reg_mode=balance) stores the balance penalty in _last_l1."""
    f = m.FFN(32, 64, activation="relu", capture_l1=True, reg_mode="balance")
    f(torch.randn(2, 8, 32))
    assert f._last_l1 is not None and float(f._last_l1) >= 0.0


def test_post_l1_sum_sums_balance_over_blocks():
    """Veritate(reg_mode=balance) sums per-block balance penalties."""
    net = m.Veritate(vocab=256, hidden=64, layers=3, ffn=64, heads=4, seq=32,
                     activation="relu", capture_l1=True, reg_mode="balance")
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


def test_neuron_balance_plugin_in_regularizer_group():
    """neuron_balance shares the regularizer group with the l1 plugins (mutually exclusive)."""
    by_id = {p["id"]: p for p in cp.REGISTRY}
    assert by_id["neuron_balance"]["group"] == cp.GROUP_REGULARIZER
    assert cp.conflicts(["neuron_balance", "l1_sparsity_light"])


def test_neuron_balance_plugin_sets_balance_args():
    """Selecting neuron_balance injects reg_mode=balance with a positive l1_lambda."""
    args = cp.args_for_selection(["neuron_balance"])
    assert args["reg_mode"] == "balance"
    assert args["l1_lambda"] > 0.0
