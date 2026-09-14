# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - ProductKeyMemory: an FFN replacement that reads a few rows of a large learned value
#   table instead of running a dense pair of projections, so capacity grows without the
#   bytes-per-token growing with it. Pinned here: the FFN output shape, the two-half key
#   search addressing sub_keys^2 slots from 2*sub_keys keys, that only top_k rows are ever
#   read, the threshold gate's straight-through behaviour (hard forward, learnable theta),
#   the constructor's refusals, and the two cost reporters the bench and the ledger quote.
# tests/core/test_model_pkm.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
import torch

from veritate_core.model_pkm import PKM_GATE_THRESHOLD, ProductKeyMemory

# ------------------------------------------------------------------------------------
# Constants

HIDDEN, FFN_WIDTH, SEQ = 16, 32, 4
SUB_KEYS, TOP_K, HEADS, KEY_DIM = 8, 4, 2, 8

# ------------------------------------------------------------------------------------
# Functions


def _pkm(**kw):
    torch.manual_seed(0)
    kw = {"sub_keys": SUB_KEYS, "top_k": TOP_K, "heads": HEADS, "key_dim": KEY_DIM, **kw}
    return ProductKeyMemory(HIDDEN, FFN_WIDTH, **kw)


def test_it_returns_the_ffn_shape():
    """Drop-in for the dense FFN inside a block."""
    x = torch.randn(2, SEQ, HIDDEN)
    assert _pkm()(x).shape == x.shape


def test_the_value_table_is_the_square_of_the_sub_keys():
    """The point of the product key: 2*sub_keys learned keys address sub_keys^2 slots, so
    capacity is quadratic in what the search actually scans."""
    m = _pkm()
    assert m.values.weight.shape == (SUB_KEYS * SUB_KEYS, HIDDEN)
    assert m.sub_key.shape == (2, HEADS, SUB_KEYS, KEY_DIM // 2)
    assert m.capacity_params() == SUB_KEYS * SUB_KEYS * HIDDEN


def test_only_top_k_rows_are_read_per_head():
    """Bytes per token must not grow with the table. The gather is top_k rows per head,
    whatever the table holds."""
    m = _pkm()
    read = m.read_bytes_per_token(weight_bytes=1)
    gather = HEADS * TOP_K * HIDDEN
    assert gather < m.capacity_params()
    assert read == (2 * HEADS * SUB_KEYS * (KEY_DIM // 2)) + gather + (HIDDEN * HEADS * KEY_DIM)


def test_a_bigger_table_does_not_cost_more_per_token():
    """Four times the slots, the same rows read: capacity is free at read time except for
    the sub-key search, which grows as the square root of the table."""
    small, big = _pkm(sub_keys=SUB_KEYS), _pkm(sub_keys=2 * SUB_KEYS)
    assert big.capacity_params() == 4 * small.capacity_params()
    grew = big.read_bytes_per_token(1) - small.read_bytes_per_token(1)
    assert grew == 2 * HEADS * SUB_KEYS * (KEY_DIM // 2)


def test_the_threshold_gate_is_hard_forward_and_still_learns():
    """Measured 15x faster than ranking (worklog 2026-08-03). The forward must be a real
    on/off so the kernel matches, while theta keeps a gradient through the soft term."""
    m = _pkm(gate=PKM_GATE_THRESHOLD)
    out = m(torch.randn(2, SEQ, HIDDEN))
    out.sum().backward()
    assert m.theta.grad is not None and torch.any(m.theta.grad != 0)
    assert m._last_fired is not None and 0 <= float(m._last_fired) <= TOP_K


def test_theta_decides_how_many_candidates_fire():
    """The gate is a threshold on the standardized score, so raising theta fires fewer."""
    x = torch.randn(2, SEQ, HIDDEN)
    m = _pkm(gate=PKM_GATE_THRESHOLD)
    m(x)
    open_gate = float(m._last_fired)
    with torch.no_grad():
        m.theta.fill_(5.0)
    m(x)
    assert float(m._last_fired) < open_gate


def test_the_default_gate_keeps_every_selected_candidate():
    """Without the threshold there is no firing count to report: the top_k softmax is the
    whole gate."""
    m = _pkm()
    m(torch.randn(2, SEQ, HIDDEN))
    assert m.theta is None and m._last_fired is None


@pytest.mark.parametrize(("kw", "msg"), [
    ({"key_dim": 7}, "key_dim must be even"),
    ({"top_k": SUB_KEYS + 1}, "exceeds sub_keys"),
    ({"gate": "sometimes"}, "unknown gate"),
])
def test_a_configuration_that_cannot_work_is_refused_at_build(kw, msg):
    """Each of these produces silent garbage rather than an error if it is let through."""
    with pytest.raises(ValueError, match=msg):
        _pkm(**kw)


def test_the_probe_points_at_the_query():
    """rule 11a: the dump suite walks probe_module; this layer has no dense up/down pair,
    so probe_weights is None and the query stands in for the activations."""
    m = _pkm()
    assert m.probe_module() is m.query
    assert m.probe_weights() is None
