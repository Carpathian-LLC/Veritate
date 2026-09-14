# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - MoEFFN, the fine-grained mixture-of-experts drop-in for the canonical FFN. What is
#   pinned here is the contract the rest of the platform relies on: the FFN output shape,
#   static tensor shapes across steps (rule 24c, the MPS kernel cache), top-k routing that
#   really selects k experts, capacity that drops the overflow instead of growing a tensor,
#   the routing bias moving only while training, and the probe_module/probe_weights pair
#   the dump, diff, pruning and export paths walk.
# tests/core/test_model_moe.py
# ------------------------------------------------------------------------------------
# Imports:

import torch

from veritate_core.model_moe import MoEFFN

# ------------------------------------------------------------------------------------
# Constants

HIDDEN, FFN_WIDTH, SEQ = 16, 32, 8

# ------------------------------------------------------------------------------------
# Functions


def _ffn(**kw):
    torch.manual_seed(0)
    return MoEFFN(HIDDEN, FFN_WIDTH, **kw)


def test_it_returns_the_ffn_shape():
    """Drop-in for the dense FFN: same in, same out."""
    x = torch.randn(2, SEQ, HIDDEN)
    assert _ffn().eval()(x).shape == x.shape


def test_every_token_is_routed_to_exactly_top_k_experts():
    """The router selects k, not k-1 and not the argmax twice: the loop masks each pick out
    of the running affinity before the next argmax."""
    m = _ffn(top_k=3)
    dispatch, _ = m._route(torch.randn(2, SEQ, HIDDEN), capacity=SEQ)
    per_token = dispatch.sum(dim=(2, 3))
    assert torch.equal(per_token, torch.full_like(per_token, 3.0))


def test_capacity_drops_the_overflow_rather_than_growing_the_tensor():
    """Capacity is the fixed third dimension of the dispatch tensor. A token past an
    expert's capacity is dropped, so the shape never depends on the routing."""
    m = _ffn()
    dispatch, combine = m._route(torch.randn(1, SEQ, HIDDEN), capacity=1)
    assert dispatch.shape[-1] == 1 and combine.shape[-1] == 1
    # [batch, token, expert, slot]: summed over tokens, no slot holds two of them
    assert dispatch.sum(dim=1).max() <= 1.0
    # and with capacity 1 and 8 tokens choosing 2 experts each, tokens really were dropped
    assert dispatch.sum() < SEQ * 2


def test_the_shapes_do_not_move_between_two_forwards_of_the_same_batch():
    """Rule 24c: a shape that changes with the data recompiles the MPS kernel cache every
    step. Two different batches of one geometry must produce identical shapes."""
    m = _ffn().eval()
    a = m(torch.randn(2, SEQ, HIDDEN))
    b = m(torch.randn(2, SEQ, HIDDEN) * 100)
    assert a.shape == b.shape


def test_the_routing_bias_moves_only_while_training():
    """The load-balance bias is nudged in the training forward and frozen at eval, so an
    evaluation cannot change how the next training step routes."""
    m = _ffn()
    x = torch.randn(2, SEQ, HIDDEN)
    m.eval()
    before = m.route_bias.clone()
    m(x)
    assert torch.equal(m.route_bias, before)
    m.train()
    m(x)
    assert not torch.equal(m.route_bias, before)


def test_the_aux_loss_and_share_are_published_for_the_trainer():
    """The trainer adds _last_aux to its loss and logs _last_share; both are set by a
    forward and the aux term carries no gradient path of its own."""
    m = _ffn()
    m.train()
    m(torch.randn(2, SEQ, HIDDEN))
    assert m._last_aux is not None and m._last_aux.ndim == 0
    assert m._last_share.shape == (m.num_experts,)


def test_l1_capture_is_off_unless_asked():
    """capture_l1 mirrors the dense FFN contract: no tensor is kept when it is off."""
    off, on = _ffn(), _ffn(capture_l1=True)
    x = torch.randn(2, SEQ, HIDDEN)
    off.eval()(x)
    on.eval()(x)
    assert off._last_l1 is None and on._last_l1 is not None


def test_the_probe_pair_points_at_the_shared_expert():
    """rule 11a: dump, diff, pruning and export reach a variant's weights only through
    probe_module / probe_weights, never through .ff.up directly."""
    m = _ffn()
    up, down = m.probe_weights()
    assert m.probe_module() is m.up
    assert up.shape == (FFN_WIDTH // 4, HIDDEN) and down.shape == (HIDDEN, FFN_WIDTH // 4)
