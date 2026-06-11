# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - mem_executor applies a MemoryPlan to a model. These pin the tier->intervention
#   mapping, prove activation checkpointing is numerically transparent (identical
#   logits, loss, and grads), and prove it actually lowers retained activation
#   memory on mps. The unmet-intervention contract is checked so a trainer cannot
#   silently under-deliver an optimizer-offload tier.
# tests/plugin_contract/test_mem_executor.py
# ------------------------------------------------------------------------------------
# Imports

import pytest
import torch

from veritate_core import model as vmodel
from veritate_core.plugin import mem_executor as me
from veritate_core.plugin import mem_planner as mp

# ------------------------------------------------------------------------------------
# Constants

SMALL = dict(vocab=vmodel.VOCAB_BYTE_LEVEL, hidden=128, layers=3, ffn=256,
             heads=4, seq=32)
BATCH = 4
GRAD_TOL = 1e-4

# ------------------------------------------------------------------------------------
# Functions


def _plan(tier):
    return mp.MemoryPlan(tier, True, 0, 0, 0, 0, 0, 0)


def _build():
    torch.manual_seed(0)
    return vmodel.Veritate(**SMALL)


def _batch(device):
    torch.manual_seed(1)
    toks = torch.randint(0, SMALL["vocab"], (BATCH, SMALL["seq"]), device=device)
    tgts = torch.randint(0, SMALL["vocab"], (BATCH, SMALL["seq"]), device=device)
    return toks, tgts


def test_none_tier_applies_nothing():
    """tier none leaves the model unwrapped and reports no unmet work."""
    applied = me.apply_plan(_build(), _plan(mp.TIER_NONE))
    assert applied.grad_checkpoint is False
    assert applied.unmet == ()


def test_checkpoint_tier_enables_checkpointing():
    """tier checkpoint_activations wraps the blocks."""
    model = _build()
    applied = me.apply_plan(model, _plan(mp.TIER_CHECKPOINT))
    assert applied.grad_checkpoint is True
    assert all(getattr(b.forward, "_grad_checkpointed", False) for b in model.blocks)


def test_page_tier_reports_unmet_optimizer_offload():
    """tier page still checkpoints but flags the unwired optimizer paging."""
    applied = me.apply_plan(_build(), _plan(mp.TIER_PAGE))
    assert applied.grad_checkpoint is True
    assert "page_optimizer_to_nvme" in applied.unmet


def test_enable_is_idempotent():
    """Re-applying checkpointing does not nest wrappers."""
    model = _build()
    me.enable_grad_checkpoint(model)
    first = [b.forward for b in model.blocks]
    me.enable_grad_checkpoint(model)
    assert [b.forward for b in model.blocks] == first


def test_checkpointing_is_numerically_transparent():
    """Checkpointed forward/backward yields identical logits, loss, and grads."""
    toks, tgts = _batch("cpu")

    plain = _build()
    plain.train()
    logits_a, loss_a = plain(toks, tgts)
    loss_a.backward()
    grads_a = {n: p.grad.clone() for n, p in plain.named_parameters()}

    ckpt = _build()
    ckpt.train()
    me.apply_plan(ckpt, _plan(mp.TIER_CHECKPOINT))
    logits_b, loss_b = ckpt(toks, tgts)
    loss_b.backward()

    assert torch.allclose(logits_a, logits_b, atol=GRAD_TOL)
    assert torch.allclose(loss_a, loss_b, atol=GRAD_TOL)
    for n, p in ckpt.named_parameters():
        assert torch.allclose(grads_a[n], p.grad, atol=GRAD_TOL), f"grad mismatch {n}"


@pytest.mark.slow
def test_checkpointing_lowers_retained_activation_memory():
    """On mps, checkpointing reduces memory retained after forward (pre-backward)."""
    from veritate_core.plugin import hardware
    if not hardware.mps_supported():
        pytest.skip("needs mps")
    dev = "mps"
    big = dict(vocab=vmodel.VOCAB_BYTE_LEVEL, hidden=512, layers=8, ffn=2048,
               heads=8, seq=256)

    def retained(checkpoint):
        torch.manual_seed(0)
        model = vmodel.Veritate(**big).to(dev)
        model.train()
        if checkpoint:
            me.apply_plan(model, _plan(mp.TIER_CHECKPOINT))
        toks = torch.randint(0, big["vocab"], (8, big["seq"]), device=dev)
        tgts = torch.randint(0, big["vocab"], (8, big["seq"]), device=dev)
        torch.mps.synchronize()
        base = torch.mps.current_allocated_memory()
        _, loss = model(toks, tgts)
        torch.mps.synchronize()
        peak = torch.mps.current_allocated_memory()
        return peak - base

    assert retained(True) < retained(False)
