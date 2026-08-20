# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - state_carry=chunks calls blocks with state=/return_state= kwargs through
#   forward_streaming; the activation-checkpoint wrappers used to drop kwargs and
#   crash the launch (wren1_3, 2026-08-18). These tests pin kwargs pass-through on
#   both wrapper sites and that gradients still flow through a wrapped, state-carried
#   window walk.
# tests/training/test_act_ckpt_state_carry.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from veritate_core.model_patched import VeritatePatched
from veritate_core.plugin import mem_executor

# ------------------------------------------------------------------------------------
# Constants

SEQ = 256   # 64 slots at PATCH_STRIDE 4: one recurrent CHUNK, smallest legal window


def _tiny():
    torch.manual_seed(0)
    return VeritatePatched(vocab=256, hidden=32, layers=2, ffn=64, heads=2, seq=SEQ,
                           global_mixer="recurrent")

# ------------------------------------------------------------------------------------
# Functions


def test_mem_executor_wrapper_passes_state_kwargs():
    """enable_grad_checkpoint must not strip state=/return_state= from block calls."""
    m = _tiny()
    mem_executor.enable_grad_checkpoint(m)
    toks = torch.randint(0, 256, (1, SEQ))
    logits, states = m.forward_streaming(toks, None)
    logits2, _ = m.forward_streaming(toks, states)
    assert logits.shape == logits2.shape == (1, SEQ, 256)


def test_trainer_style_wrapper_passes_state_kwargs():
    """The use_act_ckpt lambda in the trainer composes with forward_streaming."""
    m = _tiny()
    for blk in m.blocks:
        blk.forward = (lambda fwd: lambda x, **kw: torch.utils.checkpoint.checkpoint(
            fwd, x, use_reentrant=False, **kw))(blk.forward)
    toks = torch.randint(0, 256, (1, SEQ))
    logits, states = m.forward_streaming(toks, None)
    assert logits.shape == (1, SEQ, 256)
    assert states and all(isinstance(s, dict) for s in states)


def test_gradients_flow_through_wrapped_state_carry():
    """A state-carried window walk under checkpointing still backprops to weights."""
    m = _tiny()
    mem_executor.enable_grad_checkpoint(m)
    toks = torch.randint(0, 256, (1, SEQ))
    _, states = m.forward_streaming(toks, None)
    logits, _ = m.forward_streaming(toks, states)
    logits.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
