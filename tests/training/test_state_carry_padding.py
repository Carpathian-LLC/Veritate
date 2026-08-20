# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Padded slots used to write garbage into the carried recurrent state and, under
#   state-carried backward, received unbounded gradients that the slot-mask multiply
#   turned into nan (inf * 0), killing every training step (wren1_3, 2026-08-18).
#   Fix: torch.where for the slot mask (exact-zero gradient at padding) and k/v/la
#   masked out of the state stream inside the mixer. These tests pin the contract:
#   padding writes nothing, decays nothing, and in-window outputs are unchanged.
# tests/training/test_state_carry_padding.py
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
from veritate_core.model_recurrent import CHUNK, RecurrentMixer

# ------------------------------------------------------------------------------------
# Constants

SEQ = 256
HID = 64

# ------------------------------------------------------------------------------------
# Functions


def _model():
    torch.manual_seed(7)
    return VeritatePatched(vocab=256, hidden=HID, layers=2, ffn=128, heads=4, seq=SEQ,
                           global_mixer="recurrent").eval()


def test_fully_masked_window_is_a_state_noop():
    """A window of only padded slots must pass the carried state through unchanged:
    zero writes and identity decay."""
    torch.manual_seed(1)
    mixer = RecurrentMixer(HID, 4)
    x_live = torch.randn(1, CHUNK, HID)
    live_mask = torch.ones(1, CHUNK, 1, dtype=torch.bool)
    _, state = mixer(x_live, return_state=True, slot_mask=live_mask)
    x_pad = torch.randn(1, CHUNK, HID)
    none_mask = torch.zeros(1, CHUNK, 1, dtype=torch.bool)
    _, state2 = mixer(x_pad, state=state, return_state=True, slot_mask=none_mask)
    assert torch.allclose(state["s"], state2["s"], atol=1e-6)


def test_carried_state_ignores_padded_slot_content():
    """The state written by a window must not depend on what sits in its padded
    slots."""
    torch.manual_seed(2)
    mixer = RecurrentMixer(HID, 4)
    half = torch.ones(1, CHUNK, 1, dtype=torch.bool)
    half[:, CHUNK // 2:] = False
    xa = torch.randn(1, CHUNK, HID)
    xb = xa.clone()
    xb[:, CHUNK // 2:] = torch.randn(1, CHUNK // 2, HID)
    _, sa = mixer(xa, return_state=True, slot_mask=half)
    _, sb = mixer(xb, return_state=True, slot_mask=half)
    assert torch.allclose(sa["s"], sb["s"], atol=1e-6)


def test_streaming_window_matches_training_forward():
    """With no incoming state, forward_streaming must produce the training
    forward's logits exactly: the padding fix may not touch valid positions."""
    m = _model()
    text = b"The quick brown fox jumps over the lazy dog. " * 4
    toks = torch.tensor([list(text)], dtype=torch.long)
    with torch.no_grad():
        stream_logits, _ = m.forward_streaming(toks, None)
        train_logits, _ = m.forward(toks)
    assert torch.equal(stream_logits, train_logits)


def test_carried_backward_gradients_are_finite():
    """Backward through two state-carried windows with padded slots stays finite."""
    m = _model().train()
    text = b"word " * 100
    toks = torch.tensor([list(text[:SEQ])], dtype=torch.long)
    logits, states = m.forward_streaming(toks, None)
    logits2, _ = m.forward_streaming(toks, states)
    (logits.sum() + logits2.sum()).backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
