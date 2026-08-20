# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - regression for the delta-path decay underflow: the state update used
#   exp(a_last)/exp(a_t), and once a chunk's decay cumsum passes ~-88 both exps
#   underflow to 0 in fp32, making 0/0 = nan. Trained models reach such decays
#   routinely (killed the 10M delta run at step 2094 and wren1_4 attempt 1 at
#   step 1); random init rarely does, so this test FORCES strong decay through
#   a_proj.bias and asserts the forward and carried state stay finite. The gla
#   path always used the safe exp(a_last - a_t) form and is asserted alongside.
# tests/training/test_delta_underflow.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
import torch

from veritate_core.model_recurrent import CHUNK, RecurrentMixer

# ------------------------------------------------------------------------------------
# Constants

HID, HEADS = 64, 4
STRONG_DECAY_BIAS = 8.0   # la ~ -softplus(8) ~ -8/token -> cumsum -512/chunk

# ------------------------------------------------------------------------------------
# Functions


def _mixer(rule):
    torch.manual_seed(0)
    m = RecurrentMixer(HID, HEADS, state_rule=rule).eval()
    with torch.no_grad():
        m.a_proj.bias.fill_(STRONG_DECAY_BIAS)
    return m


@pytest.mark.parametrize("rule", ["delta", "gla", "pinned"])
def test_strong_decay_stays_finite(rule):
    m = _mixer(rule)
    x = torch.randn(2, 2 * CHUNK, HID)
    with torch.no_grad():
        o, st = m(x, return_state=True)
    assert torch.isfinite(o).all()
    assert torch.isfinite(st["s"]).all()


def test_delta_strong_decay_state_carry_finite():
    """The nan surfaced on the SECOND window (carried state feeds kc @ state):
    walk two windows with carry and assert everything stays finite."""
    m = _mixer("delta")
    with torch.no_grad():
        _, st = m(torch.randn(1, 2 * CHUNK, HID), return_state=True)
        o2, st2 = m(torch.randn(1, 2 * CHUNK, HID), state=st, return_state=True)
    assert torch.isfinite(o2).all()
    assert torch.isfinite(st2["s"]).all()


@pytest.mark.parametrize("rule", ["delta", "gla", "pinned"])
def test_strong_decay_backward_finite(rule):
    """A finite forward is not enough: exp(+large) computed pre-mask is saved
    for backward and turns masked-away entries into 0 * inf = nan GRADS
    (wren1_4 attempt 2 skipped all 1,000 steps on grad norm). Assert the
    gradients themselves stay finite under strong decay."""
    m = _mixer(rule).train()
    x = torch.randn(2, 2 * CHUNK, HID, requires_grad=True)
    o = m(x)
    o.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
