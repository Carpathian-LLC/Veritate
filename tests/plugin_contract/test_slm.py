# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - selective language modeling (the trainer's slm_ref lever): the student's loss is
#   averaged over the keep_frac share of tokens where it most exceeds a frozen reference
#   and the rest are masked out; the reference is rebuilt from its newest checkpoint's
#   own shape args and frozen.
# tests/plugin_contract/test_slm.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
import torch

from veritate_core.model import VOCAB_BYTE_LEVEL, Veritate
from veritate_core.plugin import slm

# ------------------------------------------------------------------------------------
# Constants

SHAPE = {"hidden": 16, "layers": 1, "ffn": 32, "heads": 2, "seq": 8}

# ------------------------------------------------------------------------------------
# Functions


class _Ref(torch.nn.Module):
    """A reference whose loss is 0 at even positions and huge at odd ones, so the
    student's excess is largest at even positions."""
    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def forward(self, tokens):
        B, T = tokens.shape
        logits = torch.zeros(B, T, VOCAB_BYTE_LEVEL)
        flat = logits.view(-1, VOCAB_BYTE_LEVEL)
        for i, t in enumerate(self.targets.reshape(-1).tolist()):
            if i % 2 == 0:
                flat[i, t] = 50.0          # the reference nails even positions
        return logits, None


def test_only_the_tokens_the_student_finds_hardest_relative_to_the_reference_count():
    torch.manual_seed(0)
    targets = torch.randint(0, VOCAB_BYTE_LEVEL, (1, 8))
    logits = torch.zeros(1, 8, VOCAB_BYTE_LEVEL, requires_grad=True)   # uniform student: same CE everywhere
    loss = slm.selective_loss(_Ref(targets), targets, targets, logits, keep_frac=0.5)
    loss.backward()
    grad = logits.grad[0].abs().sum(-1)
    assert (grad[0::2] > 0).all() and (grad[1::2] == 0).all()
    assert torch.isclose(loss, torch.tensor(float(torch.log(torch.tensor(float(VOCAB_BYTE_LEVEL))))))


def test_keep_frac_one_is_the_plain_mean_and_at_least_one_token_is_always_kept():
    torch.manual_seed(1)
    targets = torch.randint(0, VOCAB_BYTE_LEVEL, (2, 4))
    logits = torch.randn(2, 4, VOCAB_BYTE_LEVEL)
    plain = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_BYTE_LEVEL), targets.reshape(-1))
    assert torch.isclose(slm.selective_loss(_Ref(targets), targets, targets, logits, keep_frac=1.0), plain)
    assert torch.isfinite(slm.selective_loss(_Ref(targets), targets, targets, logits, keep_frac=0.0))


def test_the_reference_is_rebuilt_from_its_newest_checkpoint_and_frozen(tmp_path):
    torch.manual_seed(0)
    src = Veritate(vocab=VOCAB_BYTE_LEVEL, **SHAPE)
    os.makedirs(tmp_path / "checkpoints")
    for step in (3, 12, 7):
        torch.save({"model": src.state_dict(), "args": {**SHAPE, "step": step}},
                   tmp_path / "checkpoints" / f"step_{step}.pt")
    assert slm.latest_checkpoint(str(tmp_path)).endswith("step_12.pt")
    ref = slm.load_reference(str(tmp_path), "cpu")
    assert not ref.training and all(not p.requires_grad for p in ref.parameters())
    toks = torch.randint(0, VOCAB_BYTE_LEVEL, (1, 8))
    assert torch.equal(ref(toks)[0], src.eval()(toks)[0])
    with pytest.raises(FileNotFoundError):
        slm.latest_checkpoint(str(tmp_path / "empty"))
