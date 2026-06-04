# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the MultiMind plugin: attach, idempotence,
#   incompatibility rejection, and sleep adapter holdout sanity.
# tests/export/test_multimind_plugin.py
# ------------------------------------------------------------------------------------
# Imports

import math

import pytest
import torch
import torch.nn as nn

from veritate_core.model import Veritate
from veritate_core.model_mtm import VeritateMultimind
from veritate_core.multimind import MultiMindPlugin

# ------------------------------------------------------------------------------------
# Constants

SEED         = 4242
TINY_HIDDEN  = 32
TINY_LAYERS  = 2
TINY_FFN     = 64
TINY_HEADS   = 4
TINY_SEQ     = 32
BATCH        = 2
TOKENS       = 8
VOCAB        = 256
SLEEP_RANK   = 4
SLEEP_LR     = 1e-3
SLEEP_STEPS  = 3
HOLDOUT_MAX  = 0.20

# ------------------------------------------------------------------------------------
# Functions


class _DummyProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def valence(self, tokens):
        return torch.tanh(tokens.float().mean(dim=-1) / 128.0 - 1.0) * self.w


def _build_mtm():
    torch.manual_seed(SEED)
    return VeritateMultimind(hidden=TINY_HIDDEN, layers=TINY_LAYERS, ffn=TINY_FFN,
                             heads=TINY_HEADS, seq=TINY_SEQ, bias_mode=True)


def _tokens(offset=0):
    torch.manual_seed(SEED + offset)
    return torch.randint(0, VOCAB, (BATCH, TOKENS))


def _save_probe(tmp_path):
    torch.manual_seed(SEED)
    probe = _DummyProbe()
    p = tmp_path / "probe.pt"
    torch.save(probe, p)
    return str(p)


def test_plugin_attaches_to_mtm_model(tmp_path):
    """Attach installs a provider that changes routing vs no-plugin baseline."""
    m = _build_mtm()
    with torch.no_grad():
        m.gate_g.copy_(torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25, -0.25]))
    toks = _tokens()
    m(toks)
    base = m.blocks[0].ff._last_gates.clone()
    plugin = MultiMindPlugin()
    plugin.attach(m, _save_probe(tmp_path))
    m(toks)
    biased = m.blocks[0].ff._last_gates.clone()
    assert not torch.allclose(base, biased)


def test_plugin_attach_idempotent(tmp_path):
    """Two attach calls produce identical routing to a single attach."""
    probe_path = _save_probe(tmp_path)
    m1 = _build_mtm()
    p1 = MultiMindPlugin()
    p1.attach(m1, probe_path)
    toks = _tokens()
    m1(toks)
    once = m1.blocks[0].ff._last_gates.clone()
    m2 = _build_mtm()
    p2 = MultiMindPlugin()
    p2.attach(m2, probe_path)
    p2.attach(m2, probe_path)
    m2(toks)
    twice = m2.blocks[0].ff._last_gates.clone()
    assert torch.allclose(once, twice)


def test_plugin_refuses_incompatible_model(tmp_path):
    """Attach on a canonical Veritate raises TypeError with a clear message."""
    torch.manual_seed(SEED)
    canonical = Veritate(vocab=VOCAB, hidden=TINY_HIDDEN, layers=TINY_LAYERS,
                         ffn=TINY_FFN, heads=TINY_HEADS, seq=TINY_SEQ)
    plugin = MultiMindPlugin()
    with pytest.raises(TypeError, match="multimind-compatible"):
        plugin.attach(canonical, _save_probe(tmp_path))


def test_sleep_lora_preserves_holdout(tmp_path):
    """Sleep on a tiny buffer keeps holdout ppl rise under the sanity bound."""
    torch.manual_seed(SEED)
    m = _build_mtm()
    plugin = MultiMindPlugin()
    plugin.attach(m, _save_probe(tmp_path))
    train_inp = torch.randint(0, VOCAB, (BATCH, TOKENS))
    train_tgt = torch.randint(0, VOCAB, (BATCH, TOKENS))
    hold_inp  = torch.randint(0, VOCAB, (BATCH, TOKENS))
    hold_tgt  = torch.randint(0, VOCAB, (BATCH, TOKENS))
    with torch.no_grad():
        _, pre_loss = m(hold_inp, targets=hold_tgt)
    pre_ppl = math.exp(float(pre_loss))
    plugin.sleep(m, [(train_inp, train_tgt)], lr=SLEEP_LR, steps=SLEEP_STEPS,
                 rank=SLEEP_RANK, save_dir=str(tmp_path / "ckpt"))
    with torch.no_grad():
        _, post_loss = m(hold_inp, targets=hold_tgt)
    post_ppl = math.exp(float(post_loss))
    rise = (post_ppl - pre_ppl) / max(pre_ppl, 1e-9)
    assert rise < HOLDOUT_MAX
