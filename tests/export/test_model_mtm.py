# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Construction + contract tests for the multimind (MtM) Veritate variant.
# tests/export/test_model_mtm.py
# ------------------------------------------------------------------------------------
# Imports

import torch

from veritate_core.model_mtm import VeritateMultimind

# ------------------------------------------------------------------------------------
# Constants

SEED         = 1234
TINY_HIDDEN  = 32
TINY_LAYERS  = 2
TINY_FFN     = 64
TINY_HEADS   = 4
TINY_SEQ     = 64
BATCH        = 2
TOKENS       = 8
VOCAB        = 256

# ------------------------------------------------------------------------------------
# Functions


def _build(bias_mode=False):
    torch.manual_seed(SEED)
    return VeritateMultimind(hidden=TINY_HIDDEN, layers=TINY_LAYERS, ffn=TINY_FFN,
                             heads=TINY_HEADS, seq=TINY_SEQ, bias_mode=bias_mode)


def _tokens():
    torch.manual_seed(SEED)
    return torch.randint(0, VOCAB, (BATCH, TOKENS))


def test_constructs_minimal_mtm():
    """Tiny MtM constructs with non-zero parameter count."""
    m = _build()
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params > 0


def test_forward_matches_dense_shape():
    """Forward returns logits of shape (B, T, 256)."""
    m = _build()
    logits, loss = m(_tokens())
    assert logits.shape == (BATCH, TOKENS, VOCAB)
    assert loss is None


def test_bias_mode_changes_routing():
    """Sentiment-bias changes router gate weights vs neutral input."""
    m = _build(bias_mode=True)
    with torch.no_grad():
        m.gate_g.copy_(torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25, -0.25]))
    toks = _tokens()
    sent_zero = torch.zeros(BATCH)
    sent_pos  = torch.ones(BATCH)
    m(toks, sentiment=sent_zero)
    gates_zero = m.blocks[0].ff._last_gates.clone()
    m(toks, sentiment=sent_pos)
    gates_pos = m.blocks[0].ff._last_gates.clone()
    assert not torch.allclose(gates_zero, gates_pos)


def test_hook_spec_exposes_canonical_shape():
    """hook_spec().blocks[0].ff.up output has dense FFN width."""
    m = _build()
    spec = m.hook_spec()
    x = torch.randn(BATCH, TOKENS, TINY_HIDDEN)
    up_out = spec.blocks[0].ff.up(x)
    assert up_out.shape == (BATCH, TOKENS, TINY_FFN)
