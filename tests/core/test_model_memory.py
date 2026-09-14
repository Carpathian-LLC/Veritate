# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the surprise-gated memory trunk (Titans MAG class): a parallel branch at mid-depth
#   whose fast weights are written inside the forward pass, so what the model knows
#   changes with use rather than only with corpus gradient descent. What is pinned here is
#   the persistence contract the working-memory program depends on: forward() starts from a
#   blank memory, forward_carry() does not, carry_memory() detaches the state so the graph
#   is not held across windows, and reset_memory() puts it back. Also the read-before-write
#   ordering that keeps the branch causal.
# tests/core/test_model_memory.py
# ------------------------------------------------------------------------------------
# Imports:

import torch

from veritate_core.model_memory import MEM_CHUNK, NeuralMemory, VeritateMemory

# ------------------------------------------------------------------------------------
# Constants

HIDDEN, LAYERS, FFN_WIDTH, HEADS, SEQ = 32, 4, 64, 4, 128
VOCAB = 256

# ------------------------------------------------------------------------------------
# Functions


def _trunk():
    torch.manual_seed(0)
    return VeritateMemory(VOCAB, HIDDEN, LAYERS, FFN_WIDTH, HEADS, SEQ).eval()


def _tokens(n=MEM_CHUNK):
    return torch.randint(0, VOCAB, (1, n))


def test_the_fast_weights_change_during_one_forward():
    """The whole premise: reading a sequence writes memory. Without the inner update the
    branch is a fixed MLP and the trunk is an ordinary dense model."""
    mem = NeuralMemory(HIDDEN)
    mem(torch.randn(1, MEM_CHUNK, HIDDEN))
    w1, _w2, _s1, _s2 = mem.state
    assert not torch.allclose(w1[0], mem.w1_init)


def test_a_chunk_is_read_before_it_is_written():
    """Causality: the output for chunk 1 must not depend on chunk 1's own write, so a
    single-chunk read equals the read from the initial weights."""
    mem = NeuralMemory(HIDDEN).eval()
    x = torch.randn(1, MEM_CHUNK, HIDDEN)
    out = mem(x)
    mem.reset_memory()
    w1 = mem.w1_init.unsqueeze(0)
    w2 = mem.w2_init.unsqueeze(0)
    q = torch.nn.functional.normalize(mem.q_proj(x), dim=-1)
    expected = mem.o_proj(mem._read(w1, w2, q)) * torch.sigmoid(mem.gate(x))
    assert torch.allclose(out, expected, atol=1e-6)


def test_forward_starts_from_a_blank_memory_and_forward_carry_does_not():
    """Two calls of forward() on the same tokens agree; forward_carry() sees what the
    previous window wrote, which is how a document is fed in windows and quizzed later."""
    m = _trunk()
    toks = _tokens()
    a, _ = m(toks)
    b, _ = m(toks)
    assert torch.allclose(a, b, atol=1e-6)
    m.reset_memory()
    c, _ = m.forward_carry(toks)
    d, _ = m.forward_carry(toks)
    assert not torch.allclose(c, d, atol=1e-6)


def test_reset_memory_returns_the_trunk_to_its_blank_state():
    m = _trunk()
    toks = _tokens()
    first, _ = m.forward_carry(toks)
    m.reset_memory()
    again, _ = m.forward_carry(toks)
    assert torch.allclose(first, again, atol=1e-6)


def test_carry_memory_detaches_the_state():
    """Carrying the graph across windows would keep every previous window's activations
    alive; carry_memory() keeps the values and drops the history."""
    m = _trunk()
    m.train()
    m.forward_carry(_tokens())
    assert m.memory.state[0].requires_grad
    m.carry_memory()
    assert not any(s.requires_grad for s in m.memory.state)


def test_a_sequence_shorter_than_a_chunk_still_writes():
    """T is padded up to the chunk, and the output is trimmed back to T."""
    mem = NeuralMemory(HIDDEN)
    short = MEM_CHUNK // 2
    out = mem(torch.randn(1, short, HIDDEN))
    assert out.shape == (1, short, HIDDEN)
    assert not torch.allclose(mem.state[0][0], mem.w1_init)


def test_the_write_rule_is_trained_by_the_outer_optimizer():
    """TTT-style: the inner update stays inside the autograd graph, so the learning rate,
    momentum and decay of the memory are themselves learnable. It takes two chunks: the
    gradient exists only where a later chunk reads what an earlier one wrote."""
    m = _trunk()
    m.train()
    _logits, loss = m(_tokens(2 * MEM_CHUNK), _tokens(2 * MEM_CHUNK))
    loss.backward()
    assert m.memory.lr.grad is not None and m.memory.lr.grad != 0
    assert m.memory.decay.grad is not None and m.memory.decay.grad != 0


def test_one_chunk_alone_gives_the_write_rule_no_gradient():
    """The corollary of read-before-write: with a single chunk nothing reads the write, so
    the learning rate, momentum and decay cannot move. A run whose windows are one chunk
    long trains the memory projections and never the rule that writes them."""
    m = _trunk()
    m.train()
    _logits, loss = m(_tokens(MEM_CHUNK), _tokens(MEM_CHUNK))
    loss.backward()
    assert m.memory.lr.grad is None
    assert m.memory.q_proj.weight.grad is not None


def test_it_declares_no_multi_byte_decode():
    """The research trunks do not carry multi-byte heads; the loader and the decode paths
    read this rather than probing the state dict."""
    assert _trunk().supports_mtp_decode() is False
