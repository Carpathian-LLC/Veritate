# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - bench._step must be shaped like the real training step, not a proxy of it.
#   Two regressions this pins, both of which mis-sized batch in production:
#     1. no autocast -> the probe measured fp32 while training runs bf16
#     2. one seq window -> the probe missed that bptt_window chunk losses sit in
#        the graph at once, under-reporting peak activation memory
# - Uses a tiny stub model, so this asserts the SHAPE of the probe (how many
#   forwards, how many backwards, what dtype context) without needing a GPU.
# tests/training/test_bench_step_shape.py
# ------------------------------------------------------------------------------------
# Imports:

import torch

from veritate_core.plugin import bench

# ------------------------------------------------------------------------------------
# Constants

SEQ   = 8
VOCAB = 16
BATCH = 2

# ------------------------------------------------------------------------------------
# Functions


class _SpyModel(torch.nn.Module):
    """Records every forward's window width and the autocast dtype in force."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(VOCAB, VOCAB)
        self.widths = []
        self.autocast_dtypes = []

    def forward(self, toks, targets=None):
        self.widths.append(toks.size(1))
        self.autocast_dtypes.append(
            torch.get_autocast_dtype("cpu") if torch.is_autocast_enabled("cpu") else None)
        onehot = torch.nn.functional.one_hot(toks, VOCAB).float()
        logits = self.lin(onehot)
        loss = logits.float().mean()
        return logits, loss


def _opt(model):
    return torch.optim.SGD(model.parameters(), lr=0.0)


def _count_backwards(monkeypatch):
    """Count real .backward() invocations. Counting per-loss autograd hooks
    instead would count losses reduced, not backward passes: one backward over a
    window of 2 fires 2 hooks."""
    calls = []
    real = torch.Tensor.backward

    def spy(self, *a, **kw):
        calls.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    return calls


def test_probe_walks_every_chunk_of_the_step():
    """Per-step compute scales with n_chunks: 4 chunks means 4 forwards."""
    m = _SpyModel()
    bench._step(m, _opt(m), BATCH, SEQ, VOCAB, "cpu", n_chunks=4, bptt_window=2)
    assert len(m.widths) == 4
    assert set(m.widths) == {SEQ}


def test_probe_holds_bptt_window_losses_before_each_backward(monkeypatch):
    """Peak activation memory scales with bptt_window. With 4 chunks and a
    window of 2 the probe must backward twice, not four times, or it measures a
    memory peak the real run never has."""
    m = _SpyModel()
    calls = _count_backwards(monkeypatch)
    bench._step(m, _opt(m), BATCH, SEQ, VOCAB, "cpu", n_chunks=4, bptt_window=2)
    assert len(calls) == 2


def test_a_window_wider_than_the_step_backwards_once(monkeypatch):
    m = _SpyModel()
    calls = _count_backwards(monkeypatch)
    bench._step(m, _opt(m), BATCH, SEQ, VOCAB, "cpu", n_chunks=4, bptt_window=99)
    assert len(calls) == 1


def test_single_chunk_is_one_forward_one_backward(monkeypatch):
    m = _SpyModel()
    calls = _count_backwards(monkeypatch)
    bench._step(m, _opt(m), BATCH, SEQ, VOCAB, "cpu", n_chunks=1, bptt_window=1)
    assert len(m.widths) == 1
    assert len(calls) == 1


def test_probe_runs_under_autocast_when_a_dtype_is_given():
    """The probe must measure the precision the trainer trains at. Measuring
    fp32 while training runs bf16 made a GPU look 2.4x slower than it was."""
    m = _SpyModel()
    bench._step(m, _opt(m), BATCH, SEQ, VOCAB, "cpu", amp_dtype=torch.bfloat16,
                n_chunks=2, bptt_window=1)
    assert m.autocast_dtypes == [torch.bfloat16, torch.bfloat16]


def test_probe_runs_eager_when_no_dtype_is_given():
    m = _SpyModel()
    bench._step(m, _opt(m), BATCH, SEQ, VOCAB, "cpu", amp_dtype=None, n_chunks=2)
    assert m.autocast_dtypes == [None, None]


def test_throughput_counts_every_token_the_step_consumed():
    """tok/s must be over batch * seq * n_chunks. Counting one chunk's worth
    understates the step and inflates nothing, but it makes the number
    incomparable to the trainer's own tok/s line."""
    m = _SpyModel()
    _mem, tok_per_s = bench._measure_batch(m, _opt(m), BATCH, SEQ, VOCAB, "cpu",
                                           n_chunks=4, bptt_window=2)
    m2 = _SpyModel()
    _mem2, tok_per_s_single = bench._measure_batch(m2, _opt(m2), BATCH, SEQ, VOCAB, "cpu",
                                                   n_chunks=1, bptt_window=1)
    assert tok_per_s > 0 and tok_per_s_single > 0
    # 4 chunks per step does ~4x the work AND counts ~4x the tokens, so the
    # rate stays the same order rather than scaling with n_chunks.
    assert 0.25 < (tok_per_s / tok_per_s_single) < 4.0
