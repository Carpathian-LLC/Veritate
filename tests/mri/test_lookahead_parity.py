# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - end-to-end byte-parity test for the pytorch lookahead fast path. Builds a tiny
#   seeded canonical Veritate on CPU and asserts Brain.stream_fast(mode="lookahead")
#   emits the exact byte sequence single-byte Brain.stream() does under greedy, on a
#   prompt engineered to trigger real draft acceptance (so parity is not vacuous). A
#   broken accept-loop boundary or window mismatch flips the sampled bytes and fails
#   the parity assert. Also pins the C-path core fields on the fast_byte event and the
#   off-CPU drafting gate. CPU-only, seeded, tmp checkpoint; never MPS (the live
#   training run owns the GPU).
# tests/mri/test_lookahead_parity.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
import torch
from inference.backends import pytorch as pytorch_backend
from inference.backends.pytorch import LOOKAHEAD_BACKEND, Brain

from veritate_core.model import Veritate

# ------------------------------------------------------------------------------------
# Constants

INFER_DEVICE_ENV = "VERITATE_INFER_DEVICE"
VOCAB, HIDDEN, LAYERS, FFN, HEADS, SEQ = 256, 32, 2, 64, 4, 128
MODEL_SEED = 0
GEN_SEED   = 0
STEP       = 1
# Repeated substring so _ngram_draft fires; near-greedy temperature collapses the
# untrained model into a loop so drafts are accepted (parity is exercised, not vacuous).
PROMPT      = "abcdefgh abcdefgh abcdefgh abcdefgh "
TEMPERATURE = 0.02
TOP_K       = 40
# len(PROMPT) + MAX_NEW stays under SEQ - 1 so the window never slides: both paths keep
# 0-based positions over the same bytes, so CPU logits are bitwise-identical and parity
# is exact (byte-exactness holds on CPU, not guaranteed on MPS: see inference_brain.md).
MAX_NEW     = 64
DECODE_KW   = {"temperature": TEMPERATURE, "top_k_sample": TOP_K, "max_new": MAX_NEW,
               "rep_window": 0, "rep_penalty": 0.0, "no_repeat_ngram": 0}

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture(scope="module")
def brain(tmp_path_factory):
    """A tiny seeded canonical Veritate Brain forced onto CPU."""
    prev = os.environ.get(INFER_DEVICE_ENV)
    os.environ[INFER_DEVICE_ENV] = "cpu"
    try:
        torch.manual_seed(MODEL_SEED)
        model = Veritate(VOCAB, HIDDEN, LAYERS, FFN, HEADS, SEQ)
        model.eval()
        ckdir = tmp_path_factory.mktemp("model") / "checkpoints"
        ckdir.mkdir()
        ckpt = ckdir / f"step_{STEP}.pt"
        torch.save({"model": model.state_dict(), "args": {"heads": HEADS}}, str(ckpt))
        b = Brain(str(ckpt), threads=1)
        assert b.device.type == "cpu"   # never run on the training-owned GPU
        yield b
    finally:
        if prev is None:
            os.environ.pop(INFER_DEVICE_ENV, None)
        else:
            os.environ[INFER_DEVICE_ENV] = prev


def _bytes(events, kind):
    return [int(ev["byte"]) for ev in events if ev.get("kind") == kind]


def _stream_events(brain):
    torch.manual_seed(GEN_SEED)
    return list(brain.stream(PROMPT, **DECODE_KW))


def _lookahead_events(brain):
    torch.manual_seed(GEN_SEED)
    return list(brain.stream_fast(PROMPT, mode="lookahead", **DECODE_KW))


def test_lookahead_matches_stream_byte_for_byte(brain):
    """Greedy lookahead decode emits the exact byte sequence single-byte stream() does."""
    assert _bytes(_lookahead_events(brain), "fast_byte") == _bytes(_stream_events(brain), "token")


def test_lookahead_actually_accepts_drafts(brain):
    """The parity prompt triggers real draft acceptance, so the parity check is not vacuous."""
    fb = [ev for ev in _lookahead_events(brain) if ev.get("kind") == "fast_byte"]
    assert max(ev["accepted_extra_so_far"] for ev in fb) > 0


def test_lookahead_fast_byte_carries_c_core_fields(brain):
    """A lookahead fast_byte event carries the C path's core fields, backend-tagged."""
    ev0 = next(ev for ev in _lookahead_events(brain) if ev.get("kind") == "fast_byte")
    assert {"argmax_byte", "T", "backend"} <= set(ev0)
    assert ev0["backend"] == LOOKAHEAD_BACKEND
    assert ev0["T"] == len(PROMPT)   # first emitted byte's context is the prompt


def test_lookahead_argmax_byte_matches_stream(brain):
    """Per-byte argmax_byte on the fast path equals the full-telemetry stream()'s."""
    s = [int(ev["argmax_byte"]) for ev in _stream_events(brain) if ev.get("kind") == "token"]
    f = [int(ev["argmax_byte"]) for ev in _lookahead_events(brain) if ev.get("kind") == "fast_byte"]
    assert f == s


def test_drafting_disabled_off_cpu_keeps_single_byte_windows(brain, monkeypatch):
    """With the device off LOOKAHEAD_DEVICES, drafting is skipped (draft_len 0) yet output
    stays byte-identical to stream()."""
    monkeypatch.setattr(pytorch_backend, "LOOKAHEAD_DEVICES", ())
    torch.manual_seed(GEN_SEED)
    fb = [ev for ev in brain.stream_fast(PROMPT, mode="lookahead", **DECODE_KW)
          if ev.get("kind") == "fast_byte"]
    assert all(ev["draft_len"] == 0 for ev in fb)
    assert [ev["byte"] for ev in fb] == _bytes(_stream_events(brain), "token")
