# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers Brain.stream_fast(mode="stream"), the unbounded-context generation loop
#   over forward_streaming (IDEA 7/20 E3). Pins: (1) argmax parity with stream()
#   while total context stays inside one window, since both paths must produce the
#   same logits for the same bytes; (2) generation continues past seq total bytes
#   with windows committing into the carried state (the property no other fast mode
#   has — kv stops at "cache full"); (3) prompts longer than seq are consumed whole,
#   never truncated; (4) the carried state is real: a committed window's state
#   changes the next window's logits vs a state reset; (5) non-recurrent models get
#   a clean error event. Tiny seeded hybrid model, CPU-only (the training run owns
#   the GPU).
# tests/mri/test_stream_fast_streaming.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
import torch
from inference.backends.pytorch import Brain

from veritate_core.model import Veritate
from veritate_core.model_patched import VeritatePatched

# ------------------------------------------------------------------------------------
# Constants

INFER_DEVICE_ENV = "VERITATE_INFER_DEVICE"
# SEQ 256 is the smallest streaming-legal window: slots = seq/4 must be a
# multiple of the recurrent kernel's CHUNK (64) for state carry.
VOCAB, HIDDEN, LAYERS, FFN, HEADS, SEQ = 256, 32, 2, 64, 4, 256
MODEL_SEED = 0
GEN_SEED   = 0
STEP       = 1
PROMPT     = "the quick brown fox jumps over the lazy dog. "
# top_k 1 makes both paths argmax decoders: parity asserts logits alignment without
# depending on multinomial tie-breaking over near-equal probabilities.
DECODE_KW  = {"temperature": 0.7, "top_k_sample": 1,
              "rep_window": 0, "rep_penalty": 0.0, "no_repeat_ngram": 0}

# ------------------------------------------------------------------------------------
# Functions


def _make_brain(tmp_path_factory, model, extra_args=None):
    prev = os.environ.get(INFER_DEVICE_ENV)
    os.environ[INFER_DEVICE_ENV] = "cpu"
    try:
        model.eval()
        ckdir = tmp_path_factory.mktemp("model") / "checkpoints"
        ckdir.mkdir()
        ckpt = ckdir / f"step_{STEP}.pt"
        args = {"heads": HEADS}
        args.update(extra_args or {})
        torch.save({"model": model.state_dict(), "args": args}, str(ckpt))
        b = Brain(str(ckpt), threads=1)
        assert b.device.type == "cpu"   # never run on the training-owned GPU
        return b, prev
    except BaseException:
        if prev is None:
            os.environ.pop(INFER_DEVICE_ENV, None)
        else:
            os.environ[INFER_DEVICE_ENV] = prev
        raise


@pytest.fixture(scope="module")
def brain(tmp_path_factory):
    """A tiny seeded hybrid (recurrent-mixer) VeritatePatched Brain on CPU."""
    torch.manual_seed(MODEL_SEED)
    model = VeritatePatched(vocab=VOCAB, hidden=HIDDEN, layers=LAYERS, ffn=FFN,
                            heads=HEADS, seq=SEQ, activation="gelu",
                            global_mixer="recurrent")
    b, prev = _make_brain(tmp_path_factory, model,
                          {"trunk": "hybrid", "activation": "gelu"})
    yield b
    if prev is None:
        os.environ.pop(INFER_DEVICE_ENV, None)
    else:
        os.environ[INFER_DEVICE_ENV] = prev


def _bytes(events, kind):
    return [int(ev["byte"]) for ev in events if ev.get("kind") == kind]


def test_stream_mode_matches_stream_inside_one_window(brain):
    """While total context < seq (no commit), the streaming walk must argmax-decode
    the same bytes as the full-forward stream() path."""
    n = SEQ - len(PROMPT.encode("utf-8")) - 2
    torch.manual_seed(GEN_SEED)
    slow = _bytes(brain.stream(PROMPT, max_new=n, **DECODE_KW), "token")
    torch.manual_seed(GEN_SEED)
    fast = _bytes(brain.stream_fast(PROMPT, mode="stream", max_new=n, **DECODE_KW), "fast_byte")
    assert fast == slow


def test_stream_mode_generates_past_seq(brain):
    """Generation crosses the window boundary, commits state, and keeps going —
    total context exceeds seq, which no other fast mode can do."""
    torch.manual_seed(GEN_SEED)
    evs = list(brain.stream_fast(PROMPT, mode="stream", max_new=SEQ + 40, **DECODE_KW))
    fb = [ev for ev in evs if ev.get("kind") == "fast_byte"]
    assert len(fb) == SEQ + 40
    assert fb[-1]["T"] > SEQ
    assert fb[-1]["windows_committed"] >= 1
    commits = {ev["windows_committed"] for ev in fb}
    assert 0 in commits and 1 in commits   # boundary actually crossed mid-stream


def test_stream_mode_consumes_full_long_prompt(brain):
    """A prompt longer than seq is walked window-by-window, never truncated."""
    long_prompt = PROMPT * ((3 * SEQ) // len(PROMPT) + 1)
    n_bytes = len(long_prompt.encode("utf-8"))
    torch.manual_seed(GEN_SEED)
    evs = list(brain.stream_fast(long_prompt, mode="stream", max_new=4, **DECODE_KW))
    meta = next(ev for ev in evs if ev.get("kind") == "meta")
    assert len(meta["prompt_bytes"]) == n_bytes
    pre = next(ev for ev in evs if ev.get("kind") == "prefill")
    assert pre["tokens"] == n_bytes
    assert pre["windows_committed"] == n_bytes // SEQ >= 3
    assert len(_bytes(evs, "fast_byte")) == 4


def test_committed_state_changes_next_window_logits(brain):
    """The carry is real: logits over the same bytes differ with the previous
    window's state loaded vs reset (an untrained mixer does not decay to zero)."""
    m = brain.model
    torch.manual_seed(0)
    w1 = torch.randint(0, VOCAB, (1, SEQ), dtype=torch.long)
    w2 = torch.randint(0, VOCAB, (1, SEQ // 2), dtype=torch.long)
    with torch.no_grad():
        _, states = m.forward_streaming(w1)
        carried, _ = m.forward_streaming(w2, states)
        reset, _ = m.forward_streaming(w2, None)
    assert any(s["s"].abs().max() > 0 for s in states)
    assert not torch.allclose(carried, reset)


def _walk(brain, pb, max_new, sp=None):
    """Drive the internal streaming loop with raw bytes (bypasses str utf-8
    encoding, so generated bytes can be re-fed exactly)."""
    return list(brain._stream_fast_streaming(list(pb), 0.7, 1, max_new,
                                             None, None, None, state_path=sp))


def test_state_file_split_call_matches_single_call(brain, tmp_path):
    """Two calls through a persisted state file emit the exact bytes one
    continuous call does: (states, buffer) fully determine the walk position.
    Call 1 crosses a window commit so the carried tensors, not just the
    buffer, are exercised; call 2 resumes with no new prompt bytes."""
    sp = str(tmp_path / "conv.pt")
    p1 = PROMPT.encode()
    n1, n2 = SEQ, 24
    one = [ev["byte"] for ev in _walk(brain, p1, n1 + n2) if ev.get("kind") == "fast_byte"]
    a = [ev["byte"] for ev in _walk(brain, p1, n1, sp) if ev.get("kind") == "fast_byte"]
    evs = _walk(brain, b"", n2, sp)
    b = [ev["byte"] for ev in evs if ev.get("kind") == "fast_byte"]
    assert a == one[:n1]
    assert b == one[n1:]
    pre = next(ev for ev in evs if ev.get("kind") == "prefill")
    assert pre["resumed"] and pre["prior_total"] == len(p1) + n1


def test_state_file_resume_is_byte_exact_with_real_turns(brain, tmp_path):
    """With a non-empty second turn (the chat reality), split-call equals one
    call whose history is turn1 + reply1 + turn2."""
    sp = str(tmp_path / "conv.pt")
    p1, t2 = PROMPT.encode(), b" and then?"
    n1, n2 = SEQ // 2, 40
    a = [ev["byte"] for ev in _walk(brain, p1, n1, sp) if ev.get("kind") == "fast_byte"]
    b = [ev["byte"] for ev in _walk(brain, t2, n2, sp) if ev.get("kind") == "fast_byte"]
    combined = list(p1) + a + list(t2)
    one = [ev["byte"] for ev in _walk(brain, combined, n2) if ev.get("kind") == "fast_byte"]
    assert b == one


def test_state_file_checkpoint_mismatch_errors(brain, tmp_path):
    """A state written by other weights is refused, not silently used."""
    sp = str(tmp_path / "conv.pt")
    list(brain.stream_fast(PROMPT, mode="stream", max_new=4,
                           stream_state=sp, **DECODE_KW))
    s = torch.load(sp, weights_only=True)
    s["checkpoint"] = "someone_else.pt"
    torch.save(s, sp)
    evs = list(brain.stream_fast(PROMPT, mode="stream", max_new=4,
                                 stream_state=sp, **DECODE_KW))
    err = [ev for ev in evs if ev.get("kind") == "error"]
    assert err and "state" in err[0]["message"]
    assert not _bytes(evs, "fast_byte")


def test_stream_mode_rejected_on_non_recurrent_model(tmp_path_factory):
    """A dense-attention model yields a clean error event for fast=stream."""
    torch.manual_seed(MODEL_SEED)
    model = Veritate(VOCAB, HIDDEN, LAYERS, FFN, HEADS, SEQ)
    b, prev = _make_brain(tmp_path_factory, model)
    try:
        evs = list(b.stream_fast(PROMPT, mode="stream", max_new=4, **DECODE_KW))
        err = [ev for ev in evs if ev.get("kind") == "error"]
        assert err and "recurrent" in err[0]["message"]
        assert not _bytes(evs, "fast_byte")
    finally:
        if prev is None:
            os.environ.pop(INFER_DEVICE_ENV, None)
        else:
            os.environ[INFER_DEVICE_ENV] = prev
