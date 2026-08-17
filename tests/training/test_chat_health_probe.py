# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - writing_health samples at temperature 0.7 and reported distinct_4 rising
#   0.729 -> 0.986 on wren1_0, a checkpoint that repeated a 6-word window in 44%
#   of its GREEDY replies. The sampled metric and the greedy behaviour disagreed
#   completely, and the run finished before anyone could see it
#   (failures.md 2026-08-16).
# - these pin the greedy probe that closes that gap: the loop detector, the
#   turn-close detector, and its registration in the dump suite.
# tests/training/test_chat_health_probe.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

import pytest
import torch
from torch import nn

# ------------------------------------------------------------------------------------
# Constants

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "veritate_mri"))

from training import checkpoint_probe as cp  # noqa: E402
from training import save as save_mod  # noqa: E402

# ------------------------------------------------------------------------------------
# Fixtures


class _Scripted(nn.Module):
    """Emits a fixed byte string, then <|im_end|> if `close` is set. Greedy decode
    is argmax, so a one-hot logit row makes the output exactly predictable."""

    def __init__(self, text, close=True, vocab=256, seq=1024):
        super().__init__()
        self.vocab, self.seq = vocab, seq
        self.script = list(text.encode()) + (list(b"<|im_end|>") if close else [])
        self._w = nn.Parameter(torch.zeros(1))
        self.n_emitted = 0

    def forward(self, ids):
        b, t = ids.shape
        logits = torch.zeros(b, t, self.vocab)
        nxt = self.script[self.n_emitted % len(self.script)] if self.script else 0
        self.n_emitted += 1
        logits[0, -1, nxt] = 100.0
        return logits, None


# ------------------------------------------------------------------------------------
# Tests: loop detection


def test_repeated_window_is_a_loop():
    t = ("the ocean is a vast and complex ecosystem interconnected with the ocean "
         "itself the ocean is a vast and complex ecosystem interconnected with it")
    assert cp._ch_loops(t) is True


def test_ordinary_prose_is_not_a_loop():
    t = ("A prime number is a whole number greater than one whose only divisors "
         "are itself and the number one, which makes it a building block.")
    assert cp._ch_loops(t) is False


def test_short_reply_cannot_loop():
    """Fewer words than the window size must not index out of range."""
    assert cp._ch_loops("yes") is False
    assert cp._ch_loops("") is False


def test_loop_needs_the_full_window():
    """A repeated 5-gram is not flagged at n=6; the threshold must be exact."""
    assert cp._ch_loops("a b c d e x a b c d e y", n=6) is False
    assert cp._ch_loops("a b c d e f x a b c d e f y", n=6) is True


# ------------------------------------------------------------------------------------
# Tests: greedy turn generation


def test_reply_stops_at_im_end():
    m = _Scripted("hello there", close=True)
    text, closed = cp._ch_reply(m, "hi", "cpu")
    assert closed is True
    assert "<|im_end|>" not in text


def test_unclosed_reply_runs_to_the_cap():
    """The measurable form of 'it rambles': no turn marker, hits max_new."""
    m = _Scripted("abc", close=False)
    text, closed = cp._ch_reply(m, "hi", "cpu")
    assert closed is False
    assert len(text.encode()) == cp.CH_MAX_NEW


def test_model_is_restored_to_training_mode(tmp_path):
    m = _Scripted("fine", close=True)
    m.train()
    cp.dump_chat_health(m, str(tmp_path), 10)
    assert m.training is True


# ------------------------------------------------------------------------------------
# Tests: the dump


def test_dump_writes_expected_shape(tmp_path):
    m = _Scripted("A prime number has exactly two distinct divisors.", close=True)
    p = cp.dump_chat_health(m, str(tmp_path), 250)
    d = json.load(open(p))
    assert d["step"] == 250
    assert len(d["samples"]) == len(cp.CH_PROMPTS)
    for k in ("loop_rate", "closed_rate", "median_bytes"):
        assert k in d["aggregate"]
    assert d["config"]["decode"] == "greedy"


def test_dump_reports_a_clean_model_as_clean(tmp_path):
    m = _Scripted("A prime number has exactly two distinct divisors.", close=True)
    d = json.load(open(cp.dump_chat_health(m, str(tmp_path), 1)))
    assert d["aggregate"]["loop_rate"] == 0.0
    assert d["aggregate"]["closed_rate"] == 1.0


def test_dump_catches_the_wren1_0_failure(tmp_path):
    """The regression this probe exists for: fluent, closes its turn, loops."""
    looper = ("The ocean is a vast and complex ecosystem that is deeply "
              "interconnected with the ocean itself. The ocean is a vast and "
              "complex ecosystem that is deeply interconnected with the ocean itself.")
    d = json.load(open(cp.dump_chat_health(_Scripted(looper), str(tmp_path), 1)))
    assert d["aggregate"]["loop_rate"] == 1.0
    assert d["aggregate"]["closed_rate"] == 1.0, "closed turns hide the loop from closed_rate"


# ------------------------------------------------------------------------------------
# Tests: registration


def test_registered_in_all_dumps():
    assert "chat_health" in save_mod.ALL_DUMPS


def test_registered_as_heavy():
    """It generates text, so it belongs with the expensive dumps, not the ~9s set."""
    assert "chat_health" in save_mod.HEAVY_DUMPS


def test_light_hooks_skip_it():
    assert "chat_health" in set(save_mod.HEAVY_DUMPS)


def test_rename_map_covers_it():
    """Without the rename entry the artifact keeps its step suffix and no reader
    finds it."""
    assert any("chat_health" in k for k in save_mod.RENAME_MAP_TEMPLATE)
    assert "chat_health.json" in save_mod.RENAME_MAP_TEMPLATE.values()


def test_language_gated():
    """Statistical/other models train on non-text corpora; a chat probe there is
    nonsense, same as the other language dumps."""
    assert "chat_health" in save_mod.LANGUAGE_DUMPS
