# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers tools/study_recall.py, the likelihood-based closed-book recall metric. The
#   model is not loaded here; these pin the framing and the scored span.
# - the load-bearing property is the SCORED SPAN. The prompt names the chunk, so if the
#   mask let prompt bytes into the loss, a model could score well by predicting its own
#   question and the whole recall measurement would be circular.
# tests/mri/test_study_recall.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np
import torch
from tools import study_recall as sr
from training import veritate_trainer as vt

# ------------------------------------------------------------------------------------
# Functions


def _masked_span(label, text, seq=512):
    """Return the bytes that would actually contribute to the loss."""
    raw = sr.exchange_bytes(label, text)
    buf = raw + b"\n" * (seq + 1 - len(raw))
    arr = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    toks = torch.from_numpy(arr[:-1]).unsqueeze(0)
    tgts = torch.from_numpy(arr[1:].copy()).unsqueeze(0)
    masked = vt.apply_role_mask(toks, tgts)
    keep = masked[0] >= 0
    return bytes(masked[0][keep].numpy().astype(np.uint8))


def test_exchange_puts_the_chunk_in_the_assistant_turn():
    """Same framing the trainer consolidated, or the measurement is off-distribution."""
    raw = sr.exchange_bytes("m.py::f", "def f(): pass").decode()
    assert raw.startswith("<|im_start|>user\nShow me m.py::f.<|im_end|>")
    assert "<|im_start|>assistant\ndef f(): pass<|im_end|>" in raw


def test_only_the_chunk_bytes_are_scored():
    """The prompt names the chunk. If prompt bytes reached the loss, a model could
    score well by predicting its own question and recall would be circular."""
    scored = _masked_span("m.py::alpha", "def alpha(x):\n    return x + 1\n")
    assert b"def alpha(x):" in scored
    assert b"Show me" not in scored
    assert b"m.py::alpha." not in scored


def test_the_label_itself_is_never_scored():
    """A distinctive label must not leak into the scored span through the user turn."""
    scored = _masked_span("zzq_unique_label", "body text that is scored")
    assert b"body text that is scored" in scored
    assert b"zzq_unique_label" not in scored


def test_summarize_reports_nll_and_bpb():
    """bpb is nll in bits, the unit the ledgers use for val comparisons."""
    s = sr.summarize([{"nll": 0.0}, {"nll": 2 * 0.6931471805599453}])
    assert s["n"] == 2 and s["nll"] == 0.6931
    assert abs(s["bpb"] - 1.0) < 1e-3


def test_summarize_skips_unscorable_chunks():
    """A chunk that did not fit reports None and must not count as a zero."""
    s = sr.summarize([{"nll": None}, {"nll": 1.0}])
    assert s["n"] == 1 and s["nll"] == 1.0


def test_summarize_of_nothing_is_not_a_score():
    assert sr.summarize([{"nll": None}]) == {"n": 0}


def test_heldout_form_differs_from_every_trained_form():
    """Recall on a TRAINED phrasing can be surface fit to that wording. The held-out
    form is the honest test that the label is bound to the content, so it must not
    collide with anything build_study_corpus generates."""
    import random

    from tools import build_study_corpus as bsc
    forms = bsc.study_forms("L", "body text long enough to exercise every form", random.Random(0), 7)
    trained = " ".join(forms)
    assert sr.HELDOUT_PROMPT.format(label="L") not in trained
    assert sr.PROMPT.format(label="L") in trained
