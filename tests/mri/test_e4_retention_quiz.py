# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the closed-book retention quiz over the injected facts. It is the instrument the whole
#   memory program reads its numbers off, so what is pinned is the measurement contract:
#   greedy decoding (a sampled answer makes the score a coin toss), ChatML framing, the
#   reply cut at the first end-of-turn marker, substring grading in both directions, and
#   experience logging off so a quiz never becomes next night's training data. The model
#   is stubbed; no checkpoint and no weights are touched.
# tests/mri/test_e4_retention_quiz.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys
import types

import pytest
from tools import e4_retention_quiz as quiz

# ------------------------------------------------------------------------------------
# Constants

FACTS = [
    {"id": 1, "q_fwd": "Who keeps the lighthouse?", "a_fwd": "Marisol",
     "q_rev": "What does Marisol keep?", "a_rev": "lighthouse"},
    {"id": 2, "q_fwd": "Where is the bakery?", "a_fwd": "Larkfell",
     "q_rev": "What is in Larkfell?", "a_rev": "bakery"},
]

# ------------------------------------------------------------------------------------
# Functions


class _Brain:
    """Records the decode settings it was asked for and replies from a canned map."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def stream_fast(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        for b in self.replies.get(prompt, "").encode():
            yield {"kind": "fast_byte", "byte": b}


def _stub_backend(monkeypatch, brain):
    module = types.ModuleType("inference.backends.pytorch")
    module.Brain = lambda *a, **k: brain
    monkeypatch.setitem(sys.modules, "inference.backends.pytorch", module)


def _prompt(q):
    return f"{quiz.IM_S}user\n{q}{quiz.IM_E}\n{quiz.IM_S}assistant\n"


def test_the_question_is_asked_in_chatml_and_decoded_greedily():
    """Grading rule: bare greedy. A temperature-sampled answer would make the score a coin
    toss between runs of the same weights."""
    brain = _Brain({_prompt("q"): "Marisol"})
    assert quiz.ask(brain, "q") == "Marisol"
    call = brain.calls[0]
    assert call["top_k_sample"] == 1
    assert call["rep_penalty"] == 0.0
    assert call["prompt"].startswith(f"{quiz.IM_S}user\nq{quiz.IM_E}")


@pytest.mark.parametrize("reply", ["Marisol<|im_end|>\nnext", "Marisol<|im_start|>user"])
def test_the_reply_stops_at_the_turn_marker(reply):
    """A model that runs on past its turn must not have the next turn graded as its answer."""
    brain = _Brain({_prompt("q"): reply})
    assert quiz.ask(brain, "q") == "Marisol"


def test_both_directions_are_graded_by_substring(tmp_path, monkeypatch):
    """A fact counts as retained when the answer appears in the reply, case-insensitively,
    asked forwards and backwards."""
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(FACTS), encoding="utf-8")
    monkeypatch.setattr(quiz, "FACTS_PATH", str(facts))
    brain = _Brain({
        _prompt(FACTS[0]["q_fwd"]): "I think it is marisol, yes.",   # hit, lowercased
        _prompt(FACTS[0]["q_rev"]): "A bakery.",                     # miss
        _prompt(FACTS[1]["q_fwd"]): "No idea.",                      # miss
        _prompt(FACTS[1]["q_rev"]): "The bakery is there.",          # hit
    })
    _stub_backend(monkeypatch, brain)
    out = tmp_path / "r.json"
    report = quiz.run("m", 10, out_path=str(out))
    assert (report["fwd"], report["rev"]) == (1, 1)
    assert report["n"] == 2 and report["fwd_acc"] == 0.5
    assert report["model"] == "m@10"
    assert json.loads(out.read_text())["rows"][0]["fwd"] is True


def test_a_quiz_is_not_recorded_as_experience(tmp_path, monkeypatch):
    """Measured incident: a wrong closed-book answer became next night's drill. The quiz
    turns the experience log off before the model is built."""
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(FACTS[:1]), encoding="utf-8")
    monkeypatch.setattr(quiz, "FACTS_PATH", str(facts))
    monkeypatch.setenv("VERITATE_EXPERIENCE_LOG", "1")
    _stub_backend(monkeypatch, _Brain({}))
    quiz.run("m", 10)
    assert os.environ["VERITATE_EXPERIENCE_LOG"] == "0"


def test_asking_for_the_cpu_pins_the_device(tmp_path, monkeypatch):
    """The device has to be set before the backend imports, or the quiz runs on whatever
    the box picks and its timings are not comparable."""
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(FACTS[:1]), encoding="utf-8")
    monkeypatch.setattr(quiz, "FACTS_PATH", str(facts))
    monkeypatch.delenv("VERITATE_INFER_DEVICE", raising=False)
    _stub_backend(monkeypatch, _Brain({}))
    quiz.run("m", 10, device="cpu")
    assert os.environ["VERITATE_INFER_DEVICE"] == "cpu"


def test_an_empty_fact_file_is_refused(tmp_path, monkeypatch):
    """A quiz over no facts used to divide by zero at the accuracy line, after loading the
    model. Say what is wrong instead."""
    facts = tmp_path / "facts.json"
    facts.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(quiz, "FACTS_PATH", str(facts))
    _stub_backend(monkeypatch, _Brain({}))
    with pytest.raises(ValueError, match="no facts to quiz on"):
        quiz.run("m", 10)
