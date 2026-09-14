# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the two multiple-choice suites score every ending / choice through score_sequence
#   and pick the highest. With scoring stubbed to prefer one known completion the
#   bookkeeping is pinned: accuracy, letter vs text modes, per-subject rollup, limit,
#   progress callback, and the refusals.
# tests/mri/test_eval_mcq.py
# ------------------------------------------------------------------------------------
# Imports:

import json

import pytest
from eval import hellaswag, mmlu

# ------------------------------------------------------------------------------------
# Functions


def _prefers(winners):
    """A score_sequence stand-in: 1.0 when the completion is the winner for the prompt whose
    key it contains, else 0 (a prompt with no winner scores every choice equal, and the
    first choice wins the tie)."""
    def score(model, prompt_b, comp_b):
        prompt = prompt_b.decode()
        return 1.0 if any(k in prompt and comp_b.decode() in v for k, v in winners.items()) else 0.0
    return score


def test_hellaswag_picks_the_highest_scoring_ending(tmp_path, monkeypatch):
    data = tmp_path / "hs.json"
    data.write_text(json.dumps({"items": [
        {"ctx": "A man", "endings": ["eats", "sleeps", "runs", "reads"], "label": 2, "activity": "x"},
        {"ctx": "A cat", "endings": ["eats", "sleeps", "runs", "reads"], "label": 0, "activity": "y"},
        {"ctx": "A dog", "endings": ["eats", "sleeps", "runs", "reads"], "label": 1, "activity": "z"},
    ]}))
    monkeypatch.setattr(hellaswag, "score_sequence", _prefers({"A man": {" runs"}, "A cat": {" eats"}}))
    seen = []
    out = hellaswag.run_hellaswag(object(), str(data), progress_cb=lambda i, n, item: seen.append((i, n, item["pred"])))
    assert out["suite"] == "hellaswag" and out["n"] == 3 and out["accuracy"] == pytest.approx(2 / 3)
    assert seen == [(1, 3, 2), (2, 3, 0), (3, 3, 0)]
    assert hellaswag.run_hellaswag(object(), str(data), limit=1)["accuracy"] == 1.0
    with pytest.raises(FileNotFoundError):
        hellaswag.run_hellaswag(object(), str(tmp_path / "missing.json"))


def test_mmlu_prompt_lists_the_four_choices_and_ends_at_the_answer():
    p = mmlu._format_prompt("2+2?", ["3", "4", "5", "6"])
    assert p == "Question: 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"


def test_mmlu_modes_and_the_per_subject_rollup(tmp_path, monkeypatch):
    data = tmp_path / "mmlu.json"
    data.write_text(json.dumps({"questions": [
        {"question": "q1", "choices": ["w", "x", "y", "z"], "answer": 1, "subject": "math"},
        {"question": "q2", "choices": ["w", "x", "y", "z"], "answer": 3, "subject": "math"},
        {"question": "q3", "choices": ["w", "x", "y", "z"], "answer": 1, "subject": "law"},
    ]}))
    monkeypatch.setattr(mmlu, "score_sequence", _prefers({"q1": {" x", " B"}, "q3": {" x", " B"}}))   # idx 1 / B
    both = mmlu.run_mmlu(object(), str(data), mode="both")
    assert both["accuracy"] == pytest.approx(2 / 3) and both["accuracy_text"] == both["accuracy_letter"]
    assert both["by_subject"] == {"math": {"n": 2, "acc": 0.5}, "law": {"n": 1, "acc": 1.0}}
    letter = mmlu.run_mmlu(object(), str(data), mode="letter")
    assert letter["accuracy_text"] is None and letter["accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="mode"):
        mmlu.run_mmlu(object(), str(data), mode="pictures")
