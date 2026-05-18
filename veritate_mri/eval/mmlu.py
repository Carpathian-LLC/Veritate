# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - MMLU byte-level evaluation. 4-way MCQ over 57 academic subjects.
# - For each question: build 4 (prompt, completion) pairs, score via
#   score_sequence, predict highest-scoring choice. Two modes:
#     "letter": completion = " A" / " B" / " C" / " D"  (cheap baseline)
#     "text":   completion = " <answer text>"           (semantic)
# - Sample data ships ~20 questions; pass --mmlu-data for full benchmark.
# veritate_mri/eval/mmlu.py
# ------------------------------------------------------------------------------------
# Imports:

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

from readers import paths

from .score import score_sequence


# ------------------------------------------------------------------------------------
# Constants

DEFAULT_DATA = os.path.join(paths.EVAL_SAMPLES_ROOT, "mmlu_sample.json")

LETTERS = ["A", "B", "C", "D"]


# ------------------------------------------------------------------------------------
# Functions


def _format_prompt(question: str, choices: list[str]) -> str:
    """Standard MMLU few-shot-free prompt format."""
    lines = [f"Question: {question}"]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def run_mmlu(model, data_path: str = DEFAULT_DATA, mode: str = "text",
             limit: int | None = None, verbose: bool = False,
             progress_cb=None) -> dict:
    """Run MMLU. `mode` is "letter", "text", or "both" (returns both metrics).

    `progress_cb`, if supplied, is called as `progress_cb(i, n, item_dict)` after
    each question, used by the dashboard to surface live progress.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"MMLU data not found at {data_path}. See README.md for download instructions."
        )
    with open(data_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    questions = blob["questions"]
    if limit is not None:
        questions = questions[:limit]

    by_subj_correct: dict[str, int] = defaultdict(int)
    by_subj_total:   dict[str, int] = defaultdict(int)
    per_item = []
    correct_letter = 0
    correct_text = 0
    t0 = time.perf_counter()

    do_letter = mode in ("letter", "both")
    do_text   = mode in ("text", "both")
    if not (do_letter or do_text):
        raise ValueError(f"mode must be letter|text|both, got {mode!r}")

    for i, q in enumerate(questions):
        prompt = _format_prompt(q["question"], q["choices"])
        gold = int(q["answer"])
        subj = q.get("subject", "unknown")
        prompt_b = prompt.encode("utf-8")

        scores_letter = [None] * 4
        scores_text   = [None] * 4
        if do_letter:
            for ci in range(4):
                comp = (" " + LETTERS[ci]).encode("utf-8")
                scores_letter[ci] = score_sequence(model, prompt_b, comp)
        if do_text:
            for ci in range(4):
                comp = (" " + q["choices"][ci]).encode("utf-8")
                scores_text[ci] = score_sequence(model, prompt_b, comp)

        pred_letter = max(range(4), key=lambda c: scores_letter[c]) if do_letter else None
        pred_text   = max(range(4), key=lambda c: scores_text[c])   if do_text   else None

        item = {"subject": subj, "gold": gold,
                "pred_letter": pred_letter, "pred_text": pred_text}
        per_item.append(item)
        by_subj_total[subj] += 1

        # Subject-level accuracy uses the primary metric (text if available, else letter).
        if do_text:
            if pred_text == gold:
                correct_text += 1
                by_subj_correct[subj] += 1
            if do_letter and pred_letter == gold:
                correct_letter += 1
        else:  # letter only
            if pred_letter == gold:
                correct_letter += 1
                by_subj_correct[subj] += 1

        if verbose:
            print(f"  [{i+1}/{len(questions)}] {subj}: gold={LETTERS[gold]} "
                  f"pred_letter={LETTERS[pred_letter] if pred_letter is not None else '-'} "
                  f"pred_text={LETTERS[pred_text] if pred_text is not None else '-'}")
        if progress_cb is not None:
            try:
                progress_cb(i + 1, len(questions), item)
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    n = len(questions)
    subjects = {}
    for s in by_subj_total:
        subjects[s] = {
            "n": by_subj_total[s],
            "acc": by_subj_correct[s] / by_subj_total[s],
        }
    primary_correct = correct_text if do_text else correct_letter
    return {
        "suite": "mmlu",
        "n": n,
        "accuracy": primary_correct / n if n else 0.0,
        "accuracy_letter": (correct_letter / n) if (n and do_letter) else None,
        "accuracy_text":   (correct_text   / n) if (n and do_text)   else None,
        "by_subject": subjects,
        "elapsed_s": round(elapsed, 2),
        "mode": mode,
    }
