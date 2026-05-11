# ------------------------------------------------------------------------------------
# veritate_mri/eval/hellaswag.py
# ------------------------------------------------------------------------------------
# HellaSwag byte-level evaluation.
#
# HellaSwag asks "given this context, which of these 4 endings is the natural next
# sentence?" Three of the endings are adversarial distractors (machine-generated and
# human-filtered to be plausible-but-wrong). The eval is byte-level identical to
# MMLU's text-mode: score `context + ending` for each candidate, pick the highest.
#
# Data format (`hellaswag_sample.json`):
#   {
#     "items": [
#       {"ctx": "...", "endings": ["e0","e1","e2","e3"], "label": 0..3, "activity": "..."},
#       ...
#     ]
#   }
#
# The full HellaSwag val set (~10k items) lives on HuggingFace; see README.md.
# ------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import time

from .score import score_sequence


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "hellaswag_sample.json")


def run_hellaswag(model, data_path: str = DEFAULT_DATA,
                  limit: int | None = None, verbose: bool = False,
                  progress_cb=None) -> dict:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"HellaSwag data not found at {data_path}. See README.md for download instructions."
        )
    with open(data_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    items = blob["items"]
    if limit is not None:
        items = items[:limit]

    correct = 0
    per_item = []
    t0 = time.perf_counter()
    for i, it in enumerate(items):
        ctx = it["ctx"]
        endings = it["endings"]
        gold = int(it["label"])
        # HellaSwag convention: a single space joins ctx and ending.
        scores = []
        for ending in endings:
            # Split point is between ctx and " " + ending; treating the leading
            # space as part of the completion keeps the prompt boundary unambiguous.
            prompt_b = ctx.encode("utf-8")
            comp_b   = (" " + ending).encode("utf-8")
            scores.append(score_sequence(model, prompt_b, comp_b))
        pred = max(range(len(endings)), key=lambda c: scores[c])
        if pred == gold:
            correct += 1
        item = {"gold": gold, "pred": pred, "activity": it.get("activity")}
        per_item.append(item)
        if verbose:
            print(f"  [{i+1}/{len(items)}] gold={gold} pred={pred} "
                  f"scores={[round(s,3) for s in scores]}")
        if progress_cb is not None:
            try:
                progress_cb(i + 1, len(items), item)
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    n = len(items)
    return {
        "suite": "hellaswag",
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "elapsed_s": round(elapsed, 2),
    }
