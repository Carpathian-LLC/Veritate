# ------------------------------------------------------------------------------------
# veritate_mri/eval/ifeval.py
# ------------------------------------------------------------------------------------
# IFEval (Instruction-Following Eval) scaffold.
#
# Unlike MMLU/HellaSwag, IFEval is NOT a multiple-choice eval. The model is given an
# instruction with a verifiable rule ("answer in JSON", "use exactly 3 sentences",
# "do not use the letter 'e'", etc.); the model generates freely; a deterministic
# rule-checker grades pass/fail.
#
# This file ships:
#   - A minimal data schema and 3 sample prompts.
#   - A greedy-decode helper (`_generate`).
#   - Three reference rule-checkers + a `CHECKERS` registry pattern.
#   - The full pipeline (`run_ifeval`) that maps each item to its checker(s) and
#     reports per-rule and overall pass rates.
#
# To run on the real Google IFEval set (~541 prompts, 25 instruction families),
# download from https://huggingface.co/datasets/google/IFEval and write checkers
# for the remaining families. See README.md.
# ------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
import time

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "ifeval_sample.json")


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _generate(model, prompt_bytes: bytes, max_new: int = 256,
              eos_byte: int | None = None) -> bytes:
    """Greedy byte-level decode. No sampling, no top-k. Runs on whichever device
    the model's parameters already live on (CPU or MPS)."""
    device = _model_device(model)
    ids = torch.tensor(list(prompt_bytes), dtype=torch.long).unsqueeze(0)
    if ids.size(1) == 0:
        ids = torch.zeros((1, 1), dtype=torch.long)
    ids = ids.to(device)
    max_seq = getattr(model, "seq", 512)
    model.eval()
    out_bytes = bytearray()
    with torch.no_grad():
        for _ in range(max_new):
            ctx = ids if ids.size(1) <= max_seq else ids[:, -max_seq:]
            res = model(ctx)
            logits = res[0] if isinstance(res, (tuple, list)) else res
            nxt = int(logits[0, -1].argmax().item())
            out_bytes.append(nxt)
            if eos_byte is not None and nxt == eos_byte:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return bytes(out_bytes)


# ---------------- rule checkers ----------------
# Each checker takes the model's response (str) plus optional kwargs, returns bool.

def check_json(response: str, **_) -> bool:
    """Pass if the response (after trimming) parses as valid JSON."""
    try:
        json.loads(response.strip())
        return True
    except Exception:
        return False


def check_sentence_count(response: str, count: int = 3, **_) -> bool:
    """Pass if the response has exactly `count` sentences (rough heuristic)."""
    sentences = re.split(r"(?<=[.!?])\s+", response.strip())
    sentences = [s for s in sentences if s.strip()]
    return len(sentences) == count


def check_forbidden_letter(response: str, letter: str = "e", **_) -> bool:
    """Pass if the response does not contain the forbidden letter (case-insensitive)."""
    return letter.lower() not in response.lower()


CHECKERS = {
    "json":              check_json,
    "sentence_count":    check_sentence_count,
    "forbidden_letter":  check_forbidden_letter,
    # TODO: port the full Google IFEval ruleset:
    # - keywords_existence, keywords_frequency, keywords_forbidden
    # - language (detect-locale), response_length (words/sentences/paragraphs),
    # - format_constraints (title_case, all_uppercase, json_format, markdown_format),
    # - punctuation_no_comma, end_with, start_with, ...
    # Each is a deterministic str -> bool. Drop them in this dict by name.
}


def run_ifeval(model, data_path: str = DEFAULT_DATA,
               max_new: int = 256, limit: int | None = None,
               verbose: bool = False, progress_cb=None) -> dict:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"IFEval data not found at {data_path}. See README.md for download instructions."
        )
    with open(data_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    items = blob["items"]
    if limit is not None:
        items = items[:limit]

    n_total = 0
    n_pass  = 0
    by_rule_total: dict[str, int] = {}
    by_rule_pass:  dict[str, int] = {}
    per_item = []
    t0 = time.perf_counter()

    for i, it in enumerate(items):
        prompt = it["prompt"]
        rules = it.get("rules", [])
        resp_bytes = _generate(model, prompt.encode("utf-8"), max_new=max_new)
        try:
            response = resp_bytes.decode("utf-8", errors="replace")
        except Exception:
            response = ""
        item_pass = True
        rule_results = []
        for rule in rules:
            name = rule["name"]
            kwargs = {k: v for k, v in rule.items() if k != "name"}
            checker = CHECKERS.get(name)
            if checker is None:
                # Unknown rule — count as fail so the report flags the missing
                # checker rather than silently passing.
                ok = False
                rule_results.append({"name": name, "status": "no_checker"})
            else:
                ok = bool(checker(response, **kwargs))
                rule_results.append({"name": name, "pass": ok})
            by_rule_total[name] = by_rule_total.get(name, 0) + 1
            if ok:
                by_rule_pass[name] = by_rule_pass.get(name, 0) + 1
            item_pass = item_pass and ok
        n_total += 1
        if item_pass:
            n_pass += 1
        per_item.append({"prompt": prompt[:80], "response": response[:200],
                         "pass": item_pass, "rules": rule_results})
        if verbose:
            print(f"  [{i+1}/{len(items)}] pass={item_pass} rules={rule_results}")
        if progress_cb is not None:
            try:
                progress_cb(i + 1, len(items), per_item[-1])
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    by_rule = {
        r: {"n": by_rule_total[r],
            "pass_rate": by_rule_pass.get(r, 0) / by_rule_total[r]}
        for r in by_rule_total
    }
    return {
        "suite": "ifeval",
        "n": n_total,
        "pass_rate": n_pass / n_total if n_total else 0.0,
        # Mirror the multiple-choice suites' field name so the dashboard can
        # rank everything by a single "accuracy" key when convenient.
        "accuracy": n_pass / n_total if n_total else 0.0,
        "by_rule": by_rule,
        "elapsed_s": round(elapsed, 2),
        "note": "Scaffold only; full Google IFEval ruleset not yet ported. See README.",
    }
