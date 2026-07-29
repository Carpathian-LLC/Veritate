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
from readers import paths

DEFAULT_DATA = os.path.join(paths.EVAL_SAMPLES_ROOT, "ifeval_sample.json")

# A chat-trained model answers the framing it was trained on. Fed a bare instruction it
# free-associates, which scores the framing rather than the instruction following. The
# stop marker drops its closing bracket so a byte model that reproduces the marker
# approximately still halts (same rule as the serving path).
CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
CHAT_STOP     = b"<|im_end|"

# Named item sets. "default" mixes format rules with rules that also require the right
# answer; "form" grades obedience only. Report them separately: a reasoning ceiling in
# the mixed set masks an instruction-following gain.
SAMPLE_SETS = {
    "default": "ifeval_sample.json",
    "form":    "ifeval_form.json",
}


def data_path_for(name):
    fn = SAMPLE_SETS.get(name or "default")
    if fn is None:
        raise ValueError(f"unknown ifeval set {name!r}; expected one of {sorted(SAMPLE_SETS)}")
    return os.path.join(paths.EVAL_SAMPLES_ROOT, fn)


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _generate(model, prompt_bytes: bytes, max_new: int = 256,
              eos_byte: int | None = None, stop: bytes | None = None) -> bytes:
    """Greedy byte-level decode. No sampling, no top-k. Runs on whichever device
    the model's parameters already live on (CPU or MPS). `stop` halts the decode and
    is trimmed off: without it every length-based checker grades the model's reply
    plus whatever self-conversation it ran into after the turn ended."""
    device = _model_device(model)
    ids = torch.tensor(list(prompt_bytes), dtype=torch.long).unsqueeze(0)
    if ids.size(1) == 0:
        ids = torch.zeros((1, 1), dtype=torch.long)
    ids = ids.to(device)
    max_seq = model.seq
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
            if stop and out_bytes.endswith(stop):
                return bytes(out_bytes[:-len(stop)])
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


def check_word_count(response: str, maximum: int = 0, minimum: int = 0, **_) -> bool:
    """Pass if the response length in words sits inside the stated bounds."""
    n = len(response.split())
    return (maximum <= 0 or n <= maximum) and n >= minimum


def check_item_count(response: str, count: int = 3, **_) -> bool:
    """Pass if the response lists exactly `count` items, line, comma, or 'and' separated.

    ' and ' is only a separator when the list has no commas. Splitting on it
    unconditionally breaks any item that contains the word: "Ham and cheese, turkey
    club, tuna salad" is three items, not four, and counting it as four marks a
    correct answer wrong."""
    body = response.strip()
    parts = [p for p in re.split(r"\n+", body) if p.strip()]
    if len(parts) < 2:
        if "," in body or ";" in body:
            parts = re.split(r",|;", body)
        else:
            parts = re.split(r"\band\b", body)
        parts = [re.sub(r"^\s*and\b", "", p).strip(" .") for p in parts]
        parts = [p for p in parts if p]
    return len(parts) == count


def check_contains(response: str, text: str = "", **_) -> bool:
    """Pass if the response contains `text`. The compute and classify checks use
    this: the answer must appear, and surrounding working is allowed."""
    return text.lower() in response.lower()


def check_starts_with(response: str, text: str = "", **_) -> bool:
    """Pass if the response opens with `text`, ignoring case and leading punctuation.
    Pins the 'answer yes or no first' family, where a preamble is the failure."""
    return response.strip().lower().lstrip("\"'*(").startswith(text.lower())


def check_forbidden_words(response: str, words=(), **_) -> bool:
    """Pass if none of `words` appears as a whole word in the response."""
    low = response.lower()
    return not any(re.search(rf"\b{re.escape(w.lower())}\b", low) for w in words)


def check_starts_with_yes_or_no(response: str, **_) -> bool:
    """Pass if the reply opens with Yes or No. Grades the FORM the instruction asked
    for, not which answer is right: pinning the correct one turns an instruction
    check into a reasoning check, and a model that obeys perfectly still fails."""
    return check_starts_with(response, "yes") or check_starts_with(response, "no")


CHECKERS = {
    "json":              check_json,
    "starts_with_yes_or_no": check_starts_with_yes_or_no,
    "sentence_count":    check_sentence_count,
    "forbidden_letter":  check_forbidden_letter,
    "word_count":        check_word_count,
    "item_count":        check_item_count,
    "contains":          check_contains,
    "starts_with":       check_starts_with,
    "forbidden_words":   check_forbidden_words,
}


def balanced_subset(items, limit):
    """First `limit` items, round-robin over rule families instead of a raw prefix.

    The form set groups its families in contiguous blocks, so `items[:limit]` would
    hand back one family and report it as the whole score. Round-robin keeps every
    family represented at any limit and stays deterministic (no sampling seed to
    thread through, and two runs at the same limit grade the same items).
    """
    if limit is None or limit >= len(items):
        return list(items)
    buckets: dict[str, list] = {}
    for it in items:
        key = ",".join(sorted(r["name"] for r in it.get("rules", []))) or ""
        buckets.setdefault(key, []).append(it)
    out, order = [], sorted(buckets)
    while len(out) < limit:
        took = False
        for key in order:
            if buckets[key]:
                out.append(buckets[key].pop(0))
                took = True
                if len(out) >= limit:
                    break
        if not took:
            break
    return out


def run_ifeval(model, data_path: str = DEFAULT_DATA,
               max_new: int = 256, limit: int | None = None,
               verbose: bool = False, progress_cb=None, chat: bool = False) -> dict:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"IFEval data not found at {data_path}. See README.md for download instructions."
        )
    with open(data_path, encoding="utf-8") as f:
        blob = json.load(f)
    items = blob["items"]
    if limit is not None:
        items = balanced_subset(items, limit)

    n_total = 0
    n_pass  = 0
    by_rule_total: dict[str, int] = {}
    by_rule_pass:  dict[str, int] = {}
    per_item = []
    t0 = time.perf_counter()

    for i, it in enumerate(items):
        prompt = it["prompt"]
        rules = it.get("rules", [])
        framed = CHAT_TEMPLATE.format(prompt=prompt) if chat else prompt
        resp_bytes = _generate(model, framed.encode("utf-8"), max_new=max_new,
                               stop=CHAT_STOP if chat else None)
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
                # Unknown rule: count as fail so the report flags the missing
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
