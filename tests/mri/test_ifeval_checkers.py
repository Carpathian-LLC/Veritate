# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - IFEval grades instruction following programmatically, so a checker that is wrong in
#   either direction silently moves a model's score. These pin each checker on a passing
#   AND a failing response, plus the shipped sample set's shape.
# tests/mri/test_ifeval_checkers.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
from collections import Counter

from eval import ifeval
from eval.ifeval import (
    CHAT_STOP,
    CHAT_TEMPLATE,
    CHECKERS,
    DEFAULT_DATA,
    _generate,
    check_contains,
    check_forbidden_words,
    check_item_count,
    check_starts_with,
    check_word_count,
)

# ------------------------------------------------------------------------------------
# Constants

THREE_ITEMS = "red, blue, green"

# ------------------------------------------------------------------------------------
# Functions


def test_word_count_passes_inside_the_ceiling():
    assert check_word_count("a short reply here", maximum=15, minimum=2)


def test_word_count_fails_over_the_ceiling():
    assert not check_word_count(" ".join(["word"] * 16), maximum=15)


def test_word_count_fails_under_the_floor():
    """An empty or one-word reply must not pass a 'describe it' instruction."""
    assert not check_word_count("bicycle", maximum=15, minimum=2)


def test_item_count_reads_a_comma_list():
    assert check_item_count(THREE_ITEMS, count=3)


def test_item_count_reads_a_newline_list():
    assert check_item_count("boil water\nadd the egg\nwait ten minutes", count=3)


def test_item_count_fails_on_the_wrong_number_of_items():
    assert not check_item_count(THREE_ITEMS, count=4)


def test_item_count_does_not_split_an_item_containing_the_word_and():
    """'Ham and cheese' is one item. Splitting it marked correct answers wrong."""
    assert check_item_count("Ham and cheese, turkey club, tuna salad", count=3)


def test_item_count_reads_an_and_list_when_there_are_no_commas():
    assert check_item_count("salt and pepper", count=2)


def test_item_count_tolerates_the_oxford_and():
    assert check_item_count("red, blue, and green", count=3)


def test_item_count_ignores_a_trailing_period():
    assert check_item_count("Mississippi, Colorado, Rio Grande.", count=3)


def test_contains_ignores_case_and_surrounding_working():
    assert check_contains("That leaves 5 marbles.", text="5")


def test_contains_fails_when_the_answer_is_absent():
    assert not check_contains("That leaves four marbles.", text="5")


def test_starts_with_tolerates_leading_punctuation():
    assert check_starts_with('"Yes, twelve is larger."', text="yes")


def test_starts_with_fails_on_a_preamble():
    """The rule exists to catch a preamble, so a buried answer must fail."""
    assert not check_starts_with("Let me think. Yes.", text="yes")


def test_forbidden_words_fails_when_a_banned_word_appears():
    assert not check_forbidden_words("the water was calm", words=["water", "blue"])


def test_forbidden_words_allows_a_banned_word_inside_a_longer_word():
    """Whole-word matching only: 'waterfall' must not trip the 'water' ban."""
    assert check_forbidden_words("a waterfall of light", words=["water"])


def test_every_rule_named_in_the_sample_set_has_a_checker():
    """An unknown rule name scores as a fail, so a typo would silently sink the run."""
    with open(DEFAULT_DATA, encoding="utf-8") as f:
        items = json.load(f)["items"]
    named = {r["name"] for i in items for r in i["rules"]}
    assert named <= set(CHECKERS)


class _ScriptedModel:
    """Emits a fixed byte script one byte per forward, so decode-loop behavior
    (stopping, trimming) is testable without a checkpoint."""

    seq = 128

    def __init__(self, script: bytes):
        self.script = script
        self.i = 0

    def parameters(self):
        return iter(())

    def eval(self):
        return self

    def __call__(self, ctx):
        import torch
        b = self.script[self.i] if self.i < len(self.script) else 0
        self.i += 1
        logits = torch.zeros((1, ctx.size(1), 256))
        logits[0, -1, b] = 1.0
        return logits


def test_generate_stops_at_the_turn_marker_and_trims_it():
    """Without this the reply carries the marker and whatever followed it."""
    m = _ScriptedModel(b"red, blue, green" + CHAT_STOP + b"<|im_start|>user\nmore")
    assert _generate(m, b"x", max_new=80, stop=CHAT_STOP) == b"red, blue, green"


def test_generate_without_a_stop_runs_to_the_budget():
    """Base models keep the old unbounded behavior."""
    m = _ScriptedModel(b"abc" + CHAT_STOP + b"defg")
    assert _generate(m, b"x", max_new=6) == b"abc" + CHAT_STOP[:3]


def test_chat_template_puts_the_prompt_in_an_assistant_turn():
    framed = CHAT_TEMPLATE.format(prompt="List three colors.")
    assert framed.startswith("<|im_start|>user\nList three colors.<|im_end|>")
    assert framed.endswith("<|im_start|>assistant\n")


def test_chat_stop_drops_the_closing_bracket():
    """Matches the serving path: a byte model reproduces a marker approximately."""
    assert CHAT_STOP == b"<|im_end|"


def test_yes_or_no_form_check_accepts_either_answer():
    """Grades obedience, not correctness: both answers must pass the same rule."""
    from eval.ifeval import check_starts_with_yes_or_no
    assert check_starts_with_yes_or_no("Yes, it is.")
    assert check_starts_with_yes_or_no("No, it is not.")


def test_yes_or_no_form_check_rejects_a_preamble():
    from eval.ifeval import check_starts_with_yes_or_no
    assert not check_starts_with_yes_or_no("Well, let me think about that. Yes.")


def test_form_set_rules_are_all_answer_independent():
    """The point of the form set: no rule may encode which answer is correct.

    `contains` and `starts_with` are answer-pinned in the mixed set ("What is the
    capital of France?" + starts_with "Paris") and answer-INDEPENDENT here ("Begin
    your reply with the word Certainly"). The rule name cannot tell those apart, so
    the invariant is the checkable one: the required text must be dictated by the
    prompt. If the prompt names it, obeying needs no knowledge.
    """
    from eval.ifeval import data_path_for
    with open(data_path_for("form"), encoding="utf-8") as f:
        items = json.load(f)["items"]
    for it in items:
        prompt = it["prompt"].lower()
        for r in it["rules"]:
            if r["name"] in ("contains", "starts_with"):
                assert r["text"].lower() in prompt, \
                    f"answer-pinned rule in the form set: {it['prompt']!r} requires {r['text']!r}"
            for word in r.get("words", ()):
                assert word.lower() in prompt, \
                    f"forbidden word not stated in the prompt: {it['prompt']!r}"
            if r["name"] == "forbidden_letter":
                assert r["letter"].lower() in prompt, \
                    f"forbidden letter not stated in the prompt: {it['prompt']!r}"


def test_named_sets_resolve_to_files_that_exist():
    from eval.ifeval import SAMPLE_SETS, data_path_for
    for name in SAMPLE_SETS:
        assert os.path.isfile(data_path_for(name)), name


def test_an_unknown_set_name_is_rejected():
    import pytest as _pytest
    from eval.ifeval import data_path_for
    with _pytest.raises(ValueError, match="unknown ifeval set"):
        data_path_for("nope")


def test_the_sample_set_ships_with_items():
    assert os.path.isfile(DEFAULT_DATA)
    with open(DEFAULT_DATA, encoding="utf-8") as f:
        assert len(json.load(f)["items"]) >= 30


# ------------------------------------------------------------------------------------
# balanced_subset: `limit` used to be a raw prefix. The form set groups families in
# contiguous blocks, so any limited run graded one family and reported it as the score.

def _form_items():
    with open(ifeval.data_path_for("form"), encoding="utf-8") as f:
        return json.load(f)["items"]


def test_limit_covers_every_family_not_just_the_first():
    items = _form_items()
    picked = ifeval.balanced_subset(items, 18)
    fams = {r["name"] for it in picked for r in it["rules"]}
    all_fams = {r["name"] for it in items for r in it["rules"]}
    assert fams == all_fams


def test_limit_returns_exactly_that_many():
    items = _form_items()
    for n in (1, 9, 45, 279):
        assert len(ifeval.balanced_subset(items, n)) == n


def test_limit_at_or_above_size_returns_everything():
    items = _form_items()
    assert len(ifeval.balanced_subset(items, len(items) + 50)) == len(items)
    assert len(ifeval.balanced_subset(items, None)) == len(items)


def test_subset_is_deterministic():
    items = _form_items()
    a = [it["prompt"] for it in ifeval.balanced_subset(items, 27)]
    b = [it["prompt"] for it in ifeval.balanced_subset(items, 27)]
    assert a == b


def test_subset_never_repeats_an_item():
    items = _form_items()
    picked = ifeval.balanced_subset(items, 90)
    assert len({it["prompt"] for it in picked}) == 90


def test_form_set_is_powered_and_balanced():
    """26 items could not resolve a 12pt move (p=0.40). Guard the size and the floor
    per family so a future edit cannot quietly shrink the instrument."""
    items = _form_items()
    assert len(items) >= 280
    counts = Counter(r["name"] for it in items for r in it["rules"])
    assert min(counts.values()) >= 20, counts


def test_every_form_rule_has_a_checker():
    items = _form_items()
    for it in items:
        for r in it["rules"]:
            assert r["name"] in ifeval.CHECKERS, r["name"]


def test_form_prompts_are_unique():
    items = _form_items()
    assert len({it["prompt"] for it in items}) == len(items)
