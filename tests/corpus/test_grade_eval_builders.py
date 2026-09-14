# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the grammar and reasoning eval builders behind the smartness meter: seeded, so a
#   rebuild is byte-identical; every pair differs in exactly the rule it tests; every
#   reasoning item's answer follows from its prompt; main() writes one JSONL per type
#   under the eval root.
# tests/corpus/test_grade_eval_builders.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import random

from readers import paths
from training.builders.eval import build_grammar_eval as g
from training.builders.eval import build_reasoning_eval as r

# ------------------------------------------------------------------------------------
# Functions


def test_grammar_pairs_are_seeded_well_formed_and_differ_only_in_the_rule():
    for fn in (g.sv_agreement, g.articles, g.tense, g.word_order):
        a, b = fn(random.Random(1)), fn(random.Random(1))
        assert a == b and len(a) == g.N_PER_TYPE
        for pair in a:
            assert pair["correct"] != pair["incorrect"] and pair["type"] == fn.__name__
            assert pair["correct"].endswith((".", "?")) and pair["incorrect"].endswith((".", "?"))
    for pair in g.articles(random.Random(2)):
        noun = pair["correct"].split(" ", 3)[3].rstrip(".")
        assert noun in g.VOWEL_NOUNS or noun in g.CONS_NOUNS
        assert (" an " in pair["correct"]) == (noun in g.VOWEL_NOUNS)
    for pair in g.tense(random.Random(3)):
        verb_ok, verb_bad = pair["correct"].split()[2], pair["incorrect"].split()[2]
        past = pair["correct"].startswith("Yesterday")
        assert (verb_ok in g.PAST_VERBS) == past and (verb_bad in g.PRESENT_VERBS) == past


def test_reasoning_items_answer_their_own_prompts():
    for fn in (r.recall, r.pattern, r.deduction1, r.deduction_n):
        items = fn(random.Random(5))
        assert items == fn(random.Random(5)) and len(items) == r.N_PER_TIER
        assert all(i["prompt"] and i["answer"] and i["type"] == fn.__name__ for i in items)
    for item in r.deduction_n(random.Random(6)):
        names = [w.rstrip(".") for w in item["prompt"].split() if w[0].isupper() and w.rstrip(".").isalpha()]
        assert item["answer"] == item["prompt"].split(". ")[1].split()[-1] and item["answer"] in names
    for item in r.deduction1(random.Random(7)):
        assert item["prompt"].rstrip().endswith("is a") and item["answer"] in item["prompt"]


def test_main_writes_one_jsonl_per_type_under_the_eval_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "GRADE_EVAL_GRAMMAR_ROOT", str(tmp_path / "grammar"))
    monkeypatch.setattr(paths, "GRADE_EVAL_REASONING_ROOT", str(tmp_path / "reasoning"))
    assert g.main() == 0 and r.main() == 0
    for root, n in (("grammar", 4), ("reasoning", 4)):
        files = sorted((tmp_path / root).glob("*.jsonl"))
        assert len(files) == n
        for f in files:
            rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
            assert len(rows) == g.N_PER_TYPE and all(rows[0].keys() == r.keys() for r in rows)
