# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the developmental corpus builder: one event is stated four ways (active, passive,
#   who-question, what-question) so roles cannot be read off position; a held-out pair is
#   shown declaratively only and its who/what test items carry the same verb; the stream
#   is seeded; write_split takes the trailing VAL_FRACTION.
# tests/corpus/test_build_curriculum_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import build_curriculum_corpus as cc

# ------------------------------------------------------------------------------------
# Functions


def test_an_event_is_stated_four_ways_and_declaratively_only_when_held_out():
    verb = ("chase", "chases", "chased")
    full = cc.action_block("the dog", "the cat", verb)
    assert full == ("The dog chases the cat. The cat is chased by the dog. "
                    "Who chases the cat? The dog does. What does the dog chase? The cat.")
    declarative = cc.action_block("the dog", "the cat", verb, questions=False)
    assert declarative == "The dog chases the cat. The cat is chased by the dog."


def test_held_out_pairs_are_deterministic_and_their_tests_match_the_shown_verb():
    a, b = cc.holdout_verb_map(0.25), cc.holdout_verb_map(0.25)
    assert a == b and len(a) == int(len(cc.ANIMATE) * (len(cc.ANIMATE) - 1) * 0.25)
    items = cc.test_items(a)
    assert len(items) == len(a)
    for it in items:
        base, pres, _ = a[(it["gold_subj"], it["gold_obj"])]
        assert it["decl"] == f"{cc._cap(it['gold_subj'])} {pres} {it['gold_obj']}."
        assert it["who_q"].startswith(f"Who {pres} ") and it["what_q"].endswith(f" {base}?")
    assert cc.holdout_verb_map(0.0) == {}


def test_the_stream_is_seeded_and_held_out_pairs_never_get_their_questions():
    held = cc.holdout_verb_map(0.3)
    data = cc.build_stream(20_000, verbmap=held)
    assert data == cc.build_stream(20_000, verbmap=held) and len(data) >= 20_000
    text = data.decode()
    for (s, o), (base, pres, _) in held.items():
        assert f"Who {pres} {o}? {cc._cap(s)} does." not in text
        assert f"What does {s} {base}? {cc._cap(o)}." not in text
    assert cc.BLOCK_SEP in text


def test_write_split_takes_the_trailing_val_fraction(tmp_path):
    data = bytes(range(200))
    cut, n_val = cc.write_split(data, str(tmp_path / "t.bin"), str(tmp_path / "v.bin"))
    assert n_val == int(200 * cc.VAL_FRACTION) and cut == 200 - n_val
    assert (tmp_path / "t.bin").read_bytes() == data[:cut] and (tmp_path / "v.bin").read_bytes() == data[cut:]
