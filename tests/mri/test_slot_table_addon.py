# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the slot-table inference addon: it watches the bytes already generated and biases the
#   next-byte logits to suppress doc-boundary collapse, loops, n-gram echoes and
#   wrong-gender pronoun completions. It sits on the serving path, so what is pinned is
#   the addon contract (reset / observe / bias_logits), that it never mutates the logits
#   it was handed, that a blocked byte stays blocked whatever else fires, and that the
#   pronoun rule suppresses only bytes that cannot also start a correct-gender pronoun.
# tests/mri/test_slot_table_addon.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib

import torch

addon_mod = importlib.import_module("inference.addons.slot_table.addon")

# ------------------------------------------------------------------------------------
# Constants

OFF = {"ngram_penalty": 0.0, "rep_penalty": 1.0, "name_boost": 0.0, "pronoun_penalty": 0.0}

# ------------------------------------------------------------------------------------
# Functions


def _addon(**kw):
    return addon_mod.Addon(**{**OFF, **kw})


def _feed(a, text):
    for b in text.encode():
        a.observe(b)
    return a


def test_the_blocked_byte_is_always_impossible():
    """Byte 0 ends a document; sampling it mid-reply collapses the turn."""
    assert _addon().bias_logits(torch.zeros(256))[0] == float("-inf")


def test_the_input_logits_are_not_mutated():
    """The caller reuses its tensor; an in-place bias would compound every step."""
    logits = torch.zeros(256)
    _addon().bias_logits(logits)
    assert torch.equal(logits, torch.zeros(256))


def test_reset_forgets_the_window_and_what_it_learned():
    a = _feed(_addon(), "The girl was named Marisol. ")
    assert a.names and a.gender is not None
    a.reset()
    assert a.names == [] and a.gender is None and len(a.window) == 0


def test_a_named_entity_is_picked_up_and_boosted_at_a_word_start():
    """Boosting a seen name's first byte is the addon's one positive bias."""
    a = _feed(_addon(name_boost=2.0), "The girl was named Marisol and ")
    assert b"Marisol" in a.names
    out = a.bias_logits(torch.zeros(256))
    assert out[ord("M")] == 2.0


def test_the_name_boost_stays_off_at_a_sentence_start():
    """Every sentence would otherwise open with a character's name."""
    a = _feed(_addon(name_boost=2.0), "The girl was named Marisol. ")
    assert a.bias_logits(torch.zeros(256))[ord("M")] == 0.0


def test_a_repeated_byte_is_penalised_and_the_sign_is_handled():
    """Dividing a positive logit and multiplying a negative one both move it down; doing
    the same operation to both would reward the loop it is meant to break."""
    a = _feed(_addon(rep_penalty=2.0, rep_lookback=8), "aaaa")
    logits = torch.zeros(256)
    logits[ord("a")] = 4.0
    logits[ord("b")] = 4.0
    up = a.bias_logits(logits)
    assert up[ord("a")] == 2.0 and up[ord("b")] == 4.0
    logits[ord("a")] = -4.0
    down = a.bias_logits(logits)
    assert down[ord("a")] == -8.0


def test_an_ngram_that_already_happened_is_penalised():
    """The continuation that would repeat a seen 4-gram takes the hit, others do not."""
    a = _feed(_addon(ngram_n=4, ngram_penalty=3.0), "abcdabc")
    out = a.bias_logits(torch.zeros(256))
    assert out[ord("d")] == -3.0
    assert out[ord("z")] == 0.0


def test_the_wrong_gender_pronoun_byte_is_suppressed():
    """The anchor is male, so the byte that can only start "she"/"her..." is penalised.
    "h" is not: it also starts "he", "his" and "him", and the rule never suppresses a byte
    that can still reach a correct-gender pronoun."""
    a = _feed(_addon(pronoun_penalty=5.0), "The boy went out. Then ")
    assert a.gender == addon_mod.GENDER_MALE
    out = a.bias_logits(torch.zeros(256))
    assert out[ord("s")] == -5.0 and out[ord("S")] == -5.0
    assert out[ord("h")] == 0.0


def test_mid_word_the_wrong_gender_continuation_is_suppressed():
    """After "h" with a female anchor, "i" can only continue "his"/"him"/"himself"."""
    a = _feed(_addon(pronoun_penalty=5.0), "The girl went out. Then h")
    assert a.gender == addon_mod.GENDER_FEMALE
    assert a.bias_logits(torch.zeros(256))[ord("i")] == -5.0


def test_the_latest_anchor_wins():
    """Gender is whichever anchor appeared last, so a scene change is followed."""
    a = _feed(_addon(), "The girl left. The boy stayed.")
    assert a.gender == addon_mod.GENDER_MALE


def test_a_shared_continuation_is_not_suppressed():
    """Mid-word after "h", "e" continues both "he" (male) and "her" is female-only, so a
    byte that can still reach a correct-gender pronoun must survive."""
    a = _feed(_addon(pronoun_penalty=5.0), "The girl went out. Then h")
    out = a.bias_logits(torch.zeros(256))
    assert out[ord("e")] == 0.0


def test_the_window_is_bounded():
    """It biases every byte generated; an unbounded window would grow without limit."""
    a = _feed(_addon(window_bytes=16), "x" * 100)
    assert len(a.window) == 16
