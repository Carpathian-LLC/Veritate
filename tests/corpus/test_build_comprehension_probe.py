# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the reading-comprehension probe builder over a synthetic passage: items are seeded,
#   the prefix is bounded and ends right before the slot, the correct word is the slot's
#   own text, the three distractors are length-matched content words absent from the
#   recent prefix, and a passage too short to carry a band is skipped rather than
#   half-built.
# tests/corpus/test_build_comprehension_probe.py
# ------------------------------------------------------------------------------------
# Imports:

import random

from training.builders.eval import build_comprehension_probe as bc

# ------------------------------------------------------------------------------------
# Constants

WORDS = ["harbour", "lantern", "meadow", "compass", "orchard", "granite", "thistle", "saddle", "furnace", "willow",
         "canyon", "beacon", "pasture", "chimney", "hollow", "quarry", "timber", "bramble", "summit", "trellis"]

# ------------------------------------------------------------------------------------
# Functions


def _passage(n_sentences=120, seed=3):
    rng = random.Random(seed)
    sents = []
    for _ in range(n_sentences):
        a, b, c = rng.sample(WORDS, 3)
        sents.append(f"The {a} stood beside the {b} while the {c} waited in the rain.")
    return " ".join(sents).encode()


def test_items_are_seeded_bounded_and_length_matched(tmp_path):
    src = tmp_path / "grade_x_source.txt"
    src.write_bytes(_passage() + b"\n\n")
    band = bc.build_band("x", src, random.Random(1))
    again = bc.build_band("x", src, random.Random(1))
    assert band == again and band["level"] == "x" and band["n_items"] == len(band["items"]) >= bc.ITEMS_PER_BAND // 2
    text = src.read_bytes().rstrip(b"\n\r ")
    for it in band["items"]:
        s = it["slot_offset"]
        assert text[s:s + len(it["correct"])].decode() == it["correct"]
        assert it["prefix_bytes"] == text[max(0, s - bc.PREFIX_MAX_BYTES):s].decode()
        assert s >= bc.MIN_PREFIX_BYTES and len(it["prefix_bytes"]) <= bc.PREFIX_MAX_BYTES
        recent = it["prefix_bytes"][-bc.RECENT_WINDOW_BYTES:].lower()
        assert len(it["distractors"]) == 3 and len(set(it["distractors"])) == 3
        for d in it["distractors"]:
            assert d != it["correct"].lower() and abs(len(d) - len(it["correct"])) <= bc.LENGTH_TOLERANCE
            assert d not in recent and bc.is_content_word(d)


def test_a_short_or_missing_passage_is_skipped(tmp_path, capsys):
    src = tmp_path / "short.txt"
    src.write_bytes(b"The harbour stood beside the lantern.")
    assert bc.build_band("x", src, random.Random(1)) is None
    assert bc.build_band("x", tmp_path / "nope.txt", random.Random(1)) is None
    assert "[skip]" in capsys.readouterr().out
