# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - A mix whose heaviest corpus ships no _val.bin trains blind. core_50m ran 1.64M
#   steps over 27 hours on the_pile with val_bin="" and never said a word about it:
#   the eval branch is gated on a truthy val path, so the whole validation loop
#   no-oped and the run looked healthy with a train curve and nothing to check it.
# - resolve_and_weight sorts weight-DESCENDING, so validation follows the heaviest
#   corpus, not the first stem written in the spec. These tests pin that, and pin
#   that a missing bin produces a warning rather than silence.
# tests/training/test_val_path_resolution.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

TRAINER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "veritate_mri", "training")
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)

import veritate_trainer as vt

# ------------------------------------------------------------------------------------
# Helpers


def _bin(tmp_path, name, size=64):
    p = tmp_path / name
    p.write_bytes(b"\x00" * size)
    return str(p)


# ------------------------------------------------------------------------------------
# Present val bin


def test_returns_val_path_when_file_exists(tmp_path):
    train = _bin(tmp_path, "a_train.bin")
    val = _bin(tmp_path, "a_val.bin")
    got, warning = vt.resolve_val_path([(train, val, 1.0)])
    assert got == val
    assert warning == ""


def test_follows_the_heaviest_corpus_not_the_first_listed(tmp_path):
    light_val = _bin(tmp_path, "light_val.bin")
    heavy_val = _bin(tmp_path, "heavy_val.bin")
    mix = [
        (_bin(tmp_path, "heavy_train.bin"), heavy_val, 0.7),
        (_bin(tmp_path, "light_train.bin"), light_val, 0.3),
    ]
    got, warning = vt.resolve_val_path(mix)
    assert got == heavy_val
    assert warning == ""


# ------------------------------------------------------------------------------------
# Absent val bin


def test_warns_when_top_corpus_has_no_val_bin(tmp_path):
    mix = [(_bin(tmp_path, "the_pile_train.bin"), "", 1.0)]
    got, warning = vt.resolve_val_path(mix)
    assert got is None
    assert "NO validation loss" in warning


def test_warns_when_val_path_is_set_but_missing_on_disk(tmp_path):
    stale = str(tmp_path / "gone_val.bin")
    mix = [(_bin(tmp_path, "x_train.bin"), stale, 1.0)]
    got, warning = vt.resolve_val_path(mix)
    assert got is None
    assert "NO validation loss" in warning


def test_warns_when_top_corpus_lacks_val_even_though_a_lighter_one_has_it(tmp_path):
    """The core_50m shape: plenty of val bins available, none on the top weight."""
    mix = [
        (_bin(tmp_path, "the_pile_train.bin"), "", 0.9),
        (_bin(tmp_path, "fineweb_edu_train.bin"), _bin(tmp_path, "fineweb_edu_val.bin"), 0.1),
    ]
    got, warning = vt.resolve_val_path(mix)
    assert got is None
    assert "top weight" in warning


def test_empty_mix_warns_rather_than_raising():
    got, warning = vt.resolve_val_path([])
    assert got is None
    assert warning
