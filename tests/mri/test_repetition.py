# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for the shared byte-level repetition controller. covers the hard
#   no-repeat-ngram ban, the soft suffix-repeat penalty, the sliding window, and
#   the disabled (parity-off) no-op. the c engine sampler mirrors this algorithm.
# tests/mri/test_repetition.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from inference.decode.repetition import RepetitionController, REP_MIN_MATCH

# ------------------------------------------------------------------------------------
# Constants

VOCAB = 256

# ------------------------------------------------------------------------------------
# Functions


def _feed(rc, data):
    for b in data:
        rc.observe(b)
    return rc


def test_no_repeat_ngram_bans_loop_continuation():
    """no_repeat_ngram masks the byte that would complete an already-seen n-gram."""
    rc = _feed(RepetitionController(VOCAB, 256, 0.0, 8), b"abcdefghabcdefg")
    ban, _ = rc.compute()
    assert ban is not None and bool(ban[ord("h")]) is True


def test_disabled_controller_is_noop():
    """A controller with penalty=0 and ngram=0 emits no ban or penalty (parity off)."""
    rc = _feed(RepetitionController(VOCAB, 256, 0.0, 0), b"aaaaaaaaaaaaaaaa")
    assert rc.compute() == (None, None)


def test_soft_penalty_scales_with_run_length():
    """Below the ngram threshold, a run-extending byte is demoted by penalty*(run-min+1)."""
    rc = _feed(RepetitionController(VOCAB, 256, 1.0, 0), b"abcdeabcd")
    ban, soft = rc.compute()
    assert ban is None
    assert soft is not None
    assert abs(float(soft[ord("e")]) - (1.0 * (5 - REP_MIN_MATCH + 1))) < 1e-6


def test_window_bounds_the_search():
    """A repeat older than rep_window is not banned."""
    rc = _feed(RepetitionController(VOCAB, 4, 0.0, 8), b"abcdefghabcdefg")
    ban, _ = rc.compute()
    assert ban is None
