# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for _ngram_draft, the pure byte-search that drafts the next bytes
#   for prompt/n-gram lookahead decode. No model, no torch: the speculation
#   correctness (byte-exactness) is covered by the CPU parity smoke; here we pin
#   the draft-selection contract the decoder relies on.
# tests/mri/test_lookahead_draft.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from inference.backends.pytorch import _ngram_draft

# ------------------------------------------------------------------------------------
# Functions

def test_suffix_match_returns_following_bytes():
    """A suffix that recurs earlier drafts the bytes that followed the earlier hit."""
    ctx = list(b"abcdefXYabcdef")
    assert _ngram_draft(ctx, 6, 3) == list(b"XYa")


def test_most_recent_earlier_occurrence_wins():
    """With two earlier matches the draft comes from the most recent one."""
    ctx = list(b"QZ..QK..Q")
    # suffix "Q" recurs at index 0 and 4; the index-4 hit is followed by "K..".
    assert _ngram_draft(ctx, 1, 3) == list(b"K..")


def test_no_earlier_occurrence_returns_empty():
    """A suffix that never recurs earlier drafts nothing."""
    assert _ngram_draft(list(b"abcdefgh"), 4, 8) == []


def test_context_shorter_than_ngram_returns_empty():
    """A context with fewer than ngram+1 bytes drafts nothing."""
    assert _ngram_draft(list(b"abc"), 8, 8) == []


def test_zero_max_draft_returns_empty():
    """A zero draft budget drafts nothing even on a match."""
    assert _ngram_draft(list(b"abababab"), 2, 0) == []


def test_draft_is_capped_at_max_draft():
    """The draft never exceeds max_draft bytes."""
    ctx = list(b"xyzABCDEFGHxyz")
    assert _ngram_draft(ctx, 3, 2) == list(b"AB")
