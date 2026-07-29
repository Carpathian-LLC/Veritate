# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The byte-level stop in backends_routes matches a turn marker MINUS its last char,
#   so a reply always ends on a partial marker that str.split cannot cut. The buffered
#   chat path leaked "<|im_end|" into every /v1/chat/completions reply until _trim
#   learned to strip the partial too. These pin both halves.
# tests/mri/test_chat_stop_trim.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest

from veritate_mri.routes.hybrid_routes import STOP_MARKERS, _dynamic_stop_hold, _trim

# ------------------------------------------------------------------------------------
# Constants

ANSWER = "Paris."

# ------------------------------------------------------------------------------------
# Functions


@pytest.mark.parametrize("marker", STOP_MARKERS)
def test_partial_stop_marker_is_stripped(marker):
    """_trim removes a marker missing its final char, which is how the stop fires."""
    assert _trim(ANSWER + marker[:-1]) == ANSWER


@pytest.mark.parametrize("marker", STOP_MARKERS)
def test_complete_stop_marker_and_everything_after_is_cut(marker):
    """_trim cuts at a whole marker and drops the self-conversation behind it."""
    assert _trim(f"{ANSWER}{marker}\nuser\nanother question") == ANSWER


def test_text_that_merely_resembles_a_marker_survives():
    """A reply is not allowed to lose real characters to the partial-marker strip."""
    assert _trim("Use the pipe character | to chain commands.") == \
        "Use the pipe character | to chain commands."


def test_a_clean_answer_is_returned_unchanged():
    assert _trim(ANSWER) == ANSWER


def test_hold_is_zero_when_the_tail_is_not_a_marker_prefix():
    assert _dynamic_stop_hold("all done.") == 0


def test_hold_counts_the_trailing_marker_prefix():
    assert _dynamic_stop_hold(ANSWER + "<|im_") == len("<|im_")
