# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - streaming holdback in hybrid_routes.py: _dynamic_stop_hold holds back only a
#   growing stop-marker prefix, and the end-of-stream flush strips a partial marker
#   so a run that stops mid-marker never leaks "<|im" to the caller.
# tests/mri/test_streaming_stop_hold.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from routes import hybrid_routes as hr

# ------------------------------------------------------------------------------------
# Constants

MARKER          = "<|im_end|>"
MARKER_PREFIXES = [MARKER[:i] for i in range(1, len(MARKER))]
LONG_PREFIX     = "x" * (1 << 20)
BODY            = "The quick brown fox jumps over the lazy dog. " * 40

# ------------------------------------------------------------------------------------
# Functions

def _fake_events(bytes_out):
    for b in bytes_out:
        yield {"kind": "token", "byte": int(b)}
    yield {"kind": "stop"}


def _stream_to_str(bytes_out):
    return "".join(val for tag, val in hr._local_stream_items(_fake_events(bytes_out)) if tag == "text")


@pytest.mark.parametrize("text", ["", "Hello world.", "The answer is 42.", "x < 5 is true"])
def test_dynamic_stop_hold_is_zero_for_clean_tails(text):
    """_dynamic_stop_hold returns 0 when the tail is not a stop-marker prefix."""
    assert hr._dynamic_stop_hold(text) == 0


@pytest.mark.parametrize("prefix", MARKER_PREFIXES)
def test_dynamic_stop_hold_matches_partial_marker_length(prefix):
    """_dynamic_stop_hold returns the length of a growing stop-marker prefix at the tail."""
    assert hr._dynamic_stop_hold("hello " + prefix) == len(prefix)


def test_dynamic_stop_hold_holds_back_a_complete_marker():
    """A complete stop marker at the tail is held back rather than streamed."""
    assert hr._dynamic_stop_hold("answer." + MARKER) > 0


def test_dynamic_stop_hold_never_exceeds_stop_hold():
    """The held-back length never exceeds STOP_HOLD."""
    assert hr._dynamic_stop_hold("answer." + MARKER) <= hr.STOP_HOLD


def test_dynamic_stop_hold_depends_only_on_the_tail():
    """_dynamic_stop_hold reads only the tail: a megabyte of leading text never changes the result."""
    assert hr._dynamic_stop_hold(LONG_PREFIX + "<|im") == hr._dynamic_stop_hold("<|im")


def test_dynamic_stop_hold_is_zero_for_a_long_clean_string():
    """A long string with no marker prefix at the tail holds back nothing."""
    assert hr._dynamic_stop_hold(LONG_PREFIX) == 0


def test_stream_plain_text_reassembles_verbatim():
    """A byte stream with no marker streams back verbatim."""
    assert _stream_to_str(b"Hello world.") == "Hello world."


def test_stream_partial_marker_at_end_does_not_leak():
    """A stream ending mid-marker emits the body without the partial marker."""
    assert _stream_to_str(b"answer text<|im") == "answer text"


def test_stream_complete_marker_at_end_is_stripped():
    """A stream ending in a complete <|im_end|> emits the body without the marker."""
    assert _stream_to_str(b"answer text" + MARKER.encode("utf-8")) == "answer text"


@pytest.mark.parametrize("tail", MARKER_PREFIXES)
def test_stream_long_output_never_leaks_marker_prefix(tail):
    """A long output ending on any marker prefix leaks no part of the marker."""
    assert tail not in _stream_to_str(BODY.encode("utf-8") + tail.encode("utf-8"))


@pytest.mark.parametrize("tail", MARKER_PREFIXES)
def test_stream_long_output_keeps_the_body(tail):
    """A long output ending on any marker prefix still delivers the body."""
    assert _stream_to_str(BODY.encode("utf-8") + tail.encode("utf-8")).startswith(BODY.strip()[:32])
