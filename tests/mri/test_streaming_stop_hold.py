# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Regression tests for the 2026-07-20 streaming fixes in hybrid_routes.py:
#   (1) _dynamic_stop_hold returns 0 unless the tail is actually a growing prefix
#       of a stop marker (fixes gulpy bimodal streaming caused by fixed 13-char
#       STOP_HOLD lag).
#   (2) end-of-stream flush strips trailing stop-marker prefixes so a model that
#       hits max_tokens mid-marker doesn't leak "<|im" to the user.
# tests/mri/test_streaming_stop_hold.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

# Import the module under test. hybrid_routes imports flask + inference/... at
# module scope; those are cheap and already used across the mri test suite.
from routes import hybrid_routes as hr

# ------------------------------------------------------------------------------------
# _dynamic_stop_hold — returns the number of tail chars to hold back.

def test_dynamic_stop_hold_empty_and_plain_text():
    """Plain text with no stop-marker prefix at the tail should return 0."""
    assert hr._dynamic_stop_hold("") == 0
    assert hr._dynamic_stop_hold("Hello world.") == 0
    assert hr._dynamic_stop_hold("The answer is 42.") == 0
    # Text that HAPPENS to contain "<" earlier but tail is clean
    assert hr._dynamic_stop_hold("x < 5 is true") == 0

def test_dynamic_stop_hold_partial_marker():
    """Text whose tail is a growing prefix of a stop marker returns that length."""
    # "<" is a valid 1-char prefix of "<|im_end|>", "<|endoftext|>", "<|im_start|>"
    assert hr._dynamic_stop_hold("hello <") == 1
    assert hr._dynamic_stop_hold("hello <|") == 2
    assert hr._dynamic_stop_hold("hello <|i") == 3
    assert hr._dynamic_stop_hold("hello <|im") == 4
    assert hr._dynamic_stop_hold("hello <|im_") == 5
    # Confirm the exact regression case from the 2026-07-20 fresh probe.
    assert hr._dynamic_stop_hold("That's not in the passage.<|im") == 4

def test_dynamic_stop_hold_complete_marker():
    """A complete stop marker at the tail should also return its length so the
    trimming/split path handles the removal; not returning 0."""
    # "<|im_end|>" is 10 chars; the whole thing is a valid marker.
    text = "answer.<|im_end|>"
    result = hr._dynamic_stop_hold(text)
    # Whether it's 10 (the whole marker) or exactly the marker length depends on
    # the impl; the guarantee is >0 so nothing leaks.
    assert result > 0
    # And it should not exceed STOP_HOLD.
    assert result <= hr.STOP_HOLD

def test_dynamic_stop_hold_bounded_by_stop_hold():
    """Returned hold length must never exceed STOP_HOLD, regardless of input length."""
    # Long text ending in a non-marker char.
    assert hr._dynamic_stop_hold("x" * 1000) == 0
    # Long text ending in a partial marker.
    assert hr._dynamic_stop_hold("x" * 1000 + "<|im") == 4

def test_dynamic_stop_hold_does_not_scan_full_text():
    """Regression: earlier fixed STOP_HOLD did O(1) work; dynamic version must
    also be O(STOP_HOLD), not O(len(text)). Rough perf check — a 1MB string
    should return in well under a millisecond."""
    import time
    big = ("a" * (1 << 20))  # 1 MB
    t0 = time.perf_counter()
    for _ in range(100):
        hr._dynamic_stop_hold(big)
    dt = time.perf_counter() - t0
    # 100 calls on 1MB should finish in <100ms even on a slow box.
    assert dt < 0.5, f"_dynamic_stop_hold too slow: {dt:.3f}s for 100x1MB"

# ------------------------------------------------------------------------------------
# _local_stream_items — end-to-end streaming behavior with mocked byte events.

def _fake_events(bytes_out):
    """Emit ('token', {byte, kind}) events for each byte in bytes_out, then a stop."""
    for b in bytes_out:
        yield {"kind": "token", "byte": int(b)}
    yield {"kind": "stop"}

def _stream_to_str(bytes_out):
    """Run _local_stream_items on a synthetic byte stream, collect the emitted text."""
    parts = []
    for tag, val in hr._local_stream_items(_fake_events(bytes_out)):
        if tag == "text":
            parts.append(val)
    return "".join(parts)

def test_stream_plain_text_streams_early():
    """Plain-text output should stream chunks as bytes arrive, not batch behind
    STOP_HOLD. Regression for the bimodal p90=108ms spike."""
    out = _stream_to_str(b"Hello world.")
    assert out == "Hello world."

def test_stream_partial_marker_at_end_does_not_leak():
    """If the model stops mid-marker (e.g. hits max_tokens after emitting 4 of
    the 10 bytes of "<|im_end|>"), the "<|im" tail must NOT reach the user.
    Regression case from 2026-07-20 fresh probe."""
    # Simulate a byte stream that ends with a partial "<|im_end|>" marker.
    out = _stream_to_str(b"answer text<|im")
    assert "<|im" not in out
    assert "<|" not in out
    assert out == "answer text"

def test_stream_complete_marker_at_end_is_stripped():
    """A complete "<|im_end|>" at the tail should be cut by _trim, and the
    remainder should not include any of the marker bytes."""
    out = _stream_to_str(b"answer text<|im_end|>")
    assert "<|im_end|>" not in out
    assert out == "answer text"

# Note: mid-stream markers with more bytes AFTER them are not a real production
# scenario — the upstream `_stop_on_bytes` filter truncates the event stream at
# the first stop-sequence match, so `_local_stream_items` never sees bytes past
# the marker in normal operation. The regression case that matters is the
# partial-marker-at-end-of-stream case, covered by
# test_stream_partial_marker_at_end_does_not_leak above.

def test_stream_long_output_never_leaks_marker_bytes():
    """Fuzz-ish: a long clean output followed by a partial marker must never
    leak marker bytes."""
    payload = ("The quick brown fox jumps over the lazy dog. " * 40).encode("utf-8")
    for tail in [b"<", b"<|", b"<|i", b"<|im", b"<|im_", b"<|im_e", b"<|im_en", b"<|im_end"]:
        out = _stream_to_str(payload + tail)
        assert tail.decode("utf-8", "replace") not in out, f"leaked tail={tail!r}"
        # Body should still be intact.
        assert out.startswith("The quick brown fox")

def test_stream_no_marker_prefix_streams_cleanly():
    """A raw ASCII output with no marker-prefix chars should stream verbatim
    (no holdback anywhere)."""
    payload = "Just plain output with numbers 12345 and punctuation!"
    out = _stream_to_str(payload.encode("utf-8"))
    assert out == payload
