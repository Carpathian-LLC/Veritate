# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - engine stop-sequence header token and the anchored prompt cut.
# - both are pure functions on CTracedSubprocess; no subprocess is spawned.
# tests/engine/test_stop_sequences.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from inference.backends import c_engine

# ------------------------------------------------------------------------------------
# Constants

SEQ_CAP = 1024

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def stop_token():
    return c_engine.CTracedSubprocess._stop_token


@pytest.fixture
def anchor():
    return c_engine.CTracedSubprocess._anchor_prompt


def test_no_stop_sequences_emits_the_disabled_token(stop_token):
    """An empty stop list renders the sentinel that leaves engine generation unchanged."""
    assert stop_token(()) == c_engine.STOP_SEQ_NONE


def test_chatml_markers_render_comma_separated(stop_token):
    """The ChatML turn markers ride the header as one comma-separated token."""
    assert stop_token(("<|im_end|>", "<|im_start|>")) == "<|im_end|>,<|im_start|>"


def test_bytes_markers_render_as_their_bytes(stop_token):
    """A bytes stop sequence reaches the engine as its bytes, not as repr("b'...'")."""
    assert stop_token((b"<|im_end|>", b"<|im_start|>")) == "<|im_end|>,<|im_start|>"


def test_chat_stop_seq_output_survives_the_header_encoding(stop_token):
    """The markers _chat_stop_seq returns render verbatim, so the engine can match them."""
    from routes.backends_routes import _chat_stop_seq
    wire = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    token = stop_token(_chat_stop_seq(wire))
    assert "b'" not in token
    assert token.split(c_engine.STOP_SEQ_SEP) == [m.decode() for m in _chat_stop_seq(wire)]


def test_embedded_newline_is_wire_escaped(stop_token):
    """A stop sequence containing a newline is escaped the way the prompt line is."""
    assert stop_token(("\ncontext:",)) == c_engine.NEWLINE_WIRE_ESCAPE + "context:"


def test_sequence_containing_the_separator_is_dropped(stop_token):
    """A comma-bearing sequence cannot be expressed on the wire, so it is dropped."""
    assert stop_token(("a,b",)) == c_engine.STOP_SEQ_NONE


def test_over_long_sequence_is_dropped(stop_token):
    """A sequence longer than the engine's per-entry cap is dropped, never truncated."""
    assert stop_token(("x" * (c_engine.STOP_SEQ_MAX_LEN + 1),)) == c_engine.STOP_SEQ_NONE


def test_sequence_count_is_capped_at_the_engine_maximum(stop_token):
    """No more than VERITATE_MAX_STOPS sequences reach the engine."""
    token = stop_token(tuple(f"s{i}" for i in range(c_engine.STOP_SEQ_MAX + 4)))
    assert len(token.split(c_engine.STOP_SEQ_SEP)) == c_engine.STOP_SEQ_MAX


def test_prompt_within_the_cap_is_untouched(anchor):
    """A prompt that fits is passed through byte for byte."""
    body = b"a" * 100
    assert anchor(body, SEQ_CAP) is body


def test_over_long_prompt_is_cut_to_within_the_cap(anchor):
    """An over-long prompt is cut so the engine's fgets buffer cannot overflow."""
    assert len(anchor(b"a" * (SEQ_CAP * 2), SEQ_CAP)) <= SEQ_CAP


def test_cut_keeps_the_newest_bytes(anchor):
    """The cut drops from the head, so the most recent turn always survives."""
    body = bytes(range(256)) * 8
    assert anchor(body, SEQ_CAP // 2).endswith(body[-64:])


def test_growing_prompt_keeps_a_stable_prefix(anchor):
    """Appending to a prompt inside one stride leaves the surviving prefix byte-identical.

    This is the property the engine's state cache needs. A plain tail clamp shifts the
    first surviving byte on every turn, so the cached prefix never matches and each turn
    pays a full prefill."""
    cap = 512
    base = b"a" * (cap + 1)
    grown = base + b"b" * 10
    assert anchor(grown, cap).startswith(anchor(base, cap))


def test_cut_offset_is_always_a_whole_number_of_strides(anchor):
    """However far the prompt overflows, the head cut lands on a stride boundary."""
    cap = 512
    stride = cap // c_engine.PROMPT_ANCHOR_DIVISOR
    offsets = set()
    for extra in (1, 5, stride - 1, stride, stride + 1, stride * 3, stride * 7):
        body = b"a" * (cap + extra)
        offsets.add((len(body) - len(anchor(body, cap))) % stride)
    assert offsets == {0}
