# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the byte-level decode constraints behind the Generation tab's "shape" control:
#   a vocabulary mask, a stop sequence, and the streaming JSON grammar. The JSON
#   constraint is walked byte by byte over real documents (every byte must be legal
#   under the mask before it is committed) and probed at the states where a wrong mask
#   would let the model write invalid JSON.
# tests/mri/test_decode_constraints.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from inference.decode import constraints as c

# ------------------------------------------------------------------------------------
# Constants

DOCS = [
    b'{"a": [1, 2.5e-3, -0, true, null, "x\\u00e9\\"y\\n"], "b": {}, "c": []}',
    b"[]",
    b'{"k":{"k":{"k":"v"}}}',
    b"[[[-12.75E+2]], false]",
    b'"just a string"',
]

# ------------------------------------------------------------------------------------
# Functions


def _walk(doc):
    """Commit a document byte by byte, asserting each byte was allowed first."""
    j = c.JSONConstraint()
    for i, b in enumerate(doc):
        assert j.mask()[b], f"byte {chr(b)!r} at {i} refused in {doc!r}"
        assert not j.done(), f"done before the end at {i} in {doc!r}"
        j.step(b)
    return j


def _allowed(j):
    return {b for b in range(256) if j.mask()[b]}


def test_vocab_constraint_allows_exactly_the_given_bytes():
    v = c.VocabConstraint(set(range(0x61, 0x7b)) | {0x20})
    assert _allowed(v) == set(range(0x61, 0x7b)) | {0x20}
    assert c.VocabConstraint([0x141]).allowed == {0x41}      # values are bytes
    with pytest.raises(ValueError):
        c.VocabConstraint([])


def test_stop_constraint_halts_when_the_output_ends_with_the_sequence():
    s = c.StopOnConstraint(b"\n\n")
    for b in b"line\n":
        s.step(b)
    assert not s.done() and _allowed(s) == set(range(256))
    s.step(0x0a)
    assert s.done()
    s.reset()
    assert not s.done()
    with pytest.raises(ValueError):
        c.StopOnConstraint(b"")
    with pytest.raises(TypeError):
        c.StopOnConstraint("\n")


@pytest.mark.parametrize("doc", DOCS)
def test_every_byte_of_a_valid_document_is_allowed_and_the_end_is_done(doc):
    j = _walk(doc)
    assert j.done()
    assert _allowed(j) == {0x20, 0x09, 0x0a, 0x0d}          # only trailing whitespace after the value


def test_structural_states_forbid_what_would_break_the_document():
    j = c.JSONConstraint()
    j.prime(b"{")
    assert ord(":") not in _allowed(j) and ord('"') in _allowed(j) and ord("}") in _allowed(j)
    j.prime(b'"a"')
    assert _allowed(j) == {ord(":"), 0x20, 0x09, 0x0a, 0x0d}   # a key wants its colon
    j.prime(b":0")
    assert ord("1") not in _allowed(j) and ord(".") in _allowed(j)   # no leading zeros
    j.prime(b".")
    assert ord(",") not in _allowed(j) and ord("5") in _allowed(j)   # a fraction needs a digit
    j.prime(b"5")
    assert {ord(","), ord("}")} <= _allowed(j) and ord("]") not in _allowed(j)


def test_a_literal_must_be_completed_and_a_string_escape_must_be_finished():
    j = c.JSONConstraint()
    j.prime(b"[tr")
    assert _allowed(j) == {ord("u")}
    j.prime(b'ue, "a\\')
    assert _allowed(j) == set(b'"\\/bfnrtu')
    j.prime(b"u12")
    assert _allowed(j) == set(b"0123456789abcdefABCDEF")


def test_priming_a_prompt_prefix_resumes_mid_string_and_reset_starts_over():
    j = c.JSONConstraint()
    j.prime(b'{"name": "')
    assert ord("A") in _allowed(j) and ord("\n") not in _allowed(j)
    j.prime(b'x"}')
    assert j.done()
    j.reset()
    assert not j.done() and _allowed(j) >= {ord("{"), ord("["), ord('"'), ord("-"), ord("t")}
