# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Tests for tools/corpus_audit, the acceptance gate the Distillation tab runs
#   before a built corpus is offered for training.
# - The gate exists because of chat_5gb: 5.14 GB of bytes over 708 unique user
#   turns. The load-bearing test here is test_duplicate_expanded_corpus_is_caught,
#   which rebuilds that failure in miniature -- if it ever passes, the gate is
#   worthless no matter what else is green.
# - Synthetic ChatML is built inline; nothing reads data/corpus (rule 48).
# tests/corpus/test_corpus_audit.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from corpus_audit import (
    MAX_ARTIFACTS_PER_1K,
    MIN_MEDIAN_ASSISTANT_BYTES,
    MIN_UNIQUE_CONTENT_RATIO,
    MIN_UNIQUE_USER_RATIO,
    add_size_context,
    audit_bytes,
    audit_file,
)

# ------------------------------------------------------------------------------------
# Constants

LONG_REPLY = "This is a reply long enough to clear the median assistant floor. " * 5

# ------------------------------------------------------------------------------------
# Functions

def _turn(role, text):
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def _convo(i, reply=None):
    return (_turn("user", f"Question number {i} about a distinct subject.")
            + _turn("assistant", reply or f"Answer {i}. {LONG_REPLY}"))


def _corpus(n, start=0):
    return "".join(_convo(i) for i in range(start, start + n)).encode()


def _ids(report):
    return {c["id"]: c["passed"] for c in report["checks"]}


def test_a_diverse_corpus_passes_every_check():
    """The shape the pipeline is supposed to produce clears the gate."""
    r = audit_bytes(_corpus(200))
    assert r["passed"] is True
    assert all(_ids(r).values())


def test_duplicate_expanded_corpus_is_caught():
    """chat_5gb in miniature: one conversation repeated to bulk. This is the
    failure the whole module exists to catch."""
    data = _convo(0) * 500
    r = audit_bytes(data.encode())
    checks = _ids(r)
    assert r["passed"] is False
    assert checks["unique_user_turns"] is False
    assert checks["unique_content_ratio"] is False
    assert r["unique_user_turns"] == 1
    assert r["total_user_turns"] == 500


def test_size_is_never_what_makes_a_corpus_pass():
    """Repeating a bad corpus 20x makes the file bigger and the score no better."""
    small = audit_bytes((_convo(0) * 5).encode())
    large = audit_bytes((_convo(0) * 100).encode())
    assert small["passed"] is False and large["passed"] is False
    assert large["unique_content_ratio"] <= small["unique_content_ratio"]


def test_unique_content_ratio_ignores_chatml_marker_overhead():
    """The ratio measures duplication, so it is taken against turn text rather
    than file bytes -- markers are overhead, not content, and a corpus of many
    short unique turns must not be scored as if it were repetitive."""
    r = audit_bytes(_corpus(50))
    assert r["unique_content_ratio"] == pytest.approx(1.0)
    assert r["total_content_bytes"] < len(_corpus(50))


def test_short_replies_fail_the_median_floor():
    """A corpus of terse answers cannot teach sustained conversation."""
    data = "".join(_convo(i, reply="Yes.") for i in range(100)).encode()
    r = audit_bytes(data)
    assert _ids(r)["median_assistant_bytes"] is False
    assert r["median_assistant_bytes"] < MIN_MEDIAN_ASSISTANT_BYTES
    assert r["short_assistant_turns"] == 100


def test_generator_artifacts_are_counted_and_named():
    """Artifacts are reported by name so the spec's ban list can be fixed."""
    bad = _turn("user", "Who are you?") + _turn("assistant", "As an AI language model, " + LONG_REPLY)
    r = audit_bytes(_corpus(10) + bad.encode())
    assert r["artifacts"].get("ai_disclaimer") == 1


def test_artifact_rate_is_per_thousand_assistant_turns():
    """The rate must be a rate: doubling clean volume halves it."""
    bad = _turn("user", "hi") + _turn("assistant", "As an AI language model, " + LONG_REPLY)
    few = audit_bytes(_corpus(50) + bad.encode())
    many = audit_bytes(_corpus(100) + bad.encode())
    assert many["artifacts_per_1k"] < few["artifacts_per_1k"]


def test_an_artifact_flood_fails_the_gate():
    """Enough disclaimers and the corpus is rejected even though it is diverse."""
    data = "".join(
        _turn("user", f"Question {i}?") + _turn("assistant", f"As an AI language model, {i}. {LONG_REPLY}")
        for i in range(100)).encode()
    r = audit_bytes(data)
    assert _ids(r)["artifacts_per_1k"] is False
    assert r["artifacts_per_1k"] > MAX_ARTIFACTS_PER_1K


def test_turns_spanning_a_chunk_boundary_are_counted_once(monkeypatch, tmp_path):
    """Streaming uses overlapping windows; a turn on a seam must not double-count."""
    import corpus_audit
    monkeypatch.setattr(corpus_audit, "CHUNK_BYTES", 4096)
    monkeypatch.setattr(corpus_audit, "OVERLAP_BYTES", 2048)
    data = _corpus(400)
    p = tmp_path / "seam_train.bin"
    p.write_bytes(data)
    streamed = audit_file(str(p))
    whole = audit_bytes(data)
    assert streamed["total_user_turns"] == whole["total_user_turns"] == 400
    assert streamed["unique_user_turns"] == whole["unique_user_turns"] == 400


def test_a_genuinely_repeated_turn_still_counts_as_a_repeat(tmp_path):
    """Dedup by offset must not silently forgive real duplication far apart."""
    data = _corpus(100) + _convo(0).encode()
    r = audit_bytes(data)
    assert r["total_user_turns"] == 101
    assert r["unique_user_turns"] == 100


def test_size_context_is_attached_without_becoming_a_check(tmp_path):
    """File size is reported for the UI but never gates anything."""
    p = tmp_path / "ctx_train.bin"
    p.write_bytes(_corpus(30))
    r = audit_file(str(p))
    assert r["size_bytes"] == p.stat().st_size
    assert r["marker_overhead_bytes"] > 0
    assert "size_bytes" not in {c["id"] for c in r["checks"]}


def test_empty_corpus_does_not_crash_and_does_not_pass():
    """A build that produced nothing must fail loudly, not divide by zero."""
    r = audit_bytes(b"")
    assert r["passed"] is False
    assert r["total_user_turns"] == 0
    assert r["median_assistant_bytes"] == 0


def test_missing_file_raises():
    """A path the caller got wrong is an error, not a silent zero score."""
    with pytest.raises(FileNotFoundError):
        audit_file("/nonexistent/nope_train.bin")


@pytest.mark.parametrize("check_id,floor", [
    ("unique_user_turns", MIN_UNIQUE_USER_RATIO),
    ("unique_content_ratio", MIN_UNIQUE_CONTENT_RATIO),
])
def test_every_ratio_check_publishes_the_floor_it_used(check_id, floor):
    """The UI renders 'value vs floor', so the floor travels with the check."""
    r = audit_bytes(_corpus(50))
    check = next(c for c in r["checks"] if c["id"] == check_id)
    assert check["floor"] == floor
    assert check["why"]


def test_add_size_context_is_idempotent():
    """Re-attaching context must not append duplicate checks."""
    r = audit_bytes(_corpus(20))
    before = len(r["checks"])
    add_size_context(r, 1000)
    add_size_context(r, 1000)
    assert len(r["checks"]) == before


def test_assistant_register_tics_are_not_counted_in_user_turns():
    """Register artifacts belong to the ASSISTANT voice. Counting them anywhere in
    the file charged a corpus for user turns saying ordinary human things --
    measured 2026-08-20 on interview_v1, where two user turns opening "Sure, I'm
    trying to decide between..." failed an otherwise clean corpus."""
    data = (_corpus(40)
            + (_turn("user", "Sure, I'm trying to decide between two laptops.")
               + _turn("assistant", f"Both are solid choices. {LONG_REPLY}")).encode())
    r = audit_bytes(data)
    assert r["artifacts"].get("filler_opener", 0) == 0
    assert r["passed"] is True


def test_assistant_register_tics_are_still_caught_in_assistant_turns():
    """The scoping fix must not blind the check to the thing it exists for."""
    data = (_corpus(40)
            + (_turn("user", "Which laptop should I buy?")
               + _turn("assistant", f"Sure, both are solid. {LONG_REPLY}")).encode())
    r = audit_bytes(data)
    assert r["artifacts"].get("filler_opener", 0) == 1


def test_structural_damage_is_counted_anywhere_in_the_file():
    """Mojibake in a user turn is still a broken corpus, so those stay unscoped."""
    data = (_corpus(40)
            + (_turn("user", "What about caf� wall tiles?")
               + _turn("assistant", f"They are fine. {LONG_REPLY}")).encode())
    r = audit_bytes(data)
    assert r["artifacts"].get("mojibake", 0) == 1
