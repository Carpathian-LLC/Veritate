# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Unit tests for hybrid_routes._compact: conversation-memory bounding. A local
#   byte model cannot summarize itself, so it slides (drop oldest, no model call);
#   a remote model folds the head into a model-written summary. No model loads.
# tests/mri/test_chat_compaction.py
# ------------------------------------------------------------------------------------
# Imports:



from routes import hybrid_routes as H

# ------------------------------------------------------------------------------------
# Constants

LIMIT = 250

# ------------------------------------------------------------------------------------
# Functions

def _turns(n, size=100):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}" + "x" * size}
            for i in range(n)]


def test_local_slide_bounds_memory_to_limit():
    """A local model over budget slides so the kept memory fits char_limit."""
    _summary, kept = H._compact(lambda *a: "", "", _turns(8), LIMIT, "local")
    assert H._context_used("", kept) <= LIMIT


def test_local_slide_drops_the_summary():
    """A local slide returns an empty summary (byte models can't summarize)."""
    summary, _kept = H._compact(lambda *a: "", "old garbage summary", _turns(8), LIMIT, "local")
    assert summary == ""


def test_local_slide_never_calls_the_model():
    """Local compaction is mechanical: it must not invoke the selected model."""
    def complete(_messages, _system):
        raise AssertionError("local compaction must not call the model")
    H._compact(complete, "", _turns(8), LIMIT, "local")


def test_local_slide_keeps_the_newest_suffix():
    """The slid tail is the newest suffix of the original turns."""
    turns = _turns(8)
    _summary, kept = H._compact(lambda *a: "", "", turns, LIMIT, "local")
    assert kept == turns[len(turns) - len(kept):]


def test_local_slide_opens_on_a_user_turn():
    """The slid tail opens on a user turn."""
    _summary, kept = H._compact(lambda *a: "", "", _turns(8), LIMIT, "local")
    assert kept[0]["role"] == "user"


def test_under_budget_is_a_noop():
    """Memory already within budget is returned unchanged, with no model call."""
    turns = _turns(2, size=1)
    assert H._compact(None, "", turns, 10_000, "local") == ("", turns)


def test_remote_folds_head_into_model_summary():
    """A remote model over budget returns the model-written summary."""
    summary, _kept = H._compact(lambda messages, system: "SUMMARY", "", _turns(10), 200, "remote")
    assert summary == "SUMMARY"


def test_remote_keeps_the_verbatim_tail():
    """A remote compaction keeps CTX_KEEP_TAIL_TURNS turns verbatim."""
    turns = _turns(10)
    _summary, kept = H._compact(lambda messages, system: "SUMMARY", "", turns, 200, "remote")
    assert kept == turns[-H.CTX_KEEP_TAIL_TURNS:]
