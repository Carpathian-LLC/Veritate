# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Unit tests for hybrid_routes._compact: conversation-memory bounding. A local
#   byte model cannot summarize itself, so it slides (drop oldest, no model call);
#   a remote model folds the head into a model-written summary. No model loads.
# - Also hybrid_routes.fit_chat_history: prompt bounding for the turn being sent. The
#   engine cuts an over-long prompt mid-turn, so whole turns go first.
# tests/mri/test_chat_compaction.py
# ------------------------------------------------------------------------------------
# Imports:



from routes import hybrid_routes as H

# ------------------------------------------------------------------------------------
# Constants

LIMIT = 250
# A trained context small enough that a dozen 100-char turns overflow it, as they do
# on the byte models this runs on.
SEQ      = 512
MAX_NEW  = 256
IM_START = "<|im_start|>"

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


def test_fitted_history_renders_within_the_engine_budget():
    """A conversation past the context fits the prompt the engine reads whole."""
    from inference.backends.c_engine import max_prompt_chars
    fitted = H.fit_chat_history(_turns(13), "", SEQ, MAX_NEW)
    assert len(H._render_local(fitted, "")) <= max_prompt_chars(SEQ, MAX_NEW)


def test_fitted_history_keeps_the_turn_being_answered():
    """The newest user turn survives however little of the conversation fits."""
    turns = _turns(13)
    assert H.fit_chat_history(turns, "", SEQ, MAX_NEW)[-1] == turns[-1]


def test_fitted_history_keeps_the_newest_suffix():
    """What survives is the newest run of turns, not an arbitrary subset."""
    turns = _turns(13)
    fitted = H.fit_chat_history(turns, "", SEQ, MAX_NEW)
    assert fitted == turns[len(turns) - len(fitted):]


def test_fitted_history_opens_on_a_turn_marker():
    """The rendered prompt starts on a turn marker, never inside a dropped message."""
    fitted = H.fit_chat_history(_turns(13), "", SEQ, MAX_NEW)
    assert H._render_local(fitted, "").startswith(IM_START)


def test_history_within_the_budget_is_untouched():
    """A conversation that already fits is sent whole."""
    turns = _turns(5, size=1)
    assert H.fit_chat_history(turns, "", SEQ, MAX_NEW) == turns
