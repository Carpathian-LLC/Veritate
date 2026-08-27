# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers tools/study_exam.py scoring. The model itself is not loaded here; these pin
#   the grading arithmetic, because a scorer that flatters the model would make the
#   whole consolidation experiment unfalsifiable.
# tests/mri/test_study_exam.py
# ------------------------------------------------------------------------------------
# Imports:

from tools import study_exam as se

# ------------------------------------------------------------------------------------
# Functions


def test_prefix_share_measures_the_opening():
    """A partial memory should be credited for the signature it does reproduce."""
    assert se.prefix_share("def f(x):", "def f(x): return x") == round(9 / 18, 4)
    assert se.prefix_share("", "abc") == 0.0
    assert se.prefix_share("abc", "abc") == 1.0
    assert se.prefix_share("xyz", "abc") == 0.0


def test_summarize_reports_means_and_counts():
    """Aggregates are means over chunks plus an identify count, not a single score."""
    rows = [{"sim": 0.5, "prefix": 0.2, "identify": True, "raw_bytes": 30},
            {"sim": 0.1, "prefix": 0.0, "identify": False, "raw_bytes": 30}]
    s = se.summarize(rows)
    assert s["n"] == 2 and s["sim"] == 0.3 and s["prefix"] == 0.1
    assert s["identify"] == 1 and s["identify_acc"] == 0.5


def test_summarize_of_nothing_is_not_a_score():
    """An empty split reports n=0 rather than a misleading zero score."""
    assert se.summarize([]) == {"n": 0}


def test_summarize_separates_silent_from_degenerate():
    """An all-whitespace decode strips to "" and scores like a model that emitted
    nothing. Those are different failures, and reporting them as one hid a real
    result on 2026-08-25: the model was producing 64 spaces, not staying silent."""
    rows = [{"sim": 0.0, "prefix": 0.0, "identify": False, "raw_bytes": 64, "degenerate": True},
            {"sim": 0.0, "prefix": 0.0, "identify": False, "raw_bytes": 0, "degenerate": False},
            {"sim": 0.5, "prefix": 0.1, "identify": True, "raw_bytes": 90, "degenerate": False}]
    s = se.summarize(rows)
    assert s["degenerate"] == 1 and s["silent"] == 1 and s["n"] == 3
