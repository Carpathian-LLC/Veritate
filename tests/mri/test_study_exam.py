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
    rows = [{"sim": 0.5, "prefix": 0.2, "identify": True},
            {"sim": 0.1, "prefix": 0.0, "identify": False}]
    s = se.summarize(rows)
    assert s == {"n": 2, "sim": 0.3, "prefix": 0.1, "identify": 1, "identify_acc": 0.5}


def test_summarize_of_nothing_is_not_a_score():
    """An empty split reports n=0 rather than a misleading zero score."""
    assert se.summarize([]) == {"n": 0}
