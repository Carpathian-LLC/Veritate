# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the reading-level builder's own instrument: the syllable heuristic and the
#   Flesch-Kincaid grade / reading-ease scores it flags off-target bands with. Pinned
#   on words with known counts and on two passages whose grades must order correctly.
# tests/corpus/test_build_grade_evals.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from training.builders.eval import build_grade_evals as bg

# ------------------------------------------------------------------------------------
# Functions


@pytest.mark.parametrize("word, n", [("cat", 1), ("the", 1), ("apple", 2), ("banana", 3), ("jumped", 1),
                                     ("yes", 1), ("beautiful", 3), ("'hello,'", 2), ("", 0)])
def test_syllable_counts(word, n):
    assert bg._syllables(word) == n


def test_fk_scores_read_the_grade_off_sentence_and_word_length():
    simple = "The cat sat. The dog ran. I see a hat. We go up."
    dense = ("Notwithstanding considerable methodological heterogeneity, contemporary investigations "
             "consistently demonstrate substantial associations between socioeconomic circumstances and outcomes.")
    s, d = bg.fk_scores(simple), bg.fk_scores(dense)
    assert s[2] == 13 and s[3] == 4 and s[4] == 3.25         # words, sentences, words per sentence
    assert s[0] < 2 and d[0] > 15                            # grade level orders the passages
    assert s[1] > d[1]                                       # reading ease runs the other way
    assert bg.fk_scores("   ") is None and bg.fk_scores("...") is None
