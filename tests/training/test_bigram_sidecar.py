# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The writing-health probe scores PMI against a corpus's <stem>_bigrams.npz
#   sidecar. Without a sidecar it used to return None and score worse in silence;
#   these tests pin the on-demand build and the shared sidecar path.
# - Tiny in-memory corpus in tmp_path, no model, no torch forward (rule 49).
# tests/training/test_bigram_sidecar.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
from tools import build_bigram_index as bigram_index
from training import checkpoint_probe

# ------------------------------------------------------------------------------------
# Constants

CORPUS_TEXT = ("the cat sat on the mat and the dog sat on the rug " * 200).encode("utf-8")
PAIR        = ["the", "cat"]

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def corpus_bin(tmp_path):
    """A small corpus .bin with no sidecar, evicted from the probe cache after use."""
    path = tmp_path / "tiny_train.bin"
    path.write_bytes(CORPUS_TEXT)
    yield str(path)
    checkpoint_probe._WH_PMI_CACHE.pop(os.path.abspath(str(path)), None)


def test_sidecar_path_is_derived_from_the_corpus_bin(tmp_path):
    """sidecar_path swaps the .bin suffix for _bigrams.npz."""
    assert bigram_index.sidecar_path(str(tmp_path / "x_train.bin")) == str(tmp_path / "x_train_bigrams.npz")


def test_missing_sidecar_is_built_on_demand(corpus_bin):
    """Loading the PMI index for a corpus with no sidecar writes the sidecar."""
    checkpoint_probe._wh_load_pmi_index(corpus_bin)
    assert os.path.isfile(bigram_index.sidecar_path(corpus_bin))


def test_index_is_returned_when_the_sidecar_was_missing(corpus_bin):
    """The probe gets a usable index instead of None when the sidecar was absent."""
    assert checkpoint_probe._wh_load_pmi_index(corpus_bin) is not None


def test_pmi_scores_a_corpus_pair_after_the_on_demand_build(corpus_bin):
    """A word pair the corpus contains scores above the OOV floor, not nan."""
    index = checkpoint_probe._wh_load_pmi_index(corpus_bin)
    score = checkpoint_probe._wh_pmi_score(PAIR, index)
    assert score > index["oov_penalty"]
