# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers tools/build_recall_corpus.py (IDEA 20 recall curriculum). Pins: every
#   example states a codeword, buries it under at least min_span filler bytes,
#   and restates it at the end; the revision variant's final statement carries
#   the NEWEST codeword (in-place revision is what the curriculum rewards);
#   spans force the dependency across at least one window; determinism by seed.
# tests/mri/test_recall_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import re

import pytest
from tools import build_recall_corpus as brc

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture()
def corpus(tmp_path):
    filler = tmp_path / "hansard_train.bin"
    filler.write_bytes(b"the house rose at noon and resumed at two oclock " * 400)
    brc.build(n=60, min_span=1024, max_span=2048, seed=1, out_dir=str(tmp_path))
    return (tmp_path / "recall_curr_train.bin").read_bytes()


def _examples(data):
    return [e for e in data.split(b".\n") if e.strip()]


def test_examples_state_bury_restate(corpus):
    for ex in _examples(corpus):
        text = ex.decode("utf-8", "replace")
        pairs = re.findall(r"The codeword for (\w+) is (?:now )?(\w+)", text)
        assert len(pairs) >= 2
        assert pairs[0][0] == pairs[-1][0]              # same key
        first = text.index("The codeword for")
        last = text.rindex("The codeword for")
        assert last - first >= 1024                     # crosses a window


def test_revision_final_statement_uses_newest_word(corpus):
    revised = 0
    for ex in _examples(corpus):
        pairs = re.findall(r"The codeword for (\w+) is (?:now )?(\w+)",
                           ex.decode("utf-8", "replace"))
        if len(pairs) == 3:
            revised += 1
            assert pairs[-1][1] == pairs[1][1]          # final == revised value
            assert pairs[-1][1] != pairs[0][1]
    assert revised > 5                                  # variant actually present


def test_deterministic_by_seed(tmp_path):
    filler = tmp_path / "hansard_train.bin"
    filler.write_bytes(b"order order the member will resume their seat now " * 400)
    brc.build(n=20, min_span=1024, max_span=1536, seed=7, out_dir=str(tmp_path))
    a = (tmp_path / "recall_curr_train.bin").read_bytes()
    brc.build(n=20, min_span=1024, max_span=1536, seed=7, out_dir=str(tmp_path))
    assert (tmp_path / "recall_curr_train.bin").read_bytes() == a
