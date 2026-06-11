# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Locks the kb_build FILE-mode chunker quality gate: clean self-contained
#   sentences are kept; anaphora-leading, abbreviation-split, too-short, and
#   mostly-symbol fragments are dropped or merged.
# tests/rag/test_kb_chunking.py
# ------------------------------------------------------------------------------------
# Imports

import os
import sys

import pytest

RAG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "v2", "rag")
)
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

import kb_build as kb

# ------------------------------------------------------------------------------------
# Constants

CLEAN = "The Eiffel Tower is a wrought iron lattice tower located in Paris, France."
ANAPHORA = "It was completed in 1889 and stands over three hundred metres tall."
ABBR_HEAD = "The work was led by Major Robert B."
ABBR_TAIL = "Lee of the United States Army during the period."
SHORT = "It is large."
SYMBOLS = "1 @ 2 # 3 $ 4 % 5 ^ 6 & 7 * 8 ( 9 ) 0 + = - _ / 1 2 3 4 5 6 7 8 9 0 ."
HEADER = " = History = "

# ------------------------------------------------------------------------------------
# Functions

def _chunk(tmp_path, text):
    p = tmp_path / "wiki.txt"
    p.write_text(text, encoding="utf-8")
    return kb.chunk_file(str(p))


def test_keeps_clean_sentence(tmp_path):
    """Clean self-contained sentence survives the quality gate."""
    chunks = _chunk(tmp_path, CLEAN + "\n")
    assert chunks == [CLEAN]


def test_drops_anaphora_start(tmp_path):
    """Sentence starting with an unresolved referent is dropped."""
    chunks = _chunk(tmp_path, ANAPHORA + "\n")
    assert chunks == []


def test_merges_abbreviation_split(tmp_path):
    """Abbreviation split merges into one chunk instead of two fragments."""
    chunks = _chunk(tmp_path, ABBR_HEAD + " " + ABBR_TAIL + "\n")
    assert len(chunks) == 1
    assert "Robert B. Lee" in chunks[0]


def test_drops_too_short(tmp_path):
    """Below the minimum word count is dropped."""
    chunks = _chunk(tmp_path, SHORT + "\n")
    assert chunks == []


def test_drops_mostly_symbols(tmp_path):
    """Chunk that is mostly numbers and symbols is dropped."""
    chunks = _chunk(tmp_path, SYMBOLS + "\n")
    assert chunks == []


def test_drops_markup_header(tmp_path):
    """Wiki header line with '=' is dropped."""
    chunks = _chunk(tmp_path, HEADER + "\n")
    assert chunks == []


def test_mixed_input_keeps_only_clean(tmp_path):
    """Mixed prose keeps clean sentences and drops fragment lines."""
    text = " ".join([CLEAN, ANAPHORA, SHORT, SYMBOLS]) + "\n"
    chunks = _chunk(tmp_path, text)
    assert chunks == [CLEAN]


def test_limit_caps_output(tmp_path):
    """limit caps the number of returned chunks."""
    p = tmp_path / "wiki.txt"
    p.write_text((CLEAN + "\n") * 5, encoding="utf-8")
    assert len(kb.chunk_file(str(p))) == 5
    assert len(kb.chunk_file(str(p), limit=2)) == 2
