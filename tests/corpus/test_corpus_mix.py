# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - corpus.usage() mix-awareness: single-stem shape unchanged, '+' mixes return
#   per-member blocks with size-proportional weights, and a missing member is
#   flagged rather than 404/raising. Deps are stubbed so no real files are read.
# tests/corpus/test_corpus_mix.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from readers import corpus

# ------------------------------------------------------------------------------------
# Constants

SIZES = {"a": 300, "b": 100}

# ------------------------------------------------------------------------------------
# Functions

def _stub(monkeypatch, sizes):
    monkeypatch.setattr(corpus, "resolve_paths",
                        lambda s: (f"/x/{s}_train.bin", None) if s in sizes else (None, None))
    monkeypatch.setattr(corpus, "_file_block",
                        lambda p: ({"path": p, "bytes": sizes[os.path.basename(p).split("_")[0]],
                                    "mtime": 0, "sha256": "s"} if p else None))
    monkeypatch.setattr(corpus, "_models_for", lambda sha: [])


def test_single_stem_reports_its_stem(monkeypatch):
    """usage() on a single stem echoes the stem back."""
    _stub(monkeypatch, SIZES)
    assert corpus.usage("a")["stem"] == "a"


def test_single_stem_reports_train_bytes(monkeypatch):
    """usage() on a single stem reports the train file's byte count."""
    _stub(monkeypatch, SIZES)
    assert corpus.usage("a")["train"]["bytes"] == SIZES["a"]


def test_single_stem_is_not_flagged_mixed(monkeypatch):
    """usage() on a single stem carries no mixed key."""
    _stub(monkeypatch, SIZES)
    assert "mixed" not in corpus.usage("a")


def test_missing_single_returns_none(monkeypatch):
    """usage() returns None when a single stem does not resolve."""
    _stub(monkeypatch, SIZES)
    assert corpus.usage("nope") is None


def test_mix_is_flagged_mixed(monkeypatch):
    """A '+' mix is flagged mixed."""
    _stub(monkeypatch, SIZES)
    assert corpus.usage("a+b")["mixed"] is True


def test_mix_weights_members_by_train_size(monkeypatch):
    """A '+' mix weights each member by its train-file size."""
    _stub(monkeypatch, SIZES)
    members = corpus.usage("a+b")["members"]
    assert [(m["stem"], round(m["weight"], 2)) for m in members] == [("a", 0.75), ("b", 0.25)]


def test_mix_of_present_members_reports_nothing_missing(monkeypatch):
    """A '+' mix whose members all resolve reports an empty missing list."""
    _stub(monkeypatch, SIZES)
    assert corpus.usage("a+b")["missing"] == []


def test_mix_names_the_missing_member(monkeypatch):
    """A mix member that does not resolve is named in missing."""
    _stub(monkeypatch, SIZES)
    assert corpus.usage("a+gone")["missing"] == ["gone"]


def test_mix_missing_member_has_no_train_block(monkeypatch):
    """A mix member that does not resolve carries a null train block instead of raising."""
    _stub(monkeypatch, SIZES)
    assert [m["stem"] for m in corpus.usage("a+gone")["members"] if m["train"] is None] == ["gone"]
