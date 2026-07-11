# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - corpus.usage() mix-awareness: single-stem shape unchanged, '+' mixes return
#   per-member blocks with size-proportional weights, and a missing member is
#   flagged rather than 404/raising. Deps are stubbed so no real files are read.
# tests/mri/test_corpus_mix.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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


def test_single_stem_shape_unchanged(monkeypatch):
    """usage() on a single stem returns a train/val/models block, no `mixed`."""
    _stub(monkeypatch, SIZES)
    u = corpus.usage("a")
    assert u["stem"] == "a" and u["train"]["bytes"] == 300 and "mixed" not in u


def test_missing_single_returns_none(monkeypatch):
    """usage() returns None when a single stem does not resolve."""
    _stub(monkeypatch, SIZES)
    assert corpus.usage("nope") is None


def test_mix_returns_members_with_size_weights(monkeypatch):
    """A '+' mix returns per-member blocks weighted by train-file size."""
    _stub(monkeypatch, SIZES)
    u = corpus.usage("a+b")
    assert u["mixed"] is True
    assert [(m["stem"], round(m["weight"], 2)) for m in u["members"]] == [("a", 0.75), ("b", 0.25)]
    assert u["missing"] == []


def test_mix_flags_missing_member(monkeypatch):
    """A missing mix member is flagged in `missing` and does not raise."""
    _stub(monkeypatch, SIZES)
    u = corpus.usage("a+gone")
    assert u["missing"] == ["gone"]
    assert any(m["train"] is None for m in u["members"])
