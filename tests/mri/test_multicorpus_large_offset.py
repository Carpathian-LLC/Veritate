# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Regression: multicorpus.make_mixed_loader must draw start offsets from corpora
#   larger than 2^31 bytes (2.1 GB). numpy RandomState.randint defaults to int32
#   and raises "high is out of bounds for int32" when the corpus is bigger. Fix
#   is `dtype=np.int64` on the randint call. Any single stem or a real training
#   mix over ~2 GB hits this.
# - Arch-agnostic bug: same on Windows, macOS, Linux; test runs identically on all.
# tests/mri/test_multicorpus_large_offset.py
# ------------------------------------------------------------------------------------
# Imports

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from veritate_core.plugin import multicorpus


# ------------------------------------------------------------------------------------
# Constants

LARGE_SIZE = (1 << 31) + 100_000  # 2.1 GB + a hair, forces int64 in randint
SEQ        = 512
BATCH      = 2
SEED       = 7


# ------------------------------------------------------------------------------------
# Functions

class _FakeMemmap:
    """Behaves like a np.uint8 memmap of a huge file without allocating one:
    len() returns the requested size; slicing returns SEQ zero-filled bytes."""
    def __init__(self, size):
        self._size = int(size)
    def __len__(self):
        return self._size
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, _step = key.indices(self._size)
            return np.zeros(stop - start, dtype=np.uint8)
        return np.uint8(0)


def test_make_mixed_loader_handles_corpus_over_2gb(monkeypatch):
    """draw() must not raise on a corpus larger than 2^31 bytes (int32 max)."""
    monkeypatch.setattr(multicorpus.np, "memmap",
                        lambda p, dtype, mode: _FakeMemmap(LARGE_SIZE))
    paths = [("/fake/big_train.bin", None, 1.0)]
    draw, total = multicorpus.make_mixed_loader(paths, BATCH, SEQ, SEED)
    assert total == LARGE_SIZE
    toks, tgts = draw()
    assert toks.shape == (BATCH, SEQ)
    assert tgts.shape == (BATCH, SEQ)


def test_randint_int64_produces_valid_start(monkeypatch):
    """The randint dtype must be int64 so offsets in [0, LARGE_SIZE) are legal."""
    monkeypatch.setattr(multicorpus.np, "memmap",
                        lambda p, dtype, mode: _FakeMemmap(LARGE_SIZE))
    paths = [("/fake/big_train.bin", None, 1.0)]
    draw, _ = multicorpus.make_mixed_loader(paths, 1, SEQ, SEED)
    for _ in range(4):
        toks, _ = draw()
        assert toks.shape == (1, SEQ)


def test_parse_spec_single_stem_with_weight():
    """Regression: 'fineweb_edu:1.0' (single stem with WEIGHT_SEP, no LIST/MIX_SEP)
    must parse as one weighted stem, not as a bare stem name. Arch-agnostic string
    parsing bug."""
    parsed = multicorpus.parse_spec("fineweb_edu:1.0")
    assert parsed == [("fineweb_edu", 1.0)]

    parsed = multicorpus.parse_spec("chat_50mb:0.5")
    assert parsed == [("chat_50mb", 0.5)]

    # Bare stem still returns None weight
    parsed = multicorpus.parse_spec("fineweb_edu")
    assert parsed == [("fineweb_edu", None)]


def test_int32_dtype_would_have_raised():
    """Guard: prove the pre-fix code path (int32 default) is what raised. If numpy
    ever silently promotes randint high, this test flags the change so the
    dtype=np.int64 fix can be re-evaluated."""
    rng = np.random.RandomState(SEED)
    with pytest.raises(ValueError, match="high is out of bounds"):
        rng.randint(0, LARGE_SIZE, dtype=np.int32)
    rng2 = np.random.RandomState(SEED)
    val = int(rng2.randint(0, LARGE_SIZE, dtype=np.int64))
    assert 0 <= val < LARGE_SIZE
