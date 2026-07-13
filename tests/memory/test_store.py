# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Unit tests for the memory milestone store + reader (model-free): on-disk store
#   roundtrip, flat cosine top-k ranking, and deterministic invented-entity facts.
# tests/memory/test_store.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEMORY    = os.path.join(REPO_ROOT, "experiments", "v2", "memory")
for _p in (MEMORY, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_facts
import reader
import store as store_mod

# ------------------------------------------------------------------------------------
# Constants

LEAF_BYTES = store_mod.LEAF_BYTES


# ------------------------------------------------------------------------------------
# Functions

def test_store_roundtrip(tmp_path):
    """MemStore persists and reloads leaf bytes and keys unchanged."""
    leaves = [bytes([(i + b) % 256 for b in range(LEAF_BYTES)]) for i in range(8)]
    keys = np.random.default_rng(0).standard_normal((8, 16)).astype(store_mod.KEY_DTYPE)
    store_mod._save(str(tmp_path), leaves, keys, LEAF_BYTES)
    s = store_mod.load(str(tmp_path))
    assert len(s) == 8
    assert s.leaf_text(3) == leaves[3].decode("utf-8", "replace")
    assert np.array_equal(s.keys(), keys)


def test_search_topk_ranks_nearest_first():
    """search returns the k nearest keys by cosine, highest score first."""
    keys = np.eye(4, dtype=np.float16)
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float16)
    idx, scores = reader.search(keys, query, 2)
    assert list(idx) == [0, 1]
    assert scores[0] > scores[1]


def test_facts_deterministic_and_extractive():
    """facts() is seed-deterministic and every answer is a substring of its passage."""
    a = make_facts.facts(30)
    b = make_facts.facts(30)
    assert a == b
    assert len(a) == 30
    assert all(f["a"] in f["passage"] for f in a)
