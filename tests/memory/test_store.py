# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Unit tests for the addressable-memory tier in veritate_core.memory (model-free):
#   on-disk store roundtrip, flat cosine top-k ranking, and the IVF drill-down
#   returning the same nearest key as the flat scan.
# tests/memory/test_store.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np

from veritate_core.memory import hindex, reader
from veritate_core.memory import store as store_mod

# ------------------------------------------------------------------------------------
# Constants

LEAF_BYTES = store_mod.LEAF_BYTES
N_LEAVES   = 8
KEY_DIM    = 16
SEED       = 0

# ------------------------------------------------------------------------------------
# Functions

def _leaves(n=N_LEAVES):
    return [bytes([(i + b) % 256 for b in range(LEAF_BYTES)]) for i in range(n)]


def test_store_roundtrip(tmp_path):
    """MemStore persists and reloads leaf bytes and keys unchanged."""
    leaves = _leaves()
    keys = np.random.default_rng(SEED).standard_normal((N_LEAVES, KEY_DIM)).astype(store_mod.KEY_DTYPE)
    store_mod._save(str(tmp_path), leaves, keys, LEAF_BYTES)
    s = store_mod.load(str(tmp_path))
    assert len(s) == N_LEAVES
    assert s.leaf_text(3) == leaves[3].decode("utf-8", "replace")
    assert np.array_equal(s.keys(), keys)


def test_search_topk_ranks_nearest_first():
    """search returns the k nearest keys by cosine, highest score first."""
    keys = np.eye(4, dtype=np.float16)
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float16)
    idx, scores = reader.search(keys, query, 2)
    assert list(idx) == [0, 1]
    assert scores[0] > scores[1]


def test_hindex_finds_the_same_top_key_as_the_flat_scan():
    """The IVF drill-down returns the flat scan's nearest key when every cell is probed."""
    rng = np.random.default_rng(SEED)
    keys = rng.standard_normal((256, KEY_DIM)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    query = keys[7]
    idx = hindex.HIndex(keys, cell_target=32, seed=SEED)
    top, _, _ = idx.search(query, topk=1, nprobe=idx.n_cells())
    assert int(top[0]) == int(reader.search(keys, query, 1)[0][0])
