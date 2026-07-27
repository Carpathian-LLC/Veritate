# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Flat cosine retrieval over a MemStore's byte-native keys. encode_query reuses
#   store.embed, so a query key is the identical mean-pool op as a leaf key. Keys
#   are unit-normalized, so cosine is a dot product.
# - retrieve() returns (leaf_texts, scores), matching the serving-side retrieval
#   shape; search() is the array-level core. hindex.HIndex replaces the flat scan
#   when the key count makes a full scan too slow.
# veritate_core/memory/reader.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np

from veritate_core.memory import store as store_mod

# ------------------------------------------------------------------------------------
# Constants

QUERY_ENC    = "utf-8"
QUERY_ERRORS = "replace"
SCORE_DTYPE  = np.float32


# ------------------------------------------------------------------------------------
# Functions

def encode_query(model, text):
    return store_mod.embed(model, [text.encode(QUERY_ENC, QUERY_ERRORS)])[0]


def search(keys, query, k):
    """Cosine top-k over unit-normalized keys. Returns (indices, scores), best first."""
    scores = keys.astype(SCORE_DTYPE) @ query.astype(SCORE_DTYPE)
    k = min(k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]


def retrieve(memstore, model, query, k):
    idx, sc = search(memstore.keys(), encode_query(model, query), k)
    return [memstore.leaf_text(int(i)) for i in idx], [float(s) for s in sc]
