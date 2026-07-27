# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - On-disk leaf store for the addressable-memory retrieval tier. Splits a corpus
#   slice into fixed-size leaves, computes a byte-native key per leaf by mean-pooling
#   model.hidden_states and L2-normalizing, and persists leaves + offsets + fp16 keys.
# - embed() is the single key-encoding path, shared with reader.encode_query, so a
#   query key is the identical pooling op as a leaf key.
# - torch.no_grad throughout. A byte-native key is a full trunk forward, so a large
#   index is an offline build.
# - Fixed leaves need no padding; variable inputs (queries, planted passages) are
#   grouped by length so every batch is uniform-width.
# veritate_core/memory/store.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------------------------
# Constants

LEAF_BYTES   = 512
BATCH        = 8
KEY_DTYPE    = np.float16
KEYS_FILE    = "keys.npy"
LEAVES_FILE  = "leaves.bin"
OFFSETS_FILE = "offsets.npy"
LEAF_ENCODING = "utf-8"
DECODE_ERRORS = "replace"


# ------------------------------------------------------------------------------------
# Functions

def embed(model, seqs, batch=BATCH):
    """Mean-pooled, L2-normalized byte-native key per input, as an fp16 [n, H] array.
    Groups by length so every batch is uniform-width (no padding contamination)."""
    keys = [None] * len(seqs)
    by_len = {}
    for i, s in enumerate(seqs):
        by_len.setdefault(len(s), []).append(i)
    for length, idxs in by_len.items():
        if length == 0:
            continue
        for j in range(0, len(idxs), batch):
            grp = idxs[j:j + batch]
            x = torch.tensor([list(seqs[i]) for i in grp], dtype=torch.long,
                             device=next(model.parameters()).device)
            with torch.no_grad():
                h = model.hidden_states(x)
            v = F.normalize(h.mean(dim=1), dim=-1).to(torch.float16).cpu().numpy()
            for row, i in enumerate(grp):
                keys[i] = v[row]
    return np.stack(keys).astype(KEY_DTYPE)


def _read_leaves(corpus_path, start_byte, n_leaves, leaf_bytes):
    with open(corpus_path, "rb") as f:
        f.seek(start_byte)
        data = f.read(n_leaves * leaf_bytes)
    return [data[i * leaf_bytes:(i + 1) * leaf_bytes] for i in range(len(data) // leaf_bytes)]


def _save(out_dir, leaves, keys, leaf_bytes):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, LEAVES_FILE), "wb") as f:
        f.write(b"".join(leaves))
    offsets = np.arange(len(leaves) + 1, dtype=np.int64) * leaf_bytes
    np.save(os.path.join(out_dir, OFFSETS_FILE), offsets)
    np.save(os.path.join(out_dir, KEYS_FILE), keys.astype(KEY_DTYPE))


def build(corpus_path, n_bytes, leaf_bytes, out_dir, model, batch=BATCH):
    leaves = _read_leaves(corpus_path, 0, n_bytes // leaf_bytes, leaf_bytes)
    _save(out_dir, leaves, embed(model, leaves, batch), leaf_bytes)
    return MemStore(out_dir)


def load(out_dir):
    return MemStore(out_dir)


class MemStore:
    def __init__(self, out_dir):
        self._offsets = np.load(os.path.join(out_dir, OFFSETS_FILE))
        self._keys    = np.load(os.path.join(out_dir, KEYS_FILE))
        self._leaves  = np.memmap(os.path.join(out_dir, LEAVES_FILE), dtype=np.uint8, mode="r")

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys

    def leaf_text(self, i):
        a, b = int(self._offsets[i]), int(self._offsets[i + 1])
        return bytes(self._leaves[a:b]).decode(LEAF_ENCODING, DECODE_ERRORS)
