# memory store

## What it is

The addressable external-memory retrieval tier: an on-disk leaf store keyed by byte-native model embeddings, flat cosine retrieval over it, and an IVF drill-down for sub-linear search. Lives at [veritate_core/memory/](../../veritate_core/memory/) (`store.py`, `reader.py`, `hindex.py`).

## How it works

- `store.embed(model, seqs)` is the single key-encoding path: mean-pool `model.hidden_states(x)`, L2-normalize, cast fp16. Inputs are grouped by length so every batch is uniform-width and no padding contaminates a key. Runs under `torch.no_grad`.
- `store.build(corpus_path, n_bytes, leaf_bytes, out_dir, model)` splits a corpus slice into fixed-size leaves (`LEAF_BYTES`), embeds them, and persists three files in `out_dir`: `leaves.bin`, `offsets.npy`, `keys.npy`.
- `store.MemStore` memmaps the leaf bytes and loads keys plus offsets. `len()` is the leaf count, `keys()` the fp16 key matrix, `leaf_text(i)` the decoded leaf.
- `reader.encode_query` calls the same `embed`, so a query key is the identical pooling op as a leaf key. `reader.search(keys, query, k)` is a dot product over unit-normalized keys (cosine) with an argpartition top-k; `reader.retrieve` returns `(leaf_texts, scores)`.
- `hindex.HIndex(keys)` partitions the keys with spherical k-means into `N / cell_target` cells. `search(q, topk, nprobe)` scores the centroids, probes the top cells, and exact-scores only those candidates, returning `(indices, scores, candidates_scored)`. `candidates_scored` is the read cost that makes the sub-linear claim measurable.

## Dependencies

- torch for `embed` only. `reader.search` and `hindex` are numpy-only, so retrieval runs without a model once the keys exist.
- Any model exposing `hidden_states(x)`; the store never branches on the model variant (rule 11a).

## Pitfalls

- Keys must be unit-normalized for cosine to reduce to a dot product. Both `embed` and `HIndex` assume it; feeding raw vectors silently changes the ranking.
- A key is a full trunk forward, so building a large index is an offline job, not something to do inside a request.
- `HIndex` holds an fp32 copy of the keys; at large N that copy, not the search, is the memory cost.
- `MemStore` memmaps `leaves.bin`; the directory must outlive the object.
