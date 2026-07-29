# bigram index

## What it is

A unigram plus bigram count index over a corpus `.bin`, stored as a `<stem>_bigrams.npz` sidecar next to it. The writing-health probe scores generated word pairs against it (normalized PMI). Lives at [veritate_mri/tools/build_bigram_index.py](../../veritate_mri/tools/build_bigram_index.py).

## How it works

- `sidecar_path(corpus_path)` is the single owner of the sidecar name: `<stem>_train.bin` becomes `<stem>_train_bigrams.npz`, anything else gets `.bigrams.npz` appended.
- `iter_words` streams the corpus in `CHUNK_BYTES` chunks with a carry window, so a multi-GB corpus never loads whole. `build_index` counts unigrams and adjacent pairs, trims the bigram counter periodically to bound memory, keeps the top `top_uni` tokens as the vocab, and keeps the top `top_bigrams` pairs whose both tokens are in that vocab.
- `write_index(corpus_path, top_uni, top_bigrams, max_bytes)` writes the sidecar: `vocab`, `uni_c`, `bi_keys` (packed `(i<<32)|j`), `bi_c`, `n_tokens`, `n_bigrams`, `config`. `max_bytes` caps the scan.
- Built on demand: `checkpoint_probe._wh_load_pmi_index` calls `write_index` the first time a corpus needs an index it does not have, capped at `WH_PMI_MAX_SCAN_BYTES`, then caches the loaded index in process memory. There is no manual step in the PMI path.
- `--all` pre-builds uncapped indexes for every `data/corpus/*_train.bin` (pg19 is opt-in via `--include-pg19`).

## Dependencies

- `readers.paths` for `CORPUS_ROOT` and the train-bin suffix.
- [checkpoint_probe.py](../architecture/backend/checkpoint_probe.md) is the consumer.

## Pitfalls

- The on-demand build is capped, so its index covers the head of a large corpus rather than all of it. PMI values from a capped index are not comparable with those from an uncapped `--all` build of the same corpus.
- The first checkpoint after a corpus changes pays the build cost inside the save; the cap bounds it, the process cache means later checkpoints pay nothing.
- Tokens are lowercase `[a-z][a-z']*`, so digits, punctuation, and non-ASCII words are outside the index and score at the OOV floor.
