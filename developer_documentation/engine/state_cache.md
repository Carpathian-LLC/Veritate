# engine: persistent prompt/state cache

Snapshots the post-prefill v13 hybrid decode state to disk and restores it on a
prefix match, so a repeated or extended prompt skips the per-byte prefill loop
(20.8 s for a 1 KB prompt on the target box). Fully env-gated: with
`VERITATE_STATE_CACHE` unset the decode path is byte-identical to before.

Owned by [state_cache.c](../../veritate_engine/v1/src/state_cache.c) /
[state_cache.h](../../veritate_engine/v1/src/state_cache.h). All OS file I/O is
behind the shim in [fsutil.c](../../veritate_engine/v1/src/fsutil.c) (preflight
rule 34); `state_cache.c` includes no OS header.

## what it caches

The resumable per-request state is the six `hybrid_t` fields `hybrid_reset`
zeroes ([hybrid.c:664](../../veritate_engine/v1/src/hybrid.c)): `kv_k`, `kv_v`
(local-block KV over `seq`), `rec_state`, `conv_ring` (recurrent global state),
`slot_count`, `pos`. A snapshot at prefix length `L` also stores `logits`
(the next-byte distribution the sampler reads,
[model.c:1386](../../veritate_engine/v1/src/model.c)) and `u` (final normed
hidden, consumed by `hybrid_final_act_i8`). KV is stored compact: only rows
`0..L-1` per local layer, since `hybrid_step` writes row `pos` before reading it
([hybrid.c:449](../../veritate_engine/v1/src/hybrid.c)), so restored rows need no
tail zeroing.

## how it works

- `state_cache_model_id` fingerprints the bin (path + size + mtime, FNV-1a); set
  on the `hybrid_t` at load ([model.c model_load hybrid branch](../../veritate_engine/v1/src/model.c)).
- Key: two rolling hashes seeded by `model_id`, `hh[L] = hh[L-1]*P + tok`, plus a
  second hash `chk[L]` with a different prime. Both are prefix-consistent, so an
  extended prompt's `hh/chk[L]` equal the base prompt's at `L` (the extend hit).
  Filename is `hex64(hh[L])`; the header carries `chk[L]` + shape guards as a
  collision guard.
- Forward hook ([model.c forward hybrid branch](../../veritate_engine/v1/src/model.c)):
  `restored = try_restore(...)`; `if (restored == 0) hybrid_reset`; step the loop
  from `i = restored`; after the loop, `store(...)`. Both calls guarded by
  `state_cache_enabled()`.
- `try_restore` lists the dir once, scans `L` from the ceiling down to
  `SC_MIN_PREFIX`, opens the first present `hh[L]`, validates the header, restores
  the payload, and `veritate_touch`es the file (LRU refresh).
- `store` snapshots at `L = n`, skips if the key exists, temp-writes then
  `veritate_rename`s atomically, then evicts oldest-mtime files while the dir
  exceeds `VERITATE_STATE_CACHE_MB` (default 4096).

## trace safety

With a trace record the last prompt position `n-1` must be re-stepped so its
trace frame is real ([main.c:502,542](../../veritate_engine/v1/src/main.c) read
`pos = n-1` at step 0). `try_restore` therefore caps the scan ceiling to `n-1`
when `has_trace`, so the matched snapshot is always `<= n-1` and its
`rec_state`/`conv_ring`/`slot_count` stay consistent with the restored KV rows.
An exact-length snapshot (stored by a non-traced run) is simply not used by a
traced prefill.

## env knobs

- `VERITATE_STATE_CACHE=<dir>` : enable + cache directory.
- `VERITATE_STATE_CACHE_MB=<cap>` : size cap (default 4096).
- `VERITATE_STATE_CACHE_LOG=1` : stderr lines `restored L=..` / `miss` / `stored L=..`.

The Python backend
([c_engine.py](../../veritate_mri/inference/backends/c_engine.py)) `setdefault`s
`VERITATE_STATE_CACHE` to `<model_dir>/state_cache` (constructor flag
`state_cache=True`); a parent-env value still wins.

## file format

Little-endian. 80-byte header: magic `VSTC`, `format_version=1`,
`model_version=13`, `model_id`, shape guards (hidden, heads, head_dim, seq,
n_enc, n_global, n_dec, conv_kernel, vocab, dtype), `prefix_len`, `prefix_chk`.
Payload fp32: compact `kv_k`/`kv_v` (`n_local * L * H`), `rec_state`,
`conv_ring`, `logits[vocab]`, `u[hidden]`, then i32 `slot_count`, `pos`.

## dependencies

- [hybrid.h](../../veritate_engine/v1/src/hybrid.h) : `hybrid_t` (adds `bin_id`)
  and the shape fields read/written directly.
- [portability.h](../../veritate_engine/v1/src/portability.h) : `veritate_stat`,
  `veritate_dir_list`, `veritate_remove`, `veritate_rename`, `veritate_touch`,
  `veritate_mkdir`, `veritate_now_ns`.
- Built into the shared-TU list in
  [build.sh](../../veritate_engine/v1/build/build.sh) / build.bat.

## pitfalls

- Snapshots are model-specific: `model_id` mixes size + mtime, so re-exporting a
  bin to the same path invalidates its snapshots (a miss, never a stale hit).
- The cap is checked after each store; a single snapshot larger than the cap is
  still written (eviction cannot drop the file it just added below the cap).
- Restore trusts the header's shape guards; a hash collision with a mismatched
  shape is treated as a miss and the scan continues to a shorter prefix.
- Tests: [tests/engine/test_state_cache.py](../../tests/engine/test_state_cache.py).
