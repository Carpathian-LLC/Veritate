// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - persistent prompt/state cache for the v13 hybrid decode path. snapshots the
//   post-prefill hybrid_t state to disk and restores it on a longest-prefix hit,
//   so a repeated (or extended) prompt skips the per-byte prefill loop.
// - fully env-gated by VERITATE_STATE_CACHE; unset means no lookup, no store.
//   spec: developer_documentation/engine/state_cache.md.
// veritate_engine/src/state_cache.h
// ------------------------------------------------------------------------------------

#ifndef VERITATE_STATE_CACHE_H
#define VERITATE_STATE_CACHE_H

#include <stdint.h>

#include "hybrid.h"

// non-zero when VERITATE_STATE_CACHE names a directory. cheap, call per prefill.
int state_cache_enabled(void);

// stable per-file fingerprint: mixes the bin path, size, and mtime.
uint64_t state_cache_model_id(const char* bin_path);

// longest-prefix lookup + restore into h. returns the restored prefix length L
// (0 = miss). has_trace caps the usable prefix to n-1 so the last position is
// re-stepped and its trace frame is real.
int32_t state_cache_try_restore(hybrid_t* h, uint64_t model_id,
                                const int32_t* tokens, int32_t n, int has_trace);

// snapshot h at prefix length n. no-op if the key already exists; temp-write +
// atomic rename, then evict to the size cap. off the decode path.
void state_cache_store(hybrid_t* h, uint64_t model_id,
                       const int32_t* tokens, int32_t n);

#endif
