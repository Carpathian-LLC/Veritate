// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - portable primitives shim per preflight rule 31. wraps OS-specific aligned
//   allocation, thread-pool dispatch, and small toggles behind one surface.
// - per-arch kernels include this header and never include OS headers directly.
// veritate_engine/src/portability.h
// ------------------------------------------------------------------------------------

#ifndef VERITATE_PORTABILITY_H
#define VERITATE_PORTABILITY_H

#include <stddef.h>
#include <stdint.h>

// ------------------------------------------------------------------------------------
// Constants

// vector / cache-line alignment for every kernel buffer. matches the existing
// _mm_malloc(_, 64) call sites and is a multiple of every supported SIMD width
// (avx-512: 64B, neon: 16B, sve: variable but always <=64B for sane chips).
#define VERITATE_ALIGN 64

// hard cap on persistent worker threads. matches the prior VERITATE_MAX_THREADS
// in matmul_vnni.c. raise only with a matched rebench.
#define VERITATE_MAX_THREADS 32

// ------------------------------------------------------------------------------------
// Functions

// aligned allocator. `align` must be a power of two and a multiple of sizeof(void*).
// returns NULL on failure. caller pairs with veritate_aligned_free.
void* veritate_aligned_alloc(size_t bytes, size_t align);
void  veritate_aligned_free (void*  p);

// thread pool — persistent workers, generic per-call dispatch.
// veritate_pool_run wakes the first `n` workers, hands each its arg slot, and
// blocks until all callbacks return. n must be in [1, veritate_pool_size()].
typedef void (*veritate_work_fn)(void* arg, int32_t worker_idx);

int32_t veritate_pool_size(void);
void    veritate_pool_run (veritate_work_fn fn, void* const* args, int32_t n);

#endif
