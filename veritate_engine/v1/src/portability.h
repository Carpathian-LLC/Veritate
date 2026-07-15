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

// monotonic clock in nanoseconds. micro-calibration timing only.
uint64_t veritate_now_ns(void);

// file-system shim — state-cache persistence. POSIX dirent/stat/rename/utimensat,
// Win32 FindFirstFile/MoveFileEx/SetFileTime. keeps state_cache.c OS-free (rule 34).
// mtime_ns is unix-epoch nanoseconds on every OS so stat and dir_list order alike.
typedef void (*veritate_dir_fn)(const char* name, uint64_t size,
                                uint64_t mtime_ns, void* ctx);

// size + modification time. returns 0 on success, nonzero if the path is missing.
int  veritate_stat(const char* path, uint64_t* size, uint64_t* mtime_ns);
// enumerate regular files in dir, invoking cb(basename, size, mtime_ns, ctx) each.
void veritate_dir_list(const char* dir, veritate_dir_fn cb, void* ctx);
// delete a file. returns 0 on success.
int  veritate_remove(const char* path);
// rename, atomically replacing dst if present. returns 0 on success.
int  veritate_rename(const char* from, const char* to);
// set a file's modification time to now (LRU refresh). returns 0 on success.
int  veritate_touch(const char* path);
// create one directory level, ok if it already exists. returns 0 on success.
int  veritate_mkdir(const char* path);

#endif
