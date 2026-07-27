// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - aligned-alloc shim. windows uses _aligned_malloc; posix uses posix_memalign.
// - all veritate buffers route through this. constants live in portability.h.
// veritate_engine/src/alloc.c
// ------------------------------------------------------------------------------------
// Imports:

#include "portability.h"

#include <stdlib.h>

#if defined(_WIN32)
    #include <malloc.h>
#elif defined(__linux__)
    #include <sys/mman.h>
#endif

// ------------------------------------------------------------------------------------
// Constants

// ------------------------------------------------------------------------------------
// Functions

// Buffers at or above VERITATE_HUGE_MIN are raised to huge-page alignment so the
// OS can back them with huge pages: Linux needs both the alignment and the
// madvise hint when transparent_hugepage is set to madvise. Elsewhere the wider
// alignment is harmless and the allocation behaves exactly as before.
void* veritate_aligned_alloc(size_t bytes, size_t align) {
    if (bytes == 0) return NULL;
    if (bytes >= VERITATE_HUGE_MIN && align < VERITATE_HUGE_PAGE) align = VERITATE_HUGE_PAGE;
#if defined(_WIN32)
    return _aligned_malloc(bytes, align);
#else
    void* p = NULL;
    if (posix_memalign(&p, align, bytes) != 0) return NULL;
    #if defined(__linux__)
    if (bytes >= VERITATE_HUGE_MIN) madvise(p, bytes, MADV_HUGEPAGE);
    #endif
    return p;
#endif
}

void veritate_aligned_free(void* p) {
    if (!p) return;
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}
