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
#endif

// ------------------------------------------------------------------------------------
// Constants

// ------------------------------------------------------------------------------------
// Functions

void* veritate_aligned_alloc(size_t bytes, size_t align) {
    if (bytes == 0) return NULL;
#if defined(_WIN32)
    return _aligned_malloc(bytes, align);
#else
    void* p = NULL;
    if (posix_memalign(&p, align, bytes) != 0) return NULL;
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
