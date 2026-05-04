// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - portable scalar int4 path. prep, free, scalar oracle, and a delegating
//   stub for matmul_int4_vnni_prep so non-x86 builds link cleanly.
// - the build script links this TU only when no SIMD int4 backend ships for
//   the target arch. on x86 builds, the AVX-512 version owns these symbols.
// veritate_engine/kernels/scalar/matmul_int4_scalar.c
// ------------------------------------------------------------------------------------
// Imports:

#include "../../src/veritate.h"
#include "../../src/portability.h"

#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------------
// Constants

#define INT4_IDENTITY_Q24 (1 << 24)
#define INT4_LO_NIBBLE    0x0F
#define INT4_SIGN_BIT     8

// ------------------------------------------------------------------------------------
// Functions

static inline int8_t sign_ext_4(int8_t nib) {
    return (int8_t)(((int8_t)(nib & INT4_LO_NIBBLE) ^ INT4_SIGN_BIT) - INT4_SIGN_BIT);
}

void prep_b_int4(const int8_t* b, int32_t n, int32_t k, prepped_b_int4_t* out) {
    out->n          = n;
    out->k          = k;
    out->bt_packed  = (uint8_t*)veritate_aligned_alloc((size_t)n * (k / 2),         VERITATE_ALIGN);
    out->bias       = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), VERITATE_ALIGN);
    out->row_q24    = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), VERITATE_ALIGN);

    for (int32_t j = 0; j < n; j++) {
        int32_t s = 0;
        uint8_t* dst = out->bt_packed + (size_t)j * (k / 2);
        for (int32_t t = 0; t < k / 2; t++) {
            int8_t v0 = b[(2 * t + 0) * n + j];
            int8_t v1 = b[(2 * t + 1) * n + j];
            if (v0 >  7) v0 =  7; if (v0 < -8) v0 = -8;
            if (v1 >  7) v1 =  7; if (v1 < -8) v1 = -8;
            dst[t] = (uint8_t)((v0 & INT4_LO_NIBBLE) | ((v1 & INT4_LO_NIBBLE) << 4));
            s += v0 + v1;
        }
        out->bias[j]    = 128 * s;
        out->row_q24[j] = INT4_IDENTITY_Q24;
    }
}

void free_prepped_b_int4(prepped_b_int4_t* p) {
    if (p->bt_packed) veritate_aligned_free(p->bt_packed);
    if (p->bias)      veritate_aligned_free(p->bias);
    if (p->row_q24)   veritate_aligned_free(p->row_q24);
    p->bt_packed = NULL; p->bias = NULL; p->row_q24 = NULL;
}

void matmul_int4_scalar_prep(const int8_t* a, const prepped_b_int4_t* p,
                             int32_t* c, int32_t m) {
    for (int32_t i = 0; i < m; i++) {
        const int8_t* a_row = a + (size_t)i * p->k;
        for (int32_t j = 0; j < p->n; j++) {
            const uint8_t* row = p->bt_packed + (size_t)j * (p->k / 2);
            int32_t s = 0;
            for (int32_t t = 0; t < p->k / 2; t++) {
                uint8_t b  = row[t];
                int8_t  w0 = sign_ext_4((int8_t)(b & INT4_LO_NIBBLE));
                int8_t  w1 = sign_ext_4((int8_t)((b >> 4) & INT4_LO_NIBBLE));
                s += (int32_t)a_row[2 * t + 0] * (int32_t)w0;
                s += (int32_t)a_row[2 * t + 1] * (int32_t)w1;
            }
            c[i * p->n + j] = s;
        }
    }
}

// matmul_int4_vnni_prep is supplied by the per-arch TU
// (kernels/x86_64/matmul_int4.c via VNNI, kernels/arm64/matmul_int4_neon.c via
// NEON SDOT). this scalar TU intentionally does not provide it: scalar-only
// builds without a SIMD int4 path do not exist in the current build matrix.
