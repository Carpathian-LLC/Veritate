// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - AVX-512 + VNNI ternary matmul. weights packed 5-per-byte in base-3, activations
//   int8. unpack each row's trits into a per-call stack buffer, then dispatch
//   through vpdpbusd against an unsigned-shifted activation copy. bit-identical
//   to matmul_ternary_scalar_prep by construction (we use the same scalar
//   ternary_unpack_row for the weight tier and the standard +128 bias trick for
//   the activation tier).
// - perf: bounded by the scalar unpack at ~1 GB/s. follow-up work in phase B+
//   (see docs/c_engine_ternary_moe_tracking.md): vectorize unpack via a 256-
//   entry LUT and VPSHUFB, or compute base-3 via reciprocal multiplication.
// veritate_engine/kernels/x86_64/matmul_ternary_vnni.c
// ------------------------------------------------------------------------------------
// Imports

#include "../../src/veritate.h"
#include "../../src/portability.h"

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------------
// Constants

#define TERNARY_TRITS_PER_BYTE 5
#define TERNARY_BASE           3
#define VNNI_LANE_BYTES        64

// ------------------------------------------------------------------------------------
// Functions

static void ternary_unpack_row_local(const uint8_t* bytes, int32_t k, int8_t* out) {
    // bit-identical to kernels/scalar/matmul_ternary_scalar.c::ternary_unpack_row.
    // duplicated here as a static so the compiler can inline.
    int32_t n_bytes = (k + TERNARY_TRITS_PER_BYTE - 1) / TERNARY_TRITS_PER_BYTE;
    for (int32_t b = 0; b < n_bytes; b++) {
        uint32_t v = bytes[b];
        int8_t t[TERNARY_TRITS_PER_BYTE];
        for (int32_t s = TERNARY_TRITS_PER_BYTE - 1; s >= 0; s--) {
            int32_t shifted = (int32_t)(v % TERNARY_BASE);
            v = v / TERNARY_BASE;
            t[s] = (int8_t)(shifted - 1);
        }
        for (int32_t s = 0; s < TERNARY_TRITS_PER_BYTE; s++) {
            int32_t i = b * TERNARY_TRITS_PER_BYTE + s;
            if (i < k) out[i] = t[s];
        }
    }
}

static int32_t vnni_ternary_dot_1x1(const int8_t* a_biased, const int8_t* w, int32_t k) {
    __m512i acc = _mm512_setzero_si512();
    int32_t p = 0;
    for (; p + VNNI_LANE_BYTES <= k; p += VNNI_LANE_BYTES) {
        __m512i au = _mm512_loadu_si512((const __m512i*)(a_biased + p));
        __m512i bv = _mm512_loadu_si512((const __m512i*)(w        + p));
        acc = _mm512_dpbusd_epi32(acc, au, bv);
    }
    int32_t s = _mm512_reduce_add_epi32(acc);
    // tail (k not a multiple of 64). every shape we ship pads to 64 already, but be
    // robust against odd k since ternary k is bounded by ceil(k/5)*5.
    for (; p < k; p++) {
        int32_t a_signed = (int32_t)(uint8_t)a_biased[p] - 128;
        s += a_signed * (int32_t)w[p];
    }
    return s;
}

void matmul_ternary_vnni_prep(const int8_t* a, const prepped_b_ternary_t* p,
                              int32_t* c, int32_t m) {
    const int32_t k         = p->k;
    const int32_t n         = p->n;
    const int32_t row_bytes = (k + TERNARY_TRITS_PER_BYTE - 1) / TERNARY_TRITS_PER_BYTE;
    const int32_t k_padded  = ((k + VNNI_LANE_BYTES - 1) / VNNI_LANE_BYTES) * VNNI_LANE_BYTES;
    const __m512i bias_v    = _mm512_set1_epi8((char)0x80);

    // bias activations once per (i,*) tile; reused across all output columns.
    int8_t a_biased_buf[16384] __attribute__((aligned(64)));
    int8_t* a_biased  = a_biased_buf;
    int8_t* w_buf     = NULL;

    int32_t need_a = k_padded;
    if (need_a > (int32_t)sizeof(a_biased_buf)) {
        a_biased = (int8_t*)veritate_aligned_alloc((size_t)need_a, 64);
    }
    w_buf = (int8_t*)veritate_aligned_alloc((size_t)k_padded, 64);
    if (k_padded > k) {
        memset(w_buf + k, 0, (size_t)(k_padded - k));
    }

    for (int32_t i = 0; i < m; i++) {
        const int8_t* a_row = a + (size_t)i * (size_t)k;

        // shift a to unsigned by adding 128 in place (vector). 64-byte tail
        // padding so VNNI loads at the end can read past k safely; we mask the
        // partial-lane bytes to a value that contributes 0 to the dot product
        // (zero in w_buf tail times any a is 0).
        int32_t pp = 0;
        for (; pp + VNNI_LANE_BYTES <= k; pp += VNNI_LANE_BYTES) {
            __m512i av = _mm512_loadu_si512((const __m512i*)(a_row + pp));
            _mm512_storeu_si512((__m512i*)(a_biased + pp),
                                _mm512_add_epi8(av, bias_v));
        }
        for (; pp < k; pp++) {
            a_biased[pp] = (int8_t)((uint8_t)a_row[pp] + 0x80);
        }
        for (; pp < k_padded; pp++) {
            a_biased[pp] = (int8_t)0x80;  // unsigned 128. paired with w_buf tail = 0 -> 0 contrib.
        }

        for (int32_t j = 0; j < n; j++) {
            const uint8_t* packed = p->bt_packed + (size_t)j * (size_t)row_bytes;
            ternary_unpack_row_local(packed, k, w_buf);
            // s_unsigned = sum (a_biased * w_buf). bias correction:
            // s_signed = s_unsigned - 128 * sum(w_buf).
            // sum(w_buf) is small (-k..+k), compute scalar in the unpack pass to
            // avoid a second sweep. for now, recompute here -- cheap relative to
            // the dot product. fold into unpack in a follow-up pass.
            int32_t w_sum = 0;
            for (int32_t t = 0; t < k; t++) w_sum += (int32_t)w_buf[t];
            int32_t s_unsigned = vnni_ternary_dot_1x1(a_biased, w_buf, k_padded);
            c[(size_t)i * (size_t)n + (size_t)j] = s_unsigned - 128 * w_sum;
        }
    }

    if (a_biased != a_biased_buf) veritate_aligned_free(a_biased);
    veritate_aligned_free(w_buf);
}
