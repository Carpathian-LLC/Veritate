// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - x86_64 inline per-pair attention helpers. avx-512 + vnni.
// - included from src/model.c on x86_64 builds only. dispatched at compile time.
// veritate_engine/kernels/inline/attn_x86_64.h
// ------------------------------------------------------------------------------------
// Imports:

#ifndef VERITATE_ATTN_INLINE_X86_64_H
#define VERITATE_ATTN_INLINE_X86_64_H

#include <stdint.h>
#include <immintrin.h>

// ------------------------------------------------------------------------------------
// Constants

// V_HEAD_DIM-sized bias for the unsigned-shift trick in dpbusd: sum over a
// 64-byte vector that has been shifted by +128.
#define ATTN_VNNI_HSUM_BIAS 8192

// ------------------------------------------------------------------------------------
// Functions

static inline int32_t attn_hsum_inline(const int8_t* x) {
    const __m512i bias = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);
    __m512i v = _mm512_add_epi8(_mm512_loadu_si512((const __m512i*)x), bias);
    return _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), v, ones)) - ATTN_VNNI_HSUM_BIAS;
}

static inline int32_t attn_dot_inline(const int8_t* q, const int8_t* k, int32_t q_sum) {
    const __m512i bias = _mm512_set1_epi8((char)0x80);
    __m512i ku  = _mm512_add_epi8(_mm512_loadu_si512((const __m512i*)k), bias);
    __m512i qs  = _mm512_loadu_si512((const __m512i*)q);
    __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ku, qs);
    return _mm512_reduce_add_epi32(acc) - 128 * q_sum;
}

// cross-arch signature parity. x86 vnni's per-call dependency chain sits at
// 4 cycles (one dpbusd per 64-byte head); four sequential calls is the
// natural shape here. arm64 ports diverge (parallel sdot accumulators).
static inline void attn_dot_inline_4(
    const int8_t*  q,
    const int8_t*  k0, const int8_t* k1, const int8_t* k2, const int8_t* k3,
    int32_t        q_sum,
    int32_t*       out
) {
    out[0] = attn_dot_inline(q, k0, q_sum);
    out[1] = attn_dot_inline(q, k1, q_sum);
    out[2] = attn_dot_inline(q, k2, q_sum);
    out[3] = attn_dot_inline(q, k3, q_sum);
}

#endif
