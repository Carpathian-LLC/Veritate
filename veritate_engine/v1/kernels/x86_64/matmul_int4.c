// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - int4 packed matmul. 2 weights per byte. low nibble = w[2t], high = w[2t+1].
// - sign-extend 4-bit signed via (x ^ 8) - 8.
// - per-row weight scale (fp32) baked at prep, q24 requant scale on output.
// - decode shape (m=1) accumulator path: load 32 packed bytes -> 64 int8 weights.
// ------------------------------------------------------------------------------------

#include "../../src/veritate.h"
#include "../../src/portability.h"

#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------

static inline int8_t sign_ext_4(int8_t nib) {
    // nib has value in 0..15 (low nibble); sign-extend to int8 in -8..7.
    return (int8_t)(((int8_t)(nib & 0x0F) ^ 8) - 8);
}

// ------------------------------------------------------------------------------------
// prep — pack int8 b (column-major) into int4 packed rows. k must be even.
// q24 scale derived from rms(b) like int8 prep, with a 16x boost since int4 weights
// are <=8 in magnitude vs <=127. exporter writes the calibrated scale directly.
// ------------------------------------------------------------------------------------

void prep_b_int4(const int8_t* b, int32_t n, int32_t k, prepped_b_int4_t* out) {
    out->n          = n;
    out->k          = k;
    out->bt_packed  = (uint8_t*)veritate_aligned_alloc((size_t)n * (k / 2), 64);
    out->bias       = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), 64);
    out->row_q24    = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), 64);

    for (int32_t j = 0; j < n; j++) {
        int32_t s = 0;
        uint8_t* dst = out->bt_packed + (size_t)j * (k / 2);
        for (int32_t t = 0; t < k / 2; t++) {
            int8_t v0 = b[(2 * t + 0) * n + j];
            int8_t v1 = b[(2 * t + 1) * n + j];
            if (v0 >  7) v0 =  7; if (v0 < -8) v0 = -8;
            if (v1 >  7) v1 =  7; if (v1 < -8) v1 = -8;
            dst[t] = (uint8_t)((v0 & 0x0F) | ((v1 & 0x0F) << 4));
            s += v0 + v1;
        }
        out->bias[j] = 128 * s;
        out->row_q24[j] = 1 << 24;  // identity by default; loader overwrites with calibrated scales
    }
}

void free_prepped_b_int4(prepped_b_int4_t* p) {
    if (p->bt_packed) veritate_aligned_free(p->bt_packed);
    if (p->bias)      veritate_aligned_free(p->bias);
    if (p->row_q24)   veritate_aligned_free(p->row_q24);
    p->bt_packed = NULL; p->bias = NULL; p->row_q24 = NULL;
}

// ------------------------------------------------------------------------------------
// scalar oracle — m=1 only. unpacks each row, dot-products against a.
// ------------------------------------------------------------------------------------

void matmul_int4_scalar_prep(const int8_t* a, const prepped_b_int4_t* p,
                             int32_t* c, int32_t m) {
    for (int32_t i = 0; i < m; i++) {
        const int8_t* a_row = a + (size_t)i * p->k;
        for (int32_t j = 0; j < p->n; j++) {
            const uint8_t* row = p->bt_packed + (size_t)j * (p->k / 2);
            int32_t s = 0;
            for (int32_t t = 0; t < p->k / 2; t++) {
                uint8_t b  = row[t];
                int8_t  w0 = sign_ext_4((int8_t)(b & 0x0F));
                int8_t  w1 = sign_ext_4((int8_t)((b >> 4) & 0x0F));
                s += (int32_t)a_row[2 * t + 0] * (int32_t)w0;
                s += (int32_t)a_row[2 * t + 1] * (int32_t)w1;
            }
            c[i * p->n + j] = s;
        }
    }
}

// ------------------------------------------------------------------------------------
// avx-512 unpack -- load 32 packed bytes into ymm, expand to 64 int8 weights in zmm
// in sequential order [w0..w63]. relies on AVX-512BW + AVX-512F only.
//
// approach:
//   1. load 32 packed bytes -> ymm (each byte holds 2 weights)
//   2. low nibbles  = w[2t]   for t in 0..31
//      high nibbles = w[2t+1] for t in 0..31
//   3. sign-extend each via (x ^ 8) - 8  ->  two ymm of int8 in -8..7
//   4. unpacklo_epi8 / unpackhi_epi8 interleave per-128-bit lane:
//        L = [w0..w15, w32..w47]   (low 16 from lane0, low 16 from lane1)
//        H = [w16..w31, w48..w63]
//   5. permute2var_epi64 with [0,1,8,9,2,3,10,11] reorders qwords:
//        result = [L_lane0, H_lane0, L_lane1, H_lane1] = [w0..w63] sequential
// ------------------------------------------------------------------------------------

static inline __m512i unpack_int4_64(const uint8_t* packed) {
    const __m256i mask_lo = _mm256_set1_epi8(0x0F);
    const __m256i eight   = _mm256_set1_epi8(8);

    __m256i wpack = _mm256_loadu_si256((const __m256i*)packed);
    __m256i lo    = _mm256_and_si256(wpack, mask_lo);
    __m256i hi    = _mm256_and_si256(_mm256_srli_epi16(wpack, 4), mask_lo);
    lo = _mm256_sub_epi8(_mm256_xor_si256(lo, eight), eight);
    hi = _mm256_sub_epi8(_mm256_xor_si256(hi, eight), eight);

    __m256i L = _mm256_unpacklo_epi8(lo, hi);
    __m256i H = _mm256_unpackhi_epi8(lo, hi);

    __m512i Lz = _mm512_castsi256_si512(L);
    __m512i Hz = _mm512_castsi256_si512(H);

    // qword indices: 0..7 from Lz, 8..15 from Hz. only Lz[0..3], Hz[0..3] hold data.
    // want sequential [w0..w63] = [Lz[0], Lz[1], Hz[0], Hz[1], Lz[2], Lz[3], Hz[2], Hz[3]].
    const __m512i ctrl = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    return _mm512_permutex2var_epi64(Lz, ctrl, Hz);
}

// ------------------------------------------------------------------------------------
// m=1 decode hot path. activations are int8, weights int4 packed. per output column j,
// dot product over k elements via vnni dpbusd; subtract 128*sum(w) bias for the unsigned
// shift. a is biased by +128 once outside the j loop and reused across all columns.
// ------------------------------------------------------------------------------------

static void matmul_int4_vnni_decode(const int8_t* a, const prepped_b_int4_t* p, int32_t* c) {
    const int32_t k = p->k;
    const int32_t n = p->n;
    const __m512i bias_v = _mm512_set1_epi8((char)0x80);

    int8_t a_biased_buf[16384] __attribute__((aligned(64)));
    int8_t* a_biased = a_biased_buf;
    if (k > (int32_t)sizeof(a_biased_buf)) {
        a_biased = (int8_t*)veritate_aligned_alloc((size_t)k, 64);
    }
    for (int32_t p2 = 0; p2 < k; p2 += 64) {
        __m512i av = _mm512_loadu_si512((const __m512i*)(a + p2));
        _mm512_storeu_si512((__m512i*)(a_biased + p2), _mm512_add_epi8(av, bias_v));
    }

    for (int32_t j = 0; j < n; j++) {
        const uint8_t* row = p->bt_packed + (size_t)j * (k / 2);
        __m512i acc = _mm512_setzero_si512();
        for (int32_t t = 0; t < k; t += 64) {
            __m512i wv = unpack_int4_64(row + t / 2);
            __m512i au = _mm512_loadu_si512((const __m512i*)(a_biased + t));
            acc = _mm512_dpbusd_epi32(acc, au, wv);
        }
        int32_t s = _mm512_reduce_add_epi32(acc) - p->bias[j];
        c[j] = s;
    }

    if (a_biased != a_biased_buf) veritate_aligned_free(a_biased);
}

void matmul_int4_vnni_prep(const int8_t* a, const prepped_b_int4_t* p,
                           int32_t* c, int32_t m) {
    for (int32_t i = 0; i < m; i++) {
        matmul_int4_vnni_decode(a + (size_t)i * p->k, p, c + (size_t)i * p->n);
    }
}

// ------------------------------------------------------------------------------------
// hadamard apply — block-diagonal sylvester rotation, block size 64 = V_HEAD_DIM.
// uses fast walsh-hadamard transform (FWHT) butterflies. each block is 6 stages of
// add/sub on length-64 int16 vectors; final scale is 1/sqrt(64) = 1/8.
// cols must be a multiple of 64. dst may equal src.
// ------------------------------------------------------------------------------------

#define V_HD 64

// scalar reference, for correctness oracle if ever needed.
static void hadamard_block64_scalar(const int8_t* sb, int8_t* db) {
    int16_t v[V_HD];
    for (int i = 0; i < V_HD; i++) v[i] = (int16_t)sb[i];
    // FWHT in-place
    for (int sz = 1; sz < V_HD; sz <<= 1) {
        for (int base = 0; base < V_HD; base += sz * 2) {
            for (int j = 0; j < sz; j++) {
                int16_t a = v[base + j];
                int16_t b = v[base + j + sz];
                v[base + j     ] = a + b;
                v[base + j + sz] = a - b;
            }
        }
    }
    // divide by 8 (== 1/sqrt(64)) with rounding
    for (int i = 0; i < V_HD; i++) {
        int32_t r = (v[i] + (v[i] >= 0 ? 4 : -4)) / 8;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        db[i] = (int8_t)r;
    }
}

// in-int16 FWHT: 6 stages of butterfly, then divide by 8 with rounding, saturate to int8.
// avx-512 vectorizes the higher stages naturally (sz>=4 align to lane crossings); for sz=1,2
// we use scalar on int16. dominant cost on a 9800X3D is the cache traffic, not the ops.
static inline void hadamard_block64(const int8_t* sb, int8_t* db) {
    int16_t v[V_HD] __attribute__((aligned(64)));
    {
        __m512i v_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)sb));
        __m512i v_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(sb + 32)));
        _mm512_storeu_si512((__m512i*)(v +  0), v_lo);
        _mm512_storeu_si512((__m512i*)(v + 32), v_hi);
    }
    for (int sz = 1; sz < V_HD; sz <<= 1) {
        for (int base = 0; base < V_HD; base += sz * 2) {
            for (int j = 0; j < sz; j++) {
                int16_t a = v[base + j];
                int16_t b = v[base + j + sz];
                v[base + j     ] = (int16_t)(a + b);
                v[base + j + sz] = (int16_t)(a - b);
            }
        }
    }
    __m512i v0 = _mm512_loadu_si512((const __m512i*)(v +  0));
    __m512i v1 = _mm512_loadu_si512((const __m512i*)(v + 32));
    // round-half-toward-zero divide by 8: add 4 if >=0 else -4, then >>3.
    const __m512i bias_pos = _mm512_set1_epi16(4);
    const __m512i bias_neg = _mm512_set1_epi16(-4);
    __mmask32 m0 = _mm512_cmpge_epi16_mask(v0, _mm512_setzero_si512());
    __mmask32 m1 = _mm512_cmpge_epi16_mask(v1, _mm512_setzero_si512());
    v0 = _mm512_srai_epi16(_mm512_add_epi16(v0, _mm512_mask_blend_epi16(m0, bias_neg, bias_pos)), 3);
    v1 = _mm512_srai_epi16(_mm512_add_epi16(v1, _mm512_mask_blend_epi16(m1, bias_neg, bias_pos)), 3);
    __m512i packed = _mm512_packs_epi16(v0, v1);
    // packs_epi16 is per-128-bit-lane interleaved. fix by 64-bit permute.
    packed = _mm512_permutexvar_epi64(_mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0), packed);
    _mm512_storeu_si512((__m512i*)db, packed);
}

void hadamard_apply_int8(const int8_t* src, int8_t* dst, int32_t cols) {
    const int32_t blocks = cols / V_HD;
    for (int32_t b = 0; b < blocks; b++) {
        hadamard_block64(src + (size_t)b * V_HD, dst + (size_t)b * V_HD);
    }
    (void)hadamard_block64_scalar;
}
