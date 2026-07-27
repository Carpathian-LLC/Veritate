// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - batched (prefill) matmul for the v13 hybrid path on x86_64. X = [B x k]
//   row-major, out = [B x n] row-major. mirrors matvec_f32_avx2.c per (row j,
//   activation b) with the weight row hot in L1 across the b sweep, so each
//   out[b][j] is bitwise-equal to hybrid_matvec_*_avx2(w, X[b])[j]: two __m256
//   accumulators hold the 16 interleaved partials, fold
//   t_l = (s_l + s_{l+8}) + (s_{l+4} + s_{l+12}), scalar tail into lane 0,
//   reduce (t0+t1)+(t2+t3). fp16 via vcvtph2ps (F16C, exact). int8 activations
//   arrive pre-quantized (qx + a_scale, hoisted by hybrid_mm so j-split workers
//   never race the shared scratch); the widen path (vpmovsxbw + vpmaddwd) sums
//   int32 exactly. each call writes output rows [j0, j1) only. compile with
//   -mavx2 -mf16c -ffp-contract=off.
// veritate_engine/kernels/x86_64/matmul_prefill_avx2.c
// ------------------------------------------------------------------------------------
// Imports:

#include "../../src/hybrid.h"

#include <immintrin.h>

// ------------------------------------------------------------------------------------
// Functions

void hybrid_matmul_f32_avx2(const void* w, const float* X, float* out,
                            int32_t n, int32_t k, int32_t B, const int8_t* qx,
                            const float* a_scale, int32_t j0, int32_t j1) {
    (void)qx; (void)a_scale;
    const float* wf = (const float*)w;
    for (int32_t j = j0; j < j1; j++) {
        const float* row = wf + (size_t)j * k;
        for (int32_t b = 0; b < B; b++) {
            const float* x = X + (size_t)b * k;
            __m256 a_lo = _mm256_setzero_ps();
            __m256 a_hi = _mm256_setzero_ps();
            int32_t i = 0;
            for (; i + 16 <= k; i += 16) {
                a_lo = _mm256_add_ps(a_lo, _mm256_mul_ps(_mm256_loadu_ps(row + i),
                                                         _mm256_loadu_ps(x + i)));
                a_hi = _mm256_add_ps(a_hi, _mm256_mul_ps(_mm256_loadu_ps(row + i + 8),
                                                         _mm256_loadu_ps(x + i + 8)));
            }
            float s[16];
            _mm256_storeu_ps(s,     a_lo);
            _mm256_storeu_ps(s + 8, a_hi);
            float t[4];
            for (int32_t l = 0; l < 4; l++) t[l] = (s[l] + s[l + 8]) + (s[l + 4] + s[l + 12]);
            for (; i < k; i++) t[0] += row[i] * x[i];
            out[(size_t)b * n + j] = (t[0] + t[1]) + (t[2] + t[3]);
        }
    }
}

void hybrid_matmul_f16_avx2(const void* w, const float* X, float* out,
                            int32_t n, int32_t k, int32_t B, const int8_t* qx,
                            const float* a_scale, int32_t j0, int32_t j1) {
    (void)qx; (void)a_scale;
    const uint16_t* wh = (const uint16_t*)w;
    for (int32_t j = j0; j < j1; j++) {
        const uint16_t* row = wh + (size_t)j * k;
        for (int32_t b = 0; b < B; b++) {
            const float* x = X + (size_t)b * k;
            __m256 a_lo = _mm256_setzero_ps();
            __m256 a_hi = _mm256_setzero_ps();
            int32_t i = 0;
            for (; i + 16 <= k; i += 16) {
                __m256 w_lo = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + i)));
                __m256 w_hi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(row + i + 8)));
                a_lo = _mm256_add_ps(a_lo, _mm256_mul_ps(w_lo, _mm256_loadu_ps(x + i)));
                a_hi = _mm256_add_ps(a_hi, _mm256_mul_ps(w_hi, _mm256_loadu_ps(x + i + 8)));
            }
            float s[16];
            _mm256_storeu_ps(s,     a_lo);
            _mm256_storeu_ps(s + 8, a_hi);
            float t[4];
            for (int32_t l = 0; l < 4; l++) t[l] = (s[l] + s[l + 8]) + (s[l + 4] + s[l + 12]);
            for (; i < k; i++) t[0] += hybrid_f16_to_f32(row[i]) * x[i];
            out[(size_t)b * n + j] = (t[0] + t[1]) + (t[2] + t[3]);
        }
    }
}

void hybrid_matmul_i8_avx2(const void* w, const float* X, float* out,
                           int32_t n, int32_t k, int32_t B, const int8_t* qx,
                           const float* a_scale, int32_t j0, int32_t j1) {
    (void)X;
    const hybrid_w_i8_t* wi = (const hybrid_w_i8_t*)w;
    for (int32_t j = j0; j < j1; j++) {
        const int8_t* row = wi->q + (size_t)j * k;
        const float wscale = wi->scale[j];
        for (int32_t b = 0; b < B; b++) {
            if (a_scale[b] == 0.0f) { out[(size_t)b * n + j] = 0.0f; continue; }
            const int8_t* qb = qx + (size_t)b * k;
            __m256i acc = _mm256_setzero_si256();
            int32_t i = 0;
            for (; i + 16 <= k; i += 16) {
                __m256i wv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(row + i)));
                __m256i xv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(qb + i)));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wv, xv));
            }
            __m128i sm = _mm_add_epi32(_mm256_castsi256_si128(acc),
                                       _mm256_extracti128_si256(acc, 1));
            sm = _mm_add_epi32(sm, _mm_shuffle_epi32(sm, _MM_SHUFFLE(1, 0, 3, 2)));
            sm = _mm_add_epi32(sm, _mm_shuffle_epi32(sm, _MM_SHUFFLE(2, 3, 0, 1)));
            int32_t acc_s = _mm_cvtsi128_si32(sm);
            for (; i < k; i++) acc_s += (int32_t)row[i] * (int32_t)qb[i];
            out[(size_t)b * n + j] = (float)acc_s * (wscale * a_scale[b]);
        }
    }
}
