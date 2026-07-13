// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - fp32 + fp16 matvec for the v13 hybrid path on x86_64. two __m256
//   accumulators hold the scalar references' 16 interleaved partials exactly
//   (lanes 0..7 -> s[0..7], 8..15 -> s[8..15]; mul then add, no contraction):
//   fold t_l = (s_l + s_{l+8}) + (s_{l+4} + s_{l+12}), scalar tail into lane 0,
//   reduce (t0+t1)+(t2+t3). fp16 weights convert by vcvtph2ps (F16C, exact,
//   equals hybrid_f16_to_f32) then run the same fp32 math. compile this TU and
//   src/hybrid.c with -ffp-contract=off to keep the pairing exact.
// veritate_engine/kernels/x86_64/matvec_f32_avx2.c
// ------------------------------------------------------------------------------------

#include "../../src/hybrid.h"

#include <immintrin.h>

void hybrid_matvec_f32_avx2(const void* w, const float* x, float* out,
                            int32_t n, int32_t k) {
    const float* wf = (const float*)w;
    for (int32_t j = 0; j < n; j++) {
        const float* row = wf + (size_t)j * k;
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
        out[j] = (t[0] + t[1]) + (t[2] + t[3]);
    }
}

void hybrid_matvec_f16_avx2(const void* w, const float* x, float* out,
                            int32_t n, int32_t k) {
    const uint16_t* wh = (const uint16_t*)w;
    for (int32_t j = 0; j < n; j++) {
        const uint16_t* row = wh + (size_t)j * k;
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
        out[j] = (t[0] + t[1]) + (t[2] + t[3]);
    }
}
