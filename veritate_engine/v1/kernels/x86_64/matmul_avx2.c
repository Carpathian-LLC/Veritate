// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - INT8 matmul, AVX2 intrinsics. sign-extend to int16 then madd_epi16 (no saturation).
// ------------------------------------------------------------------------------------

#include "../../src/veritate.h"
#include "../../src/portability.h"
#include <immintrin.h>

void matmul_int8_avx2(
    const int8_t* a,
    const int8_t* b,
    int32_t*      c,
    int32_t       m,
    int32_t       n,
    int32_t       k
) {
    int8_t* b_col = (int8_t*)veritate_aligned_alloc((size_t)k, VERITATE_ALIGN);

    for (int32_t j = 0; j < n; j++) {
        for (int32_t p = 0; p < k; p++) b_col[p] = b[p * n + j];

        for (int32_t i = 0; i < m; i++) {
            const int8_t* a_row = a + (size_t)i * k;

            __m256i acc = _mm256_setzero_si256();
            int32_t p = 0;

            for (; p + 32 <= k; p += 32) {
                __m256i av = _mm256_loadu_si256((const __m256i*)(a_row + p));
                __m256i bv = _mm256_loadu_si256((const __m256i*)(b_col + p));

                __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(av));
                __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bv));
                __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1));
                __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv, 1));

                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_lo, b_lo));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_hi, b_hi));
            }

            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i s  = _mm_add_epi32(lo, hi);
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            int32_t sum = _mm_cvtsi128_si32(s);

            for (; p < k; p++) sum += (int32_t)a_row[p] * (int32_t)b_col[p];

            c[i * n + j] = sum;
        }
    }

    veritate_aligned_free(b_col);
}
