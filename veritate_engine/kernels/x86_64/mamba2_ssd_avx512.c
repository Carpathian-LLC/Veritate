// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - mamba-2 ssd recurrence, avx-512 fp32. parallel along head_dim, 16 lanes per vec.
// - bit-equivalent to scalar within fp32 epsilon: same op order per (s,d).
// - tail handled with mask load/store. head_dim need not be a multiple of 16.
// ------------------------------------------------------------------------------------

#include "../../src/mamba2_ssd.h"

#include <immintrin.h>
#include <math.h>

void mamba2_ssd_step_avx512(
    float*       h,
    float*       y,
    const float* x,
    const float* A_log,
    const float* dt,
    const float* B,
    const float* C,
    const float* D,
    int32_t      n_heads,
    int32_t      n_state,
    int32_t      head_dim
) {
    for (int32_t hd = 0; hd < n_heads; hd++) {
        float A    = -expf(A_log[hd]);
        float dt_h = dt[hd];
        float dA   = expf(dt_h * A);
        __m512 dA_v = _mm512_set1_ps(dA);
        __m512 Dh_v = _mm512_set1_ps(D[hd]);

        const float* x_h = x + (size_t)hd * head_dim;
        float*       y_h = y + (size_t)hd * head_dim;
        float*       h_h = h + (size_t)hd * n_state * head_dim;

        int32_t d = 0;
        for (; d + 16 <= head_dim; d += 16) {
            __m512 xv = _mm512_loadu_ps(x_h + d);
            _mm512_storeu_ps(y_h + d, _mm512_mul_ps(Dh_v, xv));
        }
        if (d < head_dim) {
            __mmask16 m = (__mmask16)((1u << (head_dim - d)) - 1u);
            __m512 xv = _mm512_maskz_loadu_ps(m, x_h + d);
            _mm512_mask_storeu_ps(y_h + d, m, _mm512_mul_ps(Dh_v, xv));
        }

        for (int32_t s = 0; s < n_state; s++) {
            float Bs = B[s] * dt_h;
            __m512 Bs_v = _mm512_set1_ps(Bs);
            __m512 Cs_v = _mm512_set1_ps(C[s]);
            float* h_row = h_h + (size_t)s * head_dim;

            d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 xv  = _mm512_loadu_ps(x_h  + d);
                __m512 hv  = _mm512_loadu_ps(h_row + d);
                __m512 yv  = _mm512_loadu_ps(y_h  + d);
                __m512 h_new = _mm512_fmadd_ps(dA_v, hv, _mm512_mul_ps(Bs_v, xv));
                _mm512_storeu_ps(h_row + d, h_new);
                _mm512_storeu_ps(y_h + d, _mm512_fmadd_ps(Cs_v, h_new, yv));
            }
            if (d < head_dim) {
                __mmask16 m = (__mmask16)((1u << (head_dim - d)) - 1u);
                __m512 xv  = _mm512_maskz_loadu_ps(m, x_h  + d);
                __m512 hv  = _mm512_maskz_loadu_ps(m, h_row + d);
                __m512 yv  = _mm512_maskz_loadu_ps(m, y_h  + d);
                __m512 h_new = _mm512_fmadd_ps(dA_v, hv, _mm512_mul_ps(Bs_v, xv));
                _mm512_mask_storeu_ps(h_row + d, m, h_new);
                _mm512_mask_storeu_ps(y_h + d, m, _mm512_fmadd_ps(Cs_v, h_new, yv));
            }
        }
    }
}

mamba2_ssd_step_fn mamba2_ssd_step = mamba2_ssd_step_scalar;

void mamba2_ssd_dispatch_init(int has_avx512f) {
    mamba2_ssd_step = has_avx512f ? mamba2_ssd_step_avx512 : mamba2_ssd_step_scalar;
}
