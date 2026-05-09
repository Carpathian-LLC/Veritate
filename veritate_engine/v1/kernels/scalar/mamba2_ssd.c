// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - reference fp32 mamba-2 ssd recurrence. single-token decode form.
// - oracle for the avx-512 kernel. correctness over speed.
// - shapes come from caller. no globals, no hardcoded dims.
// ------------------------------------------------------------------------------------

#include "../../src/mamba2_ssd.h"

#include <math.h>

void mamba2_ssd_step_scalar(
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
        float A   = -expf(A_log[hd]);
        float dt_h = dt[hd];
        float dA  = expf(dt_h * A);
        float Dh  = D[hd];
        const float* x_h = x + (size_t)hd * head_dim;
        float*       y_h = y + (size_t)hd * head_dim;
        float*       h_h = h + (size_t)hd * n_state * head_dim;

        for (int32_t d = 0; d < head_dim; d++) y_h[d] = Dh * x_h[d];

        for (int32_t s = 0; s < n_state; s++) {
            float Bs = B[s] * dt_h;
            float Cs = C[s];
            float* h_row = h_h + (size_t)s * head_dim;
            for (int32_t d = 0; d < head_dim; d++) {
                float h_new = dA * h_row[d] + Bs * x_h[d];
                h_row[d] = h_new;
                y_h[d]  += Cs * h_new;
            }
        }
    }
}
