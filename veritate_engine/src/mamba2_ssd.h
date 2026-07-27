// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - mamba-2 ssd single-token recurrence kernels. fp32. matches mamba2_block.py::step.
// - h:        [n_heads, n_state, head_dim]  state, in-place updated
// - y:        [n_heads, head_dim]           output for the token
// - x:        [n_heads, head_dim]           per-head input
// - A_log:    [n_heads]                     A = -exp(A_log)
// - dt:       [n_heads]                     softplus(raw + dt_bias)
// - B, C:     [n_state]                     per-token (shared across heads)
// - D:        [n_heads]                     skip scalar
// ------------------------------------------------------------------------------------

#ifndef VERITATE_MAMBA2_SSD_H
#define VERITATE_MAMBA2_SSD_H

#include <stdint.h>

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
);

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
);

typedef void (*mamba2_ssd_step_fn)(
    float*, float*, const float*, const float*, const float*,
    const float*, const float*, const float*,
    int32_t, int32_t, int32_t
);

extern mamba2_ssd_step_fn mamba2_ssd_step;

void mamba2_ssd_dispatch_init(int has_avx512f);

#endif
