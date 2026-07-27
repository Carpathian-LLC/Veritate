// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - arm64 inline per-pair attention helpers. neon + sdot.
// - included from src/model.c on arm64 builds only. dispatched at compile time.
// - sdot does signed-int8 dot product directly: no bias trick needed; q_sum is
//   accepted to keep the per-arch surface uniform but unused here.
// veritate_engine/kernels/inline/attn_arm64.h
// ------------------------------------------------------------------------------------
// Imports:

#ifndef VERITATE_ATTN_INLINE_ARM64_H
#define VERITATE_ATTN_INLINE_ARM64_H

#include <stdint.h>
#include <arm_neon.h>

// ------------------------------------------------------------------------------------
// Constants

// V_HEAD_DIM = 64 bytes per attn pair; consumed in 4 sdot lanes of 16 bytes.
#define ATTN_NEON_PAIR_BYTES 64
#define ATTN_NEON_LANE_BYTES 16

// ------------------------------------------------------------------------------------
// Functions

// horizontal sum hook. SDOT path's attn_dot_inline ignores q_sum, so the
// arm hsum returns a constant zero rather than burning loads + reductions
// the compiler may not eliminate. cross-arch signature parity preserved.
static inline int32_t attn_hsum_inline(const int8_t* x) {
    (void)x;
    return 0;
}

// signed int8 dot product over 64 elements via 4x sdot. q_sum is unused on
// the sdot path; the parameter survives for cross-arch signature parity.
static inline int32_t attn_dot_inline(const int8_t* q, const int8_t* k, int32_t q_sum) {
    (void)q_sum;
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t q0 = vld1q_s8(q +  0); int8x16_t k0 = vld1q_s8(k +  0);
    int8x16_t q1 = vld1q_s8(q + 16); int8x16_t k1 = vld1q_s8(k + 16);
    int8x16_t q2 = vld1q_s8(q + 32); int8x16_t k2 = vld1q_s8(k + 32);
    int8x16_t q3 = vld1q_s8(q + 48); int8x16_t k3 = vld1q_s8(k + 48);
    acc = vdotq_s32(acc, q0, k0);
    acc = vdotq_s32(acc, q1, k1);
    acc = vdotq_s32(acc, q2, k2);
    acc = vdotq_s32(acc, q3, k3);
    return (int32_t)vaddvq_s32(acc);
}

// 4-at-a-time per-pair attention dot: one Q row dotted against 4 K rows,
// 4 independent int32x4 accumulators saturating the 4-wide SDOT pipe in the
// M-series P-core. caller writes the four int32 results into out[0..3].
// breaks the per-call dependency chain that bottlenecks the decode K-loop
// (256 K rows / head / layer at end-of-context).
static inline void attn_dot_inline_4(
    const int8_t*  q,
    const int8_t*  k0, const int8_t* k1, const int8_t* k2, const int8_t* k3,
    int32_t        q_sum,
    int32_t*       out
) {
    (void)q_sum;
    int8x16_t q0 = vld1q_s8(q +  0);
    int8x16_t q1 = vld1q_s8(q + 16);
    int8x16_t q2 = vld1q_s8(q + 32);
    int8x16_t q3 = vld1q_s8(q + 48);

    int32x4_t a0 = vdupq_n_s32(0);
    int32x4_t a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0);
    int32x4_t a3 = vdupq_n_s32(0);

    a0 = vdotq_s32(a0, q0, vld1q_s8(k0 +  0));
    a0 = vdotq_s32(a0, q1, vld1q_s8(k0 + 16));
    a0 = vdotq_s32(a0, q2, vld1q_s8(k0 + 32));
    a0 = vdotq_s32(a0, q3, vld1q_s8(k0 + 48));

    a1 = vdotq_s32(a1, q0, vld1q_s8(k1 +  0));
    a1 = vdotq_s32(a1, q1, vld1q_s8(k1 + 16));
    a1 = vdotq_s32(a1, q2, vld1q_s8(k1 + 32));
    a1 = vdotq_s32(a1, q3, vld1q_s8(k1 + 48));

    a2 = vdotq_s32(a2, q0, vld1q_s8(k2 +  0));
    a2 = vdotq_s32(a2, q1, vld1q_s8(k2 + 16));
    a2 = vdotq_s32(a2, q2, vld1q_s8(k2 + 32));
    a2 = vdotq_s32(a2, q3, vld1q_s8(k2 + 48));

    a3 = vdotq_s32(a3, q0, vld1q_s8(k3 +  0));
    a3 = vdotq_s32(a3, q1, vld1q_s8(k3 + 16));
    a3 = vdotq_s32(a3, q2, vld1q_s8(k3 + 32));
    a3 = vdotq_s32(a3, q3, vld1q_s8(k3 + 48));

    out[0] = (int32_t)vaddvq_s32(a0);
    out[1] = (int32_t)vaddvq_s32(a1);
    out[2] = (int32_t)vaddvq_s32(a2);
    out[3] = (int32_t)vaddvq_s32(a3);
}

#endif
