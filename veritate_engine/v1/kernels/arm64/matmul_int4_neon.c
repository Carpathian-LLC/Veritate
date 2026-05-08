// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - INT4 packed matmul, NEON + SDOT. mirrors the x86 matmul_int4_vnni_decode
//   structure: load 32 packed bytes, unpack to 64 sign-extended int8, sdot
//   accumulate. rule-23 bitwise oracle is matmul_int4_scalar_prep.
// - decode hot path: 1 a_row x 4 b_cols (1x4 tile). saturates the M-series
//   4-wide SDOT pipe with 4 independent int32x4 accumulators.
// - k must be a multiple of 64 for the SIMD path. odd-shape callers fall
//   back to the scalar oracle. production model dims (V_HIDDEN=768,
//   V_FFN=3072) divide cleanly so the fallback is bench-only.
// veritate_engine/kernels/arm64/matmul_int4_neon.c
// ------------------------------------------------------------------------------------
// Imports:

#include "../../src/veritate.h"

#if !defined(__ARM_FEATURE_DOTPROD)
    #error "matmul_int4_neon.c requires FEAT_DotProd. build with -mcpu=apple-m1."
#endif

#include <arm_neon.h>
#include <stdint.h>

// ------------------------------------------------------------------------------------
// Constants

#define INT4_LO_NIBBLE_MASK   0x0F
#define INT4_SIGN_BIT         8
#define INT4_K_TILE_BYTES     64
#define INT4_PACKED_PER_TILE  32
#define INT4_PREFETCH_BYTES   256

// ------------------------------------------------------------------------------------
// Functions

// ------------------------------------------------------------------------------------
// unpack 32 packed nibble-bytes (= 64 int4 weights) into 4 int8x16 vectors,
// sequentially ordered: w0 = w[0..15], w1 = w[16..31], w2 = w[32..47], w3 = w[48..63].
// each input byte t holds (w[2t] in low nibble, w[2t+1] in high nibble).
// 4-bit signed values are sign-extended via ((x ^ 8) - 8).
// ------------------------------------------------------------------------------------

static inline void int4_unpack_64(
    const uint8_t* packed,
    int8x16_t*     w0,
    int8x16_t*     w1,
    int8x16_t*     w2,
    int8x16_t*     w3
) {
    const int8x16_t mask  = vdupq_n_s8(INT4_LO_NIBBLE_MASK);
    const int8x16_t eight = vdupq_n_s8(INT4_SIGN_BIT);

    uint8x16_t plo = vld1q_u8(packed +  0);
    uint8x16_t phi = vld1q_u8(packed + 16);

    int8x16_t lo_nib_lo = vandq_s8(vreinterpretq_s8_u8(plo), mask);
    int8x16_t hi_nib_lo = vreinterpretq_s8_u8(vshrq_n_u8(plo, 4));
    int8x16_t lo_nib_hi = vandq_s8(vreinterpretq_s8_u8(phi), mask);
    int8x16_t hi_nib_hi = vreinterpretq_s8_u8(vshrq_n_u8(phi, 4));

    lo_nib_lo = vsubq_s8(veorq_s8(lo_nib_lo, eight), eight);
    hi_nib_lo = vsubq_s8(veorq_s8(hi_nib_lo, eight), eight);
    lo_nib_hi = vsubq_s8(veorq_s8(lo_nib_hi, eight), eight);
    hi_nib_hi = vsubq_s8(veorq_s8(hi_nib_hi, eight), eight);

    *w0 = vzip1q_s8(lo_nib_lo, hi_nib_lo);
    *w1 = vzip2q_s8(lo_nib_lo, hi_nib_lo);
    *w2 = vzip1q_s8(lo_nib_hi, hi_nib_hi);
    *w3 = vzip2q_s8(lo_nib_hi, hi_nib_hi);
}

// ------------------------------------------------------------------------------------
// scalar fallback for k that is not a multiple of INT4_K_TILE_BYTES. matches
// matmul_int4_scalar_prep semantics so rule-23 holds regardless of shape.
// ------------------------------------------------------------------------------------

static inline int32_t int4_dot_scalar(
    const int8_t*  a_row,
    const uint8_t* row,
    int32_t        k_half
) {
    int32_t s = 0;
    for (int32_t t = 0; t < k_half; t++) {
        uint8_t b = row[t];
        int8_t  w0 = (int8_t)(((int8_t)(b & INT4_LO_NIBBLE_MASK) ^ INT4_SIGN_BIT) - INT4_SIGN_BIT);
        int8_t  w1 = (int8_t)(((int8_t)((b >> 4) & INT4_LO_NIBBLE_MASK) ^ INT4_SIGN_BIT) - INT4_SIGN_BIT);
        s += (int32_t)a_row[2 * t + 0] * (int32_t)w0;
        s += (int32_t)a_row[2 * t + 1] * (int32_t)w1;
    }
    return s;
}

// ------------------------------------------------------------------------------------
// 1x4 tile — m=1 hot path. one A row dotted into 4 packed B columns. four
// independent int32x4 accumulators saturate the SDOT pipe; the four packed
// b_cols stream from cache with prefetch hints two lines ahead.
// ------------------------------------------------------------------------------------

static inline void int4_block_1x4(
    const int8_t*   a_row,
    const uint8_t*  b0_packed,
    const uint8_t*  b1_packed,
    const uint8_t*  b2_packed,
    const uint8_t*  b3_packed,
    int32_t         k,
    int32_t*        c0,
    int32_t*        c1,
    int32_t*        c2,
    int32_t*        c3
) {
    int32x4_t s0 = vdupq_n_s32(0);
    int32x4_t s1 = vdupq_n_s32(0);
    int32x4_t s2 = vdupq_n_s32(0);
    int32x4_t s3 = vdupq_n_s32(0);

    int32_t kk = 0;
    for (; kk + INT4_K_TILE_BYTES <= k; kk += INT4_K_TILE_BYTES) {
        const int32_t kp = kk / 2;

        __builtin_prefetch(b0_packed + kp + INT4_PREFETCH_BYTES / 2, 0, 3);
        __builtin_prefetch(b1_packed + kp + INT4_PREFETCH_BYTES / 2, 0, 3);
        __builtin_prefetch(b2_packed + kp + INT4_PREFETCH_BYTES / 2, 0, 3);
        __builtin_prefetch(b3_packed + kp + INT4_PREFETCH_BYTES / 2, 0, 3);

        int8x16_t a0 = vld1q_s8(a_row + kk +  0);
        int8x16_t a1 = vld1q_s8(a_row + kk + 16);
        int8x16_t a2 = vld1q_s8(a_row + kk + 32);
        int8x16_t a3 = vld1q_s8(a_row + kk + 48);

        int8x16_t w00, w01, w02, w03;
        int8x16_t w10, w11, w12, w13;
        int8x16_t w20, w21, w22, w23;
        int8x16_t w30, w31, w32, w33;
        int4_unpack_64(b0_packed + kp, &w00, &w01, &w02, &w03);
        int4_unpack_64(b1_packed + kp, &w10, &w11, &w12, &w13);
        int4_unpack_64(b2_packed + kp, &w20, &w21, &w22, &w23);
        int4_unpack_64(b3_packed + kp, &w30, &w31, &w32, &w33);

        s0 = vdotq_s32(s0, a0, w00); s0 = vdotq_s32(s0, a1, w01);
        s0 = vdotq_s32(s0, a2, w02); s0 = vdotq_s32(s0, a3, w03);

        s1 = vdotq_s32(s1, a0, w10); s1 = vdotq_s32(s1, a1, w11);
        s1 = vdotq_s32(s1, a2, w12); s1 = vdotq_s32(s1, a3, w13);

        s2 = vdotq_s32(s2, a0, w20); s2 = vdotq_s32(s2, a1, w21);
        s2 = vdotq_s32(s2, a2, w22); s2 = vdotq_s32(s2, a3, w23);

        s3 = vdotq_s32(s3, a0, w30); s3 = vdotq_s32(s3, a1, w31);
        s3 = vdotq_s32(s3, a2, w32); s3 = vdotq_s32(s3, a3, w33);
    }

    int32_t r0 = (int32_t)vaddvq_s32(s0);
    int32_t r1 = (int32_t)vaddvq_s32(s1);
    int32_t r2 = (int32_t)vaddvq_s32(s2);
    int32_t r3 = (int32_t)vaddvq_s32(s3);

    if (kk < k) {
        const int32_t k_remaining = (k - kk) / 2;
        const int8_t* a_tail = a_row + kk;
        r0 += int4_dot_scalar(a_tail, b0_packed + kk / 2, k_remaining);
        r1 += int4_dot_scalar(a_tail, b1_packed + kk / 2, k_remaining);
        r2 += int4_dot_scalar(a_tail, b2_packed + kk / 2, k_remaining);
        r3 += int4_dot_scalar(a_tail, b3_packed + kk / 2, k_remaining);
    }

    *c0 = r0; *c1 = r1; *c2 = r2; *c3 = r3;
}

// ------------------------------------------------------------------------------------
// per-row dispatcher. iterates output columns in blocks of 4 (1x4 tile);
// any leftover n columns fall through to the scalar oracle row-by-row.
// ------------------------------------------------------------------------------------

static void int4_decode_row(
    const int8_t*           a,
    const prepped_b_int4_t* p,
    int32_t*                c
) {
    const int32_t n      = p->n;
    const int32_t k      = p->k;
    const int32_t k_half = k / 2;

    int32_t j = 0;
    for (; j + 4 <= n; j += 4) {
        int4_block_1x4(
            a,
            p->bt_packed + (size_t)(j + 0) * k_half,
            p->bt_packed + (size_t)(j + 1) * k_half,
            p->bt_packed + (size_t)(j + 2) * k_half,
            p->bt_packed + (size_t)(j + 3) * k_half,
            k,
            &c[j + 0], &c[j + 1], &c[j + 2], &c[j + 3]
        );
    }
    for (; j < n; j++) {
        c[j] = int4_dot_scalar(a, p->bt_packed + (size_t)j * k_half, k_half);
    }
}

// ------------------------------------------------------------------------------------
// public symbol. supplied by exactly one TU per build (arm64 NEON here, x86_64
// VNNI in kernels/x86_64/matmul_int4.c). m loop runs the m=1 hot path per row;
// matches the x86 layout one-for-one so weights load uniformly.
// ------------------------------------------------------------------------------------

void matmul_int4_vnni_prep(
    const int8_t*            a,
    const prepped_b_int4_t*  p,
    int32_t*                 c,
    int32_t                  m
) {
    if ((p->k & (INT4_K_TILE_BYTES - 1)) != 0) {
        matmul_int4_scalar_prep(a, p, c, m);
        return;
    }
    for (int32_t i = 0; i < m; i++) {
        int4_decode_row(a + (size_t)i * p->k, p, c + (size_t)i * p->n);
    }
}
