// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - arm64 NEON transformer hot-path kernels:
//     score_dot_v_neon       — score-weighted V row sum to int8 row.
//     softmax_rows_neon      — fp32 softmax + int16 quantize at 2^15.
//     layernorm_i16_to_i8_neon — int16 residual to int8 layernormed row.
// - matched 1:1 to the avx-512 path's numerics. correctness oracle is
//   kernels/scalar/transformer_scalar.c; this TU must produce within 1 LSB.
// - softmax fp inner loop uses 4-wide poly5 of e^r with range reduction
//   r = x - n*ln2, then 2^n via integer exponent-bias (no scalef on NEON).
// - layernorm post-RMS weight×scale loop is fully vectorized over 16
//   elements/iter; sumsq pass also vectorized.
// veritate_engine/kernels/arm64/transformer_neon.c
// ------------------------------------------------------------------------------------
// Imports:

#include "../../src/veritate.h"

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>

// ------------------------------------------------------------------------------------
// Constants

#define SDV_Q15_HALF       16384
#define SDV_Q15_SHIFT      15
#define SDV_OUT_BYTES      64
#define SDV_INT8_MAX       127
#define SDV_INT8_MIN      (-128)

#define SOFTMAX_Q15_SCALE  32768.0f
#define SOFTMAX_Q15_MAX    32767
#define SOFTMAX_Q15_MIN   (-32768)
#define SOFTMAX_EXP_CLAMP  (-87.0f)

#define LN_EPS             1e-5f
#define LN_HALF_PRESCALE   0.5f
#define LN_INT8_MAX        127
#define LN_INT8_MIN       (-128)

// ------------------------------------------------------------------------------------
// Functions

// ------------------------------------------------------------------------------------
// score_dot_v_neon — out[64] = sat_int8( ((sum_j scores[j] * v_base[j*v_stride + c]) + 16384) >> 15 ).
// scores is int16 q15 softmax. accumulator is int32 across all j.
// ------------------------------------------------------------------------------------

void score_dot_v_neon(const int16_t* scores, const int8_t* v_base,
                      int32_t v_stride, int32_t n_j, int8_t* out) {
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int32x4_t a4 = vdupq_n_s32(0), a5 = vdupq_n_s32(0);
    int32x4_t a6 = vdupq_n_s32(0), a7 = vdupq_n_s32(0);
    int32x4_t a8 = vdupq_n_s32(0), a9 = vdupq_n_s32(0);
    int32x4_t aA = vdupq_n_s32(0), aB = vdupq_n_s32(0);
    int32x4_t aC = vdupq_n_s32(0), aD = vdupq_n_s32(0);
    int32x4_t aE = vdupq_n_s32(0), aF = vdupq_n_s32(0);

    for (int32_t j = 0; j < n_j; j++) {
        int32x4_t sj = vdupq_n_s32((int32_t)scores[j]);
        const int8_t* vp = v_base + (size_t)j * v_stride;
        int8x16_t v_lo = vld1q_s8(vp +  0);
        int8x16_t v_hi = vld1q_s8(vp + 16);
        int8x16_t v_lo2 = vld1q_s8(vp + 32);
        int8x16_t v_hi2 = vld1q_s8(vp + 48);

        int16x8_t v_lo_l = vmovl_s8(vget_low_s8 (v_lo));
        int16x8_t v_lo_h = vmovl_s8(vget_high_s8(v_lo));
        int16x8_t v_hi_l = vmovl_s8(vget_low_s8 (v_hi));
        int16x8_t v_hi_h = vmovl_s8(vget_high_s8(v_hi));
        int16x8_t v_lo2_l = vmovl_s8(vget_low_s8 (v_lo2));
        int16x8_t v_lo2_h = vmovl_s8(vget_high_s8(v_lo2));
        int16x8_t v_hi2_l = vmovl_s8(vget_low_s8 (v_hi2));
        int16x8_t v_hi2_h = vmovl_s8(vget_high_s8(v_hi2));

        a0 = vmlaq_s32(a0, sj, vmovl_s16(vget_low_s16 (v_lo_l)));
        a1 = vmlaq_s32(a1, sj, vmovl_s16(vget_high_s16(v_lo_l)));
        a2 = vmlaq_s32(a2, sj, vmovl_s16(vget_low_s16 (v_lo_h)));
        a3 = vmlaq_s32(a3, sj, vmovl_s16(vget_high_s16(v_lo_h)));
        a4 = vmlaq_s32(a4, sj, vmovl_s16(vget_low_s16 (v_hi_l)));
        a5 = vmlaq_s32(a5, sj, vmovl_s16(vget_high_s16(v_hi_l)));
        a6 = vmlaq_s32(a6, sj, vmovl_s16(vget_low_s16 (v_hi_h)));
        a7 = vmlaq_s32(a7, sj, vmovl_s16(vget_high_s16(v_hi_h)));
        a8 = vmlaq_s32(a8, sj, vmovl_s16(vget_low_s16 (v_lo2_l)));
        a9 = vmlaq_s32(a9, sj, vmovl_s16(vget_high_s16(v_lo2_l)));
        aA = vmlaq_s32(aA, sj, vmovl_s16(vget_low_s16 (v_lo2_h)));
        aB = vmlaq_s32(aB, sj, vmovl_s16(vget_high_s16(v_lo2_h)));
        aC = vmlaq_s32(aC, sj, vmovl_s16(vget_low_s16 (v_hi2_l)));
        aD = vmlaq_s32(aD, sj, vmovl_s16(vget_high_s16(v_hi2_l)));
        aE = vmlaq_s32(aE, sj, vmovl_s16(vget_low_s16 (v_hi2_h)));
        aF = vmlaq_s32(aF, sj, vmovl_s16(vget_high_s16(v_hi2_h)));
    }

    int32x4_t half = vdupq_n_s32(SDV_Q15_HALF);

    int32x4_t r0 = vshrq_n_s32(vaddq_s32(a0, half), SDV_Q15_SHIFT);
    int32x4_t r1 = vshrq_n_s32(vaddq_s32(a1, half), SDV_Q15_SHIFT);
    int32x4_t r2 = vshrq_n_s32(vaddq_s32(a2, half), SDV_Q15_SHIFT);
    int32x4_t r3 = vshrq_n_s32(vaddq_s32(a3, half), SDV_Q15_SHIFT);
    int32x4_t r4 = vshrq_n_s32(vaddq_s32(a4, half), SDV_Q15_SHIFT);
    int32x4_t r5 = vshrq_n_s32(vaddq_s32(a5, half), SDV_Q15_SHIFT);
    int32x4_t r6 = vshrq_n_s32(vaddq_s32(a6, half), SDV_Q15_SHIFT);
    int32x4_t r7 = vshrq_n_s32(vaddq_s32(a7, half), SDV_Q15_SHIFT);
    int32x4_t r8 = vshrq_n_s32(vaddq_s32(a8, half), SDV_Q15_SHIFT);
    int32x4_t r9 = vshrq_n_s32(vaddq_s32(a9, half), SDV_Q15_SHIFT);
    int32x4_t rA = vshrq_n_s32(vaddq_s32(aA, half), SDV_Q15_SHIFT);
    int32x4_t rB = vshrq_n_s32(vaddq_s32(aB, half), SDV_Q15_SHIFT);
    int32x4_t rC = vshrq_n_s32(vaddq_s32(aC, half), SDV_Q15_SHIFT);
    int32x4_t rD = vshrq_n_s32(vaddq_s32(aD, half), SDV_Q15_SHIFT);
    int32x4_t rE = vshrq_n_s32(vaddq_s32(aE, half), SDV_Q15_SHIFT);
    int32x4_t rF = vshrq_n_s32(vaddq_s32(aF, half), SDV_Q15_SHIFT);

    int16x8_t s01 = vqmovn_high_s32(vqmovn_s32(r0), r1);
    int16x8_t s23 = vqmovn_high_s32(vqmovn_s32(r2), r3);
    int16x8_t s45 = vqmovn_high_s32(vqmovn_s32(r4), r5);
    int16x8_t s67 = vqmovn_high_s32(vqmovn_s32(r6), r7);
    int16x8_t s89 = vqmovn_high_s32(vqmovn_s32(r8), r9);
    int16x8_t sAB = vqmovn_high_s32(vqmovn_s32(rA), rB);
    int16x8_t sCD = vqmovn_high_s32(vqmovn_s32(rC), rD);
    int16x8_t sEF = vqmovn_high_s32(vqmovn_s32(rE), rF);

    int8x16_t b0 = vqmovn_high_s16(vqmovn_s16(s01), s23);
    int8x16_t b1 = vqmovn_high_s16(vqmovn_s16(s45), s67);
    int8x16_t b2 = vqmovn_high_s16(vqmovn_s16(s89), sAB);
    int8x16_t b3 = vqmovn_high_s16(vqmovn_s16(sCD), sEF);

    vst1q_s8(out +  0, b0);
    vst1q_s8(out + 16, b1);
    vst1q_s8(out + 32, b2);
    vst1q_s8(out + 48, b3);
}

// ------------------------------------------------------------------------------------
// neon_exp_ps — 4-wide poly5 e^x. range-reduce x = n*ln2 + r, evaluate poly5
// of e^r in [-ln2/2, ln2/2], reconstruct e^x = poly5(r) * 2^n. 2^n built from
// the IEEE754 exponent field directly (NEON has no scalef). matches the
// avx-512 sequence (1/120 → 1/24 → 1/6 → 1/2 → 1 → 1) under fma reassoc.
// ------------------------------------------------------------------------------------

static inline float32x4_t neon_exp_ps(float32x4_t x) {
    const float32x4_t inv_ln2 = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t ln2     = vdupq_n_f32(0.6931471805599453f);

    float32x4_t n = vrndnq_f32(vmulq_f32(x, inv_ln2));
    float32x4_t r = vfmsq_f32(x, n, ln2);

    float32x4_t e = vdupq_n_f32(1.0f / 120.0f);
    e = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), e, r);
    e = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f),  e, r);
    e = vfmaq_f32(vdupq_n_f32(0.5f),         e, r);
    e = vfmaq_f32(vdupq_n_f32(1.0f),         e, r);
    e = vfmaq_f32(vdupq_n_f32(1.0f),         e, r);

    int32x4_t n_int    = vcvtq_s32_f32(n);
    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n_int, vdupq_n_s32(127)), 23);
    float32x4_t two_n  = vreinterpretq_f32_s32(exp_bits);

    return vmulq_f32(e, two_n);
}

// ------------------------------------------------------------------------------------
// softmax_rows_neon — vectorized row-softmax. matches the avx-512 algorithm:
// max-subtract, clamp to -87, poly5 exp, sum, qinv = 32768/sum, requantize to
// int16. tail (cols % 4) uses libm expf to keep within 1 LSB of scalar.
// ------------------------------------------------------------------------------------

void softmax_rows_neon(float* x, int16_t* out_q, int32_t rows, int32_t cols) {
    const float32x4_t clamp = vdupq_n_f32(SOFTMAX_EXP_CLAMP);

    for (int32_t r = 0; r < rows; r++) {
        float*   row    = x     + (size_t)r * cols;
        int16_t* row_q  = out_q + (size_t)r * cols;

        // pass 1: vmax
        float32x4_t vmax = vdupq_n_f32(-1e38f);
        int32_t c = 0;
        for (; c + 4 <= cols; c += 4) {
            vmax = vmaxq_f32(vmax, vld1q_f32(row + c));
        }
        float vmax_s = vmaxvq_f32(vmax);
        for (; c < cols; c++) if (row[c] > vmax_s) vmax_s = row[c];
        float32x4_t vmaxb = vdupq_n_f32(vmax_s);

        // pass 2: e^(x - max), accumulate sum, store back in row
        float32x4_t vsum = vdupq_n_f32(0.0f);
        c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t v = vmaxq_f32(vsubq_f32(vld1q_f32(row + c), vmaxb), clamp);
            v = neon_exp_ps(v);
            vsum = vaddq_f32(vsum, v);
            vst1q_f32(row + c, v);
        }
        float sum_s = vaddvq_f32(vsum);
        for (; c < cols; c++) {
            float d = row[c] - vmax_s;
            if (d < SOFTMAX_EXP_CLAMP) d = SOFTMAX_EXP_CLAMP;
            float e = expf(d);
            row[c] = e;
            sum_s += e;
        }

        // pass 3: quantize to int16 q15
        float qinv = SOFTMAX_Q15_SCALE / (sum_s > 0.0f ? sum_s : 1.0f);
        float32x4_t vqinv = vdupq_n_f32(qinv);
        c = 0;
        for (; c + 4 <= cols; c += 4) {
            float32x4_t fq = vmulq_f32(vld1q_f32(row + c), vqinv);
            int32x4_t qi   = vcvtnq_s32_f32(fq);
            int16x4_t q16  = vqmovn_s32(qi);
            vst1_s16(row_q + c, q16);
        }
        for (; c < cols; c++) {
            float fq = row[c] * qinv;
            int32_t q = (int32_t)lrintf(fq);
            if (q > SOFTMAX_Q15_MAX) q = SOFTMAX_Q15_MAX;
            if (q < SOFTMAX_Q15_MIN) q = SOFTMAX_Q15_MIN;
            row_q[c] = (int16_t)q;
        }
    }
}

// ------------------------------------------------------------------------------------
// layernorm_i16_to_i8_neon — RMSNorm: x * w * 0.5 / sqrt(mean(x^2) + eps),
// saturated to int8. matches PyTorch RMSNorm (no mean subtraction).
// pass 1 (sumsq) and pass 2 (weight×scale) both vectorized; tail handles cols
// not divisible by 16 with the scalar oracle path.
// ------------------------------------------------------------------------------------

void layernorm_i16_to_i8_neon(const int16_t* x, int8_t* out, const int8_t* w,
                              int32_t rows, int32_t cols) {
    for (int32_t r = 0; r < rows; r++) {
        const int16_t* row_in  = x   + (size_t)r * cols;
        int8_t*        row_out = out + (size_t)r * cols;

        float32x4_t vsumsq = vdupq_n_f32(0.0f);
        int32_t cs = 0;
        for (; cs + 8 <= cols; cs += 8) {
            int16x8_t v16 = vld1q_s16(row_in + cs);
            float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (v16)));
            float32x4_t hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16)));
            vsumsq = vfmaq_f32(vsumsq, lo, lo);
            vsumsq = vfmaq_f32(vsumsq, hi, hi);
        }
        float sumsq_s = vaddvq_f32(vsumsq);
        for (; cs < cols; cs++) {
            float v = (float)row_in[cs];
            sumsq_s += v * v;
        }
        float ms = sumsq_s / (float)cols;
        float scale = LN_HALF_PRESCALE / sqrtf(ms + LN_EPS);
        float32x4_t vscale = vdupq_n_f32(scale);

        int32_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            int16x8_t v16_lo = vld1q_s16(row_in + c);
            int16x8_t v16_hi = vld1q_s16(row_in + c + 8);
            int8x16_t w8     = vld1q_s8 (w + c);
            int16x8_t w16_lo = vmovl_s8(vget_low_s8 (w8));
            int16x8_t w16_hi = vmovl_s8(vget_high_s8(w8));

            int32x4_t p0 = vmull_s16     (vget_low_s16 (v16_lo), vget_low_s16 (w16_lo));
            int32x4_t p1 = vmull_high_s16(v16_lo, w16_lo);
            int32x4_t p2 = vmull_s16     (vget_low_s16 (v16_hi), vget_low_s16 (w16_hi));
            int32x4_t p3 = vmull_high_s16(v16_hi, w16_hi);

            float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(p0), vscale);
            float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(p1), vscale);
            float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(p2), vscale);
            float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(p3), vscale);

            int32x4_t i0 = vcvtnq_s32_f32(f0);
            int32x4_t i1 = vcvtnq_s32_f32(f1);
            int32x4_t i2 = vcvtnq_s32_f32(f2);
            int32x4_t i3 = vcvtnq_s32_f32(f3);

            int16x8_t s01 = vqmovn_high_s32(vqmovn_s32(i0), i1);
            int16x8_t s23 = vqmovn_high_s32(vqmovn_s32(i2), i3);
            int8x16_t b   = vqmovn_high_s16(vqmovn_s16(s01), s23);
            vst1q_s8(row_out + c, b);
        }
        for (; c < cols; c++) {
            float v = (float)row_in[c] * (float)w[c] * scale;
            int32_t q = (int32_t)lrintf(v);
            if (q > LN_INT8_MAX) q = LN_INT8_MAX;
            if (q < LN_INT8_MIN) q = LN_INT8_MIN;
            row_out[c] = (int8_t)q;
        }
    }
}
