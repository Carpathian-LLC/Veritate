// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - fp32 + fp16 matvec for the v13 hybrid path. 4 accumulator vectors mirror
//   the scalar references' 16 partial sums exactly (mul then add, no
//   contraction): fold (A0+A2)+(A1+A3), scalar tail into lane 0, reduce
//   (t0+t1)+(t2+t3). fp16 weights convert by vcvt (exact, equals
//   hybrid_f16_to_f32) then run the same fp32 math. compile this TU and
//   src/hybrid.c with -ffp-contract=off to keep the pairing exact.
// ------------------------------------------------------------------------------------

#include "../../src/hybrid.h"

#include <arm_neon.h>

void hybrid_matvec_f32_neon(const void* w, const float* x, float* out,
                            int32_t n, int32_t k) {
    const float* wf = (const float*)w;
    for (int32_t j = 0; j < n; j++) {
        const float* row = wf + (size_t)j * k;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        int32_t i = 0;
        for (; i + 16 <= k; i += 16) {
            a0 = vaddq_f32(a0, vmulq_f32(vld1q_f32(row + i),      vld1q_f32(x + i)));
            a1 = vaddq_f32(a1, vmulq_f32(vld1q_f32(row + i + 4),  vld1q_f32(x + i + 4)));
            a2 = vaddq_f32(a2, vmulq_f32(vld1q_f32(row + i + 8),  vld1q_f32(x + i + 8)));
            a3 = vaddq_f32(a3, vmulq_f32(vld1q_f32(row + i + 12), vld1q_f32(x + i + 12)));
        }
        float32x4_t t = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
        float t0 = vgetq_lane_f32(t, 0);
        float t1 = vgetq_lane_f32(t, 1);
        float t2 = vgetq_lane_f32(t, 2);
        float t3 = vgetq_lane_f32(t, 3);
        for (; i < k; i++) t0 += row[i] * x[i];
        out[j] = (t0 + t1) + (t2 + t3);
    }
}

void hybrid_matvec_f16_neon(const void* w, const float* x, float* out,
                            int32_t n, int32_t k) {
    const __fp16* wh = (const __fp16*)w;
    for (int32_t j = 0; j < n; j++) {
        const __fp16* row = wh + (size_t)j * k;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        int32_t i = 0;
        for (; i + 16 <= k; i += 16) {
            float16x8_t h0 = vld1q_f16(row + i);
            float16x8_t h1 = vld1q_f16(row + i + 8);
            a0 = vaddq_f32(a0, vmulq_f32(vcvt_f32_f16(vget_low_f16(h0)),
                                         vld1q_f32(x + i)));
            a1 = vaddq_f32(a1, vmulq_f32(vcvt_f32_f16(vget_high_f16(h0)),
                                         vld1q_f32(x + i + 4)));
            a2 = vaddq_f32(a2, vmulq_f32(vcvt_f32_f16(vget_low_f16(h1)),
                                         vld1q_f32(x + i + 8)));
            a3 = vaddq_f32(a3, vmulq_f32(vcvt_f32_f16(vget_high_f16(h1)),
                                         vld1q_f32(x + i + 12)));
        }
        float32x4_t t = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
        float t0 = vgetq_lane_f32(t, 0);
        float t1 = vgetq_lane_f32(t, 1);
        float t2 = vgetq_lane_f32(t, 2);
        float t3 = vgetq_lane_f32(t, 3);
        for (; i < k; i++) t0 += hybrid_f16_to_f32(((const uint16_t*)row)[i]) * x[i];
        out[j] = (t0 + t1) + (t2 + t3);
    }
}

// int8 x int8 -> int32 via sdot. activations quantize through the shared
// scalar hybrid_quant_act, and int32 accumulation is exact, so output is
// bitwise-identical to hybrid_matvec_i8_scalar by construction.
void hybrid_matvec_i8_sdot(const void* w, const float* x, float* out,
                           int32_t n, int32_t k) {
    const hybrid_w_i8_t* wi = (const hybrid_w_i8_t*)w;
    int8_t qx[V_MAX_FFN];
    const float a_scale = hybrid_quant_act(x, k, qx);
    if (a_scale == 0.0f) {
        for (int32_t j = 0; j < n; j++) out[j] = 0.0f;
        return;
    }
    for (int32_t j = 0; j < n; j++) {
        const int8_t* row = wi->q + (size_t)j * k;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
        int32_t i = 0;
        for (; i + 64 <= k; i += 64) {
            a0 = vdotq_s32(a0, vld1q_s8(row + i),      vld1q_s8(qx + i));
            a1 = vdotq_s32(a1, vld1q_s8(row + i + 16), vld1q_s8(qx + i + 16));
            a2 = vdotq_s32(a2, vld1q_s8(row + i + 32), vld1q_s8(qx + i + 32));
            a3 = vdotq_s32(a3, vld1q_s8(row + i + 48), vld1q_s8(qx + i + 48));
        }
        for (; i + 16 <= k; i += 16) {
            a0 = vdotq_s32(a0, vld1q_s8(row + i), vld1q_s8(qx + i));
        }
        int32_t acc = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
        for (; i < k; i++) acc += (int32_t)row[i] * (int32_t)qx[i];
        out[j] = (float)acc * (wi->scale[j] * a_scale);
    }
}
