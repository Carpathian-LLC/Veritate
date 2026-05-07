// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - portable scalar reference for the three transformer hot-path kernels:
//     score_dot_v_scalar       — score-weighted V row sum to int8 row.
//     softmax_rows_scalar      — fp32 softmax + int16 quantize at 2^15.
//     layernorm_i16_to_i8_scalar — int16 residual to int8 layernormed row.
// - rule-23 oracle: every SIMD backend matches scalar bit-for-bit on int8 out
//   to within 1 LSB (per-platform tolerance defined in platforms.md).
// - linked into every build. dispatch.c initializes function pointers to these
//   so model.c is callable before dispatch_init runs.
// veritate_engine/kernels/scalar/transformer_scalar.c
// ------------------------------------------------------------------------------------
// Imports:

#include "../../src/veritate.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

// ------------------------------------------------------------------------------------
// Constants

// score-weighted V sum: scores are int16 q15 (scale 32768). round-half-up by
// adding 0.5 LSB before the >>15 shift.
#define SDV_Q15_HALF       16384
#define SDV_Q15_SHIFT      15
#define SDV_OUT_BYTES      64
#define SDV_INT8_MAX       127
#define SDV_INT8_MIN      (-128)

// softmax row quantizer scale: 2^15. matches the avx-512 path.
#define SOFTMAX_Q15_SCALE  32768.0f
#define SOFTMAX_Q15_MAX    32767
#define SOFTMAX_Q15_MIN   (-32768)
#define SOFTMAX_EXP_CLAMP  (-87.0f)

// layernorm: variance epsilon and the 0.5 prescale to match the avx half_inv form.
#define LN_EPS             1e-5f
#define LN_HALF_PRESCALE   0.5f
#define LN_INT8_MAX        127
#define LN_INT8_MIN       (-128)

// ------------------------------------------------------------------------------------
// Functions

static inline int8_t sat_i8(int32_t v) {
    if (v > SDV_INT8_MAX) return SDV_INT8_MAX;
    if (v < SDV_INT8_MIN) return SDV_INT8_MIN;
    return (int8_t)v;
}

void score_dot_v_scalar(const int16_t* scores, const int8_t* v_base,
                        int32_t v_stride, int32_t n_j, int8_t* out) {
    int32_t acc[SDV_OUT_BYTES];
    for (int32_t c = 0; c < SDV_OUT_BYTES; c++) acc[c] = 0;
    for (int32_t j = 0; j < n_j; j++) {
        int32_t s = (int32_t)scores[j];
        const int8_t* vp = v_base + (size_t)j * v_stride;
        for (int32_t c = 0; c < SDV_OUT_BYTES; c++) {
            acc[c] += s * (int32_t)vp[c];
        }
    }
    for (int32_t c = 0; c < SDV_OUT_BYTES; c++) {
        int32_t r = (acc[c] + SDV_Q15_HALF) >> SDV_Q15_SHIFT;
        out[c] = sat_i8(r);
    }
}

void softmax_rows_scalar(float* x, int16_t* out_q, int32_t rows, int32_t cols) {
    for (int32_t r = 0; r < rows; r++) {
        float*   row    = x     + (size_t)r * cols;
        int16_t* row_q  = out_q + (size_t)r * cols;

        float vmax = row[0];
        for (int32_t c = 1; c < cols; c++) if (row[c] > vmax) vmax = row[c];

        double sum = 0.0;
        for (int32_t c = 0; c < cols; c++) {
            float d = row[c] - vmax;
            if (d < SOFTMAX_EXP_CLAMP) d = SOFTMAX_EXP_CLAMP;
            float e = expf(d);
            row[c] = e;
            sum += (double)e;
        }
        float qinv = (float)(SOFTMAX_Q15_SCALE / (sum > 0.0 ? sum : 1.0));
        for (int32_t c = 0; c < cols; c++) {
            float fq = row[c] * qinv;
            int32_t q = (int32_t)lrintf(fq);
            if (q > SOFTMAX_Q15_MAX) q = SOFTMAX_Q15_MAX;
            if (q < SOFTMAX_Q15_MIN) q = SOFTMAX_Q15_MIN;
            row_q[c] = (int16_t)q;
        }
    }
}

void layernorm_i16_to_i8_scalar(const int16_t* x, int8_t* out, const int8_t* w,
                                int32_t rows, int32_t cols) {
    for (int32_t r = 0; r < rows; r++) {
        const int16_t* row_in  = x   + (size_t)r * cols;
        int8_t*        row_out = out + (size_t)r * cols;

        double sumsq = 0.0;
        for (int32_t c = 0; c < cols; c++) {
            double v = (double)row_in[c];
            sumsq += v * v;
        }
        float ms = (float)(sumsq / (double)cols);
        float half_inv = LN_HALF_PRESCALE / sqrtf(ms + LN_EPS);

        for (int32_t c = 0; c < cols; c++) {
            float v = (float)row_in[c] * (float)w[c] * half_inv;
            int32_t q = (int32_t)lrintf(v);
            if (q > LN_INT8_MAX) q = LN_INT8_MAX;
            if (q < LN_INT8_MIN) q = LN_INT8_MIN;
            row_out[c] = (int8_t)q;
        }
    }
}
