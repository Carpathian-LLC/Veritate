// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - portable scalar hadamard. block-diagonal sylvester rotation, block size 64
//   = V_HEAD_DIM. fast walsh-hadamard butterflies, normalized 1/sqrt(64) = 1/8.
// - dst may equal src. cols must be a multiple of HADAMARD_BLOCK.
// - linked when no SIMD hadamard kernel ships for the target arch.
// veritate_engine/kernels/scalar/hadamard_scalar.c
// ------------------------------------------------------------------------------------
// Imports:

#include "../../src/veritate.h"

#include <stdint.h>

// ------------------------------------------------------------------------------------
// Constants

#define HADAMARD_BLOCK     64
#define HADAMARD_DIVISOR    8
#define HADAMARD_ROUND_BIAS 4
#define HADAMARD_INT8_MAX  127
#define HADAMARD_INT8_MIN (-128)

// ------------------------------------------------------------------------------------
// Functions

static void hadamard_block(const int8_t* sb, int8_t* db) {
    int16_t v[HADAMARD_BLOCK];
    for (int32_t i = 0; i < HADAMARD_BLOCK; i++) v[i] = (int16_t)sb[i];
    for (int32_t sz = 1; sz < HADAMARD_BLOCK; sz <<= 1) {
        for (int32_t base = 0; base < HADAMARD_BLOCK; base += sz * 2) {
            for (int32_t j = 0; j < sz; j++) {
                int16_t a = v[base + j];
                int16_t b = v[base + j + sz];
                v[base + j     ] = (int16_t)(a + b);
                v[base + j + sz] = (int16_t)(a - b);
            }
        }
    }
    for (int32_t i = 0; i < HADAMARD_BLOCK; i++) {
        int32_t bias = v[i] >= 0 ? HADAMARD_ROUND_BIAS : -HADAMARD_ROUND_BIAS;
        int32_t r = (v[i] + bias) / HADAMARD_DIVISOR;
        if (r > HADAMARD_INT8_MAX) r = HADAMARD_INT8_MAX;
        if (r < HADAMARD_INT8_MIN) r = HADAMARD_INT8_MIN;
        db[i] = (int8_t)r;
    }
}

void hadamard_apply_int8(const int8_t* src, int8_t* dst, int32_t cols) {
    const int32_t blocks = cols / HADAMARD_BLOCK;
    for (int32_t b = 0; b < blocks; b++) {
        hadamard_block(src + (size_t)b * HADAMARD_BLOCK,
                       dst + (size_t)b * HADAMARD_BLOCK);
    }
}
