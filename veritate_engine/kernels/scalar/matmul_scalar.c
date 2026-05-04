// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - reference INT8 matmul. portable C. oracle for SIMD kernels.
// ------------------------------------------------------------------------------------

#include "../../src/veritate.h"

void matmul_int8_scalar(
    const int8_t* a,
    const int8_t* b,
    int32_t*      c,
    int32_t       m,
    int32_t       n,
    int32_t       k
) {
    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int32_t p = 0; p < k; p++) {
                acc += (int32_t)a[i * k + p] * (int32_t)b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}
