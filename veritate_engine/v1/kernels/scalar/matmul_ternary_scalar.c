// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - reference ternary matmul. portable C. oracle for SIMD kernels.
// - BitNet b1.58: trits in {-1, 0, +1}, per-tensor mean-abs scale (gamma).
//   trits packed 5-per-byte in base-3. spec: documentation/kernels/ternary.md.
// - matches PyTorch fake_quant_weight_ternary in veritate_core/qat.py.
// veritate_engine/kernels/scalar/matmul_ternary_scalar.c
// ------------------------------------------------------------------------------------
// Imports

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/veritate.h"

// ------------------------------------------------------------------------------------
// Constants

#define TERNARY_TRITS_PER_BYTE 5
#define TERNARY_BASE           3

// ------------------------------------------------------------------------------------
// Functions

void ternary_pack_row(const int8_t* trits, int32_t k, uint8_t* out_bytes) {
    int32_t n_bytes = (k + TERNARY_TRITS_PER_BYTE - 1) / TERNARY_TRITS_PER_BYTE;
    for (int32_t b = 0; b < n_bytes; b++) {
        uint32_t acc = 0;
        for (int32_t s = 0; s < TERNARY_TRITS_PER_BYTE; s++) {
            int32_t i = b * TERNARY_TRITS_PER_BYTE + s;
            int32_t t = (i < k) ? (int32_t)trits[i] : 0;
            int32_t shifted = t + 1;
            acc = acc * TERNARY_BASE + (uint32_t)shifted;
        }
        out_bytes[b] = (uint8_t)acc;
    }
}

void ternary_unpack_row(const uint8_t* bytes, int32_t k, int8_t* out_trits) {
    int32_t n_bytes = (k + TERNARY_TRITS_PER_BYTE - 1) / TERNARY_TRITS_PER_BYTE;
    for (int32_t b = 0; b < n_bytes; b++) {
        uint32_t v = bytes[b];
        int8_t t[TERNARY_TRITS_PER_BYTE];
        for (int32_t s = TERNARY_TRITS_PER_BYTE - 1; s >= 0; s--) {
            int32_t shifted = (int32_t)(v % TERNARY_BASE);
            v = v / TERNARY_BASE;
            t[s] = (int8_t)(shifted - 1);
        }
        for (int32_t s = 0; s < TERNARY_TRITS_PER_BYTE; s++) {
            int32_t i = b * TERNARY_TRITS_PER_BYTE + s;
            if (i < k) {
                out_trits[i] = t[s];
            }
        }
    }
}

void prep_b_ternary(const int8_t* b_trits, int32_t n, int32_t k,
                    float gamma, prepped_b_ternary_t* out) {
    int32_t row_bytes = (k + TERNARY_TRITS_PER_BYTE - 1) / TERNARY_TRITS_PER_BYTE;
    out->n         = n;
    out->k         = k;
    out->gamma     = gamma;
    out->bt_packed = (uint8_t*)malloc((size_t)n * (size_t)row_bytes);
    out->row_q24   = (int32_t*)malloc((size_t)n * sizeof(int32_t));

    int8_t* row_buf = (int8_t*)malloc((size_t)k * sizeof(int8_t));
    for (int32_t j = 0; j < n; j++) {
        for (int32_t p = 0; p < k; p++) {
            row_buf[p] = b_trits[(size_t)p * (size_t)n + (size_t)j];
        }
        ternary_pack_row(row_buf, k, out->bt_packed + (size_t)j * (size_t)row_bytes);
        out->row_q24[j] = 0;
    }
    free(row_buf);
}

void free_prepped_b_ternary(prepped_b_ternary_t* p) {
    if (p == NULL) return;
    if (p->bt_packed != NULL) { free(p->bt_packed); p->bt_packed = NULL; }
    if (p->row_q24   != NULL) { free(p->row_q24);   p->row_q24   = NULL; }
    p->n = 0;
    p->k = 0;
}

void matmul_ternary_scalar_prep(const int8_t* a, const prepped_b_ternary_t* p,
                                int32_t* c, int32_t m) {
    int32_t k         = p->k;
    int32_t n         = p->n;
    int32_t row_bytes = (k + TERNARY_TRITS_PER_BYTE - 1) / TERNARY_TRITS_PER_BYTE;

    int8_t* trits_buf = (int8_t*)malloc((size_t)k * sizeof(int8_t));

    for (int32_t i = 0; i < m; i++) {
        const int8_t* a_row = a + (size_t)i * (size_t)k;
        int32_t*      c_row = c + (size_t)i * (size_t)n;
        for (int32_t j = 0; j < n; j++) {
            const uint8_t* packed = p->bt_packed + (size_t)j * (size_t)row_bytes;
            ternary_unpack_row(packed, k, trits_buf);
            int32_t acc = 0;
            for (int32_t pp = 0; pp < k; pp++) {
                acc += (int32_t)a_row[pp] * (int32_t)trits_buf[pp];
            }
            c_row[j] = acc;
        }
    }

    free(trits_buf);
}
