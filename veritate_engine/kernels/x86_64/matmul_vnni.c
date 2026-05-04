// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - INT8 matmul, AVX-512 VNNI. one vpdpbusd does 64 INT8 multiply-accumulates.
// - bias trick: shift a by +128 to make it uint8, subtract 128*sum(b) per column.
// - 4x4 register tile: 4 a_rows x 4 b_cols x 16 accumulators per inner pass.
// - mt variant: persistent thread pool, pre-transposed b shared across workers.
// ------------------------------------------------------------------------------------

#include "../../src/veritate.h"
#include "../../src/portability.h"

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ------------------------------------------------------------------------------------
// 1x1 dot product — used for the m and n tail
// ------------------------------------------------------------------------------------

static inline int32_t vnni_dot_1x1(const int8_t* a_row, const int8_t* b_col, int32_t k) {
    const __m512i bias_vec = _mm512_set1_epi8((char)0x80);
    __m512i acc = _mm512_setzero_si512();
    for (int32_t p = 0; p < k; p += 64) {
        __m512i av = _mm512_loadu_si512((const __m512i*)(a_row + p));
        __m512i bv = _mm512_loadu_si512((const __m512i*)(b_col + p));
        __m512i au = _mm512_add_epi8(av, bias_vec);
        acc = _mm512_dpbusd_epi32(acc, au, bv);
    }
    return _mm512_reduce_add_epi32(acc);
}

// ------------------------------------------------------------------------------------
// 4x4 register tile — produces a 4x4 block of c
// ------------------------------------------------------------------------------------

static inline void vnni_4x4(
    const int8_t*  a0,
    const int8_t*  a1,
    const int8_t*  a2,
    const int8_t*  a3,
    const int8_t*  b0,
    const int8_t*  b1,
    const int8_t*  b2,
    const int8_t*  b3,
    int32_t        k,
    int32_t        bias0,
    int32_t        bias1,
    int32_t        bias2,
    int32_t        bias3,
    int32_t*       c00, int32_t* c01, int32_t* c02, int32_t* c03,
    int32_t*       c10, int32_t* c11, int32_t* c12, int32_t* c13,
    int32_t*       c20, int32_t* c21, int32_t* c22, int32_t* c23,
    int32_t*       c30, int32_t* c31, int32_t* c32, int32_t* c33
) {
    const __m512i bias_vec = _mm512_set1_epi8((char)0x80);

    __m512i a00 = _mm512_setzero_si512(), a01 = _mm512_setzero_si512();
    __m512i a02 = _mm512_setzero_si512(), a03 = _mm512_setzero_si512();
    __m512i a10 = _mm512_setzero_si512(), a11 = _mm512_setzero_si512();
    __m512i a12 = _mm512_setzero_si512(), a13 = _mm512_setzero_si512();
    __m512i a20 = _mm512_setzero_si512(), a21 = _mm512_setzero_si512();
    __m512i a22 = _mm512_setzero_si512(), a23 = _mm512_setzero_si512();
    __m512i a30 = _mm512_setzero_si512(), a31 = _mm512_setzero_si512();
    __m512i a32 = _mm512_setzero_si512(), a33 = _mm512_setzero_si512();

    int32_t p = 0;
    for (; p + 64 <= k; p += 64) {
        __m512i av0 = _mm512_add_epi8(_mm512_loadu_si512((const __m512i*)(a0 + p)), bias_vec);
        __m512i av1 = _mm512_add_epi8(_mm512_loadu_si512((const __m512i*)(a1 + p)), bias_vec);
        __m512i av2 = _mm512_add_epi8(_mm512_loadu_si512((const __m512i*)(a2 + p)), bias_vec);
        __m512i av3 = _mm512_add_epi8(_mm512_loadu_si512((const __m512i*)(a3 + p)), bias_vec);

        __m512i bv0 = _mm512_loadu_si512((const __m512i*)(b0 + p));
        __m512i bv1 = _mm512_loadu_si512((const __m512i*)(b1 + p));
        __m512i bv2 = _mm512_loadu_si512((const __m512i*)(b2 + p));
        __m512i bv3 = _mm512_loadu_si512((const __m512i*)(b3 + p));

        a00 = _mm512_dpbusd_epi32(a00, av0, bv0);
        a01 = _mm512_dpbusd_epi32(a01, av0, bv1);
        a02 = _mm512_dpbusd_epi32(a02, av0, bv2);
        a03 = _mm512_dpbusd_epi32(a03, av0, bv3);

        a10 = _mm512_dpbusd_epi32(a10, av1, bv0);
        a11 = _mm512_dpbusd_epi32(a11, av1, bv1);
        a12 = _mm512_dpbusd_epi32(a12, av1, bv2);
        a13 = _mm512_dpbusd_epi32(a13, av1, bv3);

        a20 = _mm512_dpbusd_epi32(a20, av2, bv0);
        a21 = _mm512_dpbusd_epi32(a21, av2, bv1);
        a22 = _mm512_dpbusd_epi32(a22, av2, bv2);
        a23 = _mm512_dpbusd_epi32(a23, av2, bv3);

        a30 = _mm512_dpbusd_epi32(a30, av3, bv0);
        a31 = _mm512_dpbusd_epi32(a31, av3, bv1);
        a32 = _mm512_dpbusd_epi32(a32, av3, bv2);
        a33 = _mm512_dpbusd_epi32(a33, av3, bv3);
    }

    int32_t s00 = _mm512_reduce_add_epi32(a00) - bias0;
    int32_t s01 = _mm512_reduce_add_epi32(a01) - bias1;
    int32_t s02 = _mm512_reduce_add_epi32(a02) - bias2;
    int32_t s03 = _mm512_reduce_add_epi32(a03) - bias3;

    int32_t s10 = _mm512_reduce_add_epi32(a10) - bias0;
    int32_t s11 = _mm512_reduce_add_epi32(a11) - bias1;
    int32_t s12 = _mm512_reduce_add_epi32(a12) - bias2;
    int32_t s13 = _mm512_reduce_add_epi32(a13) - bias3;

    int32_t s20 = _mm512_reduce_add_epi32(a20) - bias0;
    int32_t s21 = _mm512_reduce_add_epi32(a21) - bias1;
    int32_t s22 = _mm512_reduce_add_epi32(a22) - bias2;
    int32_t s23 = _mm512_reduce_add_epi32(a23) - bias3;

    int32_t s30 = _mm512_reduce_add_epi32(a30) - bias0;
    int32_t s31 = _mm512_reduce_add_epi32(a31) - bias1;
    int32_t s32 = _mm512_reduce_add_epi32(a32) - bias2;
    int32_t s33 = _mm512_reduce_add_epi32(a33) - bias3;

    for (; p < k; p++) {
        int32_t v0 = a0[p], v1 = a1[p], v2 = a2[p], v3 = a3[p];
        int32_t w0 = b0[p], w1 = b1[p], w2 = b2[p], w3 = b3[p];
        s00 += v0*w0; s01 += v0*w1; s02 += v0*w2; s03 += v0*w3;
        s10 += v1*w0; s11 += v1*w1; s12 += v1*w2; s13 += v1*w3;
        s20 += v2*w0; s21 += v2*w1; s22 += v2*w2; s23 += v2*w3;
        s30 += v3*w0; s31 += v3*w1; s32 += v3*w2; s33 += v3*w3;
    }

    *c00 = s00; *c01 = s01; *c02 = s02; *c03 = s03;
    *c10 = s10; *c11 = s11; *c12 = s12; *c13 = s13;
    *c20 = s20; *c21 = s21; *c22 = s22; *c23 = s23;
    *c30 = s30; *c31 = s31; *c32 = s32; *c33 = s33;
}

// ------------------------------------------------------------------------------------
// inner driver — fills [m_start, m_end) x [0, n) of c using the 4x4 tile
// ------------------------------------------------------------------------------------

static void vnni_block(
    const int8_t*  a,
    const int8_t*  bt,
    const int32_t* bias,
    int32_t*       c,
    int32_t        m_start,
    int32_t        m_end,
    int32_t        n,
    int32_t        k
) {
    int32_t i = m_start;
    for (; i + 4 <= m_end; i += 4) {
        const int8_t* a0 = a + (size_t)(i + 0) * k;
        const int8_t* a1 = a + (size_t)(i + 1) * k;
        const int8_t* a2 = a + (size_t)(i + 2) * k;
        const int8_t* a3 = a + (size_t)(i + 3) * k;

        int32_t j = 0;
        for (; j + 4 <= n; j += 4) {
            const int8_t* b0 = bt + (size_t)(j + 0) * k;
            const int8_t* b1 = bt + (size_t)(j + 1) * k;
            const int8_t* b2 = bt + (size_t)(j + 2) * k;
            const int8_t* b3 = bt + (size_t)(j + 3) * k;

            vnni_4x4(
                a0, a1, a2, a3,
                b0, b1, b2, b3,
                k,
                bias[j + 0], bias[j + 1], bias[j + 2], bias[j + 3],
                &c[(i+0)*n + j+0], &c[(i+0)*n + j+1], &c[(i+0)*n + j+2], &c[(i+0)*n + j+3],
                &c[(i+1)*n + j+0], &c[(i+1)*n + j+1], &c[(i+1)*n + j+2], &c[(i+1)*n + j+3],
                &c[(i+2)*n + j+0], &c[(i+2)*n + j+1], &c[(i+2)*n + j+2], &c[(i+2)*n + j+3],
                &c[(i+3)*n + j+0], &c[(i+3)*n + j+1], &c[(i+3)*n + j+2], &c[(i+3)*n + j+3]
            );
        }
        for (; j < n; j++) {
            const int8_t* b_col = bt + (size_t)j * k;
            c[(i+0)*n + j] = vnni_dot_1x1(a0, b_col, k) - bias[j];
            c[(i+1)*n + j] = vnni_dot_1x1(a1, b_col, k) - bias[j];
            c[(i+2)*n + j] = vnni_dot_1x1(a2, b_col, k) - bias[j];
            c[(i+3)*n + j] = vnni_dot_1x1(a3, b_col, k) - bias[j];
        }
    }
    for (; i < m_end; i++) {
        const int8_t* a_row = a + (size_t)i * k;
        for (int32_t j = 0; j < n; j++) {
            c[i * n + j] = vnni_dot_1x1(a_row, bt + (size_t)j * k, k) - bias[j];
        }
    }
}

// ------------------------------------------------------------------------------------
// single-threaded VNNI matmul
// ------------------------------------------------------------------------------------

void matmul_int8_vnni(
    const int8_t* a,
    const int8_t* b,
    int32_t*      c,
    int32_t       m,
    int32_t       n,
    int32_t       k
) {
    int8_t*  bt   = (int8_t*) veritate_aligned_alloc((size_t)k * n, 64);
    int32_t* bias = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), 64);

    for (int32_t j = 0; j < n; j++) {
        int32_t s = 0;
        int8_t* dst = bt + (size_t)j * k;
        for (int32_t p = 0; p < k; p++) {
            int8_t v = b[p * n + j];
            dst[p] = v;
            s += v;
        }
        bias[j] = 128 * s;
    }

    vnni_block(a, bt, bias, c, 0, m, n, k);

    veritate_aligned_free(bt);
    veritate_aligned_free(bias);
}

// ------------------------------------------------------------------------------------
// per-call worker arg. dispatch goes through src/threadpool.c shim.
// ------------------------------------------------------------------------------------

typedef struct {
    const int8_t*  a;
    const int8_t*  bt;
    const int32_t* bias;
    int32_t*       c;
    int32_t        m_start;
    int32_t        m_end;
    int32_t        n;
    int32_t        k;
} mm_arg_t;

static void vnni_worker(void* raw, int32_t worker_idx) {
    (void)worker_idx;
    const mm_arg_t* a = (const mm_arg_t*)raw;
    vnni_block(a->a, a->bt, a->bias, a->c, a->m_start, a->m_end, a->n, a->k);
}

// ------------------------------------------------------------------------------------
// pre-prepare b once. real inference loads weights once and reuses across forwards.
// ------------------------------------------------------------------------------------

void prep_b(const int8_t* b, int32_t n, int32_t k, prepped_b_t* out) {
    out->n             = n;
    out->k             = k;
    out->bt            = (int8_t*) veritate_aligned_alloc((size_t)k * n, 64);
    out->b_rowmaj      = NULL;
    out->bias          = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), 64);
    out->scale_per_col = NULL;

    int64_t sum_sq = 0;
    for (int32_t j = 0; j < n; j++) {
        int32_t s = 0;
        int8_t* dst = out->bt + (size_t)j * k;
        for (int32_t p = 0; p < k; p++) {
            int8_t v = b[p * n + j];
            dst[p] = v;
            s += v;
            sum_sq += (int64_t)v * v;
        }
        out->bias[j] = 128 * s;
    }

    // q24 requant scale from rms(b)
    double b_rms = sqrt((double)sum_sq / ((double)n * k));
    out->scale_q24 = (int32_t)(64.0 / (sqrt((double)k) * 32.0 * b_rms) * 16777216.0);
}

// prep_b plus a row-major copy of b retained for the sparse decode path.
void prep_b_keep_raw(const int8_t* b, int32_t n, int32_t k, prepped_b_t* out) {
    prep_b(b, n, k, out);
    out->b_rowmaj = (int8_t*)veritate_aligned_alloc((size_t)k * n, 64);
    memcpy(out->b_rowmaj, b, (size_t)k * n);
}

void free_prepped_b(prepped_b_t* p) {
    if (p->bt)            veritate_aligned_free(p->bt);
    if (p->b_rowmaj)      veritate_aligned_free(p->b_rowmaj);
    if (p->bias)          veritate_aligned_free(p->bias);
    if (p->scale_per_col) veritate_aligned_free(p->scale_per_col);
    p->bt = NULL; p->b_rowmaj = NULL; p->bias = NULL; p->scale_per_col = NULL;
}

// ------------------------------------------------------------------------------------
// per-inference matmul using pre-prepared b
// ------------------------------------------------------------------------------------

void matmul_int8_vnni_prep(
    const int8_t*     a,
    const prepped_b_t* p,
    int32_t*          c,
    int32_t           m
) {
    vnni_block(a, p->bt, p->bias, c, 0, m, p->n, p->k);
}

void matmul_int8_vnni_mt_prep(
    const int8_t*     a,
    const prepped_b_t* p,
    int32_t*          c,
    int32_t           m
) {
    int32_t threads = veritate_pool_size();
    if (threads > m) threads = m;
    if (threads < 1) threads = 1;

    int32_t rows_per = ((m + threads - 1) / threads + 3) & ~3;

    mm_arg_t args [VERITATE_MAX_THREADS];
    void*    argv [VERITATE_MAX_THREADS];
    for (int32_t t = 0; t < threads; t++) {
        args[t].a       = a;
        args[t].bt      = p->bt;
        args[t].bias    = p->bias;
        args[t].c       = c;
        args[t].m_start = t * rows_per;
        args[t].m_end   = (t + 1) * rows_per;
        if (args[t].m_end   > m) args[t].m_end   = m;
        if (args[t].m_start > m) args[t].m_start = m;
        args[t].n       = p->n;
        args[t].k       = p->k;
        argv[t]         = &args[t];
    }

    veritate_pool_run(vnni_worker, argv, threads);
}

// ------------------------------------------------------------------------------------
// multi-threaded VNNI — convenience wrapper, preps b inside (slower per call)
// ------------------------------------------------------------------------------------

void matmul_int8_vnni_mt(
    const int8_t* a,
    const int8_t* b,
    int32_t*      c,
    int32_t       m,
    int32_t       n,
    int32_t       k
) {
    prepped_b_t pb;
    prep_b(b, n, k, &pb);
    matmul_int8_vnni_mt_prep(a, &pb, c, m);
    free_prepped_b(&pb);
}
