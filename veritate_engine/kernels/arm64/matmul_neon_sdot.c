// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - INT8 matmul, NEON + SDOT (FEAT_DotProd). signed-int8 dot via vdotq_s32.
//   one sdot does 16 int8 multiply-accumulates into 4x int32; we run four
//   sdots per inner iteration for 64 macs / cycle / lane.
// - shape and prepped_b layout match the x86 vnni path bit-for-bit so
//   model_t weights load uniformly on both archs. p->bias is computed but
//   unused (SDOT is signed; the 128-bias trick is an x86 vnni quirk).
// - this TU defines the public matmul / prep / mt / sparse / ffn_down decode
//   symbols on arm64. mt path goes through src/threadpool.c (rule 31).
// veritate_engine/kernels/arm64/matmul_neon_sdot.c
// ------------------------------------------------------------------------------------
// Imports:

#include "../../src/veritate.h"
#include "../../src/portability.h"

#if !defined(__ARM_FEATURE_DOTPROD)
    #error "matmul_neon_sdot.c requires FEAT_DotProd. build with -mcpu=apple-m1 or -march=armv8.2-a+dotprod."
#endif

#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------------
// Constants

#define NEON_LANE_BYTES        16
#define VNNI_BIAS_SHIFT        128
#define Q24_FIXED_POINT        16777216.0
#define PREP_RMS_HEAD_DIM      64.0
#define PREP_INPUT_NORM        32.0
#define M1_PREFETCH_DISTANCE   256

// ------------------------------------------------------------------------------------
// Functions

// ------------------------------------------------------------------------------------
// 1x1 fallback — only runs when both m and n have <4 leftovers after tiling.
// production model shapes (hidden=768, ffn=3072) divide cleanly by 4, so this
// path is exercised only by the bench self-test on odd shapes.
// ------------------------------------------------------------------------------------

static inline int32_t sdot_block_1x1(const int8_t* a_row, const int8_t* b_col, int32_t k) {
    int32x4_t acc = vdupq_n_s32(0);
    int32_t p = 0;
    for (; p + NEON_LANE_BYTES <= k; p += NEON_LANE_BYTES) {
        acc = vdotq_s32(acc, vld1q_s8(a_row + p), vld1q_s8(b_col + p));
    }
    int32_t s = (int32_t)vaddvq_s32(acc);
    for (; p < k; p++) s += (int32_t)a_row[p] * (int32_t)b_col[p];
    return s;
}

// ------------------------------------------------------------------------------------
// 1x4 tile — decode hot path (m=1). 1 a_row reused across 4 b_cols. saturates
// the M-series SDOT pipe (4 sdots / cycle). 16 macs per sdot × 4 sdots per
// 16-byte k-step = 64 macs / output column / step. prefetch hints walk the
// next 128-byte cache line ahead per b_col.
// ------------------------------------------------------------------------------------

static inline void sdot_block_1x4(
    const int8_t*  a_row,
    const int8_t*  b0,
    const int8_t*  b1,
    const int8_t*  b2,
    const int8_t*  b3,
    int32_t        k,
    int32_t*       c0,
    int32_t*       c1,
    int32_t*       c2,
    int32_t*       c3
) {
    int32x4_t s0 = vdupq_n_s32(0);
    int32x4_t s1 = vdupq_n_s32(0);
    int32x4_t s2 = vdupq_n_s32(0);
    int32x4_t s3 = vdupq_n_s32(0);

    int32_t p = 0;
    for (; p + NEON_LANE_BYTES <= k; p += NEON_LANE_BYTES) {
        __builtin_prefetch(b0 + p + M1_PREFETCH_DISTANCE, 0, 3);
        __builtin_prefetch(b1 + p + M1_PREFETCH_DISTANCE, 0, 3);
        __builtin_prefetch(b2 + p + M1_PREFETCH_DISTANCE, 0, 3);
        __builtin_prefetch(b3 + p + M1_PREFETCH_DISTANCE, 0, 3);

        int8x16_t av = vld1q_s8(a_row + p);
        s0 = vdotq_s32(s0, av, vld1q_s8(b0 + p));
        s1 = vdotq_s32(s1, av, vld1q_s8(b1 + p));
        s2 = vdotq_s32(s2, av, vld1q_s8(b2 + p));
        s3 = vdotq_s32(s3, av, vld1q_s8(b3 + p));
    }

    int32_t r0 = (int32_t)vaddvq_s32(s0);
    int32_t r1 = (int32_t)vaddvq_s32(s1);
    int32_t r2 = (int32_t)vaddvq_s32(s2);
    int32_t r3 = (int32_t)vaddvq_s32(s3);
    for (; p < k; p++) {
        int32_t av = (int32_t)a_row[p];
        r0 += av * (int32_t)b0[p];
        r1 += av * (int32_t)b1[p];
        r2 += av * (int32_t)b2[p];
        r3 += av * (int32_t)b3[p];
    }
    *c0 = r0; *c1 = r1; *c2 = r2; *c3 = r3;
}

// ------------------------------------------------------------------------------------
// 4x4 tile — prefill path (m>=4). 4 a_rows reused across 4 b_cols; 16 sdots
// per 16-byte k-step into 16 int32x4 accumulators. amortizes both A and B
// loads 4x. mirrors the x86 vnni_4x4 layout one-for-one.
// ------------------------------------------------------------------------------------

static inline void sdot_block_4x4(
    const int8_t*  a0, const int8_t* a1, const int8_t* a2, const int8_t* a3,
    const int8_t*  b0, const int8_t* b1, const int8_t* b2, const int8_t* b3,
    int32_t        k,
    int32_t*       c00, int32_t* c01, int32_t* c02, int32_t* c03,
    int32_t*       c10, int32_t* c11, int32_t* c12, int32_t* c13,
    int32_t*       c20, int32_t* c21, int32_t* c22, int32_t* c23,
    int32_t*       c30, int32_t* c31, int32_t* c32, int32_t* c33
) {
    int32x4_t s00 = vdupq_n_s32(0), s01 = vdupq_n_s32(0), s02 = vdupq_n_s32(0), s03 = vdupq_n_s32(0);
    int32x4_t s10 = vdupq_n_s32(0), s11 = vdupq_n_s32(0), s12 = vdupq_n_s32(0), s13 = vdupq_n_s32(0);
    int32x4_t s20 = vdupq_n_s32(0), s21 = vdupq_n_s32(0), s22 = vdupq_n_s32(0), s23 = vdupq_n_s32(0);
    int32x4_t s30 = vdupq_n_s32(0), s31 = vdupq_n_s32(0), s32 = vdupq_n_s32(0), s33 = vdupq_n_s32(0);

    int32_t p = 0;
    for (; p + NEON_LANE_BYTES <= k; p += NEON_LANE_BYTES) {
        __builtin_prefetch(b0 + p + M1_PREFETCH_DISTANCE, 0, 3);
        __builtin_prefetch(b1 + p + M1_PREFETCH_DISTANCE, 0, 3);
        __builtin_prefetch(b2 + p + M1_PREFETCH_DISTANCE, 0, 3);
        __builtin_prefetch(b3 + p + M1_PREFETCH_DISTANCE, 0, 3);

        int8x16_t av0 = vld1q_s8(a0 + p);
        int8x16_t av1 = vld1q_s8(a1 + p);
        int8x16_t av2 = vld1q_s8(a2 + p);
        int8x16_t av3 = vld1q_s8(a3 + p);

        int8x16_t bv0 = vld1q_s8(b0 + p);
        int8x16_t bv1 = vld1q_s8(b1 + p);
        int8x16_t bv2 = vld1q_s8(b2 + p);
        int8x16_t bv3 = vld1q_s8(b3 + p);

        s00 = vdotq_s32(s00, av0, bv0); s01 = vdotq_s32(s01, av0, bv1);
        s02 = vdotq_s32(s02, av0, bv2); s03 = vdotq_s32(s03, av0, bv3);

        s10 = vdotq_s32(s10, av1, bv0); s11 = vdotq_s32(s11, av1, bv1);
        s12 = vdotq_s32(s12, av1, bv2); s13 = vdotq_s32(s13, av1, bv3);

        s20 = vdotq_s32(s20, av2, bv0); s21 = vdotq_s32(s21, av2, bv1);
        s22 = vdotq_s32(s22, av2, bv2); s23 = vdotq_s32(s23, av2, bv3);

        s30 = vdotq_s32(s30, av3, bv0); s31 = vdotq_s32(s31, av3, bv1);
        s32 = vdotq_s32(s32, av3, bv2); s33 = vdotq_s32(s33, av3, bv3);
    }

    int32_t r00 = vaddvq_s32(s00), r01 = vaddvq_s32(s01), r02 = vaddvq_s32(s02), r03 = vaddvq_s32(s03);
    int32_t r10 = vaddvq_s32(s10), r11 = vaddvq_s32(s11), r12 = vaddvq_s32(s12), r13 = vaddvq_s32(s13);
    int32_t r20 = vaddvq_s32(s20), r21 = vaddvq_s32(s21), r22 = vaddvq_s32(s22), r23 = vaddvq_s32(s23);
    int32_t r30 = vaddvq_s32(s30), r31 = vaddvq_s32(s31), r32 = vaddvq_s32(s32), r33 = vaddvq_s32(s33);

    for (; p < k; p++) {
        int32_t v0 = a0[p], v1 = a1[p], v2 = a2[p], v3 = a3[p];
        int32_t w0 = b0[p], w1 = b1[p], w2 = b2[p], w3 = b3[p];
        r00 += v0*w0; r01 += v0*w1; r02 += v0*w2; r03 += v0*w3;
        r10 += v1*w0; r11 += v1*w1; r12 += v1*w2; r13 += v1*w3;
        r20 += v2*w0; r21 += v2*w1; r22 += v2*w2; r23 += v2*w3;
        r30 += v3*w0; r31 += v3*w1; r32 += v3*w2; r33 += v3*w3;
    }

    *c00 = r00; *c01 = r01; *c02 = r02; *c03 = r03;
    *c10 = r10; *c11 = r11; *c12 = r12; *c13 = r13;
    *c20 = r20; *c21 = r21; *c22 = r22; *c23 = r23;
    *c30 = r30; *c31 = r31; *c32 = r32; *c33 = r33;
}

// ------------------------------------------------------------------------------------
// driver — fills [m_start, m_end) x [0, n) of c. expects bt = b transposed
// (n rows of length k). bias[j] is unused on the SDOT path; the prep struct
// shape is preserved for cross-arch parity.
// ------------------------------------------------------------------------------------

static void neon_sdot_block(
    const int8_t*  a,
    const int8_t*  bt,
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
            sdot_block_4x4(
                a0, a1, a2, a3,
                b0, b1, b2, b3,
                k,
                &c[(i+0)*n + j+0], &c[(i+0)*n + j+1], &c[(i+0)*n + j+2], &c[(i+0)*n + j+3],
                &c[(i+1)*n + j+0], &c[(i+1)*n + j+1], &c[(i+1)*n + j+2], &c[(i+1)*n + j+3],
                &c[(i+2)*n + j+0], &c[(i+2)*n + j+1], &c[(i+2)*n + j+2], &c[(i+2)*n + j+3],
                &c[(i+3)*n + j+0], &c[(i+3)*n + j+1], &c[(i+3)*n + j+2], &c[(i+3)*n + j+3]
            );
        }
        for (; j < n; j++) {
            const int8_t* b_col = bt + (size_t)j * k;
            c[(i+0)*n + j] = sdot_block_1x1(a0, b_col, k);
            c[(i+1)*n + j] = sdot_block_1x1(a1, b_col, k);
            c[(i+2)*n + j] = sdot_block_1x1(a2, b_col, k);
            c[(i+3)*n + j] = sdot_block_1x1(a3, b_col, k);
        }
    }
    for (; i < m_end; i++) {
        const int8_t* a_row = a + (size_t)i * k;
        int32_t j = 0;
        for (; j + 4 <= n; j += 4) {
            sdot_block_1x4(
                a_row,
                bt + (size_t)(j + 0) * k,
                bt + (size_t)(j + 1) * k,
                bt + (size_t)(j + 2) * k,
                bt + (size_t)(j + 3) * k,
                k,
                &c[i*n + j + 0], &c[i*n + j + 1], &c[i*n + j + 2], &c[i*n + j + 3]
            );
        }
        for (; j < n; j++) {
            c[i * n + j] = sdot_block_1x1(a_row, bt + (size_t)j * k, k);
        }
    }
}

// ------------------------------------------------------------------------------------
// raw int8 matmul — preps b inline, single thread.
// ------------------------------------------------------------------------------------

void matmul_int8_neon_sdot(
    const int8_t* a,
    const int8_t* b,
    int32_t*      c,
    int32_t       m,
    int32_t       n,
    int32_t       k
) {
    int8_t* bt = (int8_t*)veritate_aligned_alloc((size_t)k * n, VERITATE_ALIGN);
    for (int32_t j = 0; j < n; j++) {
        int8_t* dst = bt + (size_t)j * k;
        for (int32_t p = 0; p < k; p++) dst[p] = b[p * n + j];
    }
    neon_sdot_block(a, bt, c, 0, m, n, k);
    veritate_aligned_free(bt);
}

// ------------------------------------------------------------------------------------
// prep_b — pre-transpose b once and stash bias for cross-arch parity. real
// inference loads weights once; prep cost amortizes across every forward.
// ------------------------------------------------------------------------------------

void prep_b(const int8_t* b, int32_t n, int32_t k, prepped_b_t* out) {
    out->n             = n;
    out->k             = k;
    out->bt            = (int8_t*) veritate_aligned_alloc((size_t)k * n,             VERITATE_ALIGN);
    out->b_rowmaj      = NULL;
    out->bias          = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), VERITATE_ALIGN);
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
        out->bias[j] = VNNI_BIAS_SHIFT * s;
    }

    double b_rms = sqrt((double)sum_sq / ((double)n * k));
    out->scale_q24 = (int32_t)(PREP_RMS_HEAD_DIM / (sqrt((double)k) * PREP_INPUT_NORM * b_rms) * Q24_FIXED_POINT);
}

void prep_b_keep_raw(const int8_t* b, int32_t n, int32_t k, prepped_b_t* out) {
    prep_b(b, n, k, out);
    out->b_rowmaj = (int8_t*)veritate_aligned_alloc((size_t)k * n, VERITATE_ALIGN);
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
// per-inference matmul using pre-prepped b
// ------------------------------------------------------------------------------------

void matmul_int8_vnni_prep(
    const int8_t*     a,
    const prepped_b_t* p,
    int32_t*          c,
    int32_t           m
) {
    neon_sdot_block(a, p->bt, c, 0, m, p->n, p->k);
}

// ------------------------------------------------------------------------------------
// multi-threaded prep variant — partitions rows across the threadpool shim.
// ------------------------------------------------------------------------------------

typedef struct {
    const int8_t* a;
    const int8_t* bt;
    int32_t*      c;
    int32_t       m_start;
    int32_t       m_end;
    int32_t       n;
    int32_t       k;
} mt_arg_t;

static void neon_sdot_worker(void* raw, int32_t worker_idx) {
    (void)worker_idx;
    const mt_arg_t* w = (const mt_arg_t*)raw;
    neon_sdot_block(w->a, w->bt, w->c, w->m_start, w->m_end, w->n, w->k);
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

    mt_arg_t args[VERITATE_MAX_THREADS];
    void*    argv[VERITATE_MAX_THREADS];
    for (int32_t t = 0; t < threads; t++) {
        args[t].a       = a;
        args[t].bt      = p->bt;
        args[t].c       = c;
        args[t].m_start = t * rows_per;
        args[t].m_end   = (t + 1) * rows_per;
        if (args[t].m_end   > m) args[t].m_end   = m;
        if (args[t].m_start > m) args[t].m_start = m;
        args[t].n       = p->n;
        args[t].k       = p->k;
        argv[t]         = &args[t];
    }

    veritate_pool_run(neon_sdot_worker, argv, threads);
}

// ------------------------------------------------------------------------------------
// sparse decode + ffn_down_decode — v1 ARM port routes both to the dense path.
// the x86 sparse fast path exploits 50-90% near-zero post-GELU activations on
// the 9800X3D's L3-resident weights; on M-series UMA the sparse-vs-dense
// crossover sits much higher because the b_rowmaj scan is bandwidth-bound.
// adding a NEON sparse path is a v2 follow-up; for now correctness > clever.
// ------------------------------------------------------------------------------------

void matmul_int8_sparse_decode(const int8_t* a, const prepped_b_t* p, int32_t* c) {
    matmul_int8_vnni_prep(a, p, c, 1);
}

// ffn_down sparsity counters. process-global, mirrored from the x86 path so
// bench_mode's "ffn_down sparsity" report has the same hooks on both archs.
int32_t g_ffn_down_calls        = 0;
int64_t g_ffn_down_nz_sum       = 0;
int32_t g_ffn_down_sparse_calls = 0;

void ffn_down_decode(const int8_t* a, const prepped_b_t* p, int32_t* c) {
    int32_t n_nz = 0;
    for (int32_t i = 0; i < p->k; i++) if (a[i] != 0) n_nz++;
    g_ffn_down_calls++;
    g_ffn_down_nz_sum += n_nz;
    matmul_int8_vnni_prep(a, p, c, 1);
}

