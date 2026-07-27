// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - v13 hybrid loader + fp32 scalar forward. one byte per hybrid_step: embed,
//   n_enc local attn blocks, boundary-gated recurrent slot stack, n_dec local
//   attn blocks, tied lm_head. matches veritate_core/model_patched.py +
//   model_recurrent.py decode-form math. spec:
//   developer_documentation/engine/engine_v13_hybrid.md.
// - scalar matvec accumulates in 4 interleaved partial sums; SIMD ports must be
//   bitwise-identical (rule 24).
// ------------------------------------------------------------------------------------

#include "hybrid.h"
#include "portability.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HYBRID_RMS_EPS        1e-5f
#define HYBRID_SOFTPLUS_THR   20.0f
#define HYBRID_GELU_INV_SQRT2 0.70710678118654752440f
#define HYBRID_ACT_I16_SCALE  32.0f
#define HYBRID_EXT_INTS       8

// ------------------------------------------------------------------------------------
// matvec kernels + dispatch. 8 interleaved partial sums; the fp16 variants
// convert each half to fp32 (exact) then run identical fp32 math.
// ------------------------------------------------------------------------------------

float hybrid_f16_to_f32(uint16_t hb) {
    uint32_t sign = (uint32_t)(hb & 0x8000u) << 16;
    uint32_t exp  = (hb >> 10) & 0x1Fu;
    uint32_t man  = hb & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int32_t e = -1;
            do { man <<= 1; e++; } while (!(man & 0x400u));
            man &= 0x3FFu;
            bits = sign | ((uint32_t)(112 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

// 16 interleaved partial sums (4 SIMD lanes' worth of add chains). fold
// t_l = (s_l + s_{l+8}) + (s_{l+4} + s_{l+12}) BEFORE the scalar tail,
// matching the NEON (A0+A2)+(A1+A3) fold. final reduce (t0+t1)+(t2+t3).
void hybrid_matvec_f32_scalar(const void* w, const float* x, float* out,
                              int32_t n, int32_t k) {
    const float* wf = (const float*)w;
    for (int32_t j = 0; j < n; j++) {
        const float* row = wf + (size_t)j * k;
        float s[16] = {0};
        int32_t i = 0;
        for (; i + 16 <= k; i += 16) {
            for (int32_t l = 0; l < 16; l++) s[l] += row[i + l] * x[i + l];
        }
        float t[4];
        for (int32_t l = 0; l < 4; l++) t[l] = (s[l] + s[l + 8]) + (s[l + 4] + s[l + 12]);
        for (; i < k; i++) t[0] += row[i] * x[i];
        out[j] = (t[0] + t[1]) + (t[2] + t[3]);
    }
}

void hybrid_matvec_f16_scalar(const void* w, const float* x, float* out,
                              int32_t n, int32_t k) {
    const uint16_t* wh = (const uint16_t*)w;
    for (int32_t j = 0; j < n; j++) {
        const uint16_t* row = wh + (size_t)j * k;
        float s[16] = {0};
        int32_t i = 0;
        for (; i + 16 <= k; i += 16) {
            for (int32_t l = 0; l < 16; l++) s[l] += hybrid_f16_to_f32(row[i + l]) * x[i + l];
        }
        float t[4];
        for (int32_t l = 0; l < 4; l++) t[l] = (s[l] + s[l + 8]) + (s[l + 4] + s[l + 12]);
        for (; i < k; i++) t[0] += hybrid_f16_to_f32(row[i]) * x[i];
        out[j] = (t[0] + t[1]) + (t[2] + t[3]);
    }
}

float hybrid_quant_act(const float* x, int32_t k, int8_t* qx) {
    float amax = 0.0f;
    for (int32_t i = 0; i < k; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) return 0.0f;
    const float inv = 127.0f / amax;
    for (int32_t i = 0; i < k; i++) {
        long r = lrintf(x[i] * inv);
        if (r >  127) r =  127;
        if (r < -127) r = -127;
        qx[i] = (int8_t)r;
    }
    return amax / 127.0f;
}

void hybrid_matvec_i8_scalar(const void* w, const float* x, float* out,
                             int32_t n, int32_t k) {
    const hybrid_w_i8_t* wi = (const hybrid_w_i8_t*)w;
    int8_t qx[V_MAX_FFN];
    const float a_scale = hybrid_quant_act(x, k, qx);
    if (a_scale == 0.0f) {
        memset(out, 0, (size_t)n * sizeof(float));
        return;
    }
    for (int32_t j = 0; j < n; j++) {
        const int8_t* row = wi->q + (size_t)j * k;
        int32_t acc = 0;
        for (int32_t i = 0; i < k; i++) acc += (int32_t)row[i] * (int32_t)qx[i];
        out[j] = (float)acc * (wi->scale[j] * a_scale);
    }
}

// batched matmul scalar references. same 16-partial fold as the matvecs, run
// per (row j, activation b) with the weight row hot across the b sweep, so each
// out[b][j] is bitwise-equal to hybrid_matvec_*_scalar(w, X[b])[j]. compute the
// output rows [j0, j1) only (full range single-thread, a disjoint slice per
// worker under hybrid_mm's j-split); n is the output-row stride of out. int8
// activations are pre-quantized by the caller into qx + a_scale.
void hybrid_matmul_f32_scalar(const void* w, const float* X, float* out,
                              int32_t n, int32_t k, int32_t B, const int8_t* qx,
                              const float* a_scale, int32_t j0, int32_t j1) {
    (void)qx; (void)a_scale;
    const float* wf = (const float*)w;
    for (int32_t j = j0; j < j1; j++) {
        const float* row = wf + (size_t)j * k;
        for (int32_t b = 0; b < B; b++) {
            const float* x = X + (size_t)b * k;
            float s[16] = {0};
            int32_t i = 0;
            for (; i + 16 <= k; i += 16) {
                for (int32_t l = 0; l < 16; l++) s[l] += row[i + l] * x[i + l];
            }
            float t[4];
            for (int32_t l = 0; l < 4; l++) t[l] = (s[l] + s[l + 8]) + (s[l + 4] + s[l + 12]);
            for (; i < k; i++) t[0] += row[i] * x[i];
            out[(size_t)b * n + j] = (t[0] + t[1]) + (t[2] + t[3]);
        }
    }
}

void hybrid_matmul_f16_scalar(const void* w, const float* X, float* out,
                              int32_t n, int32_t k, int32_t B, const int8_t* qx,
                              const float* a_scale, int32_t j0, int32_t j1) {
    (void)qx; (void)a_scale;
    const uint16_t* wh = (const uint16_t*)w;
    for (int32_t j = j0; j < j1; j++) {
        const uint16_t* row = wh + (size_t)j * k;
        for (int32_t b = 0; b < B; b++) {
            const float* x = X + (size_t)b * k;
            float s[16] = {0};
            int32_t i = 0;
            for (; i + 16 <= k; i += 16) {
                for (int32_t l = 0; l < 16; l++) s[l] += hybrid_f16_to_f32(row[i + l]) * x[i + l];
            }
            float t[4];
            for (int32_t l = 0; l < 4; l++) t[l] = (s[l] + s[l + 8]) + (s[l + 4] + s[l + 12]);
            for (; i < k; i++) t[0] += hybrid_f16_to_f32(row[i]) * x[i];
            out[(size_t)b * n + j] = (t[0] + t[1]) + (t[2] + t[3]);
        }
    }
}

void hybrid_matmul_i8_scalar(const void* w, const float* X, float* out,
                             int32_t n, int32_t k, int32_t B, const int8_t* qx,
                             const float* a_scale, int32_t j0, int32_t j1) {
    (void)X;
    const hybrid_w_i8_t* wi = (const hybrid_w_i8_t*)w;
    for (int32_t j = j0; j < j1; j++) {
        const int8_t* row = wi->q + (size_t)j * k;
        const float wscale = wi->scale[j];
        for (int32_t b = 0; b < B; b++) {
            if (a_scale[b] == 0.0f) { out[(size_t)b * n + j] = 0.0f; continue; }
            const int8_t* qb = qx + (size_t)b * k;
            int32_t acc = 0;
            for (int32_t i = 0; i < k; i++) acc += (int32_t)row[i] * (int32_t)qb[i];
            out[(size_t)b * n + j] = (float)acc * (wscale * a_scale[b]);
        }
    }
}

hybrid_matvec_fn hybrid_matvec_wt = hybrid_matvec_f32_scalar;
hybrid_matvec_fn hybrid_matvec_fp = hybrid_matvec_f32_scalar;
hybrid_matmul_fn hybrid_matmul_wt = hybrid_matmul_f32_scalar;

// SIMD upgrades the scalar defaults per detected arch features.
// VERITATE_HYBRID_SCALAR=1 forces the scalar references for kernel-identity
// checks and A/B timing.
void hybrid_dispatch_init(int32_t dtype) {
    cpu_features_t feat;
    cpu_detect(&feat);
    hybrid_matvec_fp = hybrid_matvec_f32_scalar;
    hybrid_matvec_wt = dtype == VERITATE_HYBRID_DTYPE_FP16 ? hybrid_matvec_f16_scalar
                     : dtype == VERITATE_HYBRID_DTYPE_INT8 ? hybrid_matvec_i8_scalar
                     : hybrid_matvec_f32_scalar;
    hybrid_matmul_wt = dtype == VERITATE_HYBRID_DTYPE_FP16 ? hybrid_matmul_f16_scalar
                     : dtype == VERITATE_HYBRID_DTYPE_INT8 ? hybrid_matmul_i8_scalar
                     : hybrid_matmul_f32_scalar;
    const char* s = getenv("VERITATE_HYBRID_SCALAR");
    if (s && *s && *s != '0') return;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (feat.neon) {
        hybrid_matvec_fp = hybrid_matvec_f32_neon;
        hybrid_matvec_wt = dtype == VERITATE_HYBRID_DTYPE_FP16 ? hybrid_matvec_f16_neon
                         : dtype == VERITATE_HYBRID_DTYPE_INT8 ? hybrid_matvec_i8_sdot
                         : hybrid_matvec_f32_neon;
    }
#elif defined(__x86_64__) || defined(_M_X64)
    if (feat.avx2 && feat.f16c) {
        hybrid_matvec_fp = hybrid_matvec_f32_avx2;
        hybrid_matvec_wt = dtype == VERITATE_HYBRID_DTYPE_FP16 ? hybrid_matvec_f16_avx2
                         : dtype == VERITATE_HYBRID_DTYPE_INT8 ? hybrid_matvec_i8_avx2
                         : hybrid_matvec_f32_avx2;
        hybrid_matmul_wt = dtype == VERITATE_HYBRID_DTYPE_FP16 ? hybrid_matmul_f16_avx2
                         : dtype == VERITATE_HYBRID_DTYPE_INT8 ? hybrid_matmul_i8_avx2
                         : hybrid_matmul_f32_avx2;
    }
#endif
}

// ------------------------------------------------------------------------------------
// threaded matvec — row-split across the persistent spin-then-park pool,
// bitwise-identical to the single call (each row is computed once, same kernel).
// only matmuls with n * k >= HYBRID_MT_MIN_WORK are split. the worker count is
// per-box: a one-time micro-calibration at load (hybrid_threads_init) times real
// non-boundary decode steps on the real weights and stops at the diminishing-
// returns knee (see the calibration note below). the knee is memory-bandwidth/
// contention-driven, not core-count-driven, so only a measurement on the running
// box finds it. VERITATE_HYBRID_THREADS is an explicit override that skips
// calibration (1 = single-thread, up to pool size). row-split parity holds at
// every count, so the calibrated pick never changes output (rule 24).
// ------------------------------------------------------------------------------------

#define HYBRID_MT_MIN_WORK     (1 << 18)
#define HYBRID_MT_MIN_WORK_I8  (1 << 23)
#define HYBRID_MT_MAX          32

// micro-calibration: warmup + a decode burst per rung whose median matches the
// decode p50; direction-alternating passes whose medians are medianed decorrelate
// a rung's position-in-pass from transient load. the pick is the diminishing-
// returns knee: climb the 1,2,4,.. ladder while each rung improves on the
// previous by at least KNEE, stop at the first that does not. the knee sits at
// the memory-bandwidth saturation point and is reached before the over-threaded
// rungs, so their sustained collapse (visible only under sustained load, not a
// short calibration burst) is never selected and need not be measured for. KNEE
// also honors the energy target: a thread is added only when it buys real speed.
// LADDER_N caps the ladder array (powers of two under the ladder top, plus top).
// BUDGET_MS bounds total wall cost: passes run up to PASSES but stop once the
// budget is spent (a fast model keeps all passes; a slow model auto-trims), so
// startup stays well under half a second on any model size.
#define HYBRID_CALIB_WARMUP    1
#define HYBRID_CALIB_REPS      4
#define HYBRID_CALIB_PASSES    7
#define HYBRID_CALIB_KNEE      0.13
#define HYBRID_CALIB_LADDER_N  8
#define HYBRID_CALIB_BUDGET_MS 350

// thread-count cache: the calibrated pick is a per-(box, model shape) constant,
// so persist it next to the model (VERITATE_STATE_CACHE) and skip the ~1s
// calibration burst on later spawns. keyed on shape + core count; a re-export or
// core change re-derives it. off the decode path, so a stale file only costs a
// re-calibrate.
#define HYBRID_TCACHE_ENV      "VERITATE_STATE_CACHE"
#define HYBRID_TCACHE_NAME     "hybrid_threads.txt"
#define HYBRID_TCACHE_PATH_MAX 1024
#define HYBRID_TCACHE_FNV_OFF  14695981039346656037ULL
#define HYBRID_TCACHE_FNV_PRM  1099511628211ULL

typedef struct {
    hybrid_matvec_fn fn;
    const void* w;
    const float* x;
    float* out;
    int32_t n, k;
} hybrid_mv_span_t;

static void hybrid_mv_worker(void* arg, int32_t idx) {
    (void)idx;
    const hybrid_mv_span_t* s = (const hybrid_mv_span_t*)arg;
    s->fn(s->w, s->x, s->out, s->n, s->k);
}

static int32_t g_hybrid_nt = 1;
// Prefill and decode have opposite thread profiles: decode streams the whole model
// per token and is memory-bandwidth-bound, so extra threads buy nothing, while
// prefill batches positions and does scale. One calibrated number serves decode
// correctly and under-threads prefill. Zero means "use the decode count", which
// keeps every arch on exactly the previous behavior until the override is set.
static int32_t g_hybrid_nt_prefill = 0;

static int32_t hybrid_threads(void) { return g_hybrid_nt; }

// row-split a big matvec across nt workers with the current dtype kernel. each
// out[j] is computed once by one worker, so the result is bitwise-identical to
// the single-thread call at any nt (rule 24). int8 spans re-derive identical qx
// per worker (hybrid_quant_act is deterministic over the shared x).
static void hybrid_mv_split(const hybrid_t* h, const void* w, const float* x,
                            float* out, int32_t n, int32_t k, int32_t nt) {
    if (nt <= 1) { hybrid_matvec_wt(w, x, out, n, k); return; }
    const int i8 = h->dtype == VERITATE_HYBRID_DTYPE_INT8;
    hybrid_mv_span_t spans[HYBRID_MT_MAX];
    hybrid_w_i8_t    span_w[HYBRID_MT_MAX];
    void* args[HYBRID_MT_MAX];
    int32_t per = (n + nt - 1) / nt;
    int32_t used = 0;
    for (int32_t t = 0; t < nt; t++) {
        int32_t n0 = t * per;
        if (n0 >= n) break;
        int32_t n1 = n0 + per > n ? n : n0 + per;
        spans[used].fn  = hybrid_matvec_wt;
        if (i8) {
            const hybrid_w_i8_t* wi = (const hybrid_w_i8_t*)w;
            span_w[used].q     = wi->q + (size_t)n0 * k;
            span_w[used].scale = wi->scale + n0;
            spans[used].w = &span_w[used];
        } else {
            spans[used].w = (const char*)w + (size_t)n0 * k * h->wt_esz;
        }
        spans[used].x   = x;
        spans[used].out = out + n0;
        spans[used].n   = n1 - n0;
        spans[used].k   = k;
        args[used] = &spans[used];
        used++;
    }
    veritate_pool_run(hybrid_mv_worker, args, used);
}

// big-weight matvec entry: split at the calibrated worker count above the dtype
// work floor, else single-thread. int8 keeps a higher floor: the sdot kernel
// runs ~4x the fp rate, so it stays 1T-optimal until re-measured on an int8 bin.
static void hybrid_mv(const hybrid_t* h, const void* w, const float* x,
                      float* out, int32_t n, int32_t k) {
    const int i8 = h->dtype == VERITATE_HYBRID_DTYPE_INT8;
    const int64_t min_work = i8 ? HYBRID_MT_MIN_WORK_I8 : HYBRID_MT_MIN_WORK;
    int32_t nt = hybrid_threads();
    if (nt <= 1 || (int64_t)n * k < min_work) {
        hybrid_matvec_wt(w, x, out, n, k);
        return;
    }
    hybrid_mv_split(h, w, x, out, n, k, nt);
}

// ------------------------------------------------------------------------------------
// threaded batched matmul — j-split across the same pool as hybrid_mv. int8
// activation quant is hoisted onto the calling thread (workers must not race the
// shared pf_qx scratch), then each output row j is computed once by one worker
// with the identical per-(j,b) fold, so the result is bitwise-identical to
// single-thread at every worker count (rule 24).
// ------------------------------------------------------------------------------------

typedef struct {
    hybrid_matmul_fn fn;
    const void* w;
    const float* X;
    float* out;
    const int8_t* qx;
    const float* a_scale;
    int32_t n, k, B, j0, j1;
} hybrid_mm_span_t;

static void hybrid_mm_worker(void* arg, int32_t idx) {
    (void)idx;
    const hybrid_mm_span_t* s = (const hybrid_mm_span_t*)arg;
    s->fn(s->w, s->X, s->out, s->n, s->k, s->B, s->qx, s->a_scale, s->j0, s->j1);
}

static void hybrid_mm_split(const void* w, const float* X, float* out, int32_t n,
                            int32_t k, int32_t B, const int8_t* qx,
                            const float* a_scale, int32_t nt) {
    hybrid_mm_span_t spans[HYBRID_MT_MAX];
    void* args[HYBRID_MT_MAX];
    int32_t per = (n + nt - 1) / nt;
    int32_t used = 0;
    for (int32_t t = 0; t < nt; t++) {
        int32_t j0 = t * per;
        if (j0 >= n) break;
        int32_t j1 = j0 + per > n ? n : j0 + per;
        spans[used].fn      = hybrid_matmul_wt;
        spans[used].w       = w;
        spans[used].X       = X;
        spans[used].out     = out;
        spans[used].qx      = qx;
        spans[used].a_scale = a_scale;
        spans[used].n       = n;
        spans[used].k       = k;
        spans[used].B       = B;
        spans[used].j0      = j0;
        spans[used].j1      = j1;
        args[used] = &spans[used];
        used++;
    }
    veritate_pool_run(hybrid_mm_worker, args, used);
}

// batched-matmul entry: hoist int8 activation quant, then j-split at the
// calibrated worker count above the dtype work floor (a matmul's work is B times
// a matvec's, so the gate clears far more readily), else single-thread.
static void hybrid_mm(const hybrid_t* h, const void* w, const float* X,
                      float* out, int32_t n, int32_t k, int32_t B) {
    const int i8 = h->dtype == VERITATE_HYBRID_DTYPE_INT8;
    float a_scale[V_PREFILL_BMAX];
    if (i8)
        for (int32_t b = 0; b < B; b++)
            a_scale[b] = hybrid_quant_act(X + (size_t)b * k, k, h->pf_qx + (size_t)b * k);
    const int64_t min_work = i8 ? HYBRID_MT_MIN_WORK_I8 : HYBRID_MT_MIN_WORK;
    int32_t nt = hybrid_threads();
    if (nt <= 1 || (int64_t)n * k * B < min_work) {
        hybrid_matmul_wt(w, X, out, n, k, B, h->pf_qx, a_scale, 0, n);
        return;
    }
    hybrid_mm_split(w, X, out, n, k, B, h->pf_qx, a_scale, nt);
}

// ------------------------------------------------------------------------------------
// per-box thread calibration — see the threaded-matvec note above. runs once at
// load (or is skipped by the VERITATE_HYBRID_THREADS override); the pick lands in
// g_hybrid_nt for process life. output is unaffected by the pick (row-split
// parity holds at every count), so calibration variance can never change results.
// ------------------------------------------------------------------------------------

static int32_t calib_ladder(int32_t cap, int32_t* rungs) {
    int32_t n = 0;
    for (int32_t t = 1; t < cap; t *= 2) rungs[n++] = t;
    rungs[n++] = cap;
    return n;
}

static int calib_cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

// median per-step time of a non-boundary decode burst at nt workers. resets
// state so every rung starts at the same low position (attention cost grows with
// pos and would bias later rungs). the pos-0 boundary step is absorbed by the
// warmups, which also warm the weights into cache. median (not min) so the pick
// tracks sustained throughput, not best-case parallelism.
static uint64_t calib_time_rung(hybrid_t* h, int32_t nt, int32_t byte) {
    hybrid_reset(h);
    g_hybrid_nt = nt;
    for (int32_t r = 0; r < HYBRID_CALIB_WARMUP; r++) hybrid_step(h, byte, NULL);
    uint64_t t[HYBRID_CALIB_REPS];
    for (int32_t r = 0; r < HYBRID_CALIB_REPS; r++) {
        uint64_t t0 = veritate_now_ns();
        hybrid_step(h, byte, NULL);
        t[r] = veritate_now_ns() - t0;
    }
    qsort(t, HYBRID_CALIB_REPS, sizeof(uint64_t), calib_cmp_u64);
    return t[HYBRID_CALIB_REPS / 2];
}

static int32_t hybrid_calibrate(hybrid_t* h) {
    // reserve one core for the dispatching thread: veritate_pool_run busy-spins
    // on the completion count while workers run, so proposing pool_size workers
    // guarantees the dispatcher oversubscribes. the override still reaches the
    // full pool; auto-calibration never does.
    int32_t cap = veritate_pool_size() - 1;
    if (cap > HYBRID_MT_MAX) cap = HYBRID_MT_MAX;
    if (cap < 2) return 1;
    int32_t byte = 0;
    for (int32_t b = 0; b < 256; b++) if (!h->boundary[b]) { byte = b; break; }

    int32_t rungs[HYBRID_CALIB_LADDER_N];
    int32_t nr = calib_ladder(cap, rungs);
    uint64_t samp[HYBRID_CALIB_LADDER_N][HYBRID_CALIB_PASSES];
    const uint64_t start = veritate_now_ns();
    int32_t done = 0;
    for (int32_t p = 0; p < HYBRID_CALIB_PASSES; p++) {
        for (int32_t i = 0; i < nr; i++) {
            int32_t r = (p & 1) ? nr - 1 - i : i;
            samp[r][p] = calib_time_rung(h, rungs[r], byte);
        }
        done = p + 1;
        if ((veritate_now_ns() - start) / 1000000 >= HYBRID_CALIB_BUDGET_MS) break;
    }
    hybrid_reset(h);

    uint64_t med[HYBRID_CALIB_LADDER_N];
    for (int32_t r = 0; r < nr; r++) {
        qsort(samp[r], done, sizeof(uint64_t), calib_cmp_u64);
        med[r] = samp[r][done / 2];
    }
    if (getenv("VERITATE_HYBRID_CALIB_LOG"))
        for (int32_t r = 0; r < nr; r++)
            fprintf(stderr, "  rung nt=%2d med=%.3f ms/byte\n", rungs[r], (double)med[r] / 1e6);

    int32_t pick = 0;
    for (int32_t r = 1; r < nr; r++) {
        double impr = ((double)med[r - 1] - (double)med[r]) / (double)med[r - 1];
        if (impr < HYBRID_CALIB_KNEE) break;
        pick = r;
    }
    return rungs[pick];
}

// fnv-1a over the shape fields that move the calibrated pick plus the core
// count. a mismatch (re-export, different box) invalidates the cached value.
static uint64_t hybrid_shape_sig(const hybrid_t* h) {
    int32_t f[] = { h->hidden, h->layers, h->ffn, h->heads, h->seq, h->dtype,
                    h->n_enc, h->n_global, h->n_dec, veritate_pool_size() };
    uint64_t s = HYBRID_TCACHE_FNV_OFF;
    for (size_t i = 0; i < sizeof(f) / sizeof(f[0]); i++)
        s = (s ^ (uint64_t)(uint32_t)f[i]) * HYBRID_TCACHE_FNV_PRM;
    return s;
}

static int hybrid_tcache_path(char* buf, size_t n) {
    const char* d = getenv(HYBRID_TCACHE_ENV);
    if (!d || !*d) return 0;
    snprintf(buf, n, "%s/%s", d, HYBRID_TCACHE_NAME);
    return 1;
}

static int32_t hybrid_tcache_load(const hybrid_t* h, int32_t cap) {
    char path[HYBRID_TCACHE_PATH_MAX];
    if (!hybrid_tcache_path(path, sizeof(path))) return 0;
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    unsigned long long sig = 0;
    int nt = 0;
    int ok = fscanf(f, "%llu %d", &sig, &nt) == 2;
    fclose(f);
    if (!ok || sig != hybrid_shape_sig(h) || nt < 1 || nt > cap) return 0;
    return nt;
}

static void hybrid_tcache_store(const hybrid_t* h, int32_t nt) {
    char path[HYBRID_TCACHE_PATH_MAX];
    if (!hybrid_tcache_path(path, sizeof(path))) return;
    veritate_mkdir(getenv(HYBRID_TCACHE_ENV));
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "%llu %d\n", (unsigned long long)hybrid_shape_sig(h), nt);
    fclose(f);
}

static void hybrid_threads_init(hybrid_t* h) {
    int32_t cap = veritate_pool_size();
    if (cap > HYBRID_MT_MAX) cap = HYBRID_MT_MAX;
    const char* s = getenv("VERITATE_HYBRID_THREADS");
    uint64_t t0 = veritate_now_ns();
    int32_t nt;
    const char* how;
    if (s && *s) {
        nt = atoi(s);
        if (nt > cap) nt = cap;
        if (nt < 1) nt = 1;
        how = "override";
    } else if ((nt = hybrid_tcache_load(h, cap)) > 0) {
        how = "cached";
    } else {
        nt = hybrid_calibrate(h);
        hybrid_tcache_store(h, nt);
        how = "calibrated";
    }
    g_hybrid_nt = nt;

    const char* sp = getenv("VERITATE_HYBRID_PREFILL_THREADS");
    if (sp && *sp) {
        int32_t np = atoi(sp);
        if (np > cap) np = cap;
        if (np < 1) np = 1;
        g_hybrid_nt_prefill = np;
    }
    if (getenv("VERITATE_HYBRID_CALIB_LOG")) {
        fprintf(stderr, "hybrid: threads=%d (%s) prefill_threads=%d pool=%d %.1fms\n", nt, how,
                g_hybrid_nt_prefill > 0 ? g_hybrid_nt_prefill : nt,
                veritate_pool_size(), (double)(veritate_now_ns() - t0) / 1e6);
    }
}

// ------------------------------------------------------------------------------------
// fp32 primitives — match torch: F.rms_norm eps 1e-5, exact erf GELU,
// softplus threshold 20, silu.
// ------------------------------------------------------------------------------------

static void rmsnorm_f32(const float* x, const float* w, float* out, int32_t n) {
    float ms = 0.0f;
    for (int32_t i = 0; i < n; i++) ms += x[i] * x[i];
    float inv = 1.0f / sqrtf(ms / (float)n + HYBRID_RMS_EPS);
    for (int32_t i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

static void gelu_f32(float* x, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * HYBRID_GELU_INV_SQRT2));
    }
}

static float softplus_f32(float x) {
    return x > HYBRID_SOFTPLUS_THR ? x : log1pf(expf(x));
}

static float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

static void softmax_f32(float* x, int32_t n) {
    float mx = x[0];
    for (int32_t i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int32_t i = 0; i < n; i++) x[i] *= inv;
}

static float dot_f32(const float* a, const float* b, int32_t n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

static inline int8_t clamp_i8(float v) {
    long r = lrintf(v);
    if (r >  127) return  127;
    if (r < -128) return -128;
    return (int8_t)r;
}

static inline int16_t clamp_i16(float v) {
    long r = lrintf(v);
    if (r >  32767) return  32767;
    if (r < -32768) return -32768;
    return (int16_t)r;
}

// ------------------------------------------------------------------------------------
// block steps
// ------------------------------------------------------------------------------------

static void trace_attn_rows(trace_record_t* trace, const hybrid_t* h, int32_t L,
                            const float* row_probs, int32_t hd);

static void local_block_step(hybrid_t* h, const hybrid_block_t* b, int32_t li,
                             trace_record_t* trace, int32_t L) {
    const int32_t H = h->hidden, NH = h->heads, D = h->head_dim, F = h->ffn;
    const int32_t pos = h->pos, S = h->seq;
    const float scale = 1.0f / sqrtf((float)D);

    rmsnorm_f32(h->x, b->n1_w, h->u, H);
    hybrid_mv(h, b->qkv_w, h->u, h->qkv, 3 * H, H);
    float* k_base = h->kv_k + (size_t)li * S * H;
    float* v_base = h->kv_v + (size_t)li * S * H;
    memcpy(k_base + (size_t)pos * H, h->qkv + H,     (size_t)H * sizeof(float));
    memcpy(v_base + (size_t)pos * H, h->qkv + 2 * H, (size_t)H * sizeof(float));

    for (int32_t hd = 0; hd < NH; hd++) {
        const float* q = h->qkv + hd * D;
        for (int32_t j = 0; j <= pos; j++) {
            h->scores[j] = dot_f32(q, k_base + (size_t)j * H + hd * D, D) * scale;
        }
        softmax_f32(h->scores, pos + 1);
        if (trace && trace->attention_scores) trace_attn_rows(trace, h, L, h->scores, hd);
        float* o = h->attn_out + hd * D;
        memset(o, 0, (size_t)D * sizeof(float));
        for (int32_t j = 0; j <= pos; j++) {
            const float p = h->scores[j];
            const float* vr = v_base + (size_t)j * H + hd * D;
            for (int32_t d = 0; d < D; d++) o[d] += p * vr[d];
        }
    }

    hybrid_mv(h, b->proj_w, h->attn_out, h->tmp, H, H);
    for (int32_t i = 0; i < H; i++) h->x[i] += h->tmp[i];

    rmsnorm_f32(h->x, b->n2_w, h->u, H);
    hybrid_mv(h, b->up_w, h->u, h->ffn_buf, F, H);
    gelu_f32(h->ffn_buf, F);
    hybrid_mv(h, b->down_w, h->ffn_buf, h->tmp, H, F);
    for (int32_t i = 0; i < H; i++) h->x[i] += h->tmp[i];
}

static void recurrent_block_step(hybrid_t* h, const hybrid_block_t* b, int32_t gi) {
    const int32_t H = h->hidden, NH = h->heads, D = h->head_dim, F = h->ffn;
    const int32_t CK = h->conv_kernel;
    const float qscale = 1.0f / sqrtf((float)D);

    rmsnorm_f32(h->g, b->n1_w, h->u, H);
    hybrid_mv(h, b->qkv_w, h->u, h->qkv, 3 * H, H);

    float* ring = h->conv_ring + (size_t)gi * (CK - 1) * 3 * H;
    for (int32_t c = 0; c < 3 * H; c++) {
        const float* wr = b->conv_w + (size_t)c * CK;
        float s = wr[CK - 1] * h->qkv[c];
        for (int32_t i = 0; i < CK - 1; i++) s += wr[i] * ring[(size_t)i * 3 * H + c];
        h->conv_out[c] = s;
    }
    memmove(ring, ring + 3 * H, (size_t)(CK - 2) * 3 * H * sizeof(float));
    memcpy(ring + (size_t)(CK - 2) * 3 * H, h->qkv, (size_t)3 * H * sizeof(float));

    float* q = h->conv_out;
    const float* kk = h->conv_out + H;
    const float* vv = h->conv_out + 2 * H;
    for (int32_t i = 0; i < H; i++) q[i] *= qscale;

    for (int32_t hd = 0; hd < NH; hd++) {
        float la = -softplus_f32(dot_f32(b->a_proj_w + (size_t)hd * H, h->u, H)
                                 + b->a_proj_b[hd]);
        const float dec = expf(la);
        const float* k_h = kk + hd * D;
        const float* v_h = vv + hd * D;
        float* st = h->rec_state + ((size_t)gi * NH + hd) * D * D;
        for (int32_t dk = 0; dk < D; dk++) {
            const float kv = k_h[dk];
            float* row = st + (size_t)dk * D;
            for (int32_t dv = 0; dv < D; dv++) row[dv] = dec * row[dv] + kv * v_h[dv];
        }
        float* o = h->attn_out + hd * D;
        memset(o, 0, (size_t)D * sizeof(float));
        const float* q_h = q + hd * D;
        for (int32_t dk = 0; dk < D; dk++) {
            const float qv = q_h[dk];
            const float* row = st + (size_t)dk * D;
            for (int32_t dv = 0; dv < D; dv++) o[dv] += qv * row[dv];
        }
        rmsnorm_f32(o, b->o_norm_w, o, D);
    }

    hybrid_mv(h, b->gate_w, h->u, h->gate_buf, H, H);
    for (int32_t i = 0; i < H; i++) h->attn_out[i] *= silu_f32(h->gate_buf[i]);
    hybrid_mv(h, b->proj_w, h->attn_out, h->tmp, H, H);
    for (int32_t i = 0; i < H; i++) h->g[i] += h->tmp[i];

    rmsnorm_f32(h->g, b->n2_w, h->u, H);
    hybrid_mv(h, b->up_w, h->u, h->ffn_buf, F, H);
    gelu_f32(h->ffn_buf, F);
    hybrid_mv(h, b->down_w, h->ffn_buf, h->tmp, H, F);
    for (int32_t i = 0; i < H; i++) h->g[i] += h->tmp[i];
}

// ------------------------------------------------------------------------------------
// trace capture — dense-frame layout, layers = total blocks. local blocks carry
// real values; global blocks carry slot residuals on boundary bytes, byte
// residual pass-through otherwise. The logit lens is now computed for every
// block (trace_lens_compute). Global (GLA) attention stays zero: the recurrent
// decode keeps only an O(1) state summary, so per-position attention weights do
// not exist to emit without reintroducing O(seq) storage.
// ------------------------------------------------------------------------------------

static void trace_residual(trace_record_t* trace, const hybrid_t* h, int32_t L,
                           const float* res, int pre) {
    const int32_t H = h->hidden, S = h->seq;
    int16_t* base = pre ? trace->residual_pre : trace->residual_post;
    int16_t* dst = base + ((size_t)L * S + h->pos) * H;
    for (int32_t i = 0; i < H; i++) dst[i] = clamp_i16(res[i] * HYBRID_ACT_I16_SCALE);
}

static void trace_ffn(trace_record_t* trace, const hybrid_t* h, int32_t L,
                      const float* post_act) {
    const int32_t F = h->ffn, S = h->seq;
    int8_t* dst = trace->ffn_neurons + ((size_t)L * S + h->pos) * F;
    if (!post_act) { memset(dst, 0, (size_t)F); return; }
    for (int32_t i = 0; i < F; i++) dst[i] = clamp_i8(post_act[i] * HYBRID_ACT_I16_SCALE);
}

static void trace_attn_rows(trace_record_t* trace, const hybrid_t* h, int32_t L,
                            const float* row_probs, int32_t hd) {
    const int32_t S = h->seq, NH = h->heads, pos = h->pos;
    float* dst = trace->attention_scores
               + (size_t)L * NH * S * S + ((size_t)hd * S + pos) * S;
    if (row_probs) {
        memcpy(dst, row_probs, (size_t)(pos + 1) * sizeof(float));
        memset(dst + pos + 1, 0, (size_t)(S - pos - 1) * sizeof(float));
    } else {
        memset(dst, 0, (size_t)S * sizeof(float));
    }
}

// Logit lens: project a block's post-residual through the final norm + the tied
// unembed, the same path the real logits take at step end. Fills every block's
// lens slot (was zeroed); the last dec block's lens equals the real logits since
// its post-residual is exactly what n_out norms. Frame layout is unchanged, so
// the Python parser reads it transparently. rmsnorm + a vocab-sized matvec per
// block per byte is ~2-3% of the forward and only runs in the traced path.
static void trace_lens_compute(trace_record_t* trace, hybrid_t* h, int32_t L,
                               const float* res) {
    const int32_t H = h->hidden, S = h->seq, V = h->vocab;
    rmsnorm_f32(res, h->n_out_w, h->lens_u, H);
    hybrid_matvec_fp(h->tok_emb, h->lens_u, h->lens_f, V, H);
    int32_t* dst = trace->lens_logits + ((size_t)L * S + h->pos) * V;
    for (int32_t v = 0; v < V; v++)
        dst[v] = (int32_t)lrintf(h->lens_f[v] * VERITATE_HYBRID_LOGIT_SCALE);
}

// ------------------------------------------------------------------------------------
// step
// ------------------------------------------------------------------------------------

void hybrid_step(hybrid_t* h, int32_t byte, trace_record_t* trace) {
    const int32_t H = h->hidden, V = h->vocab, NH = h->heads;
    int32_t tok = byte;
    if (V > 0) tok = ((tok % V) + V) % V;

    const float* te = h->tok_emb + (size_t)tok * H;
    const float* pe = h->pos_emb + (size_t)h->pos * H;
    for (int32_t i = 0; i < H; i++) h->x[i] = te[i] + pe[i];

    int32_t L = 0;
    for (int32_t e = 0; e < h->n_enc; e++, L++) {
        if (trace) trace_residual(trace, h, L, h->x, 1);
        local_block_step(h, &h->blocks[L], e, trace, L);
        if (trace) {
            trace_residual(trace, h, L, h->x, 0);
            trace_ffn(trace, h, L, h->ffn_buf);
            if (trace->lens_logits) trace_lens_compute(trace, h, L, h->x);
        }
    }

    const int is_boundary = h->boundary[tok] || h->pos == 0;
    const int slot_live = is_boundary && h->slot_count < h->slots;
    if (slot_live) {
        const float* se = h->slot_pos_emb + (size_t)h->slot_count * H;
        for (int32_t i = 0; i < H; i++) h->g[i] = h->x[i] + se[i];
    }
    for (int32_t gidx = 0; gidx < h->n_global; gidx++, L++) {
        if (trace) trace_residual(trace, h, L, slot_live ? h->g : h->x, 1);
        if (slot_live) recurrent_block_step(h, &h->blocks[L], gidx);
        if (trace) {
            trace_residual(trace, h, L, slot_live ? h->g : h->x, 0);
            trace_ffn(trace, h, L, slot_live ? h->ffn_buf : NULL);
            if (trace->attention_scores) {
                for (int32_t hd = 0; hd < NH; hd++) trace_attn_rows(trace, h, L, NULL, hd);
            }
            if (trace->lens_logits) trace_lens_compute(trace, h, L, slot_live ? h->g : h->x);
        }
    }
    if (slot_live) {
        for (int32_t i = 0; i < H; i++) h->x[i] += h->g[i];
    }
    if (is_boundary) h->slot_count++;

    for (int32_t d = 0; d < h->n_dec; d++, L++) {
        if (trace) trace_residual(trace, h, L, h->x, 1);
        local_block_step(h, &h->blocks[L], h->n_enc + d, trace, L);
        if (trace) {
            trace_residual(trace, h, L, h->x, 0);
            trace_ffn(trace, h, L, h->ffn_buf);
            if (trace->lens_logits) trace_lens_compute(trace, h, L, h->x);
        }
    }

    rmsnorm_f32(h->x, h->n_out_w, h->u, H);
    hybrid_matvec_fp(h->tok_emb, h->u, h->logits, V, H);
    h->pos++;
}

void hybrid_logits_i32(const hybrid_t* h, int32_t* out) {
    for (int32_t v = 0; v < h->vocab; v++) {
        out[v] = (int32_t)lrintf(h->logits[v] * VERITATE_HYBRID_LOGIT_SCALE);
    }
}

void hybrid_final_act_i8(const hybrid_t* h, int8_t* out) {
    for (int32_t i = 0; i < h->hidden; i++) {
        out[i] = clamp_i8(h->u[i] * HYBRID_ACT_I16_SCALE);
    }
}

void hybrid_reset(hybrid_t* h) {
    memset(h->rec_state, 0,
           (size_t)h->n_global * h->heads * h->head_dim * h->head_dim * sizeof(float));
    memset(h->conv_ring, 0,
           (size_t)h->n_global * (h->conv_kernel - 1) * 3 * h->hidden * sizeof(float));
    h->slot_count = 0;
    h->pos = 0;
}

// ------------------------------------------------------------------------------------
// batched prefill: amortize weight streaming over a chunk of positions. local
// blocks batch the five matmuls over the chunk (weight rows hot across the b
// sweep) via hybrid_mm, which j-splits each across the pool; the recurrent
// stack, conv ring, and slot scan stay sequential per position so their float
// reductions match hybrid_step exactly. spec:
// developer_documentation/engine/engine_v13_hybrid.md.
// ------------------------------------------------------------------------------------

int32_t hybrid_prefill_batch(void) {
    static int32_t cached = -1;
    if (cached < 0) {
        const char* s = getenv("VERITATE_PREFILL_BATCH");
        int32_t b = s && *s ? atoi(s) : 1;
        if (b < 1) b = 1;
        if (b > V_PREFILL_BMAX) b = V_PREFILL_BMAX;
        cached = b;
    }
    return cached;
}

// one local (enc/dec) block over the Bc-row chunk at base position pos0. li is
// the local-block KV index (enc: 0..n_enc-1, dec: n_enc..n_enc+n_dec-1).
static void prefill_local_block(hybrid_t* h, const hybrid_block_t* b, int32_t li,
                                int32_t pos0, int32_t Bc) {
    const int32_t H = h->hidden, NH = h->heads, D = h->head_dim, F = h->ffn, S = h->seq;
    const float scale = 1.0f / sqrtf((float)D);

    for (int32_t r = 0; r < Bc; r++)
        rmsnorm_f32(h->pf_x + (size_t)r * H, b->n1_w, h->pf_u + (size_t)r * H, H);
    hybrid_mm(h, b->qkv_w, h->pf_u, h->pf_qkv, 3 * H, H, Bc);

    float* k_base = h->kv_k + (size_t)li * S * H;
    float* v_base = h->kv_v + (size_t)li * S * H;
    for (int32_t r = 0; r < Bc; r++) {
        const float* qkv = h->pf_qkv + (size_t)r * 3 * H;
        memcpy(k_base + (size_t)(pos0 + r) * H, qkv + H,     (size_t)H * sizeof(float));
        memcpy(v_base + (size_t)(pos0 + r) * H, qkv + 2 * H, (size_t)H * sizeof(float));
    }

    for (int32_t r = 0; r < Bc; r++) {
        const int32_t pos = pos0 + r;
        const float* qkv = h->pf_qkv + (size_t)r * 3 * H;
        float* aout = h->pf_attn + (size_t)r * H;
        for (int32_t hd = 0; hd < NH; hd++) {
            const float* q = qkv + hd * D;
            for (int32_t j = 0; j <= pos; j++)
                h->scores[j] = dot_f32(q, k_base + (size_t)j * H + hd * D, D) * scale;
            softmax_f32(h->scores, pos + 1);
            float* o = aout + hd * D;
            memset(o, 0, (size_t)D * sizeof(float));
            for (int32_t j = 0; j <= pos; j++) {
                const float p = h->scores[j];
                const float* vr = v_base + (size_t)j * H + hd * D;
                for (int32_t d = 0; d < D; d++) o[d] += p * vr[d];
            }
        }
    }

    hybrid_mm(h, b->proj_w, h->pf_attn, h->pf_tmp, H, H, Bc);
    for (int32_t r = 0; r < Bc; r++) {
        float* xr = h->pf_x + (size_t)r * H;
        const float* pr = h->pf_tmp + (size_t)r * H;
        for (int32_t i = 0; i < H; i++) xr[i] += pr[i];
    }

    for (int32_t r = 0; r < Bc; r++)
        rmsnorm_f32(h->pf_x + (size_t)r * H, b->n2_w, h->pf_u + (size_t)r * H, H);
    hybrid_mm(h, b->up_w, h->pf_u, h->pf_ff, F, H, Bc);
    for (int32_t r = 0; r < Bc; r++) gelu_f32(h->pf_ff + (size_t)r * F, F);
    hybrid_mm(h, b->down_w, h->pf_ff, h->pf_tmp, H, F, Bc);
    for (int32_t r = 0; r < Bc; r++) {
        float* xr = h->pf_x + (size_t)r * H;
        const float* dr = h->pf_tmp + (size_t)r * H;
        for (int32_t i = 0; i < H; i++) xr[i] += dr[i];
    }
}

// recurrent stack for one position: mirrors hybrid_step's middle section on
// h->x/h->g/h->pos, reusing recurrent_block_step so state advances identically.
static void prefill_recurrent_pos(hybrid_t* h, int32_t tok) {
    const int32_t H = h->hidden;
    const int is_boundary = h->boundary[tok] || h->pos == 0;
    const int slot_live = is_boundary && h->slot_count < h->slots;
    if (slot_live) {
        const float* se = h->slot_pos_emb + (size_t)h->slot_count * H;
        for (int32_t i = 0; i < H; i++) h->g[i] = h->x[i] + se[i];
        for (int32_t gidx = 0; gidx < h->n_global; gidx++)
            recurrent_block_step(h, &h->blocks[h->n_enc + gidx], gidx);
        for (int32_t i = 0; i < H; i++) h->x[i] += h->g[i];
    }
    if (is_boundary) h->slot_count++;
}

void hybrid_prefill(hybrid_t* h, const int32_t* tokens, int32_t n, int32_t B) {
    const int32_t H = h->hidden, V = h->vocab;
    // Run the whole prefill at the prefill thread count, then restore the decode
    // count. Row-split is output-invariant (see hybrid_calibrate), so this changes
    // wall-clock only. Restored on every exit path because the loop cannot throw.
    const int32_t nt_decode = g_hybrid_nt;
    if (g_hybrid_nt_prefill > 0) g_hybrid_nt = g_hybrid_nt_prefill;
    while (h->pos < n) {
        const int32_t pos0 = h->pos;
        const int32_t rem = n - pos0;
        const int32_t Bc = rem < B ? rem : B;

        for (int32_t r = 0; r < Bc; r++) {
            int32_t tok = tokens[pos0 + r];
            if (V > 0) tok = ((tok % V) + V) % V;
            const float* te = h->tok_emb + (size_t)tok * H;
            const float* pe = h->pos_emb + (size_t)(pos0 + r) * H;
            float* xr = h->pf_x + (size_t)r * H;
            for (int32_t i = 0; i < H; i++) xr[i] = te[i] + pe[i];
        }

        for (int32_t e = 0; e < h->n_enc; e++)
            prefill_local_block(h, &h->blocks[e], e, pos0, Bc);

        for (int32_t r = 0; r < Bc; r++) {
            int32_t tok = tokens[pos0 + r];
            if (V > 0) tok = ((tok % V) + V) % V;
            memcpy(h->x, h->pf_x + (size_t)r * H, (size_t)H * sizeof(float));
            h->pos = pos0 + r;
            prefill_recurrent_pos(h, tok);
            memcpy(h->pf_x + (size_t)r * H, h->x, (size_t)H * sizeof(float));
        }

        for (int32_t d = 0; d < h->n_dec; d++)
            prefill_local_block(h, &h->blocks[h->n_enc + h->n_global + d], h->n_enc + d, pos0, Bc);

        if (pos0 + Bc == n) {
            const float* xl = h->pf_x + (size_t)(Bc - 1) * H;
            rmsnorm_f32(xl, h->n_out_w, h->u, H);
            hybrid_matvec_fp(h->tok_emb, h->u, h->logits, V, H);
        }
        h->pos = pos0 + Bc;
    }
    g_hybrid_nt = nt_decode;
}

// ------------------------------------------------------------------------------------
// load
// ------------------------------------------------------------------------------------

// weight-load jobs: the collect pass allocates every tensor and records where its
// bytes live in the bin; the parallel pass reopens the bin per worker and fills
// them. splitting the streaming read across the pool hides the per-page
// first-touch fault cost that bottlenecks a single-thread load (rule 24: each
// job writes identical bytes to one buffer, so output is bitwise-unchanged).
typedef struct {
    int64_t  off;
    void*    dst;
    size_t   bytes;
    int32_t  convert_n;
} hybrid_load_job_t;

typedef struct {
    hybrid_load_job_t* jobs;
    int32_t            n;
    int64_t            off;
} hybrid_load_ctx_t;

typedef struct {
    const char*        path;
    hybrid_load_job_t* jobs;
    int32_t            j0, j1;
    int                ok;
} hybrid_load_span_t;

// small tensors: always upconverted to fp32. fp16 bins convert per element in
// the worker; fp32 bins copy raw.
static float* read_tensor_f32(hybrid_load_ctx_t* lc, size_t n, int32_t dtype) {
    float* out = (float*)veritate_aligned_alloc(n * sizeof(float), 64);
    if (!out) return NULL;
    hybrid_load_job_t* j = &lc->jobs[lc->n++];
    j->dst = out;
    j->off = lc->off;
    if (dtype == VERITATE_HYBRID_DTYPE_FP32) {
        j->bytes = n * sizeof(float);
        j->convert_n = 0;
    } else {
        j->bytes = n * sizeof(uint16_t);
        j->convert_n = (int32_t)n;
    }
    lc->off += (int64_t)j->bytes;
    return out;
}

// big matmul weights: kept in the bin dtype, consumed via hybrid_matvec_wt.
// int8 tensors (disk: q[n*k] then fp32 scale[n]) load into ONE aligned block
// [hybrid_w_i8_t | scales | q] so free stays a single call per tensor.
static void* read_tensor_big(hybrid_load_ctx_t* lc, int32_t n, int32_t k, int32_t dtype) {
    const size_t elems = (size_t)n * k;
    if (dtype != VERITATE_HYBRID_DTYPE_INT8) {
        size_t esz = dtype == VERITATE_HYBRID_DTYPE_FP16 ? sizeof(uint16_t) : sizeof(float);
        void* out = veritate_aligned_alloc(elems * esz, 64);
        if (!out) return NULL;
        hybrid_load_job_t* j = &lc->jobs[lc->n++];
        j->off = lc->off; j->dst = out; j->bytes = elems * esz; j->convert_n = 0;
        lc->off += (int64_t)j->bytes;
        return out;
    }
    size_t hdr = (sizeof(hybrid_w_i8_t) + 63) & ~(size_t)63;
    void* blk = veritate_aligned_alloc(hdr + (size_t)n * sizeof(float) + elems, 64);
    if (!blk) return NULL;
    hybrid_w_i8_t* wi = (hybrid_w_i8_t*)blk;
    float*  scale = (float*)((char*)blk + hdr);
    int8_t* q     = (int8_t*)(scale + n);
    hybrid_load_job_t* jq = &lc->jobs[lc->n++];
    jq->off = lc->off; jq->dst = q; jq->bytes = elems; jq->convert_n = 0;
    lc->off += (int64_t)elems;
    hybrid_load_job_t* js = &lc->jobs[lc->n++];
    js->off = lc->off; js->dst = scale; js->bytes = (size_t)n * sizeof(float); js->convert_n = 0;
    lc->off += (int64_t)js->bytes;
    wi->q = q;
    wi->scale = scale;
    return blk;
}

static int load_block(hybrid_load_ctx_t* lc, hybrid_block_t* b, int32_t H, int32_t F,
                      int32_t NH, int32_t D, int32_t CK, int32_t dtype,
                      int recurrent) {
    const int32_t sdt = dtype == VERITATE_HYBRID_DTYPE_INT8
        ? VERITATE_HYBRID_DTYPE_FP32 : dtype;
    b->is_recurrent = recurrent;
    b->n1_w  = read_tensor_f32(lc, (size_t)H, sdt);
    b->qkv_w = read_tensor_big(lc, 3 * H, H, dtype);
    if (recurrent) {
        b->conv_w   = read_tensor_f32(lc, (size_t)3 * H * CK, sdt);
        b->a_proj_w = read_tensor_f32(lc, (size_t)NH * H, sdt);
        b->a_proj_b = read_tensor_f32(lc, (size_t)NH, sdt);
        b->o_norm_w = read_tensor_f32(lc, (size_t)D, sdt);
        b->gate_w   = read_tensor_big(lc, H, H, dtype);
        if (!b->conv_w || !b->a_proj_w || !b->a_proj_b || !b->o_norm_w || !b->gate_w)
            return -1;
    }
    b->proj_w = read_tensor_big(lc, H, H, dtype);
    b->n2_w   = read_tensor_f32(lc, (size_t)H, sdt);
    b->up_w   = read_tensor_big(lc, F, H, dtype);
    b->down_w = read_tensor_big(lc, H, F, dtype);
    if (!b->n1_w || !b->qkv_w || !b->proj_w || !b->n2_w || !b->up_w || !b->down_w)
        return -1;
    return 0;
}

// one worker reads its contiguous span: reopen the bin, seek to the span's first
// job, fread each (jobs are gapless in file order so the reads stay sequential),
// upconverting fp16 tensors on the way in.
static void hybrid_load_worker(void* arg, int32_t idx) {
    (void)idx;
    hybrid_load_span_t* s = (hybrid_load_span_t*)arg;
    FILE* f = fopen(s->path, "rb");
    if (!f) { s->ok = 0; return; }
    for (int32_t i = s->j0; i < s->j1; i++) {
        const hybrid_load_job_t* j = &s->jobs[i];
        if (veritate_fseek64(f, j->off) != 0) { s->ok = 0; break; }
        if (j->convert_n > 0) {
            uint16_t* half = (uint16_t*)malloc(j->bytes);
            if (!half || fread(half, 1, j->bytes, f) != j->bytes) { free(half); s->ok = 0; break; }
            float* out = (float*)j->dst;
            for (int32_t e = 0; e < j->convert_n; e++) out[e] = hybrid_f16_to_f32(half[e]);
            free(half);
        } else if (fread(j->dst, 1, j->bytes, f) != j->bytes) { s->ok = 0; break; }
    }
    fclose(f);
}

// split the job list into byte-balanced contiguous spans and fill them across
// the pool. returns 0 iff every worker completed.
static int hybrid_load_run(const char* path, hybrid_load_job_t* jobs, int32_t n) {
    int32_t nt = veritate_pool_size();
    if (nt > n) nt = n;
    if (nt < 1) nt = 1;
    size_t total = 0;
    for (int32_t i = 0; i < n; i++) total += jobs[i].bytes;
    size_t target = (total + (size_t)nt - 1) / (size_t)nt;
    hybrid_load_span_t spans[VERITATE_MAX_THREADS];
    void* args[VERITATE_MAX_THREADS];
    int32_t t = 0, j0 = 0;
    size_t acc = 0;
    for (int32_t i = 0; i < n && t < nt - 1; i++) {
        acc += jobs[i].bytes;
        if (acc >= target) {
            spans[t].path = path; spans[t].jobs = jobs; spans[t].j0 = j0; spans[t].j1 = i + 1; spans[t].ok = 1;
            args[t] = &spans[t];
            t++; j0 = i + 1; acc = 0;
        }
    }
    spans[t].path = path; spans[t].jobs = jobs; spans[t].j0 = j0; spans[t].j1 = n; spans[t].ok = 1;
    args[t] = &spans[t];
    t++;
    for (int32_t k = t; k < nt; k++) {
        spans[k].path = path; spans[k].jobs = jobs; spans[k].j0 = n; spans[k].j1 = n; spans[k].ok = 1;
        args[k] = &spans[k];
    }
    veritate_pool_run(hybrid_load_worker, args, nt);
    int ok = 1;
    for (int32_t k = 0; k < nt; k++) ok &= spans[k].ok;
    return ok ? 0 : -1;
}

hybrid_t* hybrid_load(FILE* f, const char* path, int32_t vocab, int32_t hidden,
                      int32_t layers, int32_t ffn, int32_t heads, int32_t seq) {
    int32_t ext[HYBRID_EXT_INTS];
    if (fread(ext, sizeof(int32_t), HYBRID_EXT_INTS, f) != HYBRID_EXT_INTS) return NULL;
    const int32_t dtype = ext[0], n_enc = ext[1], n_global = ext[2], n_dec = ext[3];
    const int32_t stride = ext[4], slots = ext[5], ck = ext[6], rule = ext[7];
    if (dtype != VERITATE_HYBRID_DTYPE_FP32 && dtype != VERITATE_HYBRID_DTYPE_FP16 &&
        dtype != VERITATE_HYBRID_DTYPE_INT8) {
        fprintf(stderr, "hybrid_load: unknown dtype %d\n", dtype);
        return NULL;
    }
    if (rule != 0) {
        fprintf(stderr, "hybrid_load: state_rule %d not supported (gla only)\n", rule);
        return NULL;
    }
    if (n_enc < 1 || n_global < 1 || n_dec < 1 || n_enc + n_global + n_dec != layers ||
        stride < 1 || slots != seq / stride || ck < 2) {
        fprintf(stderr, "hybrid_load: inconsistent extension header\n");
        return NULL;
    }

    hybrid_dispatch_init(dtype);
    hybrid_t* h = (hybrid_t*)calloc(1, sizeof(hybrid_t));
    if (!h) return NULL;
    h->vocab = vocab; h->hidden = hidden; h->layers = layers; h->ffn = ffn;
    h->heads = heads; h->head_dim = hidden / heads; h->seq = seq;
    h->dtype = dtype; h->n_enc = n_enc; h->n_global = n_global; h->n_dec = n_dec;
    h->patch_stride = stride; h->slots = slots; h->conv_kernel = ck; h->state_rule = rule;
    h->wt_esz = dtype == VERITATE_HYBRID_DTYPE_FP16 ? sizeof(uint16_t)
              : dtype == VERITATE_HYBRID_DTYPE_INT8 ? sizeof(int8_t)
              : sizeof(float);
    const int32_t sdt = dtype == VERITATE_HYBRID_DTYPE_INT8
        ? VERITATE_HYBRID_DTYPE_FP32 : dtype;

    if (fread(h->boundary, 1, 256, f) != 256) { hybrid_free(h); return NULL; }

    hybrid_load_ctx_t lc;
    lc.n = 0;
    lc.off = (int64_t)ftell(f);
    lc.jobs = (hybrid_load_job_t*)malloc((size_t)(4 + layers * 16) * sizeof(hybrid_load_job_t));
    if (!lc.jobs) { hybrid_free(h); return NULL; }

    const int32_t H = hidden, D = h->head_dim;
    h->tok_emb      = read_tensor_f32(&lc, (size_t)vocab * H, sdt);
    h->pos_emb      = read_tensor_f32(&lc, (size_t)seq * H, sdt);
    h->slot_pos_emb = read_tensor_f32(&lc, (size_t)slots * H, sdt);
    h->blocks = (hybrid_block_t*)calloc((size_t)layers, sizeof(hybrid_block_t));
    if (!h->tok_emb || !h->pos_emb || !h->slot_pos_emb || !h->blocks) {
        free(lc.jobs); hybrid_free(h); return NULL;
    }
    for (int32_t L = 0; L < layers; L++) {
        int recurrent = (L >= n_enc && L < n_enc + n_global);
        if (load_block(&lc, &h->blocks[L], H, ffn, heads, D, ck, dtype, recurrent) != 0) {
            fprintf(stderr, "hybrid_load: truncated at block %d\n", L);
            free(lc.jobs); hybrid_free(h); return NULL;
        }
    }
    h->n_out_w = read_tensor_f32(&lc, (size_t)H, sdt);
    if (!h->n_out_w) { free(lc.jobs); hybrid_free(h); return NULL; }

    if (hybrid_load_run(path, lc.jobs, lc.n) != 0) {
        fprintf(stderr, "hybrid_load: parallel read failed\n");
        free(lc.jobs); hybrid_free(h); return NULL;
    }
    free(lc.jobs);

    const int32_t n_local = n_enc + n_dec;
    h->kv_k      = (float*)veritate_aligned_alloc((size_t)n_local * seq * H * sizeof(float), 64);
    h->kv_v      = (float*)veritate_aligned_alloc((size_t)n_local * seq * H * sizeof(float), 64);
    h->rec_state = (float*)veritate_aligned_alloc((size_t)n_global * heads * D * D * sizeof(float), 64);
    h->conv_ring = (float*)veritate_aligned_alloc((size_t)n_global * (ck - 1) * 3 * H * sizeof(float), 64);
    h->x        = (float*)veritate_aligned_alloc((size_t)H * sizeof(float), 64);
    h->g        = (float*)veritate_aligned_alloc((size_t)H * sizeof(float), 64);
    h->u        = (float*)veritate_aligned_alloc((size_t)H * sizeof(float), 64);
    h->qkv      = (float*)veritate_aligned_alloc((size_t)3 * H * sizeof(float), 64);
    h->conv_out = (float*)veritate_aligned_alloc((size_t)3 * H * sizeof(float), 64);
    h->attn_out = (float*)veritate_aligned_alloc((size_t)H * sizeof(float), 64);
    h->scores   = (float*)veritate_aligned_alloc((size_t)seq * sizeof(float), 64);
    h->ffn_buf  = (float*)veritate_aligned_alloc((size_t)ffn * sizeof(float), 64);
    h->gate_buf = (float*)veritate_aligned_alloc((size_t)H * sizeof(float), 64);
    h->tmp      = (float*)veritate_aligned_alloc((size_t)H * sizeof(float), 64);
    h->logits   = (float*)veritate_aligned_alloc((size_t)vocab * sizeof(float), 64);
    h->lens_u   = (float*)veritate_aligned_alloc((size_t)H * sizeof(float), 64);
    h->lens_f   = (float*)veritate_aligned_alloc((size_t)vocab * sizeof(float), 64);
    const size_t bm = V_PREFILL_BMAX;
    h->pf_x     = (float*)veritate_aligned_alloc(bm * (size_t)H * sizeof(float), 64);
    h->pf_u     = (float*)veritate_aligned_alloc(bm * (size_t)H * sizeof(float), 64);
    h->pf_qkv   = (float*)veritate_aligned_alloc(bm * (size_t)3 * H * sizeof(float), 64);
    h->pf_attn  = (float*)veritate_aligned_alloc(bm * (size_t)H * sizeof(float), 64);
    h->pf_ff    = (float*)veritate_aligned_alloc(bm * (size_t)ffn * sizeof(float), 64);
    h->pf_tmp   = (float*)veritate_aligned_alloc(bm * (size_t)H * sizeof(float), 64);
    h->pf_qx    = dtype == VERITATE_HYBRID_DTYPE_INT8
                ? (int8_t*)veritate_aligned_alloc(bm * (size_t)ffn, 64) : NULL;
    if (!h->kv_k || !h->kv_v || !h->rec_state || !h->conv_ring || !h->x || !h->g ||
        !h->u || !h->qkv || !h->conv_out || !h->attn_out || !h->scores ||
        !h->ffn_buf || !h->gate_buf || !h->tmp || !h->logits ||
        !h->lens_u || !h->lens_f || !h->pf_x || !h->pf_u || !h->pf_qkv ||
        !h->pf_attn || !h->pf_ff || !h->pf_tmp ||
        (dtype == VERITATE_HYBRID_DTYPE_INT8 && !h->pf_qx)) {
        hybrid_free(h); return NULL;
    }
    hybrid_reset(h);
    hybrid_threads_init(h);
    return h;
}

void hybrid_free(hybrid_t* h) {
    if (!h) return;
    veritate_aligned_free(h->tok_emb);
    veritate_aligned_free(h->pos_emb);
    veritate_aligned_free(h->slot_pos_emb);
    if (h->blocks) {
        for (int32_t L = 0; L < h->layers; L++) {
            hybrid_block_t* b = &h->blocks[L];
            veritate_aligned_free(b->n1_w);   veritate_aligned_free(b->qkv_w);
            veritate_aligned_free(b->proj_w); veritate_aligned_free(b->n2_w);
            veritate_aligned_free(b->up_w);   veritate_aligned_free(b->down_w);
            veritate_aligned_free(b->conv_w); veritate_aligned_free(b->a_proj_w);
            veritate_aligned_free(b->a_proj_b); veritate_aligned_free(b->o_norm_w);
            veritate_aligned_free(b->gate_w);
        }
        free(h->blocks);
    }
    veritate_aligned_free(h->n_out_w);
    veritate_aligned_free(h->kv_k);      veritate_aligned_free(h->kv_v);
    veritate_aligned_free(h->rec_state); veritate_aligned_free(h->conv_ring);
    veritate_aligned_free(h->x);        veritate_aligned_free(h->g);
    veritate_aligned_free(h->u);        veritate_aligned_free(h->qkv);
    veritate_aligned_free(h->conv_out); veritate_aligned_free(h->attn_out);
    veritate_aligned_free(h->scores);   veritate_aligned_free(h->ffn_buf);
    veritate_aligned_free(h->gate_buf); veritate_aligned_free(h->tmp);
    veritate_aligned_free(h->logits);
    veritate_aligned_free(h->lens_u);   veritate_aligned_free(h->lens_f);
    veritate_aligned_free(h->pf_x);     veritate_aligned_free(h->pf_u);
    veritate_aligned_free(h->pf_qkv);   veritate_aligned_free(h->pf_attn);
    veritate_aligned_free(h->pf_ff);    veritate_aligned_free(h->pf_tmp);
    veritate_aligned_free(h->pf_qx);
    free(h);
}
