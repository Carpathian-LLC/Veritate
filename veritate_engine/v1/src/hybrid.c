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

hybrid_matvec_fn hybrid_matvec_wt = hybrid_matvec_f32_scalar;
hybrid_matvec_fn hybrid_matvec_fp = hybrid_matvec_f32_scalar;

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
    }
#endif
}

// ------------------------------------------------------------------------------------
// threaded matvec — row-split across the pool, bitwise-identical to the single
// call (each row is computed once, same kernel). only matmuls with
// n * k >= HYBRID_MT_MIN_WORK are split. VERITATE_HYBRID_THREADS overrides the
// worker count (1 = single-thread).
// ------------------------------------------------------------------------------------

#define HYBRID_MT_MIN_WORK  (1 << 20)
#define HYBRID_MT_MAX       8

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

static int32_t hybrid_threads(void) {
    static int32_t cached = 0;
    if (cached == 0) {
        int32_t cap = veritate_pool_size();
        const char* s = getenv("VERITATE_HYBRID_THREADS");
        int32_t nt = s && *s ? atoi(s) : cap;
        if (nt > cap) nt = cap;
        if (nt > HYBRID_MT_MAX) nt = HYBRID_MT_MAX;
        cached = nt < 1 ? 1 : nt;
    }
    return cached;
}

// big-weight matvec entry: picks the dtype kernel, row-splits across the pool.
// int8 spans re-derive identical qx per worker (hybrid_quant_act is
// deterministic over the shared x), so threading stays bitwise-identical.
// int8 threads only at 8x the fp work floor: the sdot kernel runs ~4x the fp
// rate, so below that pool dispatch costs more than the split saves (measured:
// 1T beats 4T for int8 at both the 121.75M and 270M shapes).
static void hybrid_mv(const hybrid_t* h, const void* w, const float* x,
                      float* out, int32_t n, int32_t k) {
    const int i8 = h->dtype == VERITATE_HYBRID_DTYPE_INT8;
    const int64_t min_work = i8 ? (int64_t)HYBRID_MT_MIN_WORK * 8 : HYBRID_MT_MIN_WORK;
    int32_t nt = hybrid_threads();
    if (nt <= 1 || (int64_t)n * k < min_work) {
        hybrid_matvec_wt(w, x, out, n, k);
        return;
    }
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
// load
// ------------------------------------------------------------------------------------

// small tensors: always upconverted to fp32.
static float* read_tensor_f32(FILE* f, size_t n, int32_t dtype) {
    float* out = (float*)veritate_aligned_alloc(n * sizeof(float), 64);
    if (!out) return NULL;
    if (dtype == VERITATE_HYBRID_DTYPE_FP32) {
        if (fread(out, sizeof(float), n, f) != n) { veritate_aligned_free(out); return NULL; }
        return out;
    }
    uint16_t* half = (uint16_t*)malloc(n * sizeof(uint16_t));
    if (!half) { veritate_aligned_free(out); return NULL; }
    if (fread(half, sizeof(uint16_t), n, f) != n) {
        free(half); veritate_aligned_free(out); return NULL;
    }
    for (size_t i = 0; i < n; i++) out[i] = hybrid_f16_to_f32(half[i]);
    free(half);
    return out;
}

// big matmul weights: kept in the bin dtype, consumed via hybrid_matvec_wt.
// int8 tensors (disk: q[n*k] then fp32 scale[n]) load into ONE aligned block
// [hybrid_w_i8_t | scales | q] so free stays a single call per tensor.
static void* read_tensor_big(FILE* f, int32_t n, int32_t k, int32_t dtype) {
    const size_t elems = (size_t)n * k;
    if (dtype != VERITATE_HYBRID_DTYPE_INT8) {
        size_t esz = dtype == VERITATE_HYBRID_DTYPE_FP16 ? sizeof(uint16_t) : sizeof(float);
        void* out = veritate_aligned_alloc(elems * esz, 64);
        if (!out) return NULL;
        if (fread(out, esz, elems, f) != elems) { veritate_aligned_free(out); return NULL; }
        return out;
    }
    size_t hdr = (sizeof(hybrid_w_i8_t) + 63) & ~(size_t)63;
    void* blk = veritate_aligned_alloc(hdr + (size_t)n * sizeof(float) + elems, 64);
    if (!blk) return NULL;
    hybrid_w_i8_t* wi = (hybrid_w_i8_t*)blk;
    float*  scale = (float*)((char*)blk + hdr);
    int8_t* q     = (int8_t*)(scale + n);
    if (fread(q, 1, elems, f) != elems ||
        fread(scale, sizeof(float), (size_t)n, f) != (size_t)n) {
        veritate_aligned_free(blk);
        return NULL;
    }
    wi->q = q;
    wi->scale = scale;
    return blk;
}

static int load_block(FILE* f, hybrid_block_t* b, int32_t H, int32_t F,
                      int32_t NH, int32_t D, int32_t CK, int32_t dtype,
                      int recurrent) {
    const int32_t sdt = dtype == VERITATE_HYBRID_DTYPE_INT8
        ? VERITATE_HYBRID_DTYPE_FP32 : dtype;
    b->is_recurrent = recurrent;
    b->n1_w  = read_tensor_f32(f, (size_t)H, sdt);
    b->qkv_w = read_tensor_big(f, 3 * H, H, dtype);
    if (recurrent) {
        b->conv_w   = read_tensor_f32(f, (size_t)3 * H * CK, sdt);
        b->a_proj_w = read_tensor_f32(f, (size_t)NH * H, sdt);
        b->a_proj_b = read_tensor_f32(f, (size_t)NH, sdt);
        b->o_norm_w = read_tensor_f32(f, (size_t)D, sdt);
        b->gate_w   = read_tensor_big(f, H, H, dtype);
        if (!b->conv_w || !b->a_proj_w || !b->a_proj_b || !b->o_norm_w || !b->gate_w)
            return -1;
    }
    b->proj_w = read_tensor_big(f, H, H, dtype);
    b->n2_w   = read_tensor_f32(f, (size_t)H, sdt);
    b->up_w   = read_tensor_big(f, F, H, dtype);
    b->down_w = read_tensor_big(f, H, F, dtype);
    if (!b->n1_w || !b->qkv_w || !b->proj_w || !b->n2_w || !b->up_w || !b->down_w)
        return -1;
    return 0;
}

hybrid_t* hybrid_load(FILE* f, int32_t vocab, int32_t hidden, int32_t layers,
                      int32_t ffn, int32_t heads, int32_t seq) {
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

    const int32_t H = hidden, D = h->head_dim;
    h->tok_emb      = read_tensor_f32(f, (size_t)vocab * H, sdt);
    h->pos_emb      = read_tensor_f32(f, (size_t)seq * H, sdt);
    h->slot_pos_emb = read_tensor_f32(f, (size_t)slots * H, sdt);
    h->blocks = (hybrid_block_t*)calloc((size_t)layers, sizeof(hybrid_block_t));
    if (!h->tok_emb || !h->pos_emb || !h->slot_pos_emb || !h->blocks) {
        hybrid_free(h); return NULL;
    }
    for (int32_t L = 0; L < layers; L++) {
        int recurrent = (L >= n_enc && L < n_enc + n_global);
        if (load_block(f, &h->blocks[L], H, ffn, heads, D, ck, dtype, recurrent) != 0) {
            fprintf(stderr, "hybrid_load: truncated at block %d\n", L);
            hybrid_free(h); return NULL;
        }
    }
    h->n_out_w = read_tensor_f32(f, (size_t)H, sdt);
    if (!h->n_out_w) { hybrid_free(h); return NULL; }

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
    if (!h->kv_k || !h->kv_v || !h->rec_state || !h->conv_ring || !h->x || !h->g ||
        !h->u || !h->qkv || !h->conv_out || !h->attn_out || !h->scores ||
        !h->ffn_buf || !h->gate_buf || !h->tmp || !h->logits ||
        !h->lens_u || !h->lens_f) {
        hybrid_free(h); return NULL;
    }
    hybrid_reset(h);
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
    free(h);
}
