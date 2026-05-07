// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - single-block transformer forward pass. int8 throughout, fp32 only inside softmax
//   and layernorm stats. random weights for v3.
// ------------------------------------------------------------------------------------

#include "veritate.h"
#include "portability.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// per-pair attn_dot_inline / attn_hsum_inline live in a per-arch header. these
// helpers sit inside the inner attention loop; function-pointer overhead would
// dominate, so the dispatch happens at compile time. only the two supported
// archs ship inline headers; new arches add a sibling header and an #elif.
#if defined(__x86_64__) || defined(_M_X64)
    #include "../kernels/inline/attn_x86_64.h"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include "../kernels/inline/attn_arm64.h"
#else
    #error "veritate model.c: unsupported arch. add an inline attn header."
#endif

// ------------------------------------------------------------------------------------
// runtime layer cap from env, clamped per-model
// ------------------------------------------------------------------------------------

int32_t veritate_max_layers(const model_t* m) {
    static int32_t cached = -1;
    if (cached < 0) {
        const char* s = getenv("VERITATE_MAX_LAYERS");
        cached = s ? atoi(s) : 0x7fffffff;
        if (cached < 1) cached = 1;
    }
    int32_t cap = cached;
    if (cap > m->shape.layers) cap = m->shape.layers;
    return cap;
}

// ------------------------------------------------------------------------------------
// fast deterministic PRNG (xorshift) for seeded random init and tests
// ------------------------------------------------------------------------------------

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static uint32_t rng_state = 1;
static void rng_seed(unsigned s) { rng_state = s ? s : 1; }
static uint32_t rng_next(void) { return xorshift32(&rng_state); }

// ------------------------------------------------------------------------------------
// activation buffer pools — one prefill + one decode + one verify pool per model.
// allocated once at model_load based on shape, freed at model_free.
// ------------------------------------------------------------------------------------

typedef struct {
    int16_t* act;            // [seq * hidden]
    int8_t*  act_norm;       // [seq * hidden]
    int8_t*  act_norm_rot;   // [seq * hidden]
    int8_t*  ffn_up8_rot;    // [seq * ffn]
    int32_t* qkv32;          // [seq * 3 * hidden]
    int8_t*  qkv8;           // [seq * 3 * hidden]
    float*   scores;         // [heads * seq * seq]
    int16_t* scores_q;       // [heads * seq * seq]
    int8_t*  attn8;          // [seq * hidden]
    int32_t* out32;          // [seq * hidden]
    int32_t* ffn_up32;       // [seq * ffn]
    int8_t*  ffn_up8;        // [seq * ffn]
    int32_t* ffn_down32;     // [seq * hidden]
} acts_t;

typedef struct {
    int16_t* act;          // [hidden]
    int8_t*  act_norm;     // [hidden]
    int8_t*  act_norm_rot; // [hidden]
    int8_t*  ffn_up8_rot;  // [ffn]
    int32_t* qkv32;        // [3 * hidden]
    int8_t*  qkv8;         // [3 * hidden]
    int8_t*  attn8;        // [hidden]
    int32_t* out32;        // [hidden]
    int32_t* ffn_up32;     // [ffn]
    int8_t*  ffn_up8;      // [ffn]
    int32_t* ffn_down32;   // [hidden]
    float*   scores;       // [seq]
    int16_t* scores_q;     // [seq]
} decode_acts_t;

typedef struct {
    int16_t* act;       // [K * hidden]
    int8_t*  act_norm;  // [K * hidden]
    int32_t* qkv32;     // [K * 3 * hidden]
    int8_t*  qkv8;      // [K * 3 * hidden]
    int8_t*  attn8;     // [K * hidden]
    int32_t* out32;     // [K * hidden]
    int32_t* ffn_up32;  // [K * ffn]
    int8_t*  ffn_up8;   // [K * ffn]
    int32_t* ffn_dn32;  // [K * hidden]
    float*   scores;    // [seq]
    int16_t* scores_q;  // [seq]
} verify_acts_t;

typedef struct {
    acts_t        prefill;
    decode_acts_t decode;
    verify_acts_t verify;
} acts_pool_t;

static void* xalloc64(size_t bytes) {
    void* p = veritate_aligned_alloc(bytes, 64);
    if (p) memset(p, 0, bytes);
    return p;
}

static void acts_alloc(acts_t* a, const veritate_shape_t* s) {
    size_t sh   = (size_t)s->seq * s->hidden;
    size_t sf   = (size_t)s->seq * s->ffn;
    size_t s3h  = sh * 3;
    size_t hss  = (size_t)s->heads * s->seq * s->seq;
    a->act           = (int16_t*)xalloc64(sh * sizeof(int16_t));
    a->act_norm      = (int8_t*) xalloc64(sh);
    a->act_norm_rot  = (int8_t*) xalloc64(sh);
    a->ffn_up8_rot   = (int8_t*) xalloc64(sf);
    a->qkv32         = (int32_t*)xalloc64(s3h * sizeof(int32_t));
    a->qkv8          = (int8_t*) xalloc64(s3h);
    a->scores        = (float*)  xalloc64(hss * sizeof(float));
    a->scores_q      = (int16_t*)xalloc64(hss * sizeof(int16_t));
    a->attn8         = (int8_t*) xalloc64(sh);
    a->out32         = (int32_t*)xalloc64(sh * sizeof(int32_t));
    a->ffn_up32      = (int32_t*)xalloc64(sf * sizeof(int32_t));
    a->ffn_up8       = (int8_t*) xalloc64(sf);
    a->ffn_down32    = (int32_t*)xalloc64(sh * sizeof(int32_t));
}

static void decode_acts_alloc(decode_acts_t* d, const veritate_shape_t* s) {
    d->act          = (int16_t*)xalloc64((size_t)s->hidden * sizeof(int16_t));
    d->act_norm     = (int8_t*) xalloc64((size_t)s->hidden);
    d->act_norm_rot = (int8_t*) xalloc64((size_t)s->hidden);
    d->ffn_up8_rot  = (int8_t*) xalloc64((size_t)s->ffn);
    d->qkv32        = (int32_t*)xalloc64((size_t)3 * s->hidden * sizeof(int32_t));
    d->qkv8         = (int8_t*) xalloc64((size_t)3 * s->hidden);
    d->attn8        = (int8_t*) xalloc64((size_t)s->hidden);
    d->out32        = (int32_t*)xalloc64((size_t)s->hidden * sizeof(int32_t));
    d->ffn_up32     = (int32_t*)xalloc64((size_t)s->ffn * sizeof(int32_t));
    d->ffn_up8      = (int8_t*) xalloc64((size_t)s->ffn);
    d->ffn_down32   = (int32_t*)xalloc64((size_t)s->hidden * sizeof(int32_t));
    d->scores       = (float*)  xalloc64((size_t)s->seq * sizeof(float));
    d->scores_q     = (int16_t*)xalloc64((size_t)s->seq * sizeof(int16_t));
}

static void verify_acts_alloc(verify_acts_t* v, const veritate_shape_t* s) {
    size_t Kh  = (size_t)VERITATE_VERIFY_K_MAX * s->hidden;
    size_t K3h = Kh * 3;
    size_t Kf  = (size_t)VERITATE_VERIFY_K_MAX * s->ffn;
    v->act       = (int16_t*)xalloc64(Kh * sizeof(int16_t));
    v->act_norm  = (int8_t*) xalloc64(Kh);
    v->qkv32     = (int32_t*)xalloc64(K3h * sizeof(int32_t));
    v->qkv8      = (int8_t*) xalloc64(K3h);
    v->attn8     = (int8_t*) xalloc64(Kh);
    v->out32     = (int32_t*)xalloc64(Kh * sizeof(int32_t));
    v->ffn_up32  = (int32_t*)xalloc64(Kf * sizeof(int32_t));
    v->ffn_up8   = (int8_t*) xalloc64(Kf);
    v->ffn_dn32  = (int32_t*)xalloc64(Kh * sizeof(int32_t));
    v->scores    = (float*)  xalloc64((size_t)s->seq * sizeof(float));
    v->scores_q  = (int16_t*)xalloc64((size_t)s->seq * sizeof(int16_t));
}

static void acts_free(acts_t* a) {
    veritate_aligned_free(a->act); veritate_aligned_free(a->act_norm); veritate_aligned_free(a->act_norm_rot);
    veritate_aligned_free(a->ffn_up8_rot); veritate_aligned_free(a->qkv32); veritate_aligned_free(a->qkv8);
    veritate_aligned_free(a->scores); veritate_aligned_free(a->scores_q); veritate_aligned_free(a->attn8);
    veritate_aligned_free(a->out32); veritate_aligned_free(a->ffn_up32); veritate_aligned_free(a->ffn_up8);
    veritate_aligned_free(a->ffn_down32);
}
static void decode_acts_free(decode_acts_t* d) {
    veritate_aligned_free(d->act); veritate_aligned_free(d->act_norm); veritate_aligned_free(d->act_norm_rot);
    veritate_aligned_free(d->ffn_up8_rot); veritate_aligned_free(d->qkv32); veritate_aligned_free(d->qkv8);
    veritate_aligned_free(d->attn8); veritate_aligned_free(d->out32); veritate_aligned_free(d->ffn_up32);
    veritate_aligned_free(d->ffn_up8); veritate_aligned_free(d->ffn_down32);
    veritate_aligned_free(d->scores); veritate_aligned_free(d->scores_q);
}
static void verify_acts_free(verify_acts_t* v) {
    veritate_aligned_free(v->act); veritate_aligned_free(v->act_norm); veritate_aligned_free(v->qkv32);
    veritate_aligned_free(v->qkv8); veritate_aligned_free(v->attn8); veritate_aligned_free(v->out32);
    veritate_aligned_free(v->ffn_up32); veritate_aligned_free(v->ffn_up8); veritate_aligned_free(v->ffn_dn32);
    veritate_aligned_free(v->scores); veritate_aligned_free(v->scores_q);
}

static acts_pool_t* pool_alloc(const veritate_shape_t* s) {
    acts_pool_t* p = (acts_pool_t*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    acts_alloc(&p->prefill, s);
    decode_acts_alloc(&p->decode, s);
    verify_acts_alloc(&p->verify, s);
    return p;
}

static void pool_free(acts_pool_t* p) {
    if (!p) return;
    acts_free(&p->prefill);
    decode_acts_free(&p->decode);
    verify_acts_free(&p->verify);
    free(p);
}

// ------------------------------------------------------------------------------------
// kv cache lifecycle
// ------------------------------------------------------------------------------------

void kv_cache_init(kv_cache_t* c, const veritate_shape_t* s) {
    c->shape = *s;
    size_t total = (size_t)s->layers * s->seq * s->hidden;
    c->k = (int8_t*)xalloc64(total);
    c->v = (int8_t*)xalloc64(total);
    c->len = 0;
}

void kv_cache_free(kv_cache_t* c) {
    if (c->k) { veritate_aligned_free(c->k); c->k = NULL; }
    if (c->v) { veritate_aligned_free(c->v); c->v = NULL; }
    c->len = 0;
}

void kv_cache_copy(kv_cache_t* dst, const kv_cache_t* src) {
    if (!dst->k || dst->shape.layers != src->shape.layers ||
        dst->shape.seq    != src->shape.seq ||
        dst->shape.hidden != src->shape.hidden) {
        kv_cache_free(dst);
        kv_cache_init(dst, &src->shape);
    }
    size_t total = (size_t)src->shape.layers * src->shape.seq * src->shape.hidden;
    memcpy(dst->k, src->k, total);
    memcpy(dst->v, src->v, total);
    dst->len = src->len;
}

// ------------------------------------------------------------------------------------
// helper — apply per-row hadamard rotation to a batch of int8 rows in place.
// ------------------------------------------------------------------------------------

static void hadamard_rotate_rows(const int8_t* src, int8_t* dst, int32_t rows, int32_t cols) {
    for (int32_t r = 0; r < rows; r++) {
        hadamard_apply_int8(src + (size_t)r * cols, dst + (size_t)r * cols, cols);
    }
}

// multi-row int4 matmul wrapper. m=1 inner kernel called per row.
static void matmul_int4_multi(const int8_t* a, const prepped_b_int4_t* p,
                              int32_t* c, int32_t m) {
    for (int32_t r = 0; r < m; r++) {
        matmul_int4_vnni_prep(a + (size_t)r * p->k, p, c + (size_t)r * p->n, 1);
    }
}

// ------------------------------------------------------------------------------------
// saturating casts
// ------------------------------------------------------------------------------------

static inline int8_t sat_int8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static inline int16_t sat_int16(int32_t v) {
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

static inline int32_t requant(int32_t v, int32_t scale_q24) {
    return (int32_t)(((int64_t)v * scale_q24 + (1 << 23)) >> 24);
}

static inline int32_t requant_pb(int32_t v, const prepped_b_t* p, int32_t j) {
    int32_t s = p->scale_per_col ? p->scale_per_col[j] : p->scale_q24;
    return (int32_t)(((int64_t)v * s + (1 << 23)) >> 24);
}

// ------------------------------------------------------------------------------------
// mod gate — per-block scalar dot product. residual int16 dotted with int8 gate row,
// requanted to int8, sigmoid sign check (>=0 -> keep). counters are global and visible
// to callers via veritate_mod_stats_*.
// ------------------------------------------------------------------------------------

int64_t g_mod_gate_calls   = 0;
int64_t g_mod_gate_skipped = 0;

void veritate_mod_stats_reset(void) {
    g_mod_gate_calls   = 0;
    g_mod_gate_skipped = 0;
}

void veritate_mod_stats(int64_t* calls, int64_t* skipped) {
    if (calls)   *calls   = g_mod_gate_calls;
    if (skipped) *skipped = g_mod_gate_skipped;
}

// ablation state. -1 means inactive. read in forward_decode after gelu_int8;
// when active, zeros ffn_neurons[layer][pos][neuron] before trace capture and
// before ffn_down so the ablation is visible everywhere downstream.
int32_t g_ablate_layer  = -1;
int32_t g_ablate_neuron = -1;

void veritate_set_ablation(int32_t layer, int32_t neuron) {
    g_ablate_layer  = layer;
    g_ablate_neuron = neuron;
}

void veritate_get_ablation(int32_t* layer, int32_t* neuron) {
    if (layer)  *layer  = g_ablate_layer;
    if (neuron) *neuron = g_ablate_neuron;
}

// returns 1 if the gate keeps this token (run block), 0 to skip.
// residual is int16 [hidden]; gate_w is int8 [hidden]; result int8 in [-128,127].
// sign of the requanted int8 substitutes for sigmoid >= 0.5 (since sigmoid(x) >= 0.5
// iff x >= 0).
// VERITATE_MOD_OFF env var forces every gate to keep, for A/B benchmarking from the
// same bin. read once and cached.
static int mod_off_cached = -1;
static int gate_off(void) {
    if (mod_off_cached < 0) {
        const char* s = getenv("VERITATE_MOD_OFF");
        mod_off_cached = (s && *s && *s != '0') ? 1 : 0;
    }
    return mod_off_cached;
}

static inline int gate_keep(const int16_t* residual, const int8_t* gate_w,
                            int32_t hidden, int32_t scale_q24) {
    if (gate_off()) return 1;
    int64_t s = 0;
    for (int32_t i = 0; i < hidden; i++) {
        s += (int64_t)residual[i] * (int64_t)gate_w[i];
    }
    int32_t r = (int32_t)((s * scale_q24 + (1 << 23)) >> 24);
    return r >= 0 ? 1 : 0;
}

// ------------------------------------------------------------------------------------
// activation lut on int8. override via VERITATE_ACTIVATION_LUT.
// ------------------------------------------------------------------------------------

static int8_t lut_gelu[256] = {
       0,    1,    1,    2,    2,    3,    3,    4,    5,    5,    6,    7,    8,    9,    9,   10,
      11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
      27,   28,   29,   30,   31,   32,   34,   35,   36,   37,   38,   39,   40,   41,   43,   44,
      45,   46,   47,   48,   49,   50,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,
      63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,
      80,   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,
      96,   97,   98,   99,  100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,
     112,  113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,  124,  125,  126,  127,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
       0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
      -1,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -3,   -3,   -3,   -3,   -3,
      -3,   -3,   -3,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -4,   -5,   -5,   -5,   -5,   -5,
      -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,   -5,
      -5,   -5,   -5,   -4,   -4,   -4,   -4,   -4,   -3,   -3,   -3,   -2,   -2,   -1,   -1,    0,
};

#ifndef VERITATE_GELU_ZERO_THRESH
#define VERITATE_GELU_ZERO_THRESH 0
#endif

// load lut from VERITATE_ACTIVATION_LUT once. silent no-op when unset or unreadable.
static void lut_init_once(void) {
    static int32_t done = 0;
    if (done) return;
    done = 1;
    const char* path = getenv("VERITATE_ACTIVATION_LUT");
    if (!path || !*path) return;
    FILE* f = fopen(path, "rb");
    if (!f) return;
    int8_t buf[256];
    size_t got = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    if (got == sizeof(buf)) memcpy(lut_gelu, buf, sizeof(buf));
}

static void gelu_int8(int8_t* x, int32_t n) {
    lut_init_once();
    const int8_t thr = VERITATE_GELU_ZERO_THRESH;
    for (int32_t i = 0; i < n; i++) {
        int8_t v = lut_gelu[(uint8_t)x[i]];
        if (thr > 0 && v < thr && v > -thr) v = 0;
        x[i] = v;
    }
}

// ------------------------------------------------------------------------------------
// attention — multi-head self-attention
// ------------------------------------------------------------------------------------

static void attention(const veritate_shape_t* sh, const block_t* blk, acts_t* acts, profile_t* prof) {
    const int32_t S = sh->seq, H = sh->hidden, NH = sh->heads, HD = sh->head_dim;
    double tp;
    tp = prof ? now_ms() : 0;
    if (blk->use_int4) {
        hadamard_rotate_rows(acts->act_norm, acts->act_norm_rot, S, H);
        matmul_int4_multi(acts->act_norm_rot, &blk->qkv_i4, acts->qkv32, S);
        for (int32_t r = 0; r < S; r++) {
            int32_t* row32 = acts->qkv32 + (size_t)r * 3 * H;
            int8_t*  row8  = acts->qkv8  + (size_t)r * 3 * H;
            for (int32_t i = 0; i < 3 * H; i++) {
                row8[i] = sat_int8(requant(row32[i], blk->qkv_i4.row_q24[i]));
            }
        }
    } else {
        matmul_int8_vnni_mt_prep(acts->act_norm, &blk->qkv, acts->qkv32, S);
        const int32_t cols = 3 * H;
        for (int32_t r = 0; r < S; r++) {
            int32_t* row32 = acts->qkv32 + (size_t)r * cols;
            int8_t*  row8  = acts->qkv8  + (size_t)r * cols;
            for (int32_t j = 0; j < cols; j++) {
                row8[j] = sat_int8(requant_pb(row32[j], &blk->qkv, j));
            }
        }
    }
    if (prof) prof->qkv_ms += now_ms() - tp;

    const int32_t qkv_stride = 3 * H;
    const float   scale      = 1.0f / (sqrtf((float)HD) * 1024.0f);

    tp = prof ? now_ms() : 0;
    for (int32_t h = 0; h < NH; h++) {
        float*   head_scores   = acts->scores   + (size_t)h * S * S;
        int16_t* head_scores_q = acts->scores_q + (size_t)h * S * S;
        const int8_t* v_base   = acts->qkv8 + 2 * H + h * HD;

        for (int32_t i = 0; i < S; i++) {
            const int8_t* q_row = acts->qkv8 + (size_t)qkv_stride * i + h * HD;
            int32_t q_sum = attn_hsum_inline(q_row);
            float*   row   = head_scores   + (size_t)i * S;
            int16_t* row_q = head_scores_q + (size_t)i * S;
            int32_t j = 0;
            int32_t end = i + 1;
            for (; j + 4 <= end; j += 4) {
                int32_t out4[4];
                attn_dot_inline_4(q_row,
                    acts->qkv8 + (size_t)qkv_stride * (j + 0) + H + h * HD,
                    acts->qkv8 + (size_t)qkv_stride * (j + 1) + H + h * HD,
                    acts->qkv8 + (size_t)qkv_stride * (j + 2) + H + h * HD,
                    acts->qkv8 + (size_t)qkv_stride * (j + 3) + H + h * HD,
                    q_sum, out4);
                row[j + 0] = (float)out4[0] * scale;
                row[j + 1] = (float)out4[1] * scale;
                row[j + 2] = (float)out4[2] * scale;
                row[j + 3] = (float)out4[3] * scale;
            }
            for (; j < end; j++) {
                const int8_t* k_row = acts->qkv8 + (size_t)qkv_stride * j + H + h * HD;
                row[j] = (float)attn_dot_inline(q_row, k_row, q_sum) * scale;
            }
            softmax_rows(row, row_q, 1, i + 1);
            int8_t* out_row = acts->attn8 + (size_t)i * H + h * HD;
            score_dot_v(row_q, v_base, qkv_stride, i + 1, out_row);
        }
    }
    if (prof) prof->attn_ms += now_ms() - tp;

    tp = prof ? now_ms() : 0;
    if (blk->use_int4) {
        hadamard_rotate_rows(acts->attn8, acts->act_norm_rot, S, H);
        matmul_int4_multi(acts->act_norm_rot, &blk->out_proj_i4, acts->out32, S);
    } else {
        matmul_int8_vnni_mt_prep(acts->attn8, &blk->out_proj, acts->out32, S);
    }
    if (prof) prof->out_proj_ms += now_ms() - tp;
}

// ------------------------------------------------------------------------------------
// feed-forward network — 2 matmuls + GELU
// ------------------------------------------------------------------------------------

static void ffn(const veritate_shape_t* sh, const block_t* blk, acts_t* acts, profile_t* prof) {
    const int32_t S = sh->seq, H = sh->hidden, F = sh->ffn;
    double tp;
    tp = prof ? now_ms() : 0;
    if (blk->use_int4) {
        hadamard_rotate_rows(acts->act_norm, acts->act_norm_rot, S, H);
        matmul_int4_multi(acts->act_norm_rot, &blk->ffn_up_i4, acts->ffn_up32, S);
        for (int32_t r = 0; r < S; r++) {
            int32_t* row32 = acts->ffn_up32 + (size_t)r * F;
            int8_t*  row8  = acts->ffn_up8  + (size_t)r * F;
            for (int32_t i = 0; i < F; i++) {
                row8[i] = sat_int8(requant(row32[i], blk->ffn_up_i4.row_q24[i]));
            }
        }
    } else {
        matmul_int8_vnni_mt_prep(acts->act_norm, &blk->ffn_up, acts->ffn_up32, S);
        for (int32_t r = 0; r < S; r++) {
            int32_t* row32 = acts->ffn_up32 + (size_t)r * F;
            int8_t*  row8  = acts->ffn_up8  + (size_t)r * F;
            for (int32_t j = 0; j < F; j++) {
                row8[j] = sat_int8(requant_pb(row32[j], &blk->ffn_up, j));
            }
        }
    }
    if (prof) prof->ffn_up_ms += now_ms() - tp;

    tp = prof ? now_ms() : 0;
    gelu_int8(acts->ffn_up8, S * F);
    if (prof) prof->gelu_ms += now_ms() - tp;

    tp = prof ? now_ms() : 0;
    if (blk->use_int4) {
        hadamard_rotate_rows(acts->ffn_up8, acts->ffn_up8_rot, S, F);
        matmul_int4_multi(acts->ffn_up8_rot, &blk->ffn_down_i4, acts->ffn_down32, S);
    } else {
        matmul_int8_vnni_mt_prep(acts->ffn_up8, &blk->ffn_down, acts->ffn_down32, S);
    }
    if (prof) prof->ffn_down_ms += now_ms() - tp;
}

// ------------------------------------------------------------------------------------
// byte_direction — per-layer (V_FFN, V_VOCAB) table. entry [n, v] is the contribution
// of FFN neuron n at layer L to byte v's logit per unit of post-GELU activation.
// = sum_h W_ffn_down[n, h] * W_embed[v, h], in fp32.
// stored quantized to int16 with one fp32 scale per layer (bit-budget vs accuracy).
// ------------------------------------------------------------------------------------

int byte_direction_build(model_t* m) {
    const int32_t V = m->shape.vocab, H = m->shape.hidden, F = m->shape.ffn, L_n = m->shape.layers;
    for (int32_t L = 0; L < L_n; L++) m->byte_direction[L] = NULL;

    float* tmp = (float*)malloc((size_t)F * V * sizeof(float));
    if (!tmp) return -1;

    for (int32_t L = 0; L < L_n; L++) {
        const block_t* blk = &m->blocks[L];
        const int8_t* w_down = blk->ffn_down.b_rowmaj;
        if (!w_down) {
            // int4 path or weights not retained — skip table; ranking will be unavailable.
            free(tmp);
            return -1;
        }

        // BD[n, v] = sum_h w_down[n*hidden + h] * embed[v*hidden + h]
        float max_abs = 0.0f;
        for (int32_t n = 0; n < F; n++) {
            const int8_t* row_n = w_down + (size_t)n * H;
            for (int32_t v = 0; v < V; v++) {
                const int8_t* row_v = m->embed + (size_t)v * H;
                int32_t s = 0;
                for (int32_t h = 0; h < H; h++) {
                    s += (int32_t)row_n[h] * (int32_t)row_v[h];
                }
                float f = (float)s;
                tmp[n * V + v] = f;
                float a = f < 0 ? -f : f;
                if (a > max_abs) max_abs = a;
            }
        }

        float scale = max_abs > 0.0f ? max_abs / 32767.0f : 1.0f;
        m->byte_direction_scale[L] = scale;
        m->byte_direction[L] = (int16_t*)malloc((size_t)F * V * sizeof(int16_t));
        if (!m->byte_direction[L]) { free(tmp); return -1; }
        const float inv_scale = 1.0f / scale;
        for (size_t i = 0; i < (size_t)F * V; i++) {
            float q = tmp[i] * inv_scale;
            if (q >  32767.0f) q =  32767.0f;
            if (q < -32768.0f) q = -32768.0f;
            m->byte_direction[L][i] = (int16_t)(q < 0 ? q - 0.5f : q + 0.5f);
        }
    }

    free(tmp);
    return 0;
}

void byte_direction_free(model_t* m) {
    if (!m->byte_direction) return;
    for (int32_t L = 0; L < m->shape.layers; L++) {
        if (m->byte_direction[L]) {
            free(m->byte_direction[L]);
            m->byte_direction[L] = NULL;
        }
    }
}

// ------------------------------------------------------------------------------------
// decisiveness — per-layer max_abs/mean_abs of lens-logit deltas at one position.
// ------------------------------------------------------------------------------------

void decisiveness_compute(const veritate_shape_t* sh, const int32_t* lens_logits_pos, float* out) {
    const int32_t Ln = sh->layers, V = sh->vocab;
    for (int32_t L = 0; L < Ln; L++) {
        const int32_t* cur  = lens_logits_pos + (size_t)L * V;
        const int32_t* prev = (L == 0) ? NULL : lens_logits_pos + (size_t)(L - 1) * V;
        double max_a = 0.0, sum_a = 0.0;
        for (int32_t v = 0; v < V; v++) {
            int64_t d = prev ? ((int64_t)cur[v] - (int64_t)prev[v]) : (int64_t)cur[v];
            double a = d < 0 ? -(double)d : (double)d;
            if (a > max_a) max_a = a;
            sum_a += a;
        }
        double mean_a = sum_a / (double)V;
        out[L] = mean_a > 1e-8 ? (float)(max_a / mean_a) : 0.0f;
    }
}

// ------------------------------------------------------------------------------------
// dla_top — top-K (layer, neuron) by |contrib| for a target byte.
// contrib = ffn_neurons[L][n] (int8) * byte_direction[L][n][byte] (int16) -> int32.
// ------------------------------------------------------------------------------------

void dla_top(const model_t* m, const int8_t* ffn_neurons_pos, int32_t byte, dla_entry_t* out) {
    const int32_t K = VERITATE_DLA_TOPK;
    int32_t heap_abs[VERITATE_DLA_TOPK];
    dla_entry_t heap[VERITATE_DLA_TOPK];
    for (int32_t i = 0; i < K; i++) heap_abs[i] = -1;

    const int32_t V = m->shape.vocab, F = m->shape.ffn, Ln = m->shape.layers;
    int32_t bv = byte;
    if (V > 0) bv = ((bv % V) + V) % V;

    for (int32_t L = 0; L < Ln; L++) {
        const int8_t*  acts = ffn_neurons_pos + (size_t)L * F;
        const int16_t* bd   = m->byte_direction[L];
        if (!bd) continue;
        for (int32_t n = 0; n < F; n++) {
            int32_t a  = (int32_t)acts[n];
            int32_t w  = (int32_t)bd[(size_t)n * V + bv];
            int32_t c  = a * w;
            int32_t ca = c < 0 ? -c : c;

            // min-heap of |contrib|; replace root if larger.
            if (ca <= heap_abs[0]) continue;
            heap_abs[0] = ca;
            heap[0].layer   = (uint8_t)L;
            heap[0].pad     = 0;
            heap[0].neuron  = (uint16_t)n;
            heap[0].act     = a;
            heap[0].w       = w;
            heap[0].contrib = c;
            int32_t p = 0;
            while (1) {
                int32_t l = 2 * p + 1, r = 2 * p + 2, s = p;
                if (l < K && heap_abs[l] < heap_abs[s]) s = l;
                if (r < K && heap_abs[r] < heap_abs[s]) s = r;
                if (s == p) break;
                int32_t   tmp_a = heap_abs[p]; heap_abs[p] = heap_abs[s]; heap_abs[s] = tmp_a;
                dla_entry_t te  = heap[p];     heap[p]     = heap[s];     heap[s]     = te;
                p = s;
            }
        }
    }

    // selection-sort heap descending by |contrib|.
    for (int32_t i = 0; i < K; i++) {
        int32_t best = i;
        for (int32_t j = i + 1; j < K; j++) {
            if (heap_abs[j] > heap_abs[best]) best = j;
        }
        if (best != i) {
            int32_t   tmp_a = heap_abs[i]; heap_abs[i] = heap_abs[best]; heap_abs[best] = tmp_a;
            dla_entry_t te  = heap[i];     heap[i]     = heap[best];     heap[best]     = te;
        }
        out[i] = heap[i];
    }
}

// ------------------------------------------------------------------------------------
// logit lens — project an int16 residual row through the int8 embed matrix.
// out gets V_VOCAB int32 dot products. shape: embed[V_VOCAB][V_HIDDEN] @ residual[V_HIDDEN].
// ------------------------------------------------------------------------------------

static void lens_project(const veritate_shape_t* sh, const int8_t* embed,
                         const int16_t* residual, int32_t* out) {
    const int32_t V = sh->vocab, H = sh->hidden;
    for (int32_t v = 0; v < V; v++) {
        const int8_t* row = embed + (size_t)v * H;
        int32_t s = 0;
        for (int32_t h = 0; h < H; h++) s += (int32_t)residual[h] * (int32_t)row[h];
        out[v] = s;
    }
}

// ------------------------------------------------------------------------------------
// forward pass — embed, transformer block, write last-position activation
// ------------------------------------------------------------------------------------

void forward(const model_t* m, kv_cache_t* cache, const int32_t* tokens,
             int32_t real_len, int8_t* out_act, trace_record_t* trace, profile_t* prof) {
    const veritate_shape_t* sh = &m->shape;
    const int32_t S = sh->seq, H = sh->hidden, F = sh->ffn, V = sh->vocab, NH = sh->heads;
    acts_t* a = &((acts_pool_t*)m->scratch)->prefill;
    const size_t per_layer_residual = (size_t)S * H;
    const size_t per_layer_ffn      = (size_t)S * F;
    const size_t per_layer_attn     = (size_t)NH * S * S;

    double tp = prof ? now_ms() : 0;
    for (int32_t i = 0; i < S; i++) {
        int32_t tok = (i < real_len) ? tokens[i] : 0;
        if (V > 0) tok = ((tok % V) + V) % V;
        const int8_t* tok_row = m->embed     + (size_t)tok * H;
        const int8_t* pos_row = m->pos_embed + (size_t)i   * H;
        int16_t* dst = a->act + (size_t)i * H;
        for (int32_t j = 0; j < H; j++) {
            dst[j] = (int16_t)((int32_t)tok_row[j] + (int32_t)pos_row[j]);
        }
    }
    if (prof) prof->embed_ms += now_ms() - tp;

    static int8_t mod_keep_row[V_SEQ];
    const int32_t n_layers = veritate_max_layers(m);
    for (int32_t L = 0; L < n_layers; L++) {
        const block_t* blk = &m->blocks[L];

        if (trace) {
            memcpy(trace->residual_pre + L * per_layer_residual, a->act, per_layer_residual * sizeof(int16_t));
        }

        // mod gate evaluation per row. snapshot pre-block residual when any row may skip.
        int any_skip = 0;
        if (blk->has_gate) {
            for (int32_t r = 0; r < real_len; r++) {
                int k = gate_keep(a->act + (size_t)r * H, blk->gate_w, H, blk->gate_scale_q24);
                mod_keep_row[r] = (int8_t)k;
                g_mod_gate_calls++;
                if (!k) { g_mod_gate_skipped++; any_skip = 1; }
            }
            for (int32_t r = real_len; r < S; r++) mod_keep_row[r] = 1;
        }
        // pre-block residual snapshot, used to roll back skipped rows after the block math.
        int16_t* pre_snapshot = NULL;
        if (any_skip) {
            pre_snapshot = (int16_t*)malloc(per_layer_residual * sizeof(int16_t));
            memcpy(pre_snapshot, a->act, per_layer_residual * sizeof(int16_t));
        }

        tp = prof ? now_ms() : 0;
        layernorm_i16_to_i8(a->act, a->act_norm, blk->ln1_w, S, H);
        if (prof) prof->ln_ms += now_ms() - tp;

        attention(sh, blk, a, prof);

        if (trace && trace->attention_scores) {
            float* td = trace->attention_scores + L * per_layer_attn;
            memset(td, 0, per_layer_attn * sizeof(float));
            const float inv32k = 1.0f / 32768.0f;
            for (int32_t h = 0; h < NH; h++) {
                for (int32_t i = 0; i < S; i++) {
                    float*   td_row = td         + ((size_t)h * S + i) * S;
                    int16_t* sq_row = a->scores_q + ((size_t)h * S + i) * S;
                    for (int32_t j = 0; j <= i; j++) td_row[j] = (float)sq_row[j] * inv32k;
                }
            }
        }

        for (int32_t p = 0; p < real_len; p++) {
            memcpy(cache_k_row(cache, L, p), a->qkv8 + (size_t)(3 * H) * p + H,     H);
            memcpy(cache_v_row(cache, L, p), a->qkv8 + (size_t)(3 * H) * p + 2 * H, H);
        }

        if (blk->use_int4) {
            for (int32_t r = 0; r < S; r++) {
                for (int32_t i = 0; i < H; i++) {
                    int32_t idx = r * H + i;
                    a->act[idx] = sat_int16((int32_t)a->act[idx] +
                        requant(a->out32[idx], blk->out_proj_i4.row_q24[i]) * m->act_boost);
                }
            }
        } else {
            for (int32_t r = 0; r < S; r++) {
                for (int32_t j = 0; j < H; j++) {
                    int32_t idx = r * H + j;
                    a->act[idx] = sat_int16((int32_t)a->act[idx] +
                        requant_pb(a->out32[idx], &blk->out_proj, j) * m->act_boost);
                }
            }
        }

        tp = prof ? now_ms() : 0;
        layernorm_i16_to_i8(a->act, a->act_norm, blk->ln2_w, S, H);
        if (prof) prof->ln_ms += now_ms() - tp;

        ffn(sh, blk, a, prof);

        if (trace) {
            memcpy(trace->ffn_neurons + L * per_layer_ffn, a->ffn_up8, per_layer_ffn);
        }

        if (blk->use_int4) {
            for (int32_t r = 0; r < S; r++) {
                for (int32_t i = 0; i < H; i++) {
                    int32_t idx = r * H + i;
                    a->act[idx] = sat_int16((int32_t)a->act[idx] +
                        requant(a->ffn_down32[idx], blk->ffn_down_i4.row_q24[i]) * m->act_boost);
                }
            }
        } else {
            for (int32_t r = 0; r < S; r++) {
                for (int32_t j = 0; j < H; j++) {
                    int32_t idx = r * H + j;
                    a->act[idx] = sat_int16((int32_t)a->act[idx] +
                        requant_pb(a->ffn_down32[idx], &blk->ffn_down, j) * m->act_boost);
                }
            }
        }

        if (any_skip) {
            for (int32_t r = 0; r < real_len; r++) {
                if (mod_keep_row[r]) continue;
                memcpy(a->act + (size_t)r * H, pre_snapshot + (size_t)r * H,
                       (size_t)H * sizeof(int16_t));
            }
            free(pre_snapshot);
        }

        if (trace) {
            memcpy(trace->residual_post + L * per_layer_residual, a->act, per_layer_residual * sizeof(int16_t));
        }

        if (trace && trace->lens_logits) {
            for (int32_t p = 0; p < real_len; p++) {
                const int16_t* res = a->act + (size_t)p * H;
                int32_t* dst = trace->lens_logits + ((size_t)L * S + p) * V;
                lens_project(sh, m->embed, res, dst);
            }
        }
    }

    cache->len = real_len;
    {
        const int16_t* last = a->act + (size_t)(real_len - 1) * H;
        if (m->n_out_w) {
            layernorm_i16_to_i8(last, out_act, m->n_out_w, 1, H);
        } else {
            for (int32_t i = 0; i < H; i++) out_act[i] = sat_int8(last[i]);
        }
    }

    if (trace) memcpy(trace->final_act, out_act, H);
}

// ------------------------------------------------------------------------------------
// trace_write — dump trace_record_t to disk in the VRMR binary format
// ------------------------------------------------------------------------------------

int trace_write(const char* path, const veritate_shape_t* sh,
                const trace_record_t* trace, int32_t real_len) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    trace_header_t hdr;
    memcpy(hdr.magic, VERITATE_TRACE_MAGIC, 4);
    hdr.version  = VERITATE_TRACE_VERSION;
    hdr.v_layers = (uint32_t)sh->layers;
    hdr.v_seq    = (uint32_t)sh->seq;
    hdr.v_hidden = (uint32_t)sh->hidden;
    hdr.v_ffn    = (uint32_t)sh->ffn;
    hdr.v_heads  = (uint32_t)sh->heads;
    hdr.real_len = (uint32_t)real_len;
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }

    const size_t residual_bytes = (size_t)sh->layers * sh->seq * sh->hidden * sizeof(int16_t);
    const size_t ffn_bytes      = (size_t)sh->layers * sh->seq * sh->ffn;
    const size_t attn_bytes     = (size_t)sh->layers * sh->heads * sh->seq * sh->seq * sizeof(float);
    const size_t lens_bytes     = (size_t)sh->layers * sh->seq * sh->vocab * sizeof(int32_t);
    const uint8_t has_attention = trace->attention_scores ? 1 : 0;
    const uint8_t has_lens      = trace->lens_logits ? 1 : 0;

    if (fwrite(trace->residual_pre,    residual_bytes,                 1, f) != 1 ||
        fwrite(trace->residual_post,   residual_bytes,                 1, f) != 1 ||
        fwrite(trace->ffn_neurons,     ffn_bytes,                      1, f) != 1 ||
        fwrite(trace->final_act,       (size_t)sh->hidden,             1, f) != 1 ||
        fwrite(trace->prompt_bytes,    (size_t)real_len,               1, f) != 1 ||
        fwrite(trace->top_predictions, sizeof(trace_prediction_t) * VERITATE_TRACE_TOPK, 1, f) != 1 ||
        fwrite(&has_attention,         sizeof(uint8_t),                1, f) != 1) {
        fclose(f); return -1;
    }
    if (has_attention && fwrite(trace->attention_scores, attn_bytes, 1, f) != 1) {
        fclose(f); return -1;
    }
    if (fwrite(&has_lens, sizeof(uint8_t), 1, f) != 1) { fclose(f); return -1; }
    if (has_lens && fwrite(trace->lens_logits, lens_bytes, 1, f) != 1) {
        fclose(f); return -1;
    }

    fclose(f);
    return 0;
}

// ------------------------------------------------------------------------------------
// forward_decode — single-token decode using cached K/V
// ------------------------------------------------------------------------------------

void forward_decode(const model_t* m, kv_cache_t* cache, int32_t token, int8_t* out_act,
                    trace_record_t* trace) {
    const veritate_shape_t* sh = &m->shape;
    const int32_t S = sh->seq, H = sh->hidden, F = sh->ffn, V = sh->vocab;
    const int32_t NH = sh->heads, HD = sh->head_dim;
    decode_acts_t* d = &((acts_pool_t*)m->scratch)->decode;

    int32_t pos = cache->len;
    if (pos >= S) return;

    const size_t per_layer_residual = (size_t)S * H;
    const size_t per_layer_ffn      = (size_t)S * F;
    const size_t per_layer_attn     = (size_t)NH * S * S;

    int32_t tok = token;
    if (V > 0) tok = ((tok % V) + V) % V;
    const int8_t* tok_row = m->embed     + (size_t)tok * H;
    const int8_t* pos_row = m->pos_embed + (size_t)pos * H;
    for (int32_t j = 0; j < H; j++) {
        d->act[j] = (int16_t)((int32_t)tok_row[j] + (int32_t)pos_row[j]);
    }

    const float scale = 1.0f / (sqrtf((float)HD) * 1024.0f);
    const float inv32k = 1.0f / 32768.0f;

    const int32_t n_layers = veritate_max_layers(m);
    for (int32_t L = 0; L < n_layers; L++) {
        const block_t* blk = &m->blocks[L];

        if (trace) {
            int16_t* dst = trace->residual_pre + (size_t)L * per_layer_residual + (size_t)pos * H;
            memcpy(dst, d->act, (size_t)H * sizeof(int16_t));
        }

        if (blk->has_gate) {
            g_mod_gate_calls++;
            if (!gate_keep(d->act, blk->gate_w, H, blk->gate_scale_q24)) {
                g_mod_gate_skipped++;
                if (trace) {
                    int16_t* dst_post = trace->residual_post + (size_t)L * per_layer_residual + (size_t)pos * H;
                    memcpy(dst_post, d->act, (size_t)H * sizeof(int16_t));
                    int8_t* ffn_dst = trace->ffn_neurons + (size_t)L * per_layer_ffn + (size_t)pos * F;
                    memset(ffn_dst, 0, (size_t)F);
                    if (trace->lens_logits) {
                        int32_t* lens_dst = trace->lens_logits + ((size_t)L * S + pos) * V;
                        lens_project(sh, m->embed, d->act, lens_dst);
                    }
                }
                // kv cache rows must still advance for this position; copy the carried
                // residual through ln1 so attention at later positions sees a consistent K/V.
                layernorm_i16_to_i8(d->act, d->act_norm, blk->ln1_w, 1, H);
                memcpy(cache_k_row(cache, L, pos), d->act_norm, H);
                memcpy(cache_v_row(cache, L, pos), d->act_norm, H);
                continue;
            }
        }

        layernorm_i16_to_i8(d->act, d->act_norm, blk->ln1_w, 1, H);

        if (blk->use_int4) {
            hadamard_apply_int8(d->act_norm, d->act_norm_rot, H);
            matmul_int4_vnni_prep(d->act_norm_rot, &blk->qkv_i4, d->qkv32, 1);
            for (int32_t i = 0; i < 3 * H; i++) {
                d->qkv8[i] = sat_int8(requant(d->qkv32[i], blk->qkv_i4.row_q24[i]));
            }
        } else {
            matmul_int8_vnni_prep(d->act_norm, &blk->qkv, d->qkv32, 1);
            for (int32_t i = 0; i < 3 * H; i++) {
                d->qkv8[i] = sat_int8(requant_pb(d->qkv32[i], &blk->qkv, i));
            }
        }

        memcpy(cache_k_row(cache, L, pos), d->qkv8 + H,     H);
        memcpy(cache_v_row(cache, L, pos), d->qkv8 + 2 * H, H);

        const int8_t* q_row = d->qkv8;
        for (int32_t h = 0; h < NH; h++) {
            const int8_t* qh = q_row + h * HD;
            int32_t q_sum = attn_hsum_inline(qh);
            int32_t j = 0;
            int32_t end = pos + 1;
            for (; j + 4 <= end; j += 4) {
                int32_t out4[4];
                attn_dot_inline_4(qh,
                    cache_k_row(cache, L, j + 0) + h * HD,
                    cache_k_row(cache, L, j + 1) + h * HD,
                    cache_k_row(cache, L, j + 2) + h * HD,
                    cache_k_row(cache, L, j + 3) + h * HD,
                    q_sum, out4);
                d->scores[j + 0] = (float)out4[0] * scale;
                d->scores[j + 1] = (float)out4[1] * scale;
                d->scores[j + 2] = (float)out4[2] * scale;
                d->scores[j + 3] = (float)out4[3] * scale;
            }
            for (; j < end; j++) {
                const int8_t* kh = cache_k_row(cache, L, j) + h * HD;
                d->scores[j] = (float)attn_dot_inline(qh, kh, q_sum) * scale;
            }
            softmax_rows(d->scores, d->scores_q, 1, pos + 1);

            if (trace && trace->attention_scores) {
                float* td_row = trace->attention_scores
                              + (size_t)L * per_layer_attn
                              + ((size_t)h * S + pos) * S;
                for (int32_t j = 0;     j <= pos; j++) td_row[j] = (float)d->scores_q[j] * inv32k;
                for (int32_t j = pos+1; j <  S;   j++) td_row[j] = 0.0f;
            }

            int8_t* ah = d->attn8 + h * HD;
            score_dot_v(d->scores_q, cache_v_row(cache, L, 0) + h * HD, H, pos + 1, ah);
        }

        if (blk->use_int4) {
            hadamard_apply_int8(d->attn8, d->act_norm_rot, H);
            matmul_int4_vnni_prep(d->act_norm_rot, &blk->out_proj_i4, d->out32, 1);
            for (int32_t i = 0; i < H; i++) {
                d->act[i] = sat_int16((int32_t)d->act[i] + requant(d->out32[i], blk->out_proj_i4.row_q24[i]) * m->act_boost);
            }
        } else {
            matmul_int8_vnni_prep(d->attn8, &blk->out_proj, d->out32, 1);
            for (int32_t i = 0; i < H; i++) {
                d->act[i] = sat_int16((int32_t)d->act[i] + requant_pb(d->out32[i], &blk->out_proj, i) * m->act_boost);
            }
        }

        layernorm_i16_to_i8(d->act, d->act_norm, blk->ln2_w, 1, H);

        if (blk->use_int4) {
            hadamard_apply_int8(d->act_norm, d->act_norm_rot, H);
            matmul_int4_vnni_prep(d->act_norm_rot, &blk->ffn_up_i4, d->ffn_up32, 1);
            for (int32_t i = 0; i < F; i++) {
                d->ffn_up8[i] = sat_int8(requant(d->ffn_up32[i], blk->ffn_up_i4.row_q24[i]));
            }
        } else {
            matmul_int8_vnni_prep(d->act_norm, &blk->ffn_up, d->ffn_up32, 1);
            for (int32_t i = 0; i < F; i++) {
                d->ffn_up8[i] = sat_int8(requant_pb(d->ffn_up32[i], &blk->ffn_up, i));
            }
        }
        gelu_int8(d->ffn_up8, F);

        // causal ablation hook (v8). zero a single post-GELU neuron so ffn_down
        // and the trace capture both see the silenced neuron. no-op when global
        // ablation state is (-1, -1).
        if (g_ablate_layer == L && g_ablate_neuron >= 0 && g_ablate_neuron < F) {
            d->ffn_up8[g_ablate_neuron] = 0;
        }

        if (trace) {
            int8_t* dst = trace->ffn_neurons + (size_t)L * per_layer_ffn + (size_t)pos * F;
            memcpy(dst, d->ffn_up8, (size_t)F);
        }

        if (blk->use_int4) {
            hadamard_apply_int8(d->ffn_up8, d->ffn_up8_rot, F);
            matmul_int4_vnni_prep(d->ffn_up8_rot, &blk->ffn_down_i4, d->ffn_down32, 1);
            for (int32_t i = 0; i < H; i++) {
                d->act[i] = sat_int16((int32_t)d->act[i] + requant(d->ffn_down32[i], blk->ffn_down_i4.row_q24[i]) * m->act_boost);
            }
        } else {
            ffn_down_decode(d->ffn_up8, &blk->ffn_down, d->ffn_down32);
            for (int32_t i = 0; i < H; i++) {
                d->act[i] = sat_int16((int32_t)d->act[i] + requant_pb(d->ffn_down32[i], &blk->ffn_down, i) * m->act_boost);
            }
        }

        if (trace) {
            int16_t* dst = trace->residual_post + (size_t)L * per_layer_residual + (size_t)pos * H;
            memcpy(dst, d->act, (size_t)H * sizeof(int16_t));
        }

        if (trace && trace->lens_logits) {
            int32_t* dst = trace->lens_logits + ((size_t)L * S + pos) * V;
            lens_project(sh, m->embed, d->act, dst);
        }
    }

    cache->len = pos + 1;
    if (m->n_out_w) {
        layernorm_i16_to_i8(d->act, out_act, m->n_out_w, 1, H);
    } else {
        for (int32_t i = 0; i < H; i++) out_act[i] = sat_int8(d->act[i]);
    }

    if (trace) memcpy(trace->final_act, out_act, (size_t)H);
}

// ------------------------------------------------------------------------------------
// forward_verify — process K new tokens against a populated kv cache. M=K decode shape
// between forward_decode (M=1) and forward (M=V_SEQ). bit-equivalent within 1 LSB to
// running K sequential forward_decode calls. dispatches on K.
// ------------------------------------------------------------------------------------

void forward_verify(const model_t* m, kv_cache_t* cache, int32_t K,
                    const int32_t* tokens, int8_t* out_hidden_K) {
    const veritate_shape_t* sh = &m->shape;
    const int32_t S = sh->seq, H = sh->hidden, F = sh->ffn, V = sh->vocab;
    const int32_t NH = sh->heads, HD = sh->head_dim;
    if (K <= 0) return;
    if (K > VERITATE_VERIFY_K_MAX) K = VERITATE_VERIFY_K_MAX;
    if (K == 1 || m->blocks[0].use_int4) {
        // int4 path uses the per-row decode kernel; just call decode K times.
        for (int32_t r = 0; r < K; r++) {
            forward_decode(m, cache, tokens[r], out_hidden_K + (size_t)r * H, NULL);
        }
        return;
    }

    verify_acts_t* v = &((acts_pool_t*)m->scratch)->verify;
    int32_t base = cache->len;
    if (base + K > S) return;

    for (int32_t i = 0; i < K; i++) {
        int32_t tok = tokens[i];
        if (V > 0) tok = ((tok % V) + V) % V;
        const int8_t* tok_row = m->embed     + (size_t)tok * H;
        const int8_t* pos_row = m->pos_embed + (size_t)(base + i) * H;
        int16_t* dst = v->act + (size_t)i * H;
        for (int32_t j = 0; j < H; j++) {
            dst[j] = (int16_t)((int32_t)tok_row[j] + (int32_t)pos_row[j]);
        }
    }

    const float scale = 1.0f / (sqrtf((float)HD) * 1024.0f);
    const int32_t n_layers = veritate_max_layers(m);
    const int use_mt = (K >= 8);

    for (int32_t L = 0; L < n_layers; L++) {
        const block_t* blk = &m->blocks[L];

        layernorm_i16_to_i8(v->act, v->act_norm, blk->ln1_w, K, H);

        if (use_mt) matmul_int8_vnni_mt_prep(v->act_norm, &blk->qkv, v->qkv32, K);
        else        matmul_int8_vnni_prep   (v->act_norm, &blk->qkv, v->qkv32, K);

        for (int32_t r = 0; r < K; r++) {
            int32_t* row32 = v->qkv32 + (size_t)r * 3 * H;
            int8_t*  row8  = v->qkv8  + (size_t)r * 3 * H;
            for (int32_t j = 0; j < 3 * H; j++) {
                row8[j] = sat_int8(requant_pb(row32[j], &blk->qkv, j));
            }
        }

        for (int32_t r = 0; r < K; r++) {
            memcpy(cache_k_row(cache, L, base + r), v->qkv8 + (size_t)r * 3 * H + H,     H);
            memcpy(cache_v_row(cache, L, base + r), v->qkv8 + (size_t)r * 3 * H + 2 * H, H);
        }

        for (int32_t r = 0; r < K; r++) {
            int32_t pos = base + r;
            const int8_t* q_row = v->qkv8 + (size_t)r * 3 * H;
            for (int32_t h = 0; h < NH; h++) {
                const int8_t* qh = q_row + h * HD;
                int32_t q_sum = attn_hsum_inline(qh);
                int32_t j = 0;
                int32_t end = pos + 1;
                for (; j + 4 <= end; j += 4) {
                    int32_t out4[4];
                    attn_dot_inline_4(qh,
                        cache_k_row(cache, L, j + 0) + h * HD,
                        cache_k_row(cache, L, j + 1) + h * HD,
                        cache_k_row(cache, L, j + 2) + h * HD,
                        cache_k_row(cache, L, j + 3) + h * HD,
                        q_sum, out4);
                    v->scores[j + 0] = (float)out4[0] * scale;
                    v->scores[j + 1] = (float)out4[1] * scale;
                    v->scores[j + 2] = (float)out4[2] * scale;
                    v->scores[j + 3] = (float)out4[3] * scale;
                }
                for (; j < end; j++) {
                    const int8_t* kh = cache_k_row(cache, L, j) + h * HD;
                    v->scores[j] = (float)attn_dot_inline(qh, kh, q_sum) * scale;
                }
                softmax_rows(v->scores, v->scores_q, 1, pos + 1);
                int8_t* ah = v->attn8 + (size_t)r * H + h * HD;
                score_dot_v(v->scores_q, cache_v_row(cache, L, 0) + h * HD,
                                   H, pos + 1, ah);
            }
        }

        if (use_mt) matmul_int8_vnni_mt_prep(v->attn8, &blk->out_proj, v->out32, K);
        else        matmul_int8_vnni_prep   (v->attn8, &blk->out_proj, v->out32, K);
        for (int32_t r = 0; r < K; r++) {
            for (int32_t j = 0; j < H; j++) {
                int32_t idx = r * H + j;
                v->act[idx] = sat_int16((int32_t)v->act[idx] +
                    requant_pb(v->out32[idx], &blk->out_proj, j) * m->act_boost);
            }
        }

        layernorm_i16_to_i8(v->act, v->act_norm, blk->ln2_w, K, H);

        if (use_mt) matmul_int8_vnni_mt_prep(v->act_norm, &blk->ffn_up, v->ffn_up32, K);
        else        matmul_int8_vnni_prep   (v->act_norm, &blk->ffn_up, v->ffn_up32, K);
        for (int32_t r = 0; r < K; r++) {
            int32_t* row32 = v->ffn_up32 + (size_t)r * F;
            int8_t*  row8  = v->ffn_up8  + (size_t)r * F;
            for (int32_t j = 0; j < F; j++) {
                row8[j] = sat_int8(requant_pb(row32[j], &blk->ffn_up, j));
            }
        }
        gelu_int8(v->ffn_up8, K * F);

        if (use_mt) matmul_int8_vnni_mt_prep(v->ffn_up8, &blk->ffn_down, v->ffn_dn32, K);
        else        matmul_int8_vnni_prep   (v->ffn_up8, &blk->ffn_down, v->ffn_dn32, K);
        for (int32_t r = 0; r < K; r++) {
            for (int32_t j = 0; j < H; j++) {
                int32_t idx = r * H + j;
                v->act[idx] = sat_int16((int32_t)v->act[idx] +
                    requant_pb(v->ffn_dn32[idx], &blk->ffn_down, j) * m->act_boost);
            }
        }
    }

    cache->len = base + K;
    if (m->n_out_w) {
        layernorm_i16_to_i8(v->act, out_hidden_K, m->n_out_w, K, H);
    } else {
        for (int32_t r = 0; r < K; r++) {
            for (int32_t i = 0; i < H; i++) {
                out_hidden_K[r * H + i] = sat_int8(v->act[r * H + i]);
            }
        }
    }
}

// ------------------------------------------------------------------------------------
// byte-level tokenizer — one token per byte, vocab covers 0..255
// ------------------------------------------------------------------------------------

int32_t tokenize_bytes(const char* text, int32_t* tokens, int32_t max_tokens) {
    int32_t i = 0;
    while (text[i] != '\0' && i < max_tokens) {
        tokens[i] = (uint8_t)text[i];
        i++;
    }
    return i;
}

void detokenize_bytes(const int32_t* tokens, int32_t n, char* out) {
    for (int32_t i = 0; i < n; i++) {
        out[i] = (char)((uint32_t)tokens[i] & 0xFF);
    }
    out[n] = '\0';
}

// ------------------------------------------------------------------------------------
// lm head — tied to input embedding. temperature + top-k + multinomial.
// ------------------------------------------------------------------------------------

// build lm_head — transpose embed to [hidden, vocab] layout prep_b expects, then pack.
void lm_head_build(model_t* m) {
    const int32_t V = m->shape.vocab, H = m->shape.hidden;
    int8_t* embed_T = (int8_t*)malloc((size_t)H * V);
    for (int32_t v = 0; v < V; v++) {
        for (int32_t p = 0; p < H; p++) {
            embed_T[p * V + v] = m->embed[v * H + p];
        }
    }
    prep_b(embed_T, V, H, &m->lm_head);
    free(embed_T);
}

int32_t sample_token_ext(const model_t* m, const int8_t* hidden, float temp, int32_t top_k,
                         uint32_t* rng, int32_t* out_logits, int32_t* out_argmax) {
    const int32_t V = m->shape.vocab;
    int32_t* logits = (int32_t*)malloc((size_t)V * sizeof(int32_t));
    float*   fp     = (float*)  malloc((size_t)V * sizeof(float));

    matmul_int8_vnni_prep(hidden, &m->lm_head, logits, 1);

    int32_t argmax = 0;
    int32_t argmax_logit = logits[0];
    for (int32_t v = 1; v < V; v++) {
        if (logits[v] > argmax_logit) { argmax_logit = logits[v]; argmax = v; }
    }
    if (out_logits) memcpy(out_logits, logits, (size_t)V * sizeof(int32_t));
    if (out_argmax) *out_argmax = argmax;

    if (temp <= 0.0f) { free(logits); free(fp); return argmax; }

    int32_t threshold = -2147483647 - 1;
    if (top_k > 0 && top_k < V) {
        int32_t* heap = (int32_t*)malloc((size_t)top_k * sizeof(int32_t));
        for (int32_t i = 0; i < top_k; i++) heap[i] = logits[i];
        for (int32_t i = top_k / 2 - 1; i >= 0; i--) {
            int32_t p = i;
            while (1) {
                int32_t l = 2 * p + 1, r = 2 * p + 2, s = p;
                if (l < top_k && heap[l] < heap[s]) s = l;
                if (r < top_k && heap[r] < heap[s]) s = r;
                if (s == p) break;
                int32_t t = heap[p]; heap[p] = heap[s]; heap[s] = t;
                p = s;
            }
        }
        for (int32_t i = top_k; i < V; i++) {
            if (logits[i] > heap[0]) {
                heap[0] = logits[i];
                int32_t p = 0;
                while (1) {
                    int32_t l = 2 * p + 1, r = 2 * p + 2, s = p;
                    if (l < top_k && heap[l] < heap[s]) s = l;
                    if (r < top_k && heap[r] < heap[s]) s = r;
                    if (s == p) break;
                    int32_t t = heap[p]; heap[p] = heap[s]; heap[s] = t;
                    p = s;
                }
            }
        }
        threshold = heap[0];
        free(heap);
    }

    float max_fp = -1e30f;
    for (int32_t v = 0; v < V; v++) {
        if (logits[v] < threshold) {
            fp[v] = -1e30f;
        } else {
            fp[v] = (float)logits[v] / temp;
            if (fp[v] > max_fp) max_fp = fp[v];
        }
    }

    float sum = 0.0f;
    for (int32_t v = 0; v < V; v++) {
        fp[v] = expf(fp[v] - max_fp);
        sum += fp[v];
    }

    uint32_t r = xorshift32(rng);
    float u = (float)(r >> 8) / (float)(1u << 24) * sum;
    float c = 0.0f;
    int32_t pick = V - 1;
    for (int32_t v = 0; v < V; v++) {
        c += fp[v];
        if (c >= u) { pick = v; break; }
    }
    free(logits); free(fp);
    return pick;
}

int32_t sample_token(const model_t* m, const int8_t* hidden, float temp, int32_t top_k, uint32_t* rng) {
    return sample_token_ext(m, hidden, temp, top_k, rng, NULL, NULL);
}

// ------------------------------------------------------------------------------------
// random weight init — for v3 testing, no real training
// ------------------------------------------------------------------------------------

static void fill_random_b(int32_t n, int32_t k, prepped_b_t* p, int keep_raw) {
    int8_t* raw = (int8_t*)malloc((size_t)n * k);
    for (int32_t i = 0; i < n * k; i++) raw[i] = (int8_t)(rng_next() & 0x3F) - 32;
    if (keep_raw) prep_b_keep_raw(raw, n, k, p);
    else          prep_b(raw, n, k, p);
    p->scale_q24 = 524288;
    free(raw);
}

// allocate all heap-resident model storage based on m->shape. embed/pos_embed/blocks/
// byte_direction/scratch. weights inside blocks remain unfilled — caller follows up
// with model_init_random or model_load to populate.
static int model_alloc_storage(model_t* m) {
    const int32_t V = m->shape.vocab, H = m->shape.hidden, S = m->shape.seq, Ln = m->shape.layers;
    m->embed                = (int8_t*) xalloc64((size_t)V * H);
    m->pos_embed            = (int8_t*) xalloc64((size_t)S * H);
    m->blocks               = (block_t*)calloc((size_t)Ln, sizeof(block_t));
    m->byte_direction       = (int16_t**)calloc((size_t)Ln, sizeof(int16_t*));
    m->byte_direction_scale = (float*)  calloc((size_t)Ln, sizeof(float));
    if (!m->embed || !m->pos_embed || !m->blocks ||
        !m->byte_direction || !m->byte_direction_scale) return -1;
    for (int32_t L = 0; L < Ln; L++) {
        block_t* blk = &m->blocks[L];
        blk->ln1_w = (int8_t*)xalloc64((size_t)H);
        blk->ln2_w = (int8_t*)xalloc64((size_t)H);
        if (!blk->ln1_w || !blk->ln2_w) return -1;
    }
    m->scratch = pool_alloc(&m->shape);
    if (!m->scratch) return -1;
    return 0;
}

static void shape_set_default(veritate_shape_t* s) {
    s->vocab    = V_VOCAB;
    s->seq      = V_SEQ;
    s->hidden   = V_HIDDEN;
    s->heads    = V_HEADS;
    s->head_dim = V_HEAD_DIM;
    s->ffn      = V_FFN;
    s->layers   = V_LAYERS;
}

void model_init_random(model_t* m, unsigned seed) {
    memset(m, 0, sizeof(*m));
    shape_set_default(&m->shape);
    m->act_boost = 1;
    rng_seed(seed);

    if (model_alloc_storage(m) != 0) return;

    const int32_t V = m->shape.vocab, H = m->shape.hidden, S = m->shape.seq;
    const int32_t F = m->shape.ffn, Ln = m->shape.layers;

    for (int32_t i = 0; i < V * H; i++) {
        m->embed[i] = (int8_t)(rng_next() & 0x3F) - 32;
    }
    for (int32_t i = 0; i < S * H; i++) {
        m->pos_embed[i] = (int8_t)(rng_next() & 0x1F) - 16;
    }

    for (int32_t L = 0; L < Ln; L++) {
        block_t* blk = &m->blocks[L];
        blk->use_int4 = 0;
        for (int32_t i = 0; i < H; i++) {
            blk->ln1_w[i] = 64;
            blk->ln2_w[i] = 64;
        }
        fill_random_b(3 * H, H, &blk->qkv,      0);
        fill_random_b(H,     H, &blk->out_proj, 0);
        fill_random_b(F,     H, &blk->ffn_up,   0);
        fill_random_b(H,     F, &blk->ffn_down, 1);
    }
    byte_direction_build(m);
    lm_head_build(m);
    m->cw_loaded = 0;
}

// ------------------------------------------------------------------------------------
// model_load — read raw int8 weights from disk, run prep_b at load time
// ------------------------------------------------------------------------------------

static int load_b(FILE* f, int32_t n, int32_t k, prepped_b_t* p, int keep_raw) {
    int8_t* raw = (int8_t*)malloc((size_t)n * k);
    if (!raw) return -1;
    if (fread(raw, (size_t)n * k, 1, f) != 1) { free(raw); return -1; }
    if (keep_raw) prep_b_keep_raw(raw, n, k, p);
    else          prep_b(raw, n, k, p);
    free(raw);
    int32_t file_scale_q24 = 0;
    if (fread(&file_scale_q24, sizeof(int32_t), 1, f) != 1) return -1;
    if (file_scale_q24 != 0) p->scale_q24 = file_scale_q24;
    return 0;
}

// v5 — per-output-column scales. weight layout matches v3; scale block is [n] int32.
static int load_b_percol(FILE* f, int32_t n, int32_t k, prepped_b_t* p, int keep_raw) {
    int8_t* raw = (int8_t*)malloc((size_t)n * k);
    if (!raw) return -1;
    if (fread(raw, (size_t)n * k, 1, f) != 1) { free(raw); return -1; }
    if (keep_raw) prep_b_keep_raw(raw, n, k, p);
    else          prep_b(raw, n, k, p);
    free(raw);
    p->scale_per_col = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), 64);
    if (!p->scale_per_col) return -1;
    if (fread(p->scale_per_col, (size_t)n * sizeof(int32_t), 1, f) != 1) return -1;
    return 0;
}

// v10 ternary load. .bin holds 5-trits-per-byte packed weights followed by a
// per-tensor gamma_q24 (q24 of the mean-abs scale). decode trits to a
// {-1, 0, +1}-valued int8 buffer of shape [k, n] and run the existing prep_b.
// hot path stays int8; ternary win is the 5x smaller .bin on disk.
//
// file layout per ternary tensor:
//   uint8  packed_bytes[n * ceil(k/5)]
//   int32  gamma_q24                    = round(gamma * 2^24)

#define VERITATE_TRITS_PER_BYTE 5
#define VERITATE_TERNARY_PACK_STRIDE(k) (((k) + VERITATE_TRITS_PER_BYTE - 1) / VERITATE_TRITS_PER_BYTE)

static int8_t  TERNARY_LUT_[256][8] __attribute__((aligned(16)));
static int32_t TERNARY_LUT_INIT_ = 0;

static void ternary_init_lut(void) {
    if (TERNARY_LUT_INIT_) return;
    for (int32_t b = 0; b < 256; b++) {
        int32_t x = b;
        for (int32_t p = 0; p < VERITATE_TRITS_PER_BYTE; p++) {
            TERNARY_LUT_[b][p] = (int8_t)(x % 3) - 1;
            x /= 3;
        }
        TERNARY_LUT_[b][5] = 0;
        TERNARY_LUT_[b][6] = 0;
        TERNARY_LUT_[b][7] = 0;
    }
    TERNARY_LUT_INIT_ = 1;
}

static int load_b_ternary(FILE* f, int32_t n, int32_t k, prepped_b_t* p) {
    ternary_init_lut();
    int32_t pack_stride  = VERITATE_TERNARY_PACK_STRIDE(k);
    size_t  packed_bytes = (size_t)n * pack_stride;

    uint8_t* packed = (uint8_t*)malloc(packed_bytes);
    if (!packed) return -1;
    if (fread(packed, packed_bytes, 1, f) != 1) { free(packed); return -1; }

    int32_t gamma_q24 = 0;
    if (fread(&gamma_q24, sizeof(int32_t), 1, f) != 1) { free(packed); return -1; }

    int8_t* raw = (int8_t*)malloc((size_t)n * k);
    if (!raw) { free(packed); return -1; }

    for (int32_t j = 0; j < n; j++) {
        const uint8_t* row = packed + (size_t)j * pack_stride;
        for (int32_t pp = 0; pp < k; pp++) {
            int32_t bidx = pp / VERITATE_TRITS_PER_BYTE;
            int32_t bofs = pp % VERITATE_TRITS_PER_BYTE;
            raw[(size_t)pp * n + j] = TERNARY_LUT_[row[bidx]][bofs];
        }
    }

    prep_b(raw, n, k, p);
    free(raw);
    free(packed);

    p->scale_q24 = gamma_q24;
    return 0;
}

// load packed int4 block with per-row q24 multipliers. matches export_quarot_int4.py format.
static int load_b_int4(FILE* f, int32_t n, int32_t k, prepped_b_int4_t* out) {
    out->n = n;
    out->k = k;
    out->bt_packed = (uint8_t*)veritate_aligned_alloc((size_t)n * (k / 2), 64);
    out->bias      = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), 64);
    out->row_q24   = (int32_t*)veritate_aligned_alloc((size_t)n * sizeof(int32_t), 64);
    if (!out->bt_packed || !out->bias || !out->row_q24) return -1;

    if (fread(out->bt_packed, (size_t)n * (k / 2), 1, f) != 1) return -1;
    if (fread(out->row_q24,   (size_t)n * sizeof(int32_t), 1, f) != 1) return -1;

    for (int32_t j = 0; j < n; j++) {
        const uint8_t* row = out->bt_packed + (size_t)j * (k / 2);
        int32_t s = 0;
        for (int32_t t = 0; t < k / 2; t++) {
            uint8_t b = row[t];
            int8_t  w0 = (int8_t)(((((int8_t)b      ) & 0x0F) ^ 8) - 8);
            int8_t  w1 = (int8_t)(((((int8_t)(b >> 4)) & 0x0F) ^ 8) - 8);
            s += w0 + w1;
        }
        out->bias[j] = 128 * s;
    }
    return 0;
}

static int model_load_int4(model_t* m, FILE* f) {
    const int32_t V = m->shape.vocab, H = m->shape.hidden, S = m->shape.seq;
    const int32_t F = m->shape.ffn, Ln = m->shape.layers;
    if (fread(m->embed,     (size_t)V * H, 1, f) != 1) return -1;
    if (fread(m->pos_embed, (size_t)S * H, 1, f) != 1) return -1;

    for (int32_t L = 0; L < Ln; L++) {
        block_t* blk = &m->blocks[L];
        blk->use_int4 = 1;
        if (fread(blk->ln1_w, (size_t)H, 1, f) != 1) return -1;
        if (load_b_int4(f, 3 * H, H, &blk->qkv_i4)      != 0) return -1;
        if (load_b_int4(f, H,     H, &blk->out_proj_i4) != 0) return -1;
        if (fread(blk->ln2_w, (size_t)H, 1, f) != 1) return -1;
        if (load_b_int4(f, F,     H, &blk->ffn_up_i4)   != 0) return -1;
        if (load_b_int4(f, H,     F, &blk->ffn_down_i4) != 0) return -1;
    }
    return 0;
}

int model_load(model_t* m, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    model_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    if (memcmp(hdr.magic, VERITATE_MODEL_MAGIC, 4) != 0) { fclose(f); return -1; }

    memset(m, 0, sizeof(*m));
    m->act_boost      = 1;
    m->shape.vocab    = (int32_t)hdr.v_vocab;
    m->shape.seq      = (int32_t)hdr.v_seq;
    m->shape.hidden   = (int32_t)hdr.v_hidden;
    m->shape.heads    = (int32_t)hdr.v_heads;
    m->shape.head_dim = (m->shape.heads > 0) ? (m->shape.hidden / m->shape.heads) : 0;
    m->shape.ffn      = (int32_t)hdr.v_ffn;
    m->shape.layers   = (int32_t)hdr.v_layers;
    if (m->shape.vocab <= 0 || m->shape.seq <= 0 || m->shape.hidden <= 0 ||
        m->shape.heads <= 0 || m->shape.head_dim <= 0 || m->shape.ffn <= 0 ||
        m->shape.layers <= 0) { fclose(f); return -1; }
    if (m->shape.ffn > V_MAX_FFN) { fclose(f); return -1; }

    if (model_alloc_storage(m) != 0) { fclose(f); model_free(m); return -1; }

    if (hdr.version == VERITATE_MODEL_VERSION_INT4) {
        int rc = model_load_int4(m, f);
        fclose(f);
        // int4 path lacks b_rowmaj on ffn_down; decision-trace tables stay NULL.
        if (rc == 0) lm_head_build(m);
        m->cw_loaded = 0;
        return rc;
    }
    if (hdr.version != VERITATE_MODEL_VERSION &&
        hdr.version != VERITATE_MODEL_VERSION_PERCOL &&
        hdr.version != VERITATE_MODEL_VERSION_MOD &&
        hdr.version != VERITATE_MODEL_VERSION_NORM &&
        hdr.version != VERITATE_MODEL_VERSION_BOOST &&
        hdr.version != VERITATE_MODEL_VERSION_TERNARY) { fclose(f); return -1; }

    const int32_t V = m->shape.vocab, H = m->shape.hidden, S = m->shape.seq;
    const int32_t F = m->shape.ffn, Ln = m->shape.layers;

    if (hdr.version == VERITATE_MODEL_VERSION_BOOST ||
        hdr.version == VERITATE_MODEL_VERSION_TERNARY) {
        if (fread(&m->act_boost, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        if (m->act_boost < 1) m->act_boost = 1;
    }

    if (fread(m->embed,     (size_t)V * H, 1, f) != 1) { fclose(f); return -1; }
    if (fread(m->pos_embed, (size_t)S * H, 1, f) != 1) { fclose(f); return -1; }

    const int per_col  = (hdr.version == VERITATE_MODEL_VERSION_PERCOL ||
                         hdr.version == VERITATE_MODEL_VERSION_MOD);
    const int has_gate = (hdr.version == VERITATE_MODEL_VERSION_MOD);
    const int has_norm = (hdr.version == VERITATE_MODEL_VERSION_NORM ||
                         hdr.version == VERITATE_MODEL_VERSION_BOOST ||
                         hdr.version == VERITATE_MODEL_VERSION_TERNARY);
    const int ternary  = (hdr.version == VERITATE_MODEL_VERSION_TERNARY);
    for (int32_t L = 0; L < Ln; L++) {
        block_t* blk = &m->blocks[L];
        int rc = (fread(blk->ln1_w, (size_t)H, 1, f) != 1) ? -1 : 0;
        if (rc == 0) rc = ternary
            ? load_b_ternary(f, 3 * H, H, &blk->qkv)
            : per_col
                ? load_b_percol(f, 3 * H, H, &blk->qkv, 0)
                : load_b       (f, 3 * H, H, &blk->qkv, 0);
        if (rc == 0) rc = ternary
            ? load_b_ternary(f, H, H, &blk->out_proj)
            : per_col
                ? load_b_percol(f, H, H, &blk->out_proj, 0)
                : load_b       (f, H, H, &blk->out_proj, 0);
        if (rc == 0 && fread(blk->ln2_w, (size_t)H, 1, f) != 1) rc = -1;
        if (rc == 0) rc = ternary
            ? load_b_ternary(f, F, H, &blk->ffn_up)
            : per_col
                ? load_b_percol(f, F, H, &blk->ffn_up, 0)
                : load_b       (f, F, H, &blk->ffn_up, 0);
        if (rc == 0) rc = ternary
            ? load_b_ternary(f, H, F, &blk->ffn_down)
            : per_col
                ? load_b_percol(f, H, F, &blk->ffn_down, 1)
                : load_b       (f, H, F, &blk->ffn_down, 1);
        if (rc == 0 && has_gate) {
            blk->gate_w = (int8_t*)xalloc64((size_t)H);
            if (!blk->gate_w) rc = -1;
            if (rc == 0 && fread(blk->gate_w, (size_t)H, 1, f) != 1) rc = -1;
            if (rc == 0 && fread(&blk->gate_scale_q24, sizeof(int32_t), 1, f) != 1) rc = -1;
            if (rc == 0) blk->has_gate = 1;
        }
        if (rc != 0) { fclose(f); return -1; }
    }

    if (has_norm) {
        m->n_out_w = (int8_t*)xalloc64((size_t)H);
        if (!m->n_out_w) { fclose(f); return -1; }
        if (fread(m->n_out_w, (size_t)H, 1, f) != 1) { fclose(f); return -1; }
    }

    fclose(f);
    byte_direction_build(m);
    lm_head_build(m);
    m->cw_loaded = 0;
    return 0;
}

// parse minimal json: numeric value of "key". returns 1 on success, 0 on miss.
static int json_find_float(const char* s, const char* key, float* out) {
    char pat[64];
    int kn = (int)strlen(key);
    if (kn > 60) return 0;
    pat[0] = '"';
    memcpy(pat + 1, key, (size_t)kn);
    pat[kn + 1] = '"';
    pat[kn + 2] = '\0';
    const char* p = strstr(s, pat);
    if (!p) return 0;
    p += kn + 2;
    while (*p && *p != ':') p++;
    if (*p != ':') return 0;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    char* end = NULL;
    float v = strtof(p, &end);
    if (end == p) return 0;
    *out = v;
    return 1;
}

int confidence_weights_load(model_t* m, const char* path) {
    if (!m || !path) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 65536) { fclose(f); return -1; }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    buf[sz] = '\0';
    fclose(f);
    float wM = 0, wE = 0, wL = 0, wS = 0, b = 0;
    int ok = json_find_float(buf, "w_M", &wM)
           & json_find_float(buf, "w_E", &wE)
           & json_find_float(buf, "w_L", &wL)
           & json_find_float(buf, "w_S", &wS)
           & json_find_float(buf, "b",   &b);
    free(buf);
    if (!ok) return -1;
    m->cw_M = wM; m->cw_E = wE; m->cw_L = wL; m->cw_S = wS; m->cw_b = b;
    m->cw_loaded = 1;
    return 0;
}

void model_free(model_t* m) {
    if (!m) return;
    byte_direction_free(m);
    free_prepped_b(&m->lm_head);
    if (m->blocks) {
        for (int32_t L = 0; L < m->shape.layers; L++) {
            block_t* blk = &m->blocks[L];
            if (blk->use_int4) {
                free_prepped_b_int4(&blk->qkv_i4);
                free_prepped_b_int4(&blk->out_proj_i4);
                free_prepped_b_int4(&blk->ffn_up_i4);
                free_prepped_b_int4(&blk->ffn_down_i4);
            } else {
                free_prepped_b(&blk->qkv);
                free_prepped_b(&blk->out_proj);
                free_prepped_b(&blk->ffn_up);
                free_prepped_b(&blk->ffn_down);
            }
            if (blk->ln1_w) veritate_aligned_free(blk->ln1_w);
            if (blk->ln2_w) veritate_aligned_free(blk->ln2_w);
            if (blk->gate_w) { veritate_aligned_free(blk->gate_w); blk->gate_w = NULL; }
        }
        free(m->blocks);
        m->blocks = NULL;
    }
    if (m->byte_direction)       { free(m->byte_direction);       m->byte_direction = NULL; }
    if (m->byte_direction_scale) { free(m->byte_direction_scale); m->byte_direction_scale = NULL; }
    if (m->embed)     { veritate_aligned_free(m->embed);     m->embed = NULL; }
    if (m->pos_embed) { veritate_aligned_free(m->pos_embed); m->pos_embed = NULL; }
    if (m->n_out_w)   { veritate_aligned_free(m->n_out_w);   m->n_out_w = NULL; }
    if (m->scratch)   { pool_free((acts_pool_t*)m->scratch); m->scratch = NULL; }
}
