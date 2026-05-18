// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - public API for veritate. dispatch table, tensor type, kernel signatures.
// ------------------------------------------------------------------------------------

#ifndef VERITATE_H
#define VERITATE_H

#include <stdint.h>
#include <stddef.h>

#define VERITATE_VERSION "0.1.0"

// ------------------------------------------------------------------------------------
// cpu features
// ------------------------------------------------------------------------------------

typedef struct {
    // x86_64
    int avx2;
    int avx512f;
    int avx512_vnni;
    // arm64
    int neon;
    int neon_sdot;
    int neon_i8mm;
    char brand[64];
} cpu_features_t;

void cpu_detect(cpu_features_t* out);
void cpu_print(const cpu_features_t* feat);

// ------------------------------------------------------------------------------------
// matmul kernel — INT8 input, INT32 accumulator, INT32 output
// ------------------------------------------------------------------------------------

typedef void (*matmul_int8_fn)(
    const int8_t*  a,    // [M x K] row-major
    const int8_t*  b,    // [K x N] column-major
    int32_t*       c,    // [M x N] row-major
    int32_t        m,
    int32_t        n,
    int32_t        k
);

extern matmul_int8_fn matmul_int8;

// kernel implementations (one is selected at startup by dispatch)
void matmul_int8_scalar(const int8_t* a, const int8_t* b, int32_t* c,
                        int32_t m, int32_t n, int32_t k);
void matmul_int8_avx2(const int8_t* a, const int8_t* b, int32_t* c,
                      int32_t m, int32_t n, int32_t k);
void matmul_int8_vnni(const int8_t* a, const int8_t* b, int32_t* c,
                      int32_t m, int32_t n, int32_t k);
void matmul_int8_vnni_mt(const int8_t* a, const int8_t* b, int32_t* c,
                         int32_t m, int32_t n, int32_t k);
void matmul_int8_neon_sdot(const int8_t* a, const int8_t* b, int32_t* c,
                           int32_t m, int32_t n, int32_t k);

// pre-prepare weights (B) for hot-loop matmul. real inference loads weights once.
// scale_per_col, when non-null, holds n q24 multipliers — one per output column.
// when null, requant uses the uniform scale_q24.
typedef struct {
    int8_t*  bt;            // b transposed (n rows of length k)
    int8_t*  b_rowmaj;      // optional row-major copy (k rows of length n) for sparse path
    int32_t* bias;          // 128 * column-sums
    int32_t* scale_per_col; // optional [n] q24 requant scales per output column
    int32_t  scale_q24;     // q24 requant scale (uniform; used when scale_per_col == NULL)
    int32_t  n;
    int32_t  k;
} prepped_b_t;

void prep_b(const int8_t* b, int32_t n, int32_t k, prepped_b_t* out);
void prep_b_keep_raw(const int8_t* b, int32_t n, int32_t k, prepped_b_t* out);
void free_prepped_b(prepped_b_t* p);

void matmul_int8_vnni_mt_prep(const int8_t* a, const prepped_b_t* p,
                              int32_t* c, int32_t m);
void matmul_int8_vnni_prep(const int8_t* a, const prepped_b_t* p,
                           int32_t* c, int32_t m);

// sparse-aware single-row matmul. m=1, requires p->b_rowmaj non-null.
// pre-scans a for non-zeros, accumulates only non-zero contributions.
// bit-identical to matmul_int8_vnni_prep on int32 output.
void matmul_int8_sparse_decode(const int8_t* a, const prepped_b_t* p, int32_t* c);

// ffn_down decode dispatch. m=1. pre-scans a, picks sparse if n_nz < k/2 and
// p->b_rowmaj is non-null, else falls back to matmul_int8_vnni_prep.
// bit-identical int32 output regardless of branch.
void ffn_down_decode(const int8_t* a, const prepped_b_t* p, int32_t* c);

// ------------------------------------------------------------------------------------
// int4 packed weights — 2 weights per byte, sign-extended to int8 at compute time.
// per-row fp32 scale, scale_q24 for output requant matches the int8 path.
// k must be even and a multiple of 64 in the avx-512 path.
// ------------------------------------------------------------------------------------

typedef struct {
    uint8_t* bt_packed;   // [n][k/2]   row j holds 2 int4 weights per byte
    int32_t* bias;        // [n]        128 * sum(unpacked weight row)
    int32_t* row_q24;     // [n]        per-row q24 multiplier: out_int8 = sat_int8((dot * row_q24[j] + 1<<23) >> 24)
    int32_t  n;
    int32_t  k;
} prepped_b_int4_t;

// pack int8 weights (each in -8..7) of shape [n,k] (column-major, i.e. b[p*n+j]) into
// the int4 prepped_b. k must be even.
void prep_b_int4(const int8_t* b, int32_t n, int32_t k, prepped_b_int4_t* out);
void free_prepped_b_int4(prepped_b_int4_t* p);

// scalar oracle. unpacks 2 int4 per byte, sign-extends, dot-product. m=1 decode path.
void matmul_int4_scalar_prep(const int8_t* a, const prepped_b_int4_t* p,
                             int32_t* c, int32_t m);

// avx-512 fast path. m=1 decode shape. cross-lane permute interleaves low/high nibbles
// into sequential int8 then accumulates via vnni dpbusd. bit-identical to scalar.
void matmul_int4_vnni_prep(const int8_t* a, const prepped_b_int4_t* p,
                           int32_t* c, int32_t m);

// ------------------------------------------------------------------------------------
// ternary packed weights -- BitNet b1.58. trits in {-1, 0, +1} packed 5-per-byte
// (3^5 = 243 < 256). per-tensor mean-abs scale (gamma). spec at
// documentation/kernels/ternary.md.
// ------------------------------------------------------------------------------------

typedef struct {
    uint8_t* bt_packed;     // [n][ceil(k/5)]   row j holds 5 trits per byte
    int32_t* row_q24;       // [n]              per-row q24 multiplier for requant
    float    gamma;         // per-tensor mean-abs scale
    int32_t  n;
    int32_t  k;
} prepped_b_ternary_t;

// pack k trits (each in {-1,0,1}) into ceil(k/5) bytes. tail trits past k are 0.
void ternary_pack_row(const int8_t* trits, int32_t k, uint8_t* out_bytes);

// unpack ceil(k/5) bytes back into k trits in {-1,0,1}.
void ternary_unpack_row(const uint8_t* bytes, int32_t k, int8_t* out_trits);

// build a prepped ternary block from an int8 weight tensor of shape [n,k]
// (column-major, b[p*n+j]). values must be in {-1, 0, +1}. gamma is the per-
// tensor mean-abs scale produced at QAT export time.
void prep_b_ternary(const int8_t* b_trits, int32_t n, int32_t k,
                    float gamma, prepped_b_ternary_t* out);
void free_prepped_b_ternary(prepped_b_ternary_t* p);

// scalar oracle. unpacks trits, dot-products against int8 activations, writes
// int32 output. m=1 decode path; m>1 supported for prefill.
void matmul_ternary_scalar_prep(const int8_t* a, const prepped_b_ternary_t* p,
                                int32_t* c, int32_t m);

// avx-512 fast path. unpacks 5 trits per byte into a sequential int8 cache
// line, then dispatches through vnni dpbusd. bit-identical to scalar.
void matmul_ternary_vnni_prep(const int8_t* a, const prepped_b_ternary_t* p,
                              int32_t* c, int32_t m);

// ------------------------------------------------------------------------------------
// transformer hot-path kernels — runtime-dispatched per arch.
// callers go through score_dot_v / softmax_rows / layernorm_i16_to_i8 (function
// pointers); dispatch_init points each at the best impl for the live cpu.
// every backend matches these signatures bit-for-bit (rule 23).
// ------------------------------------------------------------------------------------

typedef void (*score_dot_v_fn)(const int16_t* scores, const int8_t* v_base,
                               int32_t v_stride, int32_t n_j, int8_t* out);
typedef void (*softmax_rows_fn)(float* x, int16_t* out_q,
                                int32_t rows, int32_t cols);
typedef void (*layernorm_i16_to_i8_fn)(const int16_t* x, int8_t* out, const int8_t* w,
                                       int32_t rows, int32_t cols);

extern score_dot_v_fn         score_dot_v;
extern softmax_rows_fn        softmax_rows;
extern layernorm_i16_to_i8_fn layernorm_i16_to_i8;

// x86_64 backends (avx-512 + bw + vnni)
void score_dot_v_avx512(const int16_t* scores, const int8_t* v_base,
                        int32_t v_stride, int32_t n_j, int8_t* out);
void softmax_rows_avx512(float* x, int16_t* out_q, int32_t rows, int32_t cols);
void layernorm_i16_to_i8_avx512(const int16_t* x, int8_t* out, const int8_t* w,
                                int32_t rows, int32_t cols);

// arm64 backends (neon + sdot)
void score_dot_v_neon(const int16_t* scores, const int8_t* v_base,
                      int32_t v_stride, int32_t n_j, int8_t* out);
void softmax_rows_neon(float* x, int16_t* out_q, int32_t rows, int32_t cols);
void layernorm_i16_to_i8_neon(const int16_t* x, int8_t* out, const int8_t* w,
                              int32_t rows, int32_t cols);

// scalar oracle. used as the rule-23 correctness reference and as the default
// before dispatch_init has run.
void score_dot_v_scalar(const int16_t* scores, const int8_t* v_base,
                        int32_t v_stride, int32_t n_j, int8_t* out);
void softmax_rows_scalar(float* x, int16_t* out_q, int32_t rows, int32_t cols);
void layernorm_i16_to_i8_scalar(const int16_t* x, int8_t* out, const int8_t* w,
                                int32_t rows, int32_t cols);

// ------------------------------------------------------------------------------------
// dispatch
// ------------------------------------------------------------------------------------

typedef struct {
    const char* matmul_backend;
} dispatch_info_t;

void dispatch_init(const cpu_features_t* feat, dispatch_info_t* out);

// ------------------------------------------------------------------------------------
// timing
// ------------------------------------------------------------------------------------

double now_ms(void);

// ------------------------------------------------------------------------------------
// v3 — transformer forward pass
// ------------------------------------------------------------------------------------

// default shape — bin header overrides at load time. kept for legacy callers and
// for static buffer sizing in trace/sample paths bounded by the hardest case.
#define V_VOCAB     256
#define V_SEQ       256
#define V_HIDDEN    768
#define V_HEADS     12
#define V_HEAD_DIM  (V_HIDDEN / V_HEADS)
#define V_FFN       3072
#define V_LAYERS    12

// hard cap on runtime ffn dim. sizes the sparse-decode prescan buffers in
// kernels/x86_64/transformer_avx512.c. raise to admit wider models; the only
// cost is BSS footprint (two int32 arrays of V_MAX_FFN). models with
// shape.ffn > V_MAX_FFN are rejected at load time.
#define V_MAX_FFN   8192

// runtime shape. populated by model_load from the bin header.
typedef struct {
    int32_t vocab;
    int32_t seq;
    int32_t hidden;
    int32_t heads;
    int32_t head_dim;
    int32_t ffn;
    int32_t layers;
} veritate_shape_t;

// per-row scale tracks the fp32 quantum each int8 represents
typedef struct {
    int8_t* data;
    float*  scale;     // one per row
    int32_t rows;
    int32_t cols;
} qtensor_t;

// layer weights. int8 values + per-row scales + bias int32 cache.
// when use_int4 is set, the *_i4 fields hold prepped int4 weights and the int8 fields
// are unallocated. dispatch checks use_int4 at forward time.
// when has_gate is set, gate_w/gate_scale_q24 hold a per-token mod (mixture-of-depths)
// gate. gate is hidden -> 1 int8 dot product followed by sigmoid; the block is skipped
// when sigmoid(g) < 0.5.
typedef struct {
    prepped_b_t qkv;       // [3*hidden x hidden]   q, k, v stacked
    prepped_b_t out_proj;  // [hidden x hidden]
    prepped_b_t ffn_up;    // [ffn x hidden]   (n_experts=1 path)
    prepped_b_t ffn_down;  // [hidden x ffn]   (n_experts=1 path)
    prepped_b_int4_t qkv_i4;
    prepped_b_int4_t out_proj_i4;
    prepped_b_int4_t ffn_up_i4;
    prepped_b_int4_t ffn_down_i4;
    int8_t*     ln1_w;     // [hidden]
    int8_t*     ln2_w;     // [hidden]
    int32_t     use_int4;  // 0 = int8 path, 1 = int4 + quarot path
    int32_t     has_gate;  // 0 = no mod gate, 1 = use mod gate
    int8_t*     gate_w;    // [hidden] int8 row, present when has_gate
    int32_t     gate_scale_q24;  // q24 requant for gate dot
    // MoE fields (v10+). when n_experts == 1, experts_up / experts_down are
    // NULL and the standard ffn_up / ffn_down above are used. when
    // n_experts > 1, the per-expert blocks below replace the standard pair
    // and router holds a [n_experts x hidden] matrix that produces routing
    // logits per token.
    int32_t      n_experts;     // 1 = no MoE
    int32_t      router_topk;   // experts active per token; 1 = sticky
    prepped_b_t  router;        // [n_experts x hidden]
    prepped_b_t* experts_up;    // [n_experts] of [ffn x hidden]
    prepped_b_t* experts_down;  // [n_experts] of [hidden x ffn]
} block_t;

// runtime-shaped model. embed, pos_embed, blocks, byte_direction[*], scratch all
// heap-allocated by model_load / model_init_random based on shape.
typedef struct {
    veritate_shape_t shape;
    int8_t*  embed;            // [vocab * hidden]
    int8_t*  pos_embed;        // [seq * hidden]
    int32_t  act_boost;        // residual stream scale = act_boost * ACT_INT8_SCALE
    int32_t  quant_mode;       // VERITATE_QUANT_*. v9 and earlier load as INT8.
    int32_t  n_experts;        // 1 = no MoE. v10+ only; older versions default to 1.
    int32_t  router_topk;      // experts active per token. 1 = sticky single-expert.
    block_t* blocks;           // [layers]
    int16_t** byte_direction;  // [layers] of [ffn * vocab] (NULL until built)
    float*   byte_direction_scale; // [layers]
    int8_t*  n_out_w;          // [hidden] final RMSNorm weight, NULL when absent
    prepped_b_t lm_head;
    // v12 MTP byte-0 head. mtp_present == 0 on v11 and earlier; lm_head is then
    // tied-from-embed in lm_head_build. mtp_present == 1 means lm_head was loaded
    // untied from the file and project_byte0 must be applied before the lm_head
    // matmul to recover logits faithfully.
    int32_t      mtp_present;
    prepped_b_t  mtp_transform0;   // [hidden x hidden] linear, byte-0 MTP head
    int8_t*      mtp_norm0_w;      // [hidden] RMSNorm scale-64, byte-0 MTP head
    int32_t*     mtp_scratch_i32;  // [hidden] reusable int32 buffer for project_byte0
    int16_t*     mtp_scratch_i16;  // [hidden] reusable int16 buffer for project_byte0
    void*    scratch;          // shape-sized acts pool, opaque to callers
    // calibrated confidence head. loaded from confidence_weights.json next to the
    // model bin or via VERITATE_CONFIDENCE_WEIGHTS env var. cw_loaded == 0 falls
    // back to the placeholder formula in chat_traced_loop.
    float    cw_M;
    float    cw_E;
    float    cw_L;
    float    cw_S;
    float    cw_b;
    int32_t  cw_loaded;
} model_t;

void model_init_random(model_t* m, unsigned seed);
void model_free(model_t* m);

// on-disk model format — header + raw int8 weights, prepared at load time.
// returns 0 on success, nonzero on failure (file missing, magic mismatch, shape mismatch).
typedef struct {
    char     magic[4];    // "VRTE"
    uint32_t version;
    uint32_t v_vocab;
    uint32_t v_hidden;
    uint32_t v_layers;
    uint32_t v_ffn;
    uint32_t v_heads;
    uint32_t v_seq;
} model_header_t;

#define VERITATE_MODEL_MAGIC "VRTE"
#define VERITATE_MODEL_VERSION 3
#define VERITATE_MODEL_VERSION_INT4 4
#define VERITATE_MODEL_VERSION_PERCOL 5
#define VERITATE_MODEL_VERSION_MOD 6
#define VERITATE_MODEL_VERSION_NORM 8
#define VERITATE_MODEL_VERSION_BOOST 9
// v10 was assigned twice on different branches (MoE-on-dev vs ternary-on-experimental)
// and was retired during the merge. v11 is the unified successor. v9 BOOST and earlier
// load unchanged.
#define VERITATE_MODEL_VERSION_QAT 11
// v12: v11 body + MTP byte-0 head (mtp.transforms[0], mtp.norms[0]) + untied lm_head.
#define VERITATE_MODEL_VERSION_MTP 12

// quant_mode values stored in the v11 header.
#define VERITATE_QUANT_INT8    0
#define VERITATE_QUANT_INT4    1
#define VERITATE_QUANT_TERNARY 2

// per-head sylvester hadamard, size 64 = V_HEAD_DIM, normalized 1/sqrt(64).
// applies block-diagonally along the in-channel axis. cols must be a multiple of 64.
// in-place (dst may equal src). int8 in/out, internal float math.
void hadamard_apply_int8(const int8_t* src, int8_t* dst, int32_t cols);

int model_load(model_t* m, const char* path);

// load 4 weights + bias from a json blob {"w_M":..,"w_E":..,"w_L":..,"w_S":..,"b":..}.
// returns 0 on success and sets m->cw_loaded = 1.
int confidence_weights_load(model_t* m, const char* path);

// kv cache — per-layer K/V over up to seq positions, runtime-shaped.
// k and v are flat layers*seq*hidden buffers; index via cache_kv_row.
typedef struct {
    veritate_shape_t shape;
    int8_t*  k;        // [layers * seq * hidden]
    int8_t*  v;        // [layers * seq * hidden]
    int32_t  len;
} kv_cache_t;

void kv_cache_init(kv_cache_t* c, const veritate_shape_t* s);
void kv_cache_free(kv_cache_t* c);
void kv_cache_copy(kv_cache_t* dst, const kv_cache_t* src);

static inline int8_t* cache_k_row(kv_cache_t* c, int32_t L, int32_t p) {
    return c->k + ((size_t)L * c->shape.seq + (size_t)p) * c->shape.hidden;
}
static inline int8_t* cache_v_row(kv_cache_t* c, int32_t L, int32_t p) {
    return c->v + ((size_t)L * c->shape.seq + (size_t)p) * c->shape.hidden;
}

// project mri — interpretability trace
#define VERITATE_TRACE_TOPK 5
#define VERITATE_DLA_TOPK   12
// v8: count of next-byte candidates that get a per-candidate DLA. matches the
// dashboard's `cand` length so dla_cand[i] pairs with cand[i] by index.
#define VERITATE_CAND_TOPK  12

typedef struct {
    uint8_t token;
    int32_t logit;
} trace_prediction_t;

// v8 decision-trace entry. layer + neuron identify the source; act/w/contrib are the
// raw int values (act = post-GELU int8 activation, w = int16 byte_direction, contrib = act*w).
typedef struct {
    uint8_t  layer;
    uint8_t  pad;
    uint16_t neuron;
    int32_t  act;
    int32_t  w;
    int32_t  contrib;
} dla_entry_t;

// build per-layer int16 byte_direction tables from ffn_down (b_rowmaj) + embed.
// allocates m->byte_direction[L] and sets m->byte_direction_scale[L]. callable
// once after weights are populated. on failure returns nonzero and frees nothing.
int  byte_direction_build(model_t* m);
void byte_direction_free(model_t* m);

// per-layer decisiveness from lens_logits at a single position. fills out[layers]
// with max_abs(delta) / mean_abs(delta) where delta is logits[L] minus logits[L-1]
// (and logits[0] alone for L=0). lens_logits points at the [layers][vocab] slice
// for the chosen position.
void decisiveness_compute(const veritate_shape_t* sh, const int32_t* lens_logits_pos, float* out);

// top-K (layer, neuron) contributors to byte's logit. ffn_neurons_pos points at the
// [V_LAYERS][V_FFN] slice for the chosen position. fills out[VERITATE_DLA_TOPK]
// sorted by |contrib| descending. requires m->byte_direction populated.
void dla_top(const model_t* m, const int8_t* ffn_neurons_pos, int32_t byte,
             dla_entry_t* out);

typedef struct {
    int16_t* residual_pre;       // [layers][seq][hidden]
    int16_t* residual_post;      // [layers][seq][hidden]
    int8_t*  ffn_neurons;        // [layers][seq][ffn]
    int8_t*  final_act;          // [hidden]
    uint8_t* prompt_bytes;       // [real_len]
    trace_prediction_t* top_predictions;  // [VERITATE_TRACE_TOPK]
    float*   attention_scores;   // [layers][heads][seq][seq] post-softmax. NULL = skip.
    int32_t* lens_logits;        // [layers][seq][vocab] embed @ residual_post. NULL = skip.
} trace_record_t;

// per-stage profile. ms accumulators.
typedef struct {
    double embed_ms;
    double ln_ms;
    double qkv_ms;
    double attn_ms;
    double out_proj_ms;
    double ffn_up_ms;
    double gelu_ms;
    double ffn_down_ms;
} profile_t;

// prefill real_len <= V_SEQ tokens into cache. trace + prof optional — pass NULL to skip.
void forward(const model_t* m, kv_cache_t* cache, const int32_t* tokens,
             int32_t real_len, int8_t* out_act, trace_record_t* trace, profile_t* prof);

// decode one new token at position cache->len. cache->len must be < V_SEQ.
// trace optional — when non-null, residual_pre/post, ffn_neurons, attention_scores at the
// new position, and final_act, are written into the buffer.
void forward_decode(const model_t* m, kv_cache_t* cache, int32_t token, int8_t* out_act,
                    trace_record_t* trace);

// process K new tokens (1 <= K <= VERITATE_VERIFY_K_MAX) given a cache populated up to
// cache->len. writes hidden states for each of the K positions to out_hidden_K, row-major
// [K, V_HIDDEN]. cache->len advances by K on success. bit-equivalent within 1 LSB to
// running K sequential forward_decode calls. dispatches on K: K=1 routes to forward_decode,
// K in [2,7] uses single-thread batched matmul, K>=8 uses multi-thread.
#define VERITATE_VERIFY_K_MAX 16
void forward_verify(const model_t* m, kv_cache_t* cache, int32_t K,
                    const int32_t* tokens, int8_t* out_hidden_K);

// active layer cap, read once from VERITATE_MAX_LAYERS env var, clamped to
// [1, m->shape.layers]. callers pass a model_t to get the per-model clamp.
int32_t veritate_max_layers(const model_t* m);

// mod gate counters. process-global. updated only on forward_decode paths.
void veritate_mod_stats(int64_t* calls, int64_t* skipped);
void veritate_mod_stats_reset(void);

// causal ablation (v8). when (layer, neuron) is non-negative, forward_decode
// zeros ffn_neurons[layer][pos][neuron] post-GELU before ffn_down on every
// position, every turn, until reset. process-global; chat_traced reads/writes
// it per turn from the stdin header. layer or neuron == -1 disables ablation.
void veritate_set_ablation(int32_t layer, int32_t neuron);
void veritate_get_ablation(int32_t* layer, int32_t* neuron);

#define VERITATE_TRACE_MAGIC              "VRMR"
#define VERITATE_TRACE_VERSION            8
#define VERITATE_TRACE_VERSION_ATTN_QUANT 6
#define VERITATE_TRACE_VERSION_CONFIDENCE 7
#define VERITATE_TRACE_VERSION_CAND_DLA   8

typedef struct {
    char     magic[4];        // "VRMR"
    uint32_t version;
    uint32_t v_layers;
    uint32_t v_seq;
    uint32_t v_hidden;
    uint32_t v_ffn;
    uint32_t v_heads;
    uint32_t real_len;        // number of valid token positions
} trace_header_t;

int trace_write(const char* path, const veritate_shape_t* sh,
                const trace_record_t* trace, int32_t real_len);

// tied-embedding lm head + sampler. temp<=0 argmax. top_k<=0 no filter.
int32_t sample_token(const model_t* m, const int8_t* hidden, float temp, int32_t top_k, uint32_t* rng);
int32_t sample_token_ext(const model_t* m, const int8_t* hidden, float temp, int32_t top_k,
                         uint32_t* rng, int32_t* out_logits, int32_t* out_argmax);
void    lm_head_build(model_t* m);

// v12: byte-0 MTP head projector. When mtp_present == 0, copies h_in to h_out.
// When mtp_present == 1, applies mtp.transforms[0] (hidden->hidden linear) then
// mtp.norms[0] (RMSNorm scale-64), producing the int8 hidden the lm_head matmul
// consumes. Must be called before any matmul against m->lm_head when emitting
// real byte-0 logits (sample_token_ext does this internally).
void    model_project_byte0(const model_t* m, const int8_t* h_in, int8_t* h_out);

// byte-level tokenizer — one token per byte, vocab maps directly to 0..255
int32_t tokenize_bytes(const char* text, int32_t* tokens, int32_t max_tokens);
void    detokenize_bytes(const int32_t* tokens, int32_t n, char* out);

#endif
