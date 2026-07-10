// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - v13 hybrid trunk: fp32 decode path. local attention blocks on every byte,
//   gla recurrent global blocks on boundary-anchored slots. O(1) state per byte:
//   bounded KV over seq for local blocks, [heads x d x d] state + conv ring per
//   global block. spec: developer_documentation/engine/engine_v13_hybrid.md.
// ------------------------------------------------------------------------------------

#ifndef VERITATE_HYBRID_H
#define VERITATE_HYBRID_H

#include <stdint.h>
#include <stdio.h>

#include "veritate.h"

#define VERITATE_HYBRID_DTYPE_FP32   0
#define VERITATE_HYBRID_DTYPE_FP16   1
#define VERITATE_HYBRID_DTYPE_INT8   2
#define VERITATE_HYBRID_LOGIT_SCALE  1024.0f

// int8 big-weight tensor: per-output-row symmetric int8 + fp32 scales. small
// tensors and embeddings stay fp32 in int8 bins.
typedef struct {
    const int8_t* q;      // [n x k] row-major
    const float*  scale;  // [n]
} hybrid_w_i8_t;

// big matmul weights stay in the bin dtype (fp32 or fp16) and are consumed
// through the dtype-dispatched matvec; small tensors are upconverted to fp32
// at load.
typedef struct {
    float*  n1_w;       // [H]
    void*   qkv_w;      // [3H x H] row-major [out][in], bin dtype
    void*   proj_w;     // [H x H], bin dtype
    float*  n2_w;       // [H]
    void*   up_w;       // [F x H], bin dtype
    void*   down_w;     // [H x F], bin dtype
    // recurrent-only; NULL on attention blocks
    float*  conv_w;     // [3H x conv_kernel]
    float*  a_proj_w;   // [heads x H]
    float*  a_proj_b;   // [heads]
    float*  o_norm_w;   // [head_dim]
    void*   gate_w;     // [H x H], bin dtype
    int32_t is_recurrent;
} hybrid_block_t;

typedef struct {
    int32_t vocab, hidden, layers, ffn, heads, head_dim, seq;
    int32_t dtype, n_enc, n_global, n_dec;
    int32_t patch_stride, slots, conv_kernel, state_rule;
    size_t  wt_esz;         // bytes per big-weight element (2 fp16 / 4 fp32)
    uint8_t boundary[256];
    float*  tok_emb;        // [vocab x H]
    float*  pos_emb;        // [seq x H]
    float*  slot_pos_emb;   // [slots x H]
    hybrid_block_t* blocks; // [layers]
    float*  n_out_w;        // [H]
    // decode state
    float*  kv_k;           // [(n_enc+n_dec) x seq x H]
    float*  kv_v;
    float*  rec_state;      // [n_global x heads x d x d]
    float*  conv_ring;      // [n_global x (conv_kernel-1) x 3H] oldest-first
    int32_t slot_count;
    int32_t pos;
    // scratch, allocated once; logits refreshed every step
    float*  x;              // [H]
    float*  g;              // [H]
    float*  u;              // [H]
    float*  qkv;            // [3H]
    float*  conv_out;       // [3H]
    float*  attn_out;       // [H]
    float*  scores;         // [seq]
    float*  ffn_buf;        // [ffn]
    float*  gate_buf;       // [H]
    float*  tmp;            // [H]
    float*  logits;         // [vocab] fp32, valid after each step
} hybrid_t;

// matvec kernel contract: out[j] = dot(w_row_j, x). w is fp32 or fp16 by
// kernel. scalar references accumulate in 8 interleaved partial sums reduced
// as ((s0+s4)+(s1+s5)) + ((s2+s6)+(s3+s7)); SIMD ports must be bitwise equal.
// fp16 kernels convert each weight to fp32 (exact) before the same fp32 math.
typedef void (*hybrid_matvec_fn)(const void* w, const float* x, float* out,
                                 int32_t n, int32_t k);
// hybrid_matvec_wt: bin-dtype weights (the five big per-block matrices).
// hybrid_matvec_fp: always-fp32 weights (tied lm_head / tok_emb).
extern hybrid_matvec_fn hybrid_matvec_wt;
extern hybrid_matvec_fn hybrid_matvec_fp;
void hybrid_matvec_f32_scalar(const void* w, const float* x, float* out,
                              int32_t n, int32_t k);
void hybrid_matvec_f32_neon(const void* w, const float* x, float* out,
                            int32_t n, int32_t k);
void hybrid_matvec_f16_scalar(const void* w, const float* x, float* out,
                              int32_t n, int32_t k);
void hybrid_matvec_f16_neon(const void* w, const float* x, float* out,
                            int32_t n, int32_t k);
// int8 kernels: w is a hybrid_w_i8_t*. activations quantize per call through
// hybrid_quant_act (shared, so scalar and sdot see identical int8 inputs);
// int32 accumulation is exact, so scalar/sdot outputs are bitwise-identical.
void hybrid_matvec_i8_scalar(const void* w, const float* x, float* out,
                             int32_t n, int32_t k);
void hybrid_matvec_i8_sdot(const void* w, const float* x, float* out,
                           int32_t n, int32_t k);
// symmetric per-call activation quant: qx = round(x * 127/absmax) clamped.
// returns absmax/127; 0 when x is all-zero (caller zeroes the output).
float hybrid_quant_act(const float* x, int32_t k, int8_t* qx);
float hybrid_f16_to_f32(uint16_t hb);
void hybrid_dispatch_init(int has_neon, int32_t dtype);

// load: f positioned after the 32-byte model header. returns NULL on failure.
hybrid_t* hybrid_load(FILE* f, int32_t vocab, int32_t hidden, int32_t layers,
                      int32_t ffn, int32_t heads, int32_t seq);
void hybrid_free(hybrid_t* h);

// reset stream state (turn boundary): zero recurrent states, conv rings,
// slot ordinal, position.
void hybrid_reset(hybrid_t* h);

// one byte through the full stack. advances pos. logits land in h->logits.
// trace optional: local-block residuals / ffn acts / attention rows are filled
// at the step position, global-block sections and lens stay zeroed.
void hybrid_step(hybrid_t* h, int32_t byte, trace_record_t* trace);

// int32 view of h->logits at VERITATE_HYBRID_LOGIT_SCALE for the shared
// sampler / telemetry surface.
void hybrid_logits_i32(const hybrid_t* h, int32_t* out);

// final normed hidden as int8 at activation scale 32 (trace display only).
void hybrid_final_act_i8(const hybrid_t* h, int8_t* out);

#endif
