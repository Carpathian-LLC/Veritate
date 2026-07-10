# engine v13: hybrid trunk (.bin format + decode path)

v13 is the engine format for the hybrid trunk (`VeritatePatched` with
`global_mixer="recurrent"`, `state_rule="gla"`): local attention blocks on every
byte, constant-state recurrent global blocks on boundary-anchored patch slots.
Written by `export_checkpoint` in [veritate_mri/training/export.py](../../veritate_mri/training/export.py)
when `config.training_args.trunk == "hybrid"`; loaded and decoded by
[veritate_engine/v1/src/hybrid.c](../../veritate_engine/v1/src/hybrid.c).
Canonical dense formats v3-v12 are untouched: v13 is a new version byte with its
own loader and forward path, dispatched in `model_load`
([model.c](../../veritate_engine/v1/src/model.c)).

## numeric contract

The hybrid checkpoints are NOT QAT-trained; the int16-residual / int8-activation
pipeline of v3-v12 is not quality-safe for them. v13 computes in fp32
end-to-end. Weights are stored fp32 or fp16 (header `dtype`); fp16 is
dequantized to fp32 at load. Recurrent state, conv ring, KV caches, residuals,
logits: all fp32. Quantization beyond fp16 requires the quality gates below.

- Scalar C path is the reference; SIMD kernels must be bitwise-identical to it
  (preflight rule 24). The scalar matvec accumulates in 4 interleaved partial
  sums so a 4-lane SIMD port reproduces it exactly.
- Logits cross the existing int32 telemetry/sampler surface scaled by
  `VERITATE_HYBRID_LOGIT_SCALE` (1024), mirroring the v9 `1/1024` convention in
  `ppl_mode`. The sampler folds that scale into the temperature
  (`t_eff = temp * 1024` for hybrid) so the softmax sees true `logit / temp`
  units; without the fold, sampling was effectively `temp / 1024` (near-greedy
  at any setting). The repetition soft penalty subtracts after the temperature
  division, in the scale-free nat units `repetition.py` defines, so the fold
  also lands the penalty at its designed strength for v13.

### quantization plan + quality gates (measured, chat80m_80m step_48000)

Measured against the PyTorch fp32 reference (same checkpoint, same prompts).
Parity = greedy decode, 3 chat prompts x 64 bytes; bpb = 20 chunks x 1024 on
fineweb_edu_val / chat_v1_val:

| mode | storage | compute | gate result |
|---|---|---|---|
| fp32 | 487.0 MB | fp32 | PASS: parity 192/192; bpb 1.4447 / 0.9899 (PyTorch: 1.44466 / 0.98989) |
| fp16 | 243.5 MB | fp32 (exact convert) | PASS: parity 192/192; bpb identical to 4 decimals. **shipping default** |
| int8 (dtype=2) | 126.5 MB | int8 sdot + fp32 requant | PASS (step 51000): bpb +0.0040 fineweb / +0.0020 chat_v1 vs fp16 (gate < 0.005); greedy transcripts coherent. dynamic per-call activation quant (`hybrid_quant_act`, absmax/127) + per-output-row fp32 weight scales; small tensors + embeddings stay fp32. sdot kernel ~3.9x the fp16 kernel (at the bandwidth ceiling) |

## .bin layout (little-endian)

```
model_header_t              32 B   "VRTE", version=13, vocab, hidden,
                                   layers=TOTAL blocks (16), ffn, heads, seq
int32  dtype                0=fp32, 1=fp16, 2=int8
int32  n_local_enc          2
int32  n_global             12
int32  n_local_dec          2      (n_local_enc + n_global + n_local_dec == layers)
int32  patch_stride         4
int32  slots                seq / patch_stride (256)
int32  conv_kernel          4
int32  state_rule           0=gla (only value accepted)
uint8  boundary[256]        1 = boundary byte (from model_patched._boundary_table
                                   + "position 0 is always a boundary" handled at runtime)
tensors, raw, PyTorch [out, in] row-major, header dtype. int8 (dtype=2) bins
store only the five big per-block matrices (qkv/gate/proj/ff_up/ff_down) as
int8: q[n*k] followed by fp32 scale[n] (per-output-row symmetric, absmax/127);
every other tensor is fp32:
  tok_emb      [vocab,  hidden]
  pos_emb      [seq,    hidden]
  slot_pos_emb [slots,  hidden]
  blocks 0..layers-1 in order (enc attn, global recurrent, dec attn):
    attn block:      n1[H] qkv[3H,H] proj[H,H] n2[H] ff_up[F,H] ff_down[H,F]
    recurrent block: n1[H] qkv[3H,H] conv[3H,conv_kernel] a_proj_w[heads,H]
                     a_proj_b[heads] o_norm[head_dim] gate[H,H] proj[H,H]
                     n2[H] ff_up[F,H] ff_down[H,F]
  n_out [H]
```

lm_head is tied to tok_emb (exporter verifies and refuses untied). Weights stay
in [out, in] row-major because the decode hot loop is row-dot matvec; no
transpose at export or load (the v3-v12 `prep_b` transpose convention does not
apply).

## decode dataflow

Per byte at position `pos` (prefill = the same step repeated over the prompt;
exactness relies on causal attention + the sequential form of the chunkwise
recurrence, which are mathematically identical to the training forward):

1. `x = tok_emb[byte] + pos_emb[pos]` (fp32, [H]).
2. 2 local enc blocks: pre-norm attention (RMSNorm eps 1e-5, qkv matvec, causal
   attention against that block's fp32 KV cache over positions 0..pos, softmax
   fp32, out-proj, residual add) then FFN (RMSNorm, up, exact-GELU via erf,
   down, residual add). KV bounded by seq: cache is [seq, H] per block per K/V,
   decode refuses pos >= seq exactly like the dense path.
3. Boundary test: `boundary[byte] || pos == 0`. Non-boundary bytes skip to 5.
4. Boundary byte with slot ordinal `s = n_boundaries_so_far - 1 < slots`:
   `g = x + slot_pos_emb[s]`, then 12 recurrent blocks, each:
   - `u = rmsnorm(g, n1)`
   - `qkv_raw = W_qkv @ u`; depthwise conv over slot time from a ring of the
     last `conv_kernel-1` raw qkv vectors (zeros at stream start):
     `c[j] = sum_i conv_w[j,i] * ring[t-3+i][j]`, newest aligns with `conv_w[:,3]`
   - split q,k,v per head (head_dim = H/heads); `q *= head_dim^-0.5`
   - per-head decay `a = exp(-softplus(a_proj_w @ u + a_proj_b))`
   - state update (the O(1) core): `S_h = a_h * S_h + k_h (outer) v_h`;
     `o_h = q_h @ S_h`   (S_h is [head_dim, head_dim] fp32)
   - `o_h = rmsnorm(o_h, o_norm)` per head; concat; `o *= silu(W_gate @ u)`;
     `g += W_proj @ o`; then FFN residual as in step 2.
   Slot ordinals >= slots: no state update, no scatter (matches training's
   `slot_of < S` mask). The result `g` is added back: `x += g`.
5. 2 local dec blocks (same as step 2, own KV caches).
6. `logits = tok_emb @ rmsnorm(x, n_out)` ([vocab]); greedy = fp32 argmax.

### O(1)-state contract

Per-byte state is fixed at any position: 4 attention KV caches bounded by
`seq*H` each, 12 recurrent states of `heads*head_dim^2` + conv rings of
`(conv_kernel-1)*3H`. No allocation in the decode loop; total hybrid state at
h=768/seq=1024 is ~28 MB fp32. Per-byte compute is bounded: ~56 MFLOP
(non-boundary) / ~244 MFLOP (boundary, +12 recurrent blocks), plus the
attention dot term growing linearly to the seq cap.

## engine integration

- `model_t` gains an opaque `hybrid` pointer; v13 in `model_load` routes to
  `hybrid_load`. `forward` / `forward_decode` / `forward_verify` /
  `sample_token_ext` branch to the hybrid path when set, so `chat`,
  `chat_greedy`, `chat_traced`, and `bench` run unmodified.
- `chat_traced` frames stay format-complete (the Python parser sizes sections
  from the bin header: 16 per-layer sections): local-block residuals, FFN
  neurons, and attention rows carry real (fp-converted) values; global-block
  sections, lens logits, and DLA entries are zeros (slot-level telemetry is not
  meaningful in the dense per-layer frame; same caveat as model_patched.py).
- Header `layers` = 16 total blocks so every shape-derived consumer
  (c_engine.py parser, dashboard meta) sees the real section count.

## sampler: repetition control (chat_traced)

`sample_token_ext` ([model.c](../../veritate_engine/v1/src/model.c)) takes an optional
`rep_ctx_t*` (NULL disables). `chat_traced` accumulates the turn's generated bytes
and passes them each step; the wire header gains three optional trailing fields
`rep_window rep_penalty no_repeat_ngram` (0/0/0 = off, so old headers and the
greedy/spec paths are unaffected). The mechanism mirrors the PyTorch decoder
([veritate_mri/inference/decode/repetition.py](../../veritate_mri/inference/decode/repetition.py)):
a byte-level no-repeat-ngram **hard ban** (offending logits set to `INT32_MIN`
before the top-k heap) plus a **soft** suffix-repeat penalty subtracted from the
post-temperature float logits in the softmax loop (`fp[v] -= rep_soft[v]`), so the
demotion is scale-free despite the `x1024` hybrid logit view. Constants
(`VERITATE_REP_MIN_MATCH=4`, `VERITATE_REP_MATCH_CAP=64`) are kept in sync with
`repetition.py`. With `rep` disabled `rep_soft` is never read and logits are
untouched, so the sampler is bitwise-identical to the pre-feature build — the v9
greedy golden ([tests/engine/test_v13_compat.py](../../tests/engine/test_v13_compat.py))
still matches, which is the penalty-off parity assertion.

## build: V_SEQ=1024

`V_SEQ` in veritate.h becomes `#ifndef`-guarded; build.sh / build.bat pass
`-DV_SEQ=1024`. V_SEQ only sizes static fallback buffers (`mod_keep_row`) and
the random-init default shape; runtime shapes come from the bin header, so
raising it is compat-neutral for v3-v12 bins.

## compat + parity test plan

1. **Canonical fixture regression**: a tiny v9 fixture bin checked against a
   recorded greedy transcript before and after the v13 changes
   (`tests/engine/test_v13_compat.py`); byte-identical output required.
2. **Exporter round-trip** (`tests/export/test_export_v13.py`): write v13 from
   a small synthetic hybrid state dict, re-parse with a Python reader, verify
   header fields, tensor shapes, boundary table, and byte counts.
3. **Byte-parity gate (S3)**: greedy decode of >= 3 fixed prompts (>= 64 bytes
   each) from `chat80m_80m/step_48000` compared byte-for-byte against the
   PyTorch fp32 greedy reference (full `forward()` per step). Any divergence is
   located (byte index, logit gap at the flip) and justified before proceeding.
4. **Kernel identity (S4)**: every SIMD kernel bitwise-equal to the scalar
   reference on randomized shapes (rule 24).

## measured performance (M3 Ultra, 64-byte greedy generations from chat prompts)

Threaded matvec splits rows across the pool for matmuls with n*k >= 2^20
elements (int8: >= 2^23 — the sdot kernel runs ~4x the fp rate, so smaller
splits lose to pool dispatch; measured 1T >= 4T for int8 at both shapes below).
Row-splits are bitwise-identical to single-thread; `VERITATE_HYBRID_THREADS`
overrides (default 4). p50 is a non-boundary byte, p95 a boundary byte (the
recurrent stack fires). Prompts whose replies are boundary-dense read p50 near
p95. Kernel bench (768x3072 matvec, single-thread): fp32 0.118 ms, fp16
0.112 ms, int8 sdot 0.030 ms (~79 GB/s, bandwidth ceiling).

121.75M chat80m (step 51000; quiet-machine fp16-4T at step 48000 measured
1.08-1.12 p50):

| config | p50 ms/byte | p95 ms/byte | peak tok/s |
|---|---|---|---|
| fp16 4T | 1.58-1.60 | 6.6-6.7 | 418 |
| fp16 1T | 1.98-2.00 | 8.2-8.6 | 339 |
| int8 4T | 1.22-1.25 | 3.4-3.5 | 639 |
| **int8 1T** | **1.17-1.18** | **3.35-3.36** | **660** |
| int8 1T E-core (`taskpolicy -b`) | 3.8-4.1 | 14.1-14.5 | 187 |

200m-class shape (h1024, 4 local + 16 global blocks, ffn 4096, heads 16,
seq 1024; 270.8M params by the same manifest-shape convention; random weights,
latency-only):

| config | p50 ms/byte | p95 ms/byte |
|---|---|---|
| fp16 4T | 2.06-2.11 | 11.0-11.1 |
| fp16 1T | 3.36-3.46 | 17.5-17.8 |
| **int8 1T/4T** | **1.86-1.91** (1.73-1.79 lighter load) | 6.6-6.7 |
| int8 1T E-core | 3.2-3.4 | 15.2-16.4 |
| fp16 4T E-core | 3.7-5.1 | 19-27 |

Verdict vs the <= 2 ms/byte 200m-class target: **int8 meets it on M-class
performance cores, single-threaded** (fp16 misses at 2.1-3.5). Efficiency-core
class silicon sits at ~3.2-3.4 ms p50 with int8 — not yet ms-class there;
remaining levers are boundary-density-aware scheduling and fp16 activations in
the attention KV path.

## pitfalls

- `readers/bin.py::act_boost` must return None for v13 (first extension field
  is `dtype`, not act_boost) or the dashboard mislabels fp16 bins as boosted
  non-QAT and warns about gibberish.
- The boundary table is baked at export from Python `chr(b).isalnum()`
  semantics; do not re-derive it in C (Latin-1 alnum handling differs).
- Chunkwise (training) vs sequential (decode) recurrence are algebraically
  identical but round differently in fp32; parity is defined at greedy-argmax
  level, not bitwise logits.
- `trace_alloc` at seq=1024/layers=16 allocates ~900 MB for the attention trace
  in `chat_traced`; acceptable on dev boxes, not on low-power targets. Low-power
  serving uses `chat`/`chat_greedy` (no trace).
- Confidence/trace helpers that touch int8 model internals (`m->embed`,
  `m->lm_head`) must branch on `m->hybrid`; `compute_residual_stab` and
  `trace_top_predictions` do. A missed one is a NULL+offset segfault on the
  first traced token.
- Dense-path fixtures must use head_dim 64 (see the `model_load` guard): a
  head_dim-16 fixture put heap-corrupting UB in `score_dot_v` behind every
  cross-build transcript comparison until root-caused with ASan.
- greedy argmax runs on the fp32 logits, not the x1024 int32 telemetry view
  (a <0.001-logit near-tie would round to equal ints and flip the pick).
