# RWKV-7 port investigation -- v5 architectural pivot analysis

Research deliverable, 2026-04-28. Investigates whether porting an RWKV-7
"Goose" block to Veritate's INT8 + AVX-512 VNNI infrastructure is the right
v5 architectural pivot. Companion prototype lives at
`experiments/10_rwkv_prototype/`.

# ------------------------------------------------------------------------------------
# One-line verdict
# ------------------------------------------------------------------------------------

**RWKV-7 is the wrong pivot for Veritate at 80M / V_SEQ=256.** It is slower
per-token than our existing transformer at short context and roughly the same
or worse on engineering complexity. **Mamba-2 is the right v5 pivot:** fewer
matmuls per layer, an existing INT8 reference (BitMamba-2), a single new
selective-scan kernel to write, and a clear quality story at adjacent scales.

If a long-context use case emerges (V_SEQ >> 256), revisit RWKV-7 because its
state is per-channel, not per-position. At our current shape that advantage
is dormant.

# ------------------------------------------------------------------------------------
# Context: what triggered this investigation
# ------------------------------------------------------------------------------------

`docs/RESEARCH.md` ranks RWKV-7 #2 (after Mamba-2) among architectural
alternatives, with a thesis that rwkv.cpp's existing CPU runtime makes it the
lowest-friction port. `docs/results/EXPERIMENTS_TRACKER.md` H15 promoted the question
to a research task.

The mission: validate that thesis with a concrete prototype, decode-cost
projection on the 9800X3D, and a candid take on whether RWKV-7 beats Mamba-2
for our specific constraints.

# ------------------------------------------------------------------------------------
# Smallest available RWKV-7 model
# ------------------------------------------------------------------------------------

Public weights are released by BlinkDL on Hugging Face under
`https://huggingface.co/BlinkDL/`. The RWKV-7 "Goose" model suite includes:

- **0.19B** (192M): smallest production checkpoint. ~12 layers, hidden 768.
- 0.4B, 1.5B, 2.9B in increasing size.
- Pile / Pile-v2 / Multilingual training corpora.

Actual file: `RWKV-x070-Pile-168M-20241120.pth` (or current dated variant).

Download path validated on huggingface.co; `rwkv.cpp` ships scripts to
convert `.pth` to its quantized `.bin` format. Loading a 0.19B model in
fp16 needs ~400 MB; INT8 ~200 MB; INT4 ~100 MB. All comfortably below the
9800X3D's 96 MB L3 only at INT4, which matters for the same L3-resident
argument as our INT8 80M.

**For Veritate the natural workflow is:** distill 0.19B Goose down to a
80M-class student trained natively against linear-recurrence math, on the
TinyStories corpus. The 0.19B teacher provides logit supervision; nothing
about the 0.19B weights ports directly because Veritate is byte-level
(V_VOCAB=256) while Goose is BPE.

# ------------------------------------------------------------------------------------
# RWKV-7 block math, precise
# ------------------------------------------------------------------------------------

From the Goose paper (arXiv:2503.14456). Notation: `x_t` is the residual
stream input at position t, all weights are per-layer, per-head where
relevant. HD = 64, HEADS = 12, HIDDEN = 768.

## Time-mixing (replaces attention)

```
r_t       = x_t W_r              # receptance, role of query
w_t       = exp(-exp(x_t W_w))   # data-dependent decay, in (0, 1) per channel
k_t       = x_t W_k              # key
v_t       = x_t W_v              # value
a_t       = sigmoid(x_t W_a)     # in-context learning rate
g_t       = sigmoid(x_t W_g)     # output gate
kappa_t   = k_t * l2_norm(x_t W_kap)  # removal key, scales k by a unit-norm direction

# Per-head matrix state update (HD x HD), this is the recurrence:
S_t = S_{t-1} (diag(w_t) - kappa_t^T (a_t * kappa_t)) + v_t^T k_t

# Per-head output:
o_t = (S_t r_t) * g_t            # matvec then elementwise gate

# Per-layer combination:
y_t = LN(concat over heads of o_t) W_o + x_t
```

Two structural notes:

- The state update is rank-2: a diagonal decay plus a rank-1 outer product
  (the kappa-based removal) plus a rank-1 outer product (the v-k write).
  This is the "generalized delta rule" of the paper.
- The state is a *matrix*, not a vector. `S_t` has shape HD x HD per head.
  At V_HIDDEN=768, V_HEADS=12, V_LAYERS=12: total state = 12 * 12 * 64 * 64
  fp32 = 2.25 MB. Compare KV cache at V_SEQ=256: 12 * 256 * 768 * 2 (K and
  V) bytes int8 = 4.5 MB. **2x reduction at fp32, 8x at int8.**

## Channel-mixing (replaces FFN)

```
xs_t      = x_t (1 - mu) + x_{t-1} mu     # static token-shift (one prev-token mix)
k_t       = relu(xs_t W_k)^2              # squared-ReLU expansion, FFN dim
v_t       = k_t W_v                       # contraction
y_t       = (sigmoid(x_t W_r)) * v_t      # gated output, residual added
```

Almost identical to a gated GeGLU FFN. The squared ReLU is the only deviation
from the standard FFN we already have in `model.c`.

# ------------------------------------------------------------------------------------
# Integer-friendliness per operation
# ------------------------------------------------------------------------------------

| Op                          | Type          | Cost / new code                        |
|-----------------------------|---------------|----------------------------------------|
| 7 input projections (r/w/k/v/a/g/kappa) | matmul | reuse VNNI matmul, 7 calls instead of 3 (qkv) per layer |
| `exp(-exp(.))` for w_t      | elementwise   | int8->int8 LUT, 256 entries, 1 layer of indirection    |
| sigmoid for a_t, g_t        | elementwise   | int8->int8 LUT, same primitive                         |
| l2 normalize for kappa      | reduce + scalar div | small SIMD reduce; can be folded with a per-channel scale at training time |
| State decay `S * diag(w)`   | elementwise   | HD*HD multiply per head per layer, fully SIMD-vectorizable |
| Rank-1 update `v^T k`       | outer product | one VNNI dot per row, equivalent to a tiny matmul     |
| Removal `(S kappa) * a_kappa` | matvec + outer | one HD-dim matvec + one outer product per head      |
| State readout `S r`         | matvec        | HD*HD matvec per head, exactly the same shape as a single VNNI dot row |
| Output gate `* g_t`         | elementwise   | int8 saturating multiply, trivial                     |
| Squared ReLU                | elementwise   | LUT or `max(x,0); x*x` on int16 intermediate          |
| Token shift                 | data movement | one residual buffer of width HIDDEN, no compute       |

**Conclusion on int-friendliness:** every operation has a precedent in the
existing codebase. Three new LUTs (`exp(-exp())`, sigmoid, squared-ReLU);
otherwise the matmul kernel and a small "state step" function cover the
forward.

The honest concern is **state quantization**, not op coverage. State accumulates
outer products across the sequence; quantization noise compounds. rwkv.cpp
keeps state in fp16 even with INT4 weights. A naive int8 state will diverge.

# ------------------------------------------------------------------------------------
# Decode latency model on the 9800X3D
# ------------------------------------------------------------------------------------

Baseline (transformer, 80M, V_SEQ=256, INT8 + VNNI):
- 1.0 ms per-token decode at pos=10.
- 1.18 ms per-token decode at pos=250.
- Breakdown: 4 matmuls/layer x 12 layers = 48 matmuls = ~0.95 ms.
  Attention 0.04-0.22 ms growing with pos.

RWKV-7 projection at the same shape:

| Component                   | RWKV-7 cost            | vs transformer |
|-----------------------------|------------------------|----------------|
| Time-mix input projections  | 7 matmuls/layer x 12   | +3 matmuls vs qkv -> +0.6 ms |
| Time-mix state update + readout | <0.01 ms total     | replaces attention scan (-0.04 to -0.22 ms) |
| Output projection           | 1 matmul/layer x 12    | same as today |
| Channel-mix (FFN replacement) | 2 matmuls/layer x 12 (one is squared) | same as today |
| LN, residuals, LUTs         | <0.05 ms               | comparable |

Total RWKV-7 estimate: **~1.6 to ~2.0 ms / token at any context length.**

That's 60-100% slower than today's 1.0 ms transformer at pos=10. The
crossover point where RWKV-7 becomes faster than transformer is when
attention dominates. Today attention is 4-22% of decode. The crossover sits
around V_SEQ ~1500-2000 (extrapolating the linear attention growth from
exp 03's breakdown).

**At V_SEQ=256 byte-level the crossover never happens.** RWKV-7 is strictly
slower for our current configuration. The 0.09 ms moonshot target is moved
*further away*, not closer, by this pivot.

# ------------------------------------------------------------------------------------
# State size at V_HIDDEN=768, V_LAYERS=12
# ------------------------------------------------------------------------------------

```
state per head per layer: HD x HD = 64 x 64 = 4096 elements
total state count       : 12 layers x 12 heads x 4096 = 589824 elements
total state bytes       : 2.25 MB fp32, 1.13 MB fp16, 576 KB int8

current KV cache bytes  : 12 layers x 256 positions x 768 hidden x 2 (K+V) bytes int8
                        = 4.5 MB

savings                 : 2x at fp32, 4x at fp16, 8x at int8
```

For the 9800X3D's 96 MB L3, neither cache is meaningful pressure -- both fit
comfortably alongside the 80 MB INT8 weights. The state-size advantage
becomes structural at long context: V_SEQ=8K transformer KV cache = 144 MB,
RWKV-7 state still 2.25 MB.

# ------------------------------------------------------------------------------------
# Quantization compatibility -- candid literature view
# ------------------------------------------------------------------------------------

What rwkv.cpp ships:
- INT4/5/8 weight quantization, all variants.
- **fp16 state, fp16 hot-path math.**
- Known issue: weight outliers break naive Q4_0; needs per-group scales.

Published int8 RWKV-7 quality numbers: none in the public literature as of
2026-04. RWKV-4 has been quantized to int8 with ~1% perplexity loss (rwkv.cpp
benches). RWKV-7 is too new (March 2025 paper) for systematic post-training
quantization studies to have appeared.

The one analog: **BitMamba-2** (https://github.com/jserv/bitmamba.c) ships
1.58-bit weights *and* int8-equivalent state via "Decoupled Scale
Quantization" (DSQ). 112.9 tok/s @ 255M on Xeon AVX-512 (~9 ms/tok). DSQ
scales the state per-channel and re-calibrates every K tokens. Adapts to
RWKV-7 in principle but has not been implemented anywhere public.

**Recommendation:** start with int8 weights + fp16 state. Open question:
whether a per-channel int16 state with periodic recalibration matches fp16
quality. That is a v6 research item, not a v5 deliverable.

# ------------------------------------------------------------------------------------
# What a v5 RWKV-7 port would touch
# ------------------------------------------------------------------------------------

Files in `engine/src/`:

- `model.c`: replace `forward()` and `forward_decode()` attention block with
  `rwkv7_time_mix()`. Keep token + position embed. Replace FFN with
  `rwkv7_channel_mix()`. Net: ~400-600 new lines, ~250 lines removed.
- `veritate.h`: change `kv_cache_t` to `rwkv_state_t`. New shape:
  ```c
  typedef struct {
      float S[V_LAYERS][V_HEADS][V_HEAD_DIM][V_HEAD_DIM];  // fp16 in production
      int8_t prev_token_residual[V_HIDDEN];                 // for token-shift
  } rwkv_state_t;
  ```
- `dispatch.c`: no change. matmul dispatch is reused for all 7 input projections.

New kernels (in `engine/kernels/x86_64/`):

- `rwkv_state_step`: takes `S[HD x HD]`, `r/w/k/v/a/kappa`, scratch, writes
  updated `S` and output `o[HD]`. Per-head, AVX-512.
- `rwkv_lut_apply`: int8 -> int8 elementwise via LUT (sigmoid, double-exp,
  squared-ReLU). One generic kernel parameterized by LUT pointer.

New tooling (in `scripts/`):

- `gen_rwkv_luts.py`: precompute the three LUTs (sigmoid, exp(-exp(x)),
  x^2 if relu(x)>0) as int8->int8 tables.
- `convert_rwkv_weights.py`: load Goose 0.19B fp16 weights, distill or
  quantize per-tensor int8, emit Veritate `.bin` format. Tied to a training
  pipeline change.

On-disk weight format (extension of current header):

```
magic    "VRTE"
version  4   (was 3 for transformer)
arch_id  1   (0 = transformer, 1 = rwkv-7)
v_vocab, v_hidden, v_layers, v_heads, v_seq, v_ffn  (unchanged)
state_dtype  (1 = fp16, 2 = int16)
+ weights: 7 input proj per block + out_proj per block + 2 channel-mix per
  block + LN weights + embed + (no separate lm_head; tied to embed)
```

Rough byte budget at 80M, INT8: identical to current (input projection
matrices are the same shape regardless of 4 vs 7 of them; total params
governed by hidden^2 * layers, same shape).

What stays identical:
- dispatch + matmul kernels
- `score_dot_v`, `softmax_rows`, `layernorm_i16_to_i8` are gone (attention)
  but the layernorm primitive is reused for the post-time-mix LN
- byte tokenizer, sampler, MRI trace shape
- bench harness, build pipeline

# ------------------------------------------------------------------------------------
# Why Mamba-2 wins on engineering simplicity
# ------------------------------------------------------------------------------------

Side-by-side at the same scale (both replacing attention only; FFN unchanged):

| Axis                        | RWKV-7         | Mamba-2          |
|-----------------------------|----------------|------------------|
| Input projections per layer | 7 (r/w/k/v/a/g/kap) | 3 (Delta, B, C) -- four if you count input gate |
| State shape                 | HD x HD per head | N x N per channel-group, N=16-64 |
| State update                | rank-2 (decay + rank-1 removal + rank-1 write) | rank-1 (decay + rank-1 write); SSD form is matmul-equivalent |
| Non-trivial elementwise ops | exp(-exp), sigmoid, l2 normalize | just SiLU and softplus; both LUT-able |
| INT8 reference impl         | none public, fp16 state default | BitMamba-2: INT8 weights + INT8 state (DSQ) |
| Quality at <500M            | par with Mamba-2 per Goose paper | matches Transformer++ |
| Crossover w/ transformer @ V_SEQ=256 | never (slower) | similar, attention scan small at our shape |

**Mamba-2's selective-scan in the SSD form is literally a matmul.** Per the
paper, the recurrence
  `h_t = A_t h_{t-1} + B_t x_t; y_t = C_t h_t`
with A_t = a_t * I (scalar-times-identity) reduces to a sequence of weighted
sums that can be expressed as matmul against a structured matrix. This means
**zero new SIMD primitives**: our existing VNNI matmul kernel covers the
recurrence. The new code is the per-token state update wrapper, ~50 lines.

RWKV-7's rank-2 state update is more expressive (it has the kappa-based
removal term that Mamba-2 lacks). That expressiveness is what gives RWKV-7
+0.6pp on language modeling vs Mamba-2 in the Goose paper. **At 80M scale
the +0.6pp likely doesn't survive the noise of byte-level training on
TinyStories.**

# ------------------------------------------------------------------------------------
# Hybrid alternative: Hymba pattern
# ------------------------------------------------------------------------------------

`docs/RESEARCH.md` flags Hymba (NVIDIA, ICLR 2025): per layer, run a Mamba-2
head and a sliding-window (~64 token) attention head in parallel, sum
outputs.

For Veritate this is the **best published quality bet at 80M** per the
research scan. Engineering cost is higher than pure Mamba-2 because we need
both the selective-scan kernel and a windowed-attention kernel. The windowed
attention is a tiny modification of our existing causal-mask attention.

Decision deferred: pursue pure Mamba-2 for v5, evaluate hybrid in v5.5 if
quality is below transformer baseline on TinyStories perplexity. The sliding
window restores recall that pure recurrent models lose.

# ------------------------------------------------------------------------------------
# Concrete v5 plan -- recommended path
# ------------------------------------------------------------------------------------

**Phase 1 (2 weeks): scalar Mamba-2 reference + Python training**
- Implement Mamba-2 block in PyTorch trainer. 10M-class model, train on
  TinyStories. Capture validation perplexity.
- Compare to current transformer perplexity at matched compute.
- Decision gate: ppl within 10% of transformer -> proceed. Else revisit.

**Phase 2 (1.5 weeks): C reference implementation**
- Add `selective_scan_scalar` in `engine/kernels/scalar/`.
- New `block_t` variant (Mamba-2 block) selectable at compile time.
- Bit-match scalar oracle vs PyTorch fp32 reference.

**Phase 3 (1.5 weeks): VNNI selective-scan kernel**
- Hand-tuned AVX-512 selective-scan, INT8 weights, fp16 state.
- Verify against scalar reference (1 LSB tolerance per FINDINGS contract).
- Bench end-to-end decode at 9800X3D.

**Phase 4 (1 week): integration + workbook entry**
- Wire dispatch for the new kernel.
- Run code-review + anti-overengineering agents.
- Bench against transformer baseline at pos=10 / 100 / 250.
- Decide whether to keep transformer code path or remove it.

**Total: 6 weeks.** RWKV-7 would be similar, plus one week for the additional
input projections and state-quantization research. Mamba-2's BitMamba-2
reference saves time exactly equal to that overhead.

# ------------------------------------------------------------------------------------
# Open questions
# ------------------------------------------------------------------------------------

1. **Quality on TinyStories at 80M.** No published Mamba-2 or RWKV-7
   experiment at this exact scale + corpus exists. Phase 1 of the plan is
   the cheapest answer. Without this number, the architecture choice is
   speculative.
2. **State quantization stability.** fp16 state is the safe default. int16
   per-channel with periodic recalibration is the next frontier; no public
   numbers at our scale. Open research direction for v6.
3. **Multi-layer state propagation in INT8.** Single-layer math is clean.
   12-layer composition with int8 weights and fp16 state has not been
   benchmarked at this scale. May surface as a quality bug only after
   Phase 3.
4. **Whether 80M is large enough to demonstrate the recurrent advantage.**
   Both Mamba-2 and RWKV-7 papers report quality benefits scaling with
   parameter count. At 80M we are in the territory where the literature's
   advantage is small or noisy. The "killer chat on old hardware" goal
   (PLATFORMS.md) cares about quality at 80M-300M; this is exactly where
   the data is sparsest.
5. **Whether the gibberish bug (H11) is architecture-related.** If the
   transformer C path is failing at int8 numerics in a way that recurrent
   architectures would be more sensitive to (state accumulation), the
   pivot makes the quality bug worse, not better. Fix H11 first.

# ------------------------------------------------------------------------------------
# Decision summary
# ------------------------------------------------------------------------------------

- **v5 architectural pivot: Mamba-2 (SSD form), not RWKV-7.**
- **Estimated weeks of work: 6 weeks** for the full Mamba-2 port (training,
  scalar, VNNI, integration).
- RWKV-7 deferred; revisit only if Mamba-2 quality is materially below
  baseline at 80M, or if a long-context use case (V_SEQ >> 256) becomes the
  product target.
- Hybrid (Hymba-style) deferred to v5.5 conditional on Phase 1 quality gate.
- Sub-ms decode is **not** unlocked by either architectural pivot at our
  current shape. The lever for the 0.09 ms target remains weight reduction
  (INT4 + QuaRot) and model size reduction (distillation).

This investigation does not recommend the RWKV-7 port that the research
brief proposed. The research is correct that RWKV-7 has the lowest *runtime
adoption* friction (rwkv.cpp), but lowest adoption friction is not the same
as lowest *Veritate-port* friction. For our specific shape (V_SEQ=256, 80M,
INT8 VNNI matmul already hand-tuned), Mamba-2 is the cleaner port.
