# BitNet b1.58 -- ternary weight plan

Research and design spec for adopting ternary {-1, 0, +1} weights as an
alternative to the current INT8 QAT2 pipeline. Doc-only; no code in flight.

**Status:** proposal. Decision at the bottom.
**Started:** 2026-04-29.
**Owner:** dev-box Claude.
**Composes with:** QuaRot rotation, INT4 KV cache, Mamba-2 SSD, MoD gate.

# ------------------------------------------------------------------------------------
# Why
# ------------------------------------------------------------------------------------

The decode hot path is one matmul-per-projection per token. Today it runs
INT8xINT8 through `vpdpbusd` -- 64 multiply-accumulates per cycle per lane.
With ternary weights every multiply degenerates to `+/- a` or `0`, so the
inner kernel becomes signed-add over a sparse mask: no multiplier, no
`vpdpbusd` dependency, fewer operand bytes through L2/L3. Storage drops
from 8 bits/weight to 1.58 bits/weight (5x), so the 80M model's weight
working set falls from 80 MB to ~16 MB and fits comfortably in the
9800X3D's 96 MB L3. Item 6 of `docs/PLAN.md` claims a 0.3-0.5x decode
budget multiplier; that is the prize. The cost is a from-scratch retrain
or a multi-stage warm-start, plus a kernel rewrite.

# ------------------------------------------------------------------------------------
# Reference
# ------------------------------------------------------------------------------------

- *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*,
  Ma et al. 2024. https://arxiv.org/abs/2402.17764
- *1-bit AI Infra: Fast and Lossless BitNet b1.58 Inference on CPUs*,
  Microsoft 2024. https://arxiv.org/abs/2410.16144
- *BitNet b1.58 Reloaded -- small networks*, 2024.
  https://arxiv.org/abs/2407.09527
- *BitNet b1.58 2B4T Technical Report*, 2025.
  https://arxiv.org/abs/2504.12285

Headline numbers from the original paper at 100B training tokens:

| Size  | LLaMA FP16 ppl | BitNet b1.58 ppl | Gap     |
|-------|----------------|------------------|---------|
| 700M  | 12.33          | 12.87            | +4.4%   |
| 1.3B  | 11.25          | 11.29            | +0.4%   |
| 3B    | 10.04          | 9.91             | -1.3%   |

Quality matches FP16 starting around 3B. Below 1B the gap is real;
*BitNet b1.58 Reloaded* shows small models need approximately doubled
hidden width to recover, which is a problem for the 80M target.

# ------------------------------------------------------------------------------------
# Training plan
# ------------------------------------------------------------------------------------

Mirrors `training/qat_v2.py` structurally: single block class with
fake-quant ops, drop-in for the existing trainer, exporter writes a new
binary version tag.

## Weight quantization

Absmean ternary, per BitNet b1.58:

```
gamma = mean(|W|)               # scalar per linear layer
W_tilde = clip(round(W / (gamma + eps)), -1, 1)
```

Gradients flow straight-through (STE), identical to the existing
`_RoundSTE` in `qat_v2.py`. The forward pass uses `W_tilde * gamma`
so activations stay calibrated; the engine stores `W_tilde` (1.58 bits)
and a single fp32 `gamma` per layer.

Per-row gamma is a refinement worth testing: today's INT8 path uses
per-output-row scales (`quantize_int8_per_row`) and got a 54% perplexity
drop from that change alone (see `docs/PAPER.md` per-channel section).
Default to per-row gamma in v1 of the BitNet trainer.

## Activation quantization

Keep the existing INT8 activation path: `ACT_INT8_SCALE = 32`, INT16
residual stream. BitNet b1.58 uses 8-bit per-token absmax; ours is a
fixed scale because the binary is the model. No change required.

## LayerNorm

BitNet uses RMSNorm. Veritate uses LayerNorm without bias and folds
`ln_w` into the matmul. RMSNorm is mean-free; the 80M base trained with
LayerNorm. Keep LayerNorm to avoid an architectural change inside the
BitNet experiment. Re-evaluate if the val-loss penalty exceeds 5% --
RMSNorm is the next variable to ablate.

## Optimizer and schedule

- AdamW, betas (0.9, 0.95), weight_decay 0.05.
  *BitNet b1.58 Reloaded* found 5% weight decay best for small models.
- Learning rate: 1e-3 peak (small-model rate from Reloaded paper),
  cosine to 1e-4, 2000-step warmup. Our current QAT2 fine-tune uses
  3e-4 -- the BitNet ternary forward is more sensitive to LR, raise it.
- Total steps: 60k at batch 64, V_SEQ 256. ~1B tokens. Budget is one
  overnight on the dev box.
- Gradient clipping at 1.0. STE saturation is real; clip aggressively.

## Warm-start vs from-scratch

The BitNet paper says from-scratch is required for full quality at 3B+.
*BitNet b1.58 Reloaded* trained exclusively from scratch and made no
warm-start claim. HuggingFace's 2024 ramp-up technique gradually anneals
quantization width during fine-tuning; the published gap is real but
small (single-digit perplexity percent).

**Decision:** warm-start from the trained 80M QAT2 checkpoint. Reasoning:

1. The 80M model is already coherent on TinyStories. Discarding it to
   prove a quantization technique loses a known-good baseline.
2. We can compare warm-started vs from-scratch directly on the same
   corpus, same eval, same hardware. This is the experiment the BitNet
   literature has not run.
3. If warm-start fails (loss diverges or perplexity gap > 30%), the
   from-scratch run is then a 2-day fallback, not a wasted week.

Schedule:

- Steps 0-10k: linear ramp of ternary fraction from 0 to 1. Each linear
  layer mixes `(1-alpha) * fq_int8(W) + alpha * fq_ternary(W)` in the
  forward; alpha = step/10000.
- Steps 10k-50k: full ternary, low LR.
- Steps 50k-60k: 0.5 LR cosine tail. Eval every 2k steps.

# ------------------------------------------------------------------------------------
# Storage format
# ------------------------------------------------------------------------------------

Two options. Pick one.

## Option A -- 5-trit packing (1.58 bits/weight true)

Five ternary values fit in one byte: 3^5 = 243 < 256. Encode
`b = w0 + 3*w1 + 9*w2 + 27*w3 + 81*w4` with each `w` shifted from
{-1, 0, +1} to {0, 1, 2}. Decode by repeated divmod, or a 243-entry
lookup table that yields five int8s. K must be a multiple of 5 (or
padded). Memory: ~1.6 bits/weight.

## Option B -- 2-bit padded packing (2 bits/weight, simpler)

Four ternary values per byte, 2 bits each. Decode is a shift-and-mask
identical in shape to `unpack_int4_64` already shipped in
`engine/kernels/x86_64/matmul_int4.c`. Memory: 2 bits/weight (21%
overhead vs the theoretical floor).

## Pick

**Option B.** Reasoning:

- The 21% storage overhead is irrelevant. 80M weights at 2 bits = 20 MB
  vs. 16 MB at 1.58 bits. Both fit in L3.
- Decode SIMD cost dominates the kernel. Option A's 243-entry LUT
  gather costs more cycles per byte than a shift+mask, every cycle of
  every matmul. Option B's `vpsrlw` + `vpand` is two instructions on
  any 512-bit lane.
- We already shipped the unpack pattern for INT4. Reusing the shape
  means one new prep function and one new inner loop, not a new
  algorithm.
- K=64-divisible already holds in the 80M architecture. Four
  weights/byte means the same tile granularity as INT8.

`prepped_b_ternary_t` mirrors `prepped_b_t` and `prepped_b_int4_t`:
`bt_packed` (n * k/4 bytes), `gamma_q24` (per-row int32 scale),
`pos_count` (per-row int32, popcount of +1 positions used for the
sign-bias correction).

# ------------------------------------------------------------------------------------
# Kernel sketch
# ------------------------------------------------------------------------------------

The INT8 VNNI kernel (see `engine/kernels/x86_64/matmul_vnni.c`,
`vnni_dot_1x1`) is `vpdpbusd(au, bv) -> int32 acc`, 64 MACs per cycle
per lane, with a +128 bias trick to make `a` unsigned. Every iteration
reads 64 bytes of activation and 64 bytes of weight.

Ternary kernel:

```
# decode(m=1) for one output column j over k inputs

unpack 16 bytes packed -> 64 trits in {-1, 0, +1} as int8 lane
split into pos_mask  = (trit ==  1)   # __mmask64
           neg_mask  = (trit == -1)   # __mmask64

# per 64-element block:
load 64 bytes of activation a[t..t+64]              # __m512i av
acc_pos = _mm512_mask_add_epi8_to_int32(acc_pos, pos_mask, av)
acc_neg = _mm512_mask_add_epi8_to_int32(acc_neg, neg_mask, av)

# at the end of k:
s_j = horizontal_sum(acc_pos) - horizontal_sum(acc_neg)
c[j] = s_j * gamma_q24[j] >> 24
```

The two key moves:

1. *Sign-split, not multiply.* No `vpdpbusd`. The mask register selects
   which activations contribute and to which accumulator. AVX-512BW's
   `_mm512_mask_add_epi8` plus a widening conversion gives the same
   throughput envelope without the multiplier.

2. *Per-block popcount-zero short-circuit.* If `pos_mask | neg_mask` is
   zero for a 64-element block, skip the add entirely. With absmean
   quantization roughly 30-50% of trits are zero in trained BitNet
   weights -- this is real sparsity, not statistical.

Expected uop budget per output column at K=512:
- Today (INT8 VNNI): 8 `vpdpbusd` + 8 loads = ~16 uops.
- Ternary (Option B): 8 unpacks + 8 mask-builds + 16 mask-adds + 1
  reduce + 1 scale = ~34 uops, but each uop is half the latency of
  `vpdpbusd` on Zen 5 and the loads are 4x smaller. Net: 1.5-2x
  faster, not the 3x the BitNet paper hopes for, because the kernel
  is now load-bound on activations rather than compute-bound on
  weights.

The 3x decode multiplier in `docs/PLAN.md` row 5 assumes the kernel
becomes weight-bandwidth-bound and the 5x storage cut translates
directly. This is wrong on Zen 5 with 96 MB L3 -- weight bandwidth is
not the bottleneck for an 80M model. The realistic multiplier is
1.5-2.0x decode latency; update the plan after the bench.

ARM64 NEON port follows the same shape: SDOT replaced by signed
mask-add, popcount short-circuit identical. Scalar oracle is two
nested loops with an `if (trit == 1) s += a; else if (trit == -1)
s -= a;`.

# ------------------------------------------------------------------------------------
# Integration roadmap
# ------------------------------------------------------------------------------------

Six milestones. Effort estimates assume one Claude on the dev box,
parallel-reviewed by the code-review and anti-overengineering agents.

1. **Trainer prototype** -- 3 days.
   Add `qat_ternary.py` mirroring `qat_v2.py`. New `fq_weight_ternary`
   replaces `fq_weight_per_row`. New block class. Alpha-mix warm-start
   schedule. Validate forward bit-match against a numpy oracle on
   random weights.

2. **80M warm-start training run** -- 1 day wall, 1 day analysis.
   Overnight on dev box. Eval every 2k steps. Decision gate: val
   perplexity within 30% of QAT2 baseline (1.64). If yes, proceed.
   If no, fall back to from-scratch (4 days additional).

3. **Storage format + scalar engine** -- 2 days.
   `prepped_b_ternary_t` definition, `prep_b_ternary` packer, scalar
   oracle `matmul_ternary_scalar_prep`. Bit-match scalar against the
   trainer's PyTorch ternary forward. Per memory rule: scalar oracle
   gates everything downstream.

4. **AVX-512 VNNI kernel** -- 4 days.
   Sign-split inner loop with popcount-zero short-circuit. Per-row
   gamma_q24 scaling. Bit-match against the scalar oracle (1 LSB
   tolerance on the post-scale int32). Bench against the existing
   INT8 VNNI matmul on the 9800X3D.

5. **Engine integration** -- 2 days.
   New binary version tag. Loader path for ternary weights. Dispatch
   table addition. End-to-end TinyStories chat smoke test. C engine
   perplexity matches PyTorch QAT2-ternary (cosine distance < 0.05
   per layer).

6. **MRI + decode-tracing parity** -- 1 day.
   FFN brain panel reads ternary activations; decision-trace fields
   already populated. Ensure no MRI tab broken.

Total: ~13 days serial. With the agent fan-out (kernel work on the
dev box, training on the same machine overnight) the wall-clock is
closer to 8 days.

# ------------------------------------------------------------------------------------
# Risks
# ------------------------------------------------------------------------------------

1. **Catastrophic loss spike at the warm-start ramp.** The BitNet paper
   trained from scratch; alpha-mixing INT8 weights into ternary may
   diverge. Mitigation: cap LR at 1e-4 during ramp, fall back to
   from-scratch on divergence.

2. **80M is too small.** *BitNet b1.58 Reloaded* needed roughly doubled
   hidden width on small models. Our 80M is borderline. The val-loss
   may settle at +30-50% over QAT2 baseline. Mitigation: accept the
   gap if the decode speedup composes; bigger models in a later
   sprint amortize it.

3. **Kernel speedup is 1.5x not 3x.** Realistic on Zen 5 (analysis
   above). The PLAN.md decode budget projection (0.10 ms -> 0.03 ms)
   does not land. Item 5 still pulls 1.5x; the 0.03 ms moonshot needs
   another factor from elsewhere (tighter Mamba-2 path, MoD layer
   skip).

4. **KV cache still INT8.** Decode reads K and V of all past tokens
   per step. At V_SEQ=256, that's 4 MB of KV reads vs ~16 MB of
   ternary weight reads. KV becomes the bottleneck. Mitigation:
   compose with the planned INT4 KV cache; ternary weights make KV
   the next leverage point.

5. **Mask-add throughput on Zen 5.** `_mm512_mask_add_epi8` may not
   sustain the same per-cycle throughput as `vpdpbusd`. Microbench
   first thing in milestone 4. If it's worse than 1.2x the INT8
   path, ship the from-scratch model on INT4+QuaRot and shelve
   ternary until v3 silicon.

6. **Quality drift from absmean scaling on per-row.** Per-row gamma
   is one float per row; the engine stores `gamma_q24` as int32. The
   q24 fixed-point may saturate on rows with very small absmean.
   Add a `gamma_q24` calibration pass identical to the INT8 q24
   path.

# ------------------------------------------------------------------------------------
# Decision
# ------------------------------------------------------------------------------------

**Ship.** Conditional on milestone 2 (warm-start training) hitting
val perplexity within 30% of the QAT2 baseline.

Reasoning:

- The composability story holds. Ternary, MoD, Mamba-2, and INT4 KV
  are independent on different axes (weight bits, layer count,
  layer arch, KV bits). Shipping ternary unlocks two of the five
  decode-budget rows in `docs/PLAN.md`.
- The realistic 1.5-2x kernel speedup is meaningful. 0.10 ms ->
  0.06 ms is the pre-Mamba budget; that alone is worth the work.
- Milestone 2 is a 2-day decision gate. Cheap to fail fast.
- If milestone 2 fails, falling back to from-scratch is 4 days. If
  from-scratch also fails at 80M, *then* shelve until the model
  scales up.

Primary blocker if shelved: the 80M-quality risk (point 2). Smaller
models pay a real ternary tax that the BitNet papers do not hide.

# ------------------------------------------------------------------------------------
# Open questions
# ------------------------------------------------------------------------------------

- Is per-row gamma worth the storage vs. per-tensor? Test in
  milestone 1.
- Does QuaRot rotation compose with ternary weights? The Hadamard
  transform expects bounded magnitudes; ternary post-rotation is
  no longer ternary. Skip QuaRot on the ternary path; the rotation's
  win was an INT4-specific outlier-suppression trick.
- xIELU drop-in: BitNet stack typically uses SwiGLU. Defer the
  activation question to after milestone 2.
- Mamba-2 + ternary: BitNet weights inside an SSD block are an open
  research question. Cross that bridge when item 3 of the in-flight
  list lands.
