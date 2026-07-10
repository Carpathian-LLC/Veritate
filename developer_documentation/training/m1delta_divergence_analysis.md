# m1delta divergence analysis (2026-07-06)

Race arm `m1delta` (10M hybrid trunk, `state_rule=delta`, muon, fp32, bs32/seq512, chat 40/40/20,
cosine base 6e-4) trained in line with baseline `m0gla` to step 4400 (val 0.7728 vs 0.7683),
logged one grad-norm spike at step 4420 (0.574 vs gla's 0.343 on the identical batch), then
produced only non-finite losses for the remaining ~7,570 steps. Root cause below; a numerics-only
stabilizer shipped in `veritate_core/model_recurrent.py::RecurrentMixer._tri_inverse`.

## Root cause

fp32 catastrophic blowup in the chunkwise WY inverse (hypothesis "nilpotent formula exact in
exact arithmetic, fp32 error amplifies as ||M|| grows"), triggered by beta-gate saturation on the
slowest-decay head. The delta math itself never became unstable.

Evidence chain (diagnostics on the replayed step-4401 batch, real chat bytes, CPU; scripts in
session scratchpad, numbers reproducible from `models/m1delta_10m_qat/checkpoints/`):

1. **The exact math is stable at every checkpoint.** Per-token update
   `S <- a(I - beta k kᵀ)S + beta k vᵀ` with `||k|| = 1`, `a < 1`, `beta < 1` is a contraction
   (factor eigenvalues `a` and `a(1-beta)`), so the exact `(I + M)⁻¹` is bounded: float64 max
   entry = 1.000 for every block, chunk, and checkpoint (800/4000/4400), exact `U` max 0.18-0.49.
2. **The fp32 whole-chunk nilpotent expansion is what exploded.** `_tri_inverse` formed
   `Π(I + N^{2ⁱ})` up to `N^32`. At blk9 the measured intermediates hit `|N^32| ~ 1e17-2e17`;
   the alternating cancellation that should collapse them to <= 1 is impossible at fp32's 7
   digits. fp32 inverse max entry: 1.3e8 (step 800) -> 3.4e10 (4000) -> 5.2e10 (4400) where the
   true value is <= 1.000.
3. **Driver: beta saturation + slow decay + key alignment on blk9 head 0** (the slowest-decay
   head by `a_proj` bias init, i.e. the designated long-memory head). At step 4000/4400:
   beta_mean 0.93, beta_max 0.999, 33-37% of its gate outputs > 0.99, per-token decay
   a ~ 0.998, in-chunk |k_i . k_j| mean 0.55. M entries reach 0.994-0.998 with no decay
   suppression, M spectral norm 37-39 (healthy blocks: 5-12).
4. **Compounding to overflow.** The two chunks of one forward multiply the error: fp32 `U`
   error 1e10 (chunk 0) seeds chunk 1's rhs, giving `U` ~ 5e20 and state ~ 2.8e20 at step 4400
   (exact state: ~0.15). Backward through the squaring chain squares magnitudes past fp32 max
   (3.4e38) -> inf -> `0*inf` NaN -> NaN grads; `clip_grad_norm_` scales by NaN, one `opt.step()`
   poisons all weights permanently. Forward loss stayed finite until then because `o_norm`
   (RMSNorm) renormalized the garbage mixer output.
5. **blk9 was already numerically broken at step 800** (fp32 output error 1e7 vs float64 oracle
   on real bytes). The arm trained around it: downstream RMSNorm bounded the damage and val
   tracked gla. Every "healthy" block also carried real error: old inverse absolute error
   0.03-1.4 on blocks 2-8 (exact max entry 1.0). The step-4400 needle result would have been
   invalid even without the crash: the long-memory head was emitting fp32 noise all run.

Hypotheses (b) and (d) checked and rejected as primary cause:

- (b) state growth via projections: healthy blocks' state stayed ~0.15 at all checkpoints;
  the exact-oracle state at blk9 also stayed small. Only the fp32-corrupted state grew.
- (d) muon interaction: per-param grad norms at step 4400 on the identical batch, delta vs gla:
  totals 0.340 vs 0.346, `b_proj` grads ~2e-3, worst per-param ratio 3.7x (a conv weight).
  No outliers. Muon at most walked `b_proj`/`qkv` into the saturated regime at the same pace any
  optimizer would; the failure needed the delta inverse.

## CPU repro (math vs MPS kernel)

Resume from `step_4400.pt` (weights + muon state) with the byte-identical replayed batch
sequence (same `RandomState(0)` draw order) and the trainer-faithful loop
(`chunked_step`, bptt_window 4, grad_clip 1.0, cosine lr ~4.45e-4), on CPU:

- Old (whole-chunk) inverse, steps 4401-4450: all 50 steps finite. The first ~10 steps track
  the MPS run to 3-4 decimals (4410: CPU loss 0.7580 vs MPS 0.7583), then low-order arithmetic
  drift separates the trajectories; CPU shows the same class of grad-norm spike (0.51 at step
  4446 vs 0.31-0.36 baseline; MPS spiked 0.57 at 4420, one step before dying) but does not
  cross overflow inside the window.
- Verdict: the instability is NOT MPS-specific: the broken fp32 numerics are fully present on
  CPU (items 2-5 above are all CPU measurements; the saturated-regime construct reaches 2.4e31
  in one forward on CPU). The terminal NaN is a threshold event: error compounding must cross
  fp32 max (3.4e38) inside one forward+backward, and backend-dependent rounding decides which
  batch/step first crosses it. MPS crossed at step 4421; CPU rode the same knife edge (grad
  spikes, 1e20-1e31 internals) without crossing in 50 steps. No MPS kernel bug is implicated,
  and no MPS-side investigation is warranted: the stabilizer removes the mechanism on both
  backends.

## Stabilizer decision

The evidence supports a numerics-only fix: compute the same exact inverse without large
intermediates. Rejected: beta cap (`beta_max` 0.98 leaves `0.98^31 * C(62,31)` ~ 1e17: does not
fix the expansion, changes the math), eps in the inverse (changes the math, does not remove the
cancellation), state-norm clamp (the exact state never grew; clamping treats a symptom).

Shipped: block-recursive triangular inversion in `_tri_inverse`. Halve to `DELTA_INV_BLOCK=8`
diagonal blocks, invert those by the nilpotent product (max path weight C(6,3)=20 vs
C(62,31)~4.5e17 whole-chunk), combine as `[[T1,0],[T2 N21 T1, T2]]`. Every intermediate is a
principal-block exact inverse, bounded by the same contraction argument, so fp32 error stays at
rounding scale. Matmul/cat only, fixed shapes (MPS rule 24c), same signature, gla/pinned paths
untouched.

Cost: parity on CPU at both training shape (mixer fwd+bwd [32,128,320]: 7.05 s old vs 7.11 s
new, contended box, ratio ~1.01) and decode shape (bs1/T45 full-model forward: 878-886 ms old
vs 820-878 ms new). The recursion issues ~46 small matmuls+cats per chunk versus 11 large ones,
so re-measure tok/s on the first MPS delta run (launch-overhead sensitivity; kill line is 70%
of gla).

Verification (CPU):

1. `SMOKE_RESULTS/m1m2_memory_smoke.py`: all tests pass, `errors: []`. Chunkwise-vs-naive
   oracle 7.7e-7 (tolerance 1e-4); gla baseline bit-identical (param_diff 0, loss_diff 0.0);
   full dump battery zero failures (delta + pinned). New regression test
   `test_delta_saturated` (beta forced to 0.9997, decay ~1, near-repeated inputs so in-chunk
   keys align): fixed inverse matches the naive oracle to 5.4e-7; the whole-chunk squaring
   produces 2.4e31 on the same inputs and fails. The test discriminates: it would have caught
   this bug before launch.
2. Step-4400 repro with the fix: all 50 steps (4401-4450) finite, grad norms 0.31-0.37
   throughout: flat through the MPS death window (4421-4430) and flat at step 4446 where the
   old inverse spiked to 0.51 on the identical batch (old 0.5145 vs fixed 0.3238). The spikes
   are the inverse's error, not the data. Loss band matches the healthy trajectory
   (4450: 0.7706 fixed vs 0.7712 old).
3. No-op where the old path was accurate: new inverse vs float64 exact <= 4.5e-6 on every
   block/chunk/checkpoint including saturated blk9 (old: up to 5.2e10). Where the old path was
   already accurate (e.g. blk7 step 4400, error 3.9e-5) old and new agree at that scale. Full
   forward at step 800: loss 1.304702 -> 1.304929 (delta 2.3e-4); step 4400: 0.858679 ->
   0.859135 (delta 4.6e-4). Not bitwise vs the old code because the old code was wrong even on
   healthy states; bitwise-matching it would preserve the bug.

## Recommendation

Retry the arm (m1delta2) from scratch with the stabilized inverse; do not kill M1 on
stability. The mechanism was never actually tested: blk9's long-memory head emitted fp32
noise from step 800 on, so the m1delta needle result would have been invalid even without the
crash. The failure was implementation numerics, not the delta rule (the float64 oracle is
unconditionally stable at these weights), and the mechanism's HIGH prior stands.

- Do NOT resume from step_4400: those weights adapted for 4400 steps around a broken block.
- Re-check tok/s on the first MPS run against the 70%-of-gla kill line (CPU shows parity;
  MPS kernel-launch overhead for the recursion's smaller matmuls is the one unmeasured risk).
- The saturated regime (beta ~ 1, decay ~ 1 on the slow head) is where the delta rule does its
  long-memory work; with exact numerics it is a contraction and needs no beta cap, eps, or
  clamp.
