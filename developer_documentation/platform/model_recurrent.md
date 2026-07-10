# model_recurrent (constant-state byte trunk)

## what it is

`veritate_core/model_recurrent.py::VeritateRecurrent`. Research variant of the canonical trunk where attention is replaced by a gated linear recurrence (RWKV/GLA/Mamba-2 lineage): each head carries one fixed `[d, d]` state matrix updated per byte with a learned scalar-per-head decay. Decode memory and per-byte cost are O(1) in position, versus a KV cache that grows without bound; at conversation lengths (tens of KB) this is the dominant cost of the dense trunk.

## how it works

- `RecurrentMixer` (`model_recurrent.py:47`): combined `qkv` QuantLinear + depthwise causal conv (kernel 4) + silu output gate + per-head RMSNorm + `proj`. Decay per token per head: `log a = -softplus(a_proj(x))`, biases initialized log-spaced so heads start with different memory horizons.
- Chunkwise-parallel training (`model_recurrent.py:71`): sequential scan over 64-byte chunks; within a chunk, decay-weighted attention-shaped matmuls in log space (`exp(logA_i - logA_j)`, always <= 1, nothing explodes); state carried between chunks. Exact: verified against the naive per-token recurrence to 4e-11. Inputs not divisible by 64 are right-padded internally and sliced back, so dump prompts and generation work at any length.
- Mixer module is named `attn` with `.qkv`/`.proj`, blocks expose `.ff`, `hook_spec()` returns self: the full dump suite writes all artifacts (verified, 13 artifacts).
- Selected per run via the `trunk` reserved flag (`vanilla_trainer.py::RESERVED_STR_FLAGS`), value `recurrent`.

## state_rule (memory mechanisms, opt-in)

`RecurrentMixer(hidden, heads, state_rule=...)` selects how the per-head state is updated. Threaded through `RecurrentBlock`, `VeritateRecurrent`, and `VeritatePatched` (the hybrid trunk's global mixer), and exposed to training via the `state_rule` reserved flag (`vanilla_trainer.py::RESERVED_STR_FLAGS`, default `gla`, injected into the model only for `trunk` in {`hybrid`, `hybrid_moe`, `recurrent`} and only when non-default). Part of the long-context memory campaign (`developer_documentation/training/long_context_memory_campaign.md`), mechanisms M1 and M2.

- `gla` (default): the uniform per-head scalar decay above. Bit-identical to the pre-mechanism model: same params, same forward. The `delta`/`pinned` params (`b_proj`, `pin_key`, `sal_proj`) are only allocated when their rule is selected, so the default path's parameter set and init RNG stream are untouched. Verified: hybrid + pure-recurrent trunks reproduce pre-edit param count and forward loss exactly (`SMOKE_RESULTS/m1m2_memory_smoke.py::test_baseline_intact`).
- `delta` (M1, Gated DeltaNet): the state update becomes error-correcting instead of decay-then-add. Per token `S <- alpha(I - beta k kᵀ)S + beta k vᵀ` with L2-normalized q/k, a learned per-head decay `alpha = exp(-softplus(a_proj))` and write strength `beta = sigmoid(b_proj)`; a stale key's slot is overwritten before the new value is written rather than every slot fading uniformly. Chunkwise-parallel: within a 64-token chunk the coupled per-token corrections are solved as `U = (I + M)⁻¹ rhs`, `M` strictly-lower-triangular; the inverse is block-recursive (`_tri_inverse`): halve down to `DELTA_INV_BLOCK`-sized diagonal blocks, invert those by the finite nilpotent product `Π (I + N^{2ⁱ})` (`N = -M` restricted to the block), combine as `[[T1, 0], [T2·N21·T1, T2]]`. Same exact inverse as a whole-chunk nilpotent expansion, but no large intermediates: whole-chunk expansion forms `N^{2ⁱ}` terms that reach ~1e17 and must cancel, which fp32 cannot do once `beta` saturates and in-chunk keys align (measured divergence: `developer_documentation/training/m1delta_divergence_analysis.md`). Fixed tensor shapes, matmul/cat only; the recurrence + inverse run in fp32 under `autocast(enabled=False)`. Verified exact against a naive per-token oracle to <1e-6 (`test_delta_correctness`) and against a float64 exact inverse to <5e-6 on saturated real-checkpoint states.
- `pinned` (M2, pinned memory register): the `gla` decaying state plus `PIN_SLOTS` fixed decay-exempt key/value slots per head. Reads add a softmax attention over the pins (learned `pin_key` addresses); writes are a per-chunk decay-free convex update gated by a learned saliency scalar (`sal_proj`), so unwritten pins never change and written pins interpolate toward salient content (bounded, no unbounded growth). Directly targets "important facts shouldn't fade." Pins are read as-of-chunk-start (one-chunk latency, causal at chunk granularity, consistent with the inter-chunk state read).

## streaming (inference-only state carry)

`RecurrentMixer.forward(x, state=None, return_state=False)`: `return_state=True` also returns the final recurrence state as a dict `{"s": [B,H,d,d] state matrix, "conv": last kernel-1 pre-conv qkv columns[, "pin": pin registers (pinned only)]}`; passing that dict as `state` seeds the next window. Carrying both the state matrix and the conv tail makes window-carry exact: for all three `state_rule`s, two CHUNK-multiple windows with carry reproduce the one-pass output and final state bitwise on CPU (verified 2026-07-05, max_abs_diff 0.0). `delta` states stay fp32 (produced inside the `autocast(enabled=False)` block; never downcast them). State carry requires `T % CHUNK == 0` (raises otherwise: a padded tail would fold pad rows into the state). The default call (`state=None, return_state=False`) is bit-identical to the stateless path; training is unchanged. `RecurrentBlock.forward` threads the same kwargs.

`VeritateRecurrent.forward_streaming(tokens, states=None) -> (logits, states)` runs one fixed-shape window and carries the per-block state list; `supports_streaming()` returns True. Approximation: `pos_emb` is indexed 0..T-1 per window (window-local positions); only the recurrent state crosses window boundaries. Consumed by `experiments/v2/longctx/needle_bench.py` for past-window needle distances, and by the trainer when `state_carry=chunks` (below).

## state_carry (train-time chunk carry, opt-in)

Reserved trainer flag `state_carry` (`vanilla_trainer.py::RESERVED_STR_FLAGS`, default `off`), values `off` | `chunks`. Valid only for `trunk` in {`hybrid`, `hybrid_moe`, `recurrent`} (the trainer raises otherwise); injected per run via the `/trainers/run` request body like `state_rule`. Motivation: the stateless training forward resets the recurrent state every window, so retrieval beyond one window is never trained (needle recall past ~512 B is 0.00 at every scale measured); carrying state across the chunks of a training step makes cross-window retrieval trainable at all (campaign doc, route 4; E4b precedent for the memory trunk).

Contract with `chunks`:

- Within one training step, `chunked_step` processes the `n_chunks` contiguous chunks of each row left-to-right through `forward_streaming`, threading the recurrent global state from chunk to chunk; the trainer computes the CE loss on the returned logits (same `ignore_index=-1` as `forward`).
- Gradients flow THROUGH the carried state within a `bptt_window`; at each backward boundary the carry is detached (mandatory: the graph is freed by `backward()`). `bptt_window=n_chunks` = full backprop-through-carry for the step; `bptt_window=1` = detached carry (forward-only information flow, still creates retrieval training pressure). Measured at the 10M hybrid shape on CPU (batch 8, 4x512, two rounds): grad-through at `bptt_window=4` costs ~0-1 percent step time and ~2-3 percent peak RSS over the no-carry baseline (the deferred-backward window already holds all chunk graphs); `bptt_window` itself is the memory knob (4 -> 1 cuts peak RSS 5.7 -> 1.9 GB at this shape).
- State resets at step boundaries: rows are unrelated stream offsets, only the chunks WITHIN a step are contiguous bytes of one stream row.
- `pos_emb` (and `slot_pos_emb` in the hybrid) stay window-local per chunk, exactly matching the inference streaming approximation, so train and decode see the same regime.
- Validation (`evaluate`) uses the same `state_carry` as training.
- `state_carry=off` (default) is bit-identical to the pre-flag trainer: params, forward loss, and grads reproduce the pre-edit baseline exactly (`SMOKE_RESULTS/state_carry_smoke.py`, pinned-number comparison).
- Requires the per-chunk global sequence to be a CHUNK (64) multiple (the mixer raises otherwise): `seq` multiple of 64 for `trunk=recurrent`, `seq >= 256` and a 256-multiple for the hybrid (slots = seq/4).

## dependencies

`veritate_core/model.py` (RMSNorm, QuantLinear, FFN, constants), `veritate_core/qat.py`.

## pitfalls

- ~8 percent more params than the dense trunk at the same manifest shape (the output gate); disclose in any A/B comparison.
- Not `.bin`-exportable and no Brain/load branch yet; a v12 engine path is plausible (decode = matvec + rank-1 state update; SSD kernel prototype exists in-tree).
- `dump_generation` per-layer telemetry assumes dense attention internals; per-layer panels are structurally walkable but not semantically meaningful for the mixer.
- `pos_emb` is kept for contract parity; the recurrence itself is positional, so lengths beyond `seq` are a future change (drop or extend pos_emb).
- `delta`/`pinned` are unvalidated on the recall curve: they are wired and MPS/dump-verified, not yet won. Pre-registered kill conditions (campaign doc): M1 (`delta`) is killed if `recall(L)` is not flatter than the `gla` baseline by the needle margin at matched params / >=2 seeds, or throughput <70% of `gla` on MPS. M2 (`pinned`) is killed if pinned recall is no better than `gla` at the same slot budget, or the pins collapse to unused (share ~ 0). The 10M A/B on the conversation-needle harness (M0 vs M1 vs M2) decides; only a winner folds into the 80M chat model.
- `delta` L2-normalizes q/k and washes out the shared `_project` `1/sqrt(d)` query scale; the `1/sqrt(d)` still governs `gla`/`pinned`. The delta inverse recursion assumes power-of-two chunk sizes; `DELTA_INV_BLOCK` must stay small (8): the base-block nilpotent expansion is the numerically fragile part and its error grows combinatorially with block size.
