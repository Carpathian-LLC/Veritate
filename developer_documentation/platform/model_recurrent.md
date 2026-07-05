# model_recurrent (constant-state byte trunk)

## what it is

`veritate_core/model_recurrent.py::VeritateRecurrent`. Research variant of the canonical trunk where attention is replaced by a gated linear recurrence (RWKV/GLA/Mamba-2 lineage): each head carries one fixed `[d, d]` state matrix updated per byte with a learned scalar-per-head decay. Decode memory and per-byte cost are O(1) in position, versus a KV cache that grows without bound; at conversation lengths (tens of KB) this is the dominant cost of the dense trunk.

## how it works

- `RecurrentMixer` (`model_recurrent.py:47`): combined `qkv` QuantLinear + depthwise causal conv (kernel 4) + silu output gate + per-head RMSNorm + `proj`. Decay per token per head: `log a = -softplus(a_proj(x))`, biases initialized log-spaced so heads start with different memory horizons.
- Chunkwise-parallel training (`model_recurrent.py:71`): sequential scan over 64-byte chunks; within a chunk, decay-weighted attention-shaped matmuls in log space (`exp(logA_i - logA_j)`, always <= 1, nothing explodes); state carried between chunks. Exact: verified against the naive per-token recurrence to 4e-11. Inputs not divisible by 64 are right-padded internally and sliced back, so dump prompts and generation work at any length.
- Mixer module is named `attn` with `.qkv`/`.proj`, blocks expose `.ff`, `hook_spec()` returns self: the full dump suite writes all artifacts (verified, 13 artifacts).
- Selected per run via the `trunk` reserved flag (`vanilla_trainer.py::RESERVED_STR_FLAGS`), value `recurrent`.

## dependencies

`veritate_core/model.py` (RMSNorm, QuantLinear, FFN, constants), `veritate_core/qat.py`.

## pitfalls

- ~8 percent more params than the dense trunk at the same manifest shape (the output gate); disclose in any A/B comparison.
- Not `.bin`-exportable and no Brain/load branch yet; a v12 engine path is plausible (decode = matvec + rank-1 state update; SSD kernel prototype exists in-tree).
- `dump_generation` per-layer telemetry assumes dense attention internals; per-layer panels are structurally walkable but not semantically meaningful for the mixer.
- `pos_emb` is kept for contract parity; the recurrence itself is positional, so lengths beyond `seq` are a future change (drop or extend pos_emb).
