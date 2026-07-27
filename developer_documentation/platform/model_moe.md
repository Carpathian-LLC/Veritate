# model_moe (DeepSeekMoE fine-grained FFN)

## what it is

`veritate_core/model_moe.py::MoEFFN`. A drop-in replacement for the canonical `FFN`, used inside the hybrid trunk's global blocks. DeepSeekMoE fine-grained mixture-of-experts: 8 routed experts each at 1/4 the dense `ffn` width + 1 always-on shared expert at 1/4 width, top-2 routed. More FFN parameters at near-baseline active per-token FLOPs (active width = 2*(1/4) routed + 1*(1/4) shared = 0.75x dense). Selected per run via `trunk=hybrid_moe`.

## how it works

- Experts (`model_moe.py`): the 8 routed experts are canonical `FFN`s at quarter width (`self.experts`, so QAT rides along via `QuantLinear`); the shared expert is `self.up`/`self.down` (also `QuantLinear`) run inline on every token. The router `self.gate` is a plain `nn.Linear` (kept fp, not quantized).
- Capacity-based DENSE dispatch (GShard/Switch form, `_route`), grouped PER-SEQUENCE (group = one sequence of `T` positions, matching DeepSeekMoE's sequence-wise balance): softmax affinity -> top-2 select -> normalized combine gates -> per-expert capacity slot via `cumsum` -> one-hot `dispatch`/`combine` tensors `[B, T, E, C]` -> `einsum("btec,bth->bech")` scatter into `[B, E, C, H]` -> batched expert GEMM (fixed 8-iter loop) -> `einsum("btec,bech->bth")` combine. `capacity = ceil(1.25 * T * top_k / num_experts)`; overflow dropped, underflow zero-padded. Output = shared + routed.
- MPS fixed-shape (rule 24c): every tensor shape is static across steps given fixed `(batch, seq)`. No `sort`, no bool advanced indexing: top-k via iterative `argmax` + integer-comparison one-hot, positions via `cumsum` + integer-comparison one-hot. Routing bookkeeping runs in fp32 under a local `autocast(enabled=False)` (bf16 `cumsum` over `T>256` would lose integer precision).
- Aux-loss-free load balance (DeepSeek-V3 2412.19437): `route_bias` is a registered buffer (not a Parameter) added to the top-k SELECTION score only (never into the combine weight). After each TRAINING forward it is nudged `+-1e-3` per expert by over/under-load under `no_grad` (no-op at eval). Verified to revive dead experts.
- Sequence-wise balance aux-loss (DeepSeekMoE 2401.06066, alpha=0.01): `_last_aux = alpha * E * mean_b(sum_i f_i * P_i)` with `f_i` detached (count) and `P_i` the mean router prob (carries the gradient). Exposed on the model via `VeritatePatched.moe_aux_sum()`; the trainer adds it in the backward path only (`chunked_step`), so val bpb stays clean. Per-expert token share is `_last_share`, surfaced via `moe_expert_share()` for dashboard logging.
- Dump/hook compatibility: `.up`/`.down` (the shared expert) give the dump suite a real, always-firing FFN activation to hook (`blk.ff.up`) and a `down.weight` to read; `_last_l1` mirrors the FFN contract (mean abs of the combined activation, set only under `capture_l1`). Because the global blocks are narrower than the local dense blocks, `dump_generation` pads per-layer FFN activations/weights up to `model.ffn` (uniform-width models are unaffected: the pad is a no-op).

## wiring

- `VeritatePatched.__init__(..., global_ffn="dense")` (`model_patched.py`): default `"dense"` keeps existing checkpoints loading `strict=True`; `"moe"` swaps each global block's `.ff` for a `MoEFFN` before parameter init.
- Trainer: `trunk=hybrid_moe` in `trainers/common/vanilla_trainer.py` builds `VeritatePatched(global_mixer="recurrent", global_ffn="moe")` (single delta on top of `trunk=hybrid`). `chunked_step` adds `model.moe_aux_sum()` to the loss when present (training only).

## dependencies

`veritate_core/model.py` (`FFN`, `QuantLinear`, `_ACT_FNS`, constants), `veritate_core/model_patched.py` (host trunk), `veritate_core/qat.py`. Dump padding lives in `veritate_mri/training/checkpoint_probe.py::_capture_full` / `dump_generation`.

## pitfalls

- Capacity-factor FLOP tax the box cannot hide: the dispatch/combine einsums and the expert GEMM over `num_experts*capacity` slots (including padded/dropped) are real extra FLOPs that a sparse-kernel GPU avoids and MPS does not. Compare arms at EQUAL WALL-CLOCK, never equal active-FLOPs; the realized throughput ratio vs hybrid is the number to compare on.
- Dispatch is grouped per-sequence (`[B, E, C, H]`), not per-global-batch (`[E, C, H]`): a flat group of `N=B*T` tokens makes the dispatch einsum O(N^2), pathological at `bs*seq`. Per-sequence keeps capacity small and the einsum affordable, and matches DeepSeekMoE's sequence-wise balancing.
- Total FFN params ~3x the dense trunk (8+1 quarter-experts vs 1 dense); the model is not `.bin`-exportable (non-canonical trunk, preflight 40b) and has no Brain/load branch.
- Dead-expert risk at small active-param scale (10M): the pre-registered E8 falsifier kills the lever if any expert's share is sustained <0.02 or >0.5 past step 4000. Log per-expert share each eval.
- Per-layer FFN telemetry panels for the global blocks show the shared expert only (padded with zero neurons); not semantically meaningful, same caveat as the other patched-trunk variants.
