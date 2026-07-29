# model_patched (boundary-patched byte trunk)

## what it is

`veritate_core/model_patched.py::VeritatePatched`. SpaceByte-style research variant of the canonical trunk: local blocks run on every byte position, global blocks run only on patch slots anchored at spacelike boundary bytes, then feed back into the byte stream. Global attention/FFN cost scales with slots (`seq / PATCH_STRIDE`), not bytes, buying more parameters per FLOP than the dense trunk.

## how it works

- Boundary rule (`_boundary_table`, `model_patched.py:42`): a byte is a boundary unless it is ASCII alphanumeric or a UTF-8 continuation byte (0x80-0xBF). Position 0 is forced.
- Forward (`model_patched.py:123`): embed -> `N_LOCAL_ENC` local blocks over all bytes -> gather boundary positions into a FIXED `seq/PATCH_STRIDE` slot tensor (pad slots masked, overflow boundaries dropped) -> slot pos emb -> `layers` global blocks over slots -> scatter global output back at boundary positions -> `N_LOCAL_DEC` local blocks -> tied head.
- One hidden size everywhere; every block is a canonical `model.Block`, all exposed via `self.blocks` (enc + global + dec), so `hook_spec()` returns `self` and the full dump suite walks it unchanged (verified against a real checkpoint at the real shape: probe, lens, surprise, quant_kl, concepts, writing_health, generation, neuron_memory all write).
- `_boundary_slots(tokens)` centralizes the boundary/slot mapping (`is_b`, `slot_of`, `bpos`, `slot_mask`, `S`) shared by `forward`, `forward_streaming`, and `probe_columns`.
- `probe_columns(tokens)` (hooks contract, rule 23): per-block column the dumper snapshots at. Local enc/dec blocks read the last byte; global blocks read the last **live** slot (`slot_of[-1]`), because the global stack runs on the slot stream whose trailing slots are masked padding (exactly zero). Without this the global blocks' residual norms, logit lens, and top neurons all read zero in `probe.json`/`lens.npz` and `generation.json` even though the forward computes real values.
- Effective slot count is `min(seq/PATCH_STRIDE, T)`: training inputs (fixed T=seq) keep one fixed shape (MPS kernel-cache rule); dump prompts shorter than the slot count get consistent smaller tensors (preflight 24d exists because the fixed-S version broke 7 dumps silently).
- Causality verified exact (perturbing byte p leaves logits before p bitwise unchanged).
- Selected per run via the `trunk` reserved flag (`veritate_trainer.py::RESERVED_STR_FLAGS`), values `dense` (default) | `patched` | `hybrid`. Manifest `layers` sets the GLOBAL block count; total depth is `layers + N_LOCAL_ENC + N_LOCAL_DEC`.
- `trunk=hybrid` = `global_mixer="recurrent"`: the global blocks are `RecurrentBlock`s (constant-state gated linear recurrence from `model_recurrent.py`) instead of attention, running on patch slots. Composition of the E2 and E3 winners: patch-level compute cost plus O(1) global state. Constructor default `global_mixer="attn"` keeps existing patched checkpoints loading `strict=True`. Full dump battery verified at real shape for both mixers.
- `trunk=looped` = `global_loops=4`: the global stack shrinks to `layers/2` unique weight-tied blocks iterated R times on the patch slots, with per-loop learned input injection (`loop_inj`, index clamped so `eval_loops` beyond `LOOP_MAX` reuses the last injection weight). R is sampled uniformly from 1..4 per training forward (Huginn-style); eval uses `eval_loops` (default = `global_loops`, settable for R sweeps). Params land within 0.5 percent of the dense trunk; the loop FLOPs ride the seq/4 slot positions. Loop-count changes iteration count, not tensor shapes (MPS-safe).
- `trunk=hybrid_moe` = `global_mixer="recurrent"` + `global_ffn="moe"`: the global RecurrentBlocks keep the constant-state mixer but swap their dense FFN for `MoEFFN` (DeepSeekMoE fine-grained MoE, see `model_moe.md`). Constructor default `global_ffn="dense"` keeps existing checkpoints loading `strict=True`; the swap happens before parameter init. Aux-loss surfaced via `moe_aux_sum()` (trainer adds it in the backward path), per-expert share via `moe_expert_share()`. Dump battery verified at real shape.
- Looped verdict (E7, 2026-07-05, failures ledger): beats dense at matched params (0.9920 vs 0.9990, 1.63x) but loses to patched and hybrid on quality and wall-clock; test-time R sweep peaks at the training-mean depth and degrades beyond it (no think-longer effect); random-R training makes single val evals noisy (tail stdev ~0.009). Not the trunk to scale.

## streaming (state carry across windows)

`forward_streaming(tokens, states=None) -> (logits, states)` runs one fixed-shape `[B, window]` window and carries each global `RecurrentBlock`'s recurrence state (see `model_recurrent.md`, streaming section) across windows; `states` is a flat list in invocation order (loop iterations included), opaque to the caller. `supports_streaming()` returns True only for `global_mixer="recurrent"`. The training `forward` is a separate path, bit-identical to the non-streaming baseline (max_abs_diff 0.0). Approximations: `pos_emb` and `slot_pos_emb` are window-local, local attention stays within-window, and the per-window slot budget resets: only the constant-size global state crosses window boundaries. Requires the slot count (`seq / PATCH_STRIDE`) to be a CHUNK (64) multiple, i.e. `seq >= 256`. Consumers: `experiments/v2/longctx/needle_bench.py` (past-window needle distances) and the trainer's opt-in `state_carry=chunks` train-time chunk carry (contract in `model_recurrent.md`).

## dependencies

`veritate_core/model.py` (Block, RMSNorm, QuantLinear, constants), `veritate_core/qat.py`.

## pitfalls

- PyTorch-backend servable: `veritate_core/load.py::load_from_state_dict` dispatches on `training_args.trunk` and builds this class (`state_rule` threaded, `strict=True`), so checkpoints serve through `/generate` on the brain. Still not `.bin`-exportable (non-canonical trunk, preflight 40b); `export_checkpoint` refuses trunk != dense.
- Attention panels are empty for the global blocks when the mixer is recurrent: no per-position attention weights exist (blocks pad to CHUNK=64 internally). Both the brain and `dump_generation` now emit an empty per-layer attention entry (`attn[L] = []`) for them, keyed off `probe_columns` (global blocks read a non-last column).
- `dump_probe` and `dump_generation` snapshot the global blocks at the last live slot (`probe_columns`), so residual norms, logit lens, top neurons, and the panoramic FFN grid are non-zero for L`N_LOCAL_ENC`..L`N_LOCAL_ENC+glob_layers-1`. The slot residual is the deepest global state, not a per-byte value; read it as slot-stream depth, not byte-position telemetry.
- QAT rides along (all blocks carry the `qat` flag), but INT8 export claims should not be made until an engine path exists.
