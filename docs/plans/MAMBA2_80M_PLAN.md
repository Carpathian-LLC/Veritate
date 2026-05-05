# Mamba-2 SSD scale-up to 80M -- training plan

Status: pre-launch (script ready, not yet kicked off).
Owner: training-side work only. No engine changes in this task.
Baseline to beat: transformer 80M `tinystories-80m`, val 0.486 at step 45000.

# ------------------------------------------------------------------------------------
# Background
# ------------------------------------------------------------------------------------

Exp 26 trained a 7.6M Mamba-2 SSD on byte-level TinyStories and beat a same-shape
transformer 0.594 vs 0.831 val (ppl ratio 0.78x). Exp 29 scaled to 20M
(`data/models/mamba2_20M`, hidden=640 layers=8) and reached val 0.557 / ppl 1.75 in
50K steps at B=4 on the RTX 5070, wall 6126 s. The architectural pivot is
working; next checkpoint is matching the 80M transformer at the same parameter
count to confirm the advantage holds at scale.

# ------------------------------------------------------------------------------------
# Target architecture
# ------------------------------------------------------------------------------------

Like-for-like vs transformer (vocab=256 hidden=768 layers=12 ffn=3072 heads=12)
is constrained by the SSD scan tensor `[B, T, n_heads, n_state, head_dim]`.
At hidden=768 the natural `head_dim=64` gives n_heads=24 which inflates the
scan tensor to ~1.6 GB per layer per saved activation at B=8. Bumping
`head_dim=128` halves n_heads and roughly halves activation memory while
keeping d_inner identical.

Final config:

| field    | value | notes |
| ---      | ---   | ---   |
| vocab    | 256   | byte-level, matches transformer |
| hidden   | 1024  | scaled up from 768 to recover param count |
| layers   | 12    | matches transformer depth |
| head_dim | 128   | doubled from default 64 to keep n_heads small |
| n_heads  | 16    | derived: d_inner / head_dim = 2048 / 128 |
| n_state  | 64    | matches exp 26 / exp 29 |
| expand   | 2     | matches exp 26 / exp 29 |
| seq      | 256   | matches transformer |

Param count (verified by instantiation): **77,567,552 (~77.6M)**. Within the 80M
envelope; close enough to the transformer's 79.7M for a fair quality compare.

# ------------------------------------------------------------------------------------
# Memory budget on 12 GB RTX 5070
# ------------------------------------------------------------------------------------

The training-form forward materializes per-layer:

- `M[B, T, T, H]`              decay matrix      = 4 * B * 65536 * 16 bytes
- `u_t[B, T, H, N, D]`         input outer-prod  = 4 * B * 256 * 16 * 64 * 128 bytes
- `h_full[B, T, H, N, D]`      state tensor      = same as u_t

At B=4 each layer saves ~270 MB (u_t + h_full) for autograd, ~3.2 GB across 12
layers. Param + grad + AdamW state ~1.0 GB. Steady-state peak ~6-7 GB,
leaving headroom for cudnn workspace and the M decay matrix. **B=4 is the cap;
B=8 was OOM at 20M with smaller activations and will OOM here.** Use
`--grad_accum N` to scale effective batch without raising peak memory.

A previous run OOM'd at B=128 because the 5D scan tensor was ~25 GB; at B=32
it stalled mid-forward. The numbers above are consistent with that observation.

# ------------------------------------------------------------------------------------
# Training schedule
# ------------------------------------------------------------------------------------

Mirror exp 29 hyperparameters:

```
batch_size    4
grad_accum    1   (raise to 2 or 4 if loss is noisy; doubles wall time)
total_steps   50000
warmup_steps  500
base_lr       3e-4
min_lr        3e-5  (cosine decay)
weight_decay  0.1
grad_clip     1.0
dtype         bfloat16
```

CSV path: `experiments/29_mamba2_scaleup/mamba2_80m.csv` (or any path the user
prefers; the script accepts `--csv`). Checkpoint dir: `data/models/mamba2_80M/`
to match the existing 20M directory naming.

Launch command:

```
py training/mamba2_train.py \
   --name mamba2-80m \
   --ckpt_dir data/models/mamba2_80M \
   --csv experiments/29_mamba2_scaleup/mamba2_80m.csv \
   --hidden 1024 --layers 12 --head_dim 128 --n_state 64 --expand 2 \
   --batch_size 4 --total_steps 50000
```

# ------------------------------------------------------------------------------------
# Wall-clock estimate
# ------------------------------------------------------------------------------------

Exp 29 (20M, hidden=640 layers=8) wall: 6126 s for 50K steps at B=4 on RTX 5070.

The 80M config has ~3.9x params and ~2.5x per-layer activation volume (n_heads
halved but d_inner up 33%, layers up 50%). Training-form Mamba dominated by the
T^2 decay matmul plus the outer-product writes; total flops scale roughly with
layers * d_inner * (T + N * D). Net per-step expected to be ~2.5-3.5x slower
than exp 29.

**Estimate: 5-6 hours for 50K steps.** Add 30-50% if grad_accum=2 is needed for
loss stability. Conservative budget: **8 hours overnight**.

# ------------------------------------------------------------------------------------
# Success criterion
# ------------------------------------------------------------------------------------

Primary: **val loss <= 0.486** at any step <= 50000 (matches transformer 80M
final). The ppl-ratio target from exp 26 was 0.78x; achieving val 0.486 at the
same parameter count would carry that ratio (ppl 1.626 vs the txfm's 1.626 at
val 0.486). Better than that is a ratio < 1.0 and the architectural argument
holds at scale.

Secondary: val curves at 5K, 10K, 25K, 50K should be monotone-decreasing and
not plateau before 30K. The 20M run plateaued after 30K but kept descending
slowly to 50K; the 80M should follow the same shape with a deeper floor.

Failure mode to watch: if val plateaus at >0.55 by step 25K, abort and
investigate (hyperparam, init, scan numerics).

# ------------------------------------------------------------------------------------
# Path to the C engine
# ------------------------------------------------------------------------------------

Out of scope for this task. High-level wiring once the model trains:

1. Add a `mamba2_block_t` alongside the transformer block in `engine/src/`.
   The decode-form `step()` is constant-memory O(1) per token and maps directly
   to a kernel: per-head A * h_prev decay (one `exp` or LUT), outer-product
   update `dt * B_t * x_t`, bilinear readout `C_t * h`. All FMAs, all
   vectorize. Per-arch SIMD (`engine/kernels/<arch>/ssm_step.*`).
2. Quantize the recurrent state to int8 with calibration similar to the
   transformer activation calibration. State is fp32 in this reference; the
   8-layer 20M model's int8 state is 384 KB total. The 12-layer 80M state at
   n_state=64 head_dim=128 n_heads=16 is `12 * 16 * 64 * 128 = 1.57 MB int8`,
   which still beats the txfm's seq-256 KV cache (~1.15 MB).
3. Weight loader extension: same int8 / per-row scale convention as the
   transformer, applied to in_proj / out_proj. The A_log, D, dt_bias and
   norm parameters stay fp32 (small, off the hot path).
4. Bench target: per-token decode under 0.09 ms (the standing bar). The 20M
   PyTorch reference runs at ~2.5 ms; a hand-written int8 SSM step kernel
   should land much closer to the bar than the transformer once the matmuls
   are removed from the per-token critical path.

# ------------------------------------------------------------------------------------
# Files in scope for this task
# ------------------------------------------------------------------------------------

- `training/mamba2_train.py`        -- updated CLI; --hidden, --layers,
                                       --head_dim, --n_state, --expand exposed;
                                       --grad_accum, --resume, --summary added.
- `training/mamba2_block.py`        -- unchanged.
- `docs/plans/MAMBA2_80M_PLAN.md`   -- this file.

Out of scope: `engine/`, `mri/`, `training/train.py`, `training/qat_v2*.py`,
`training/prep_curriculum.py`.
