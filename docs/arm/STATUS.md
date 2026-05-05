# ARM port — status

Tracks ARM64-specific kernel and trainer work. Live artifact, append as tiers
land. Numbers from M1 base (4P+4E, 8 GB UMA, macOS 14.6).

# ------------------------------------------------------------------------------------
# Shipped
# ------------------------------------------------------------------------------------

## Tier 1 — NEON kernel parity with the AVX-512 hot path

`kernels/arm64/transformer_neon.c`
- `softmax_rows_neon` — 4-wide poly5 e^r with range reduction `r = x - n·ln2`,
  reconstruct `e^x = poly5(r) · 2^n` via the IEEE754 exponent-bias trick (NEON
  has no `scalef`). Matches AVX-512 numerics; tail uses libm `expf`.
- `layernorm_i16_to_i8_neon` — full SIMD over the post-RMS `int16 × int8 × scale`
  loop, 16 elements per iteration. Sumsq pass also vectorized.
- `score_dot_v_neon` — unchanged.

`kernels/arm64/matmul_neon_sdot.c`
- `matmul_int8_sparse_decode` — NEON port of the sparse accumulate. Prescan,
  then per non-zero entry broadcast `val[i]` into 16-wide int32 columns.
- `ffn_down_decode` — threshold-gated sparse-or-dense
  (`p->b_rowmaj && n_nz * 2 < p->k`), mirrors the AVX-512 path.

Correctness (M1, default and bench modes):
- `decode vs full forward: max int8 diff = 0`
- `forward_verify vs K decodes: max int8 diff = 0, K in {1,2,4,8,16}`
- `int4 packed: bit-match`

M1 random-init bench (seq=256, V_HIDDEN=768, V_FFN=3072, 12 layers):
- Forward: p50 125 ms (was 125 ms — within noise)
- Decode:  p50 2.84 ms (was 2.77 ms — slight regression from prescan overhead
  when sparse doesn't fire)
- ffn_down sparsity on random init: 75.1 % nonzero → sparse fires 0.1 % of
  decode calls. The win surfaces on trained models with GELU zero-clamp where
  sparsity is 50–90 % zero per the WORKBOOK.

## Tier 3 — MPS+bf16 AMP for all multimind plugins

`plugins/` is gitignored (see commit `1bbab49`); these edits are applied to
the local working tree but do not ship via git.

`plugins/multimind_{mega,m1,m3}/plugin.py`
- New `pick_device()` prefers cuda → mps → fail.
- `chunked_step` and `evaluate` take a `device_type` arg passed to
  `torch.autocast`. No more hard-coded `"cuda"`.
- `fused=(device == "cuda")` keeps fused AdamW on cuda only (MPS has no fused).
- mega: `--use_8bit_adam` falls back to AdamW on non-cuda (bitsandbytes is
  cuda-only) with a one-line notice.

Smoke test on M1: mega 200m ternary preset, n_experts=2 (smoke-only),
batch=1 seq=64 ran step 1+2 at ~6 s/step. Real moonshot run uses the manifest
defaults (200m, 8 experts, top-1, batch=2, seq=256, n_chunks=12).

# ------------------------------------------------------------------------------------
# Deferred — and why
# ------------------------------------------------------------------------------------

## Tier 2 — SMMLA (FEAT_I8MM) matmul

The ARM mirror of VNNI: `vmmlaq_s32` is a 4×4 INT8 matrix-multiply-accumulate,
~2–4× SDOT throughput. Available on Apple M2+, Cortex-X2+/A715+, Neoverse V1+.

**Status:** not landed. Rule 23 forbids shipping unvalidated kernels. The
local dev box is M1 (`hw.optional.arm.FEAT_I8MM = 0`); SMMLA cannot be tested
here. Land when M2+ access is available; the dispatch hook is already in place
(`feat->neon_i8mm` is detected in `dispatch.c`).

## BFMMLA (FEAT_BF16) prefill matmul

Same situation: M1 has `FEAT_BF16 = 0`. Lower priority than SMMLA — would only
benefit a bf16 inference path, which the engine does not have today.

## Apple AMX

Per `docs/research/amx_probe.md` (2026-05-01): AMX is reachable but the MATINT
operand bit-fields for `int8 × int8 → int32` are unverified. `/tmp/test_amx.c`
confirms LDX/STX roundtrip; `/tmp/test_amx_matint.c` does not match the
expected outer product. Defer until reverse-engineered (Corsix + Dougall
references) and validated against the scalar oracle.

The bandwidth-bound vector for M1 is INT4 packed weights, not AMX.

## Twin-core decode (e/p split)

Architectural change. Needs design doc + cancellation primitive before code.

## SLC residency hints (`prfm pldl3keep` etc.)

Needs profiling first. Premature without bench data on the trained model.

# ------------------------------------------------------------------------------------
# Moonshot training config (multimind_mega 200m ternary)
# ------------------------------------------------------------------------------------

```
python plugins/multimind_mega/plugin.py \
    --corpus <stem> \
    --description "<one-line>" \
    --size 200m \
    --precision bf16 \
    --quant_mode ternary \
    --n_experts 8 \
    --router_topk 1 \
    --qat_enabled
```

Shape on 200m: hidden=1024 layers=12 ffn=2048 heads=8 → 250 M total / 100 M
active per byte → ~20 MB ternary L3 footprint. Manifest default
`--use_8bit_adam` is auto-disabled on MPS; AdamW is used instead.

Engine deployment requires the ternary INT8 kernel (`documentation/kernels/ternary.md`)
and the MoE router (`documentation/kernels/moe.md`). Both not yet in
`veritate_engine/`; until they land, the trained model runs in PyTorch only.

# ------------------------------------------------------------------------------------
# Per-platform decode targets (M-series, current)
# ------------------------------------------------------------------------------------

| Chip | FEAT_I8MM | FEAT_BF16 | Decode p50 (random init, 80M shape) |
|------|-----------|-----------|--------------------------------------|
| M1 base | no | no | 2.8 ms |
| M2 base | yes | yes | TBD — needs SMMLA build + bench |
| M3 / M4 | yes | yes | TBD |

Numbers will land here as platforms are accessed. The PLATFORMS.md target for
M1 is 0.5–1 ms; current 2.8 ms decode reflects bandwidth-bound matmul on UMA.
The unmeasured headroom is sparse ffn_down on a real (high-zero) model.
