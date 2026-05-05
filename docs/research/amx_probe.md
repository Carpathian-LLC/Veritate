# Apple AMX probe — 2026-05-01

Standalone probe of Apple's undocumented AMX coprocessor on M1, macOS 14.6.
Goal: determine feasibility of an AMX-backed matmul kernel for veritate.

## What landed

`/tmp/test_amx.c` — minimal probe. Confirms:

1. `AMX_SET` (`.long 0x00201220`, op 17 imm5 0) does not trap in a normal
   userspace process on this M1.
2. `LDX` (`.long 0x00201000`, op 0 gpr 0) loads 64 bytes from a pointer into
   X register 0.
3. `STX` (`.long 0x00201040`, op 2 gpr 0) stores X register 0 back to memory.
4. `AMX_CLR` (`.long 0x00201221`, op 17 imm5 1) closes AMX state cleanly.

The 64-byte source/destination roundtrip matches byte-for-byte. AMX is real
and reachable. **Apple's claim that AMX is private undocumented hardware is
true at the docs level but false at the runtime gating level.**

## Where it stalled

`/tmp/test_amx_matint.c` — MATINT (op 20, `.long 0x00201280`) operand
encoding. The probe sweeps several plausible bit-field placements for
`x_type` / `y_type` / output type. Operand `0x4800000000000` (bits 47 and 50
set) writes 32 nonzero Z rows. **But the values are not the int8 outer
product we expect.**

Input: `a[i] = i` for i=0..63, `b[j] = 1` for j=0..63.
Expected outer product: `Z[i][j] = i * 1 = i` (row i = i,i,i,i...).
Observed first row: `15, 13, 13, 11, 13, 11, 11, 9, 13, 11, 11, 9, 11, 9, 9, 7, ...`

Repeating pattern of `{15, 13, 11, 9}`. Does not match int8 outer product.
Likely interpretations the operand we hit actually selected:
- a different reduction (vector dot, not outer product)
- a different input width (treating `a` and `b` as int16 pairs)
- one of the alu_mode bits we didn't set, picking up a saturate or shift

## What it would take to ship

A correct, bit-equivalent AMX int8 matmul needs:

1. The exact operand bit-field layout for MATINT in int8/int8→int32 mode.
   Not in any Apple documentation. Reverse-engineered references exist
   (Corsix's `amx` repo on GitHub, Dougall Johnson's notes) but the field
   positions for `x_type` / `y_type` / output bit-width have shifted
   between AMX1 (M1) and AMX2 (M2+) and are not consistent across public
   write-ups.
2. A validation harness that compares Z output to a scalar oracle for at
   least three input regimes (small magnitudes, max int8, mixed signs).
3. A study of Z layout — output may not be simply `Z[i] = row i of result`;
   M1 AMX is known to interleave columns by lane width.
4. Per-chip detection. Some M1 SKUs reportedly trap on certain MATINT
   variants; the engine must runtime-guard.

Estimated effort: 1-3 sessions of focused reverse engineering on top of
Corsix's docs, plus a test sweep on M1 / M1 Pro / M2 if accessible.

## Recommendation

Defer the AMX kernel. The NEON SDOT 4x4 register tile shipped with this
turn already gives a 1.5x prefill speedup on M1. The remaining decode gap
to the 9800X3D's 0.09 ms target is bandwidth-bound, not compute-bound;
AMX would only help if it also unlocks an int4 path (it does not — AMX
operates on int8 / int16 / fp).

The bandwidth-bound vector for M1 is **NEON int4 packed weights**
(`kernels/arm64/matmul_int4_neon.c`, mirroring `matmul_int4_vnni_decode`
on x86). Halves weight reads, ~2x decode floor. Strictly higher confidence
of payoff than AMX given the doc situation.

## Files

`/tmp/test_amx.c`        — basic SET/LDX/STX/CLR probe (passes)
`/tmp/test_amx_matint.c` — MATINT operand sweep (partial, does not validate
                           int8 outer product semantics)

Both excluded from the engine build; they live in `/tmp` so they don't
ship a half-validated kernel into a repo that enforces rule 23.
