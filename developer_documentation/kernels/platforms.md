# Platforms

Cross-platform scope for Veritate. Live tracking artifact. Updated as
ports land and bench numbers come in.

# ------------------------------------------------------------------------------------
# Why this exists
# ------------------------------------------------------------------------------------

Veritate's strategic bet: **run capable AI on hardware people already
own.** Every old laptop, Mac mini, Raspberry Pi, and dusty desktop becomes a
coherent local LLM instead of e-waste. This sidesteps the "AI needs a
$3000 GPU" constraint that's gating real local-AI adoption.

The architecture commitment that makes this work:

- **Per-arch hand-tuned kernels behind function pointers.** No abstraction
  layer; each kernel is hand-written intrinsics for its arch. No
  compromise for portability.
- **Runtime dispatch picks the best at startup.** Within one x86_64
  binary, an AVX-512 + VNNI chip runs the VNNI path; a Skylake laptop
  runs AVX2; a Pentium 4 runs scalar. Same binary, no chip pays for
  capabilities it can't use.
- **One binary per OS+major-ISA.** `veritate-x86_64.exe`,
  `veritate-x86_64-linux`, `veritate-arm64-macos`, `veritate-arm64-linux`.
  Separate builds. **No fat universal binary that flattens performance.**
- **The 9800X3D stays the speed champion.** All optimization on the dev
  box gets banked. Other platforms inherit the architecture and ship
  their own custom-tuned versions hitting their own ceilings.
- **"Binary IS the model" still holds.** Architecture (V_HIDDEN,
  V_LAYERS, etc.) is compile-time per the CLAUDE.md ethos. A different
  model shape means a recompile.

# ------------------------------------------------------------------------------------
# Tier matrix
# ------------------------------------------------------------------------------------

Ranked by realistic install base in 2026. Each row is a kernel
implementation we ship.

| # | Tier                                  | Hardware example                            | Install base                  | Status |
|---|---------------------------------------|---------------------------------------------|-------------------------------|--------|
| 1 | x86_64 + AVX-512 + VNNI               | Ryzen 9800X3D, Sapphire Rapids+, Zen 4+     | Modern dev/server             | **Done — current target** |
| 2 | x86_64 + AVX2                         | Intel Mac mini 2018, Haswell→Ice Lake       | 10+ years of consumer x86     | Matmul done; rest of model.c not |
| 3 | ARM64 + NEON SDOT                     | Apple M1+, modern Android, Cortex-A76+      | Apple Silicon, modern phones  | **Initial port landed** (matmul + transformer hot-path; bench TBD) |
| 4 | ARM64 + NEON only (no SDOT)           | Pi 4, older Android, M1 base path           | Cheap Linux SBCs, embedded    | matmul_int8_neon raw only; prep/transformer ride NEON SDOT TU |
| 5 | ARM64 + AMX (Apple matrix coprocessor)| M-series Macs                               | Mac-only, stretch goal        | Empty |
| 6 | scalar C                              | RISC-V, old Intel, anything                 | Universal correctness baseline| Matmul done; rest of model.c not |

Skipped intentionally:

- **AVX-512 without VNNI** — tiny install base (Skylake-X, Cannon Lake).
  AVX2 fallback covers them.
- **Pre-2013 SSE-only x86** — fading. AVX2 covers anything from 2013+.
- **WASM** — entirely separate target, deferred.
- **GPU backends (CUDA, Metal, Vulkan compute)** — out of scope. Veritate
  is the CPU-first inference engine. GPU is the *training* substrate.

# ------------------------------------------------------------------------------------
# Refactor status — extracted from model.c
#

Five hot-path primitives split out of `engine/src/model.c` into
`engine/kernels/<arch>/`:

| Function                | Binding         | x86_64 location                          |
|-------------------------|-----------------|------------------------------------------|
| `attn_dot_inline`       | inline-per-arch | `model.c` (avx-512 + vnni)               |
| `attn_hsum_inline`      | inline-per-arch | `model.c` (avx-512 + vnni)               |
| `score_dot_v_avx512`    | direct call     | `kernels/x86_64/transformer_avx512.c`    |
| `softmax_rows_avx512`   | direct call     | `kernels/x86_64/transformer_avx512.c`    |
| `layernorm_i16_to_i8_avx512` | direct call | `kernels/x86_64/transformer_avx512.c`    |

`attn_dot` and `attn_hsum` stay inlined: 64-element int8 helpers
called millions of times per prefill. Function-pointer dispatch
overhead exceeds their bodies. Per-arch ports provide their own inline
versions via header swap at compile time.

The other three are direct calls today (one ISA tier, AVX-512). When
a second arch lands (NEON / AVX2), introduce a function-pointer
typedef + dispatch on first need — the call sites in `model.c` swap
to the indirect call at that point. ~30 seconds of wiring.

GELU LUT stays in `model.c` — already pure C, portable.

# ------------------------------------------------------------------------------------
# Cross-platform signatures (locked)
# ------------------------------------------------------------------------------------

These signatures are the cross-platform interface. Today only x86_64
implements them; other-arch ports must match exactly. When a second
arch lands, a `typedef` + runtime dispatch goes in front. Until then
the call sites in `model.c` invoke the AVX-512 versions directly.

```c
// score-weighted V row sum into 64 int8 outputs.
// scores is int16 quantized softmax (scale 32768).
void score_dot_v(const int16_t* scores, const int8_t* v_base,
                 int32_t v_stride, int32_t n_j, int8_t* out);

// softmax over a float row, writing int16 quantized result alongside.
// in-place modifies float row. cols may not be 16-aligned (decode path).
void softmax_rows(float* x, int16_t* out_q, int32_t rows, int32_t cols);

// int16 residual stream -> int8 layernorm output. cols always V_HIDDEN.
void layernorm_i16_to_i8(const int16_t* x, int8_t* out, const int8_t* w,
                         int32_t rows, int32_t cols);
```

For the per-pair attention helpers (`attn_dot`, `attn_hsum`), each port
defines `static inline` versions in `model.c` (or an arch-specific
header included from it). These run inside the `for (j ...)` inner
loop — function-pointer overhead dominates their tiny bodies. Compile-
time binding is the only viable design.

```c
// per-pair signatures (inline only, NOT function pointers)
static inline int32_t attn_dot_inline(const int8_t* q, const int8_t* k, int32_t q_sum);
static inline int32_t attn_hsum_inline(const int8_t* x);
```

`q_sum` is precomputed `sum(q[0..63])` as int32. x86 VNNI uses it for
the signed/unsigned correction; ARM SDOT can ignore it.

# ------------------------------------------------------------------------------------
# Adding the second arch — three-step dispatch wiring
# ------------------------------------------------------------------------------------

When a second arch ships its first kernel (NEON, SDOT, AVX2, etc.),
introduce runtime dispatch in one commit. Until then the x86_64 build
calls the `_avx512` versions directly to avoid wrapping a single impl.

**Step 1 — `engine/src/veritate.h`:** add the typedef and extern next
to the matmul block.

```c
typedef void (*score_dot_v_fn)(const int16_t* scores, const int8_t* v_base,
                               int32_t v_stride, int32_t n_j, int8_t* out);
extern score_dot_v_fn score_dot_v;

void score_dot_v_avx512(const int16_t* scores, const int8_t* v_base,
                        int32_t v_stride, int32_t n_j, int8_t* out);
void score_dot_v_neon  (const int16_t* scores, const int8_t* v_base,
                        int32_t v_stride, int32_t n_j, int8_t* out);
```

(Repeat for `softmax_rows` and `layernorm_i16_to_i8` if the second
arch lands those too. Same shape: `_fn` typedef, extern, forward decls
for each implementation.)

**Step 2 — `engine/src/dispatch.c`:** add the global default and the
runtime selection.

```c
score_dot_v_fn score_dot_v = score_dot_v_avx512;  // or _scalar default

void dispatch_init(const cpu_features_t* feat, dispatch_info_t* out) {
    // ...existing matmul dispatch...

    if (feat->avx512_vnni) score_dot_v = score_dot_v_avx512;
    else if (feat->neon)   score_dot_v = score_dot_v_neon;
}
```

**Step 3 — `engine/src/model.c`:** swap call sites from the direct
name to the dispatch name.

```c
// before
score_dot_v_avx512(row_q, v_base, qkv_stride, i + 1, out_row);
// after
score_dot_v(row_q, v_base, qkv_stride, i + 1, out_row);
```

`grep -n score_dot_v_avx512 engine/src/model.c` finds them; same drill
for `softmax_rows_avx512` and `layernorm_i16_to_i8_avx512`.

That's the whole wiring. About 30 seconds of editing. The AVX-512
implementations stay where they are — `engine/kernels/x86_64/transformer_avx512.c`
— and become one of two (or more) backends behind the dispatch.

# ------------------------------------------------------------------------------------
# Kernel correctness contract
# ------------------------------------------------------------------------------------

Every kernel implementation must satisfy:

1. **Bitwise oracle match for matmul.** All matmul kernels produce
   identical int32 output to `matmul_int8_scalar`. Already enforced in
   `main.c` `verify_match`.
2. **≤1 LSB int8 diff for the rest.** Attention helpers, softmax,
   layernorm: scalar reference is the oracle; SIMD versions are
   considered correct if final int8 output differs by at most 1 LSB
   per element across the full forward pass.
3. **Decode bit-equivalence within tolerance.** `forward_decode`
   appended to a cached prefill must match a fresh `forward` over the
   appended sequence to ≤1 LSB. Enforced by `VERITATE_VERIFY_DECODE`.
4. **Sub-millisecond matmul gate** on the platform's stated bench
   target. The number is per-platform — not all chips can hit 0.09 ms.

# ------------------------------------------------------------------------------------
# Per-platform bench targets
# ------------------------------------------------------------------------------------

The 0.09 ms standard is on the 9800X3D. Other platforms have their own
ceilings determined by memory bandwidth and SIMD width.

| Platform                          | Decode p50 target | Limiting factor                |
|-----------------------------------|-------------------|--------------------------------|
| Ryzen 9800X3D, AVX-512 + VNNI     | **0.09 ms**       | L3 bandwidth (96 MB cache)     |
| Modern Intel + AVX-512 (no 3D-V$) | 0.5-1 ms          | DRAM bandwidth, smaller L3     |
| Apple Silicon M4 (NEON + SDOT + AMX) | 0.1-0.3 ms     | UMA bandwidth (~200+ GB/s)     |
| Apple Silicon M1 base             | 0.5-1 ms          | UMA bandwidth (~70 GB/s)       |
| Intel Mac mini 2018 (AVX2)        | 3-5 ms            | DDR4-2400 bandwidth, no L3 fit |
| Intel Mac mini 2014 (AVX2 + DDR3) | 5-10 ms           | DDR3 bandwidth                 |
| Raspberry Pi 4 (Cortex-A72, NEON) | 50-100 ms         | DRAM, no SDOT, narrow SIMD     |
| Pi 5 (Cortex-A76, NEON + SDOT)    | 10-30 ms          | DRAM bandwidth, SDOT helps     |

Numbers are estimates pending real benches. **Each platform port
adds its measured number to this table.** Workbook entries cite the
specific platform for every bench number going forward.

The "killer performance on old hardware" threshold is **coherent text +
sub-100 ms decode**. Anything in that range feels instant in a chat
context (human perception floor is ~150 ms). We hit that easily on
2014-era hardware once the AVX2 port lands.

# ------------------------------------------------------------------------------------
# Build matrix
# ------------------------------------------------------------------------------------

Today:

- `build.bat` — Windows x86_64. Working.
- `build.sh` — POSIX (Linux + macOS, x86_64 + arm64). Working.
- `setup.ps1` — Windows toolchain (LLVM-mingw + NASM via winget). Working.
- `setup.sh` — POSIX toolchain check (Apple clang on macOS, distro
  package on Linux). Working.

`build.sh` detects host via `uname -s/-m`, picks the matching kernel
TUs (x86_64 → `kernels/x86_64/*`; arm64 → `kernels/arm64/*` plus the
scalar int4 + hadamard fallbacks), and invokes clang with the right
arch flags (`-mcpu=apple-m1` on Apple Silicon, `-mavx512vnni` on
x86_64). Output: `bin/<os>/<arch>/veritate`.

# ------------------------------------------------------------------------------------
# Refactor order (one-time cost, then platforms compose)
# ------------------------------------------------------------------------------------

1. **Lock function-pointer signatures.** Done above in this doc.
2. **Extract from `model.c` to `kernels/x86_64/`** — keep current
   AVX-512 work, refactor as standalone `.c` files exposing the
   function-pointer signatures. Add `kernels/scalar/` references for
   each (correctness oracle).
3. **Wire dispatch.** Extend `dispatch.c` to pick attention / softmax /
   layernorm kernels in addition to matmul. CPU-feature detection
   already does the work.
4. **Verify zero regression** on the 9800X3D. Bench numbers in
   workbook should be identical (within noise) to pre-refactor.
5. **AVX2 port** — implement the same five functions in
   `kernels/x86_64/avx2.c` style. Old Mac mini Intel + old PC.
6. **NEON SDOT port** — `kernels/arm64/sdot.c`. Apple Silicon mini, M4
   Studio, modern Android.
7. **NEON-only port** — `kernels/arm64/neon.c`. Pi 4 baseline.
8. **AMX port (stretch)** — `kernels/arm64/amx.c`. M-series stretch goal.

After step 4, every subsequent platform is contained work — same
skeleton, different intrinsics. No platform port can regress another
because they're separate translation units.

# ------------------------------------------------------------------------------------
# What this unlocks
# ------------------------------------------------------------------------------------

- **A 2018 Mac mini** ($300 used) becomes a useful local AI box.
- **A Pi 5 with 8 GB RAM** ($80) runs a coherent chatbot for hobbyist
  projects.
- **An old gaming laptop** with AVX2 + 16 GB RAM runs Veritate at
  ~5 ms/token — chat-instant, no internet, no API key.
- **The Mac Studio M4 inbound** becomes the production performance
  target alongside the 9800X3D, with NEON SDOT + AMX competing for
  the speed crown.

The strategic angle: **Veritate doesn't need users to upgrade
hardware**. The market for "AI on hardware you already own" is much
bigger than the market for "AI on a brand-new GPU rig." This is the
same constraint-driven framing as the 0.09 ms target — make the
software so efficient that the hardware ceases to be a problem.
