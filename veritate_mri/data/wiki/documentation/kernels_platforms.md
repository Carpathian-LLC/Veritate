---
title: platforms
date: 2026-05-05
tags: [kernels, platforms, x86_64, arm64, avx512, neon]
summary: Which CPUs Veritate runs on, why, and what each tier hits in ms/byte.
---

> Friendly summary. The canonical contract is `developer_documentation/kernels/platforms.md`.

## the bet

Run capable AI on hardware people already own. Old laptops, Mac minis, Raspberry Pis, dusty desktops. The architecture commitment that makes this work:

- **Hand-tuned kernels per arch** behind function pointers. No portability layer, no compromise.
- **Runtime dispatch** picks the best at startup. AVX-512 + VNNI on a Zen 4, AVX2 on a Skylake laptop, scalar on a Pentium 4 — same binary.
- **One binary per OS + major-ISA.** No fat universal binaries that flatten performance.
- **The 9800X3D stays the speed champion.** All optimization on the dev box gets banked; other platforms inherit the architecture and ship their own ceilings.

## tiers

Ranked by 2026 install base. Each row is a kernel implementation.

| # | tier | hardware | install base | status |
|---|---|---|---|---|
| 1 | x86_64 + AVX-512 + VNNI | Ryzen 9800X3D, Zen 4+, Sapphire Rapids | modern dev/server | done — current target |
| 2 | x86_64 + AVX2 | Haswell → Ice Lake, Intel Mac mini 2018 | 10+ yrs of consumer x86 | matmul done; rest of model.c not |
| 3 | ARM64 + NEON SDOT | Apple M1+, modern Android, Cortex-A76+ | Apple Silicon, modern phones | initial port landed; bench TBD |
| 4 | ARM64 + NEON only | Pi 4, older Android | cheap Linux SBCs | matmul only; transformer rides SDOT TU |
| 5 | ARM64 + AMX | Apple M-series matrix coprocessor | Mac-only stretch | empty |
| 6 | scalar C | RISC-V, anything | universal correctness baseline | matmul done |

Skipped on purpose: AVX-512 without VNNI (tiny install base, AVX2 covers them); pre-2013 SSE-only x86 (fading); WASM (separate target); GPU (training substrate, not inference scope).

## per-platform decode targets

The 0.09 ms/byte target is on the 9800X3D. Other platforms have their own ceilings set by memory bandwidth and SIMD width. Numbers are estimates pending real benches.

| platform | decode p50 | limiting factor |
|---|---|---|
| Ryzen 9800X3D, AVX-512 + VNNI | 0.09 ms | L3 bandwidth (96 MB cache) |
| modern Intel + AVX-512, no 3D-V$ | 0.5 – 1 ms | DRAM, smaller L3 |
| Apple M4 (NEON SDOT + AMX) | 0.1 – 0.3 ms | UMA bandwidth (~200+ GB/s) |
| Apple M1 base | 0.5 – 1 ms | UMA bandwidth (~70 GB/s) |
| Intel Mac mini 2018 (AVX2) | 3 – 5 ms | DDR4-2400, no L3 fit |
| Intel Mac mini 2014 (AVX2 + DDR3) | 5 – 10 ms | DDR3 bandwidth |
| Pi 4 (Cortex-A72, NEON only) | 50 – 100 ms | DRAM, no SDOT |
| Pi 5 (Cortex-A76, NEON SDOT) | 10 – 30 ms | DRAM, SDOT helps |

The "killer perf on old hardware" bar is **coherent text + sub-100 ms decode**. That feels instant in chat (human floor ~150 ms). 2014 hardware hits it once the AVX2 port lands.

## what each port unlocks

- A 2018 Mac mini ($300 used) becomes a useful local AI box.
- A Pi 5 with 8 GB RAM ($80) runs a coherent chatbot for hobbyists.
- An old gaming laptop with AVX2 + 16 GB hits ~5 ms/byte — chat-instant, no internet, no API key.
- The Mac Studio M4 becomes a production target alongside the 9800X3D.

The strategic angle: Veritate doesn't need users to upgrade. The market for "AI on hardware you already own" dwarfs the market for "AI on a brand-new GPU rig."

## adding the second arch

Three tiny edits, ~30 seconds of wiring.

1. **`veritate_engine/v1/src/veritate.h`** — add the function-pointer typedef and `extern` next to the matmul block; forward-declare each implementation.
2. **`veritate_engine/v1/src/dispatch.c`** — add the global default and the runtime selection (`if feat->avx512_vnni: use _avx512; else if feat->neon: use _neon`).
3. **`veritate_engine/v1/src/model.c`** — swap call sites from the direct name (`score_dot_v_avx512`) to the dispatch name (`score_dot_v`).

Until a second arch lands, the x86_64 build calls `_avx512` versions directly to avoid wrapping a single impl. Five hot-path primitives have the dispatch hooks ready: `attn_dot_inline` and `attn_hsum_inline` stay inlined per-arch (function-pointer overhead exceeds their bodies); `score_dot_v`, `softmax_rows`, `layernorm_i16_to_i8` go through pointers.

## correctness contract

Every kernel implementation must satisfy:

| rule | what |
|---|---|
| matmul | bitwise oracle match against `matmul_int8_scalar`. Enforced by `verify_match` in `main.c`. |
| attention helpers, softmax, layernorm | ≤1 LSB INT8 diff vs scalar reference across the full forward pass. |
| decode | `forward_decode` appended to a cached prefill matches a fresh `forward` to ≤1 LSB. Enforced by `VERITATE_VERIFY_DECODE`. |
| sub-ms gate | matmul under the platform's stated bench target. Per-platform — not all chips hit 0.09 ms. |

## build matrix

| script | what | status |
|---|---|---|
| `build.bat` | Windows x86_64 | working |
| `build.sh` | POSIX (Linux + macOS, x86_64 + arm64) | working |
| `setup.ps1` | Windows toolchain (LLVM-mingw + NASM via winget) | working |
| `setup.sh` | POSIX toolchain check | working |

`build.sh` detects host via `uname -s/-m`, picks the matching kernel TUs, and invokes clang with the right arch flags (`-mcpu=apple-m1` on Apple Silicon, `-mavx512vnni` on x86_64). Output: `bin/<os>/<arch>/veritate`.

## refactor order

1. Lock function-pointer signatures. Done.
2. Extract from `model.c` into `kernels/x86_64/` as standalone TUs exposing the signatures. Add scalar references for each.
3. Wire dispatch for attention / softmax / layernorm in addition to matmul.
4. Verify zero regression on the 9800X3D.
5. AVX2 port (`veritate_engine/v1/kernels/x86_64/matmul_avx2.c`). Old Mac mini Intel + old PC.
6. NEON SDOT port (`veritate_engine/v1/kernels/arm64/matmul_neon_sdot.c`). Apple Silicon, M4 Studio, modern Android.
7. NEON-only port (`veritate_engine/v1/kernels/arm64/transformer_neon.c`). Pi 4 baseline.
8. AMX stretch. M-series stretch goal, no kernel yet.

After step 4, every subsequent platform is contained work. Separate translation units mean no platform port can regress another.
