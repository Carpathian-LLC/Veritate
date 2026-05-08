---
title: "Build 4: ternary kernel, v10 format, addons end-to-end, gibberish guard"
date: 2026-05-07
tags: [build, engine, kernels, addons, format, gibberish]
summary: Engine gains a ternary scalar oracle and AVX-512 + VNNI ternary matmul kernel. New v10 model format reserves header fields for MoE (n_experts, router_topk, quant_mode); load path is wired but the MoE forward itself lands in a follow-up. Inference-time addons now run on both backends with per-request selection. Non-QAT models are refused at load time with a clear error. New smartness meter added.

---

## versions

- build: 4
- engine: v2.1.0
- mri: v0.1.2
- format: v0.2.0
- plugins: v0.1.1

## what changed

### engine kernels

- **Ternary scalar oracle.** `kernels/scalar/matmul_ternary_scalar.c` ships the BitNet b1.58 reference: trits in {-1, 0, +1}, packed 5-per-byte in base-3, per-tensor mean-abs scale. Matches `veritate/qat.py::fake_quant_weight_ternary` exactly.
- **Ternary AVX-512 + VNNI path.** `kernels/x86_64/matmul_ternary_vnni.c` unpacks each row's trits into a per-call int8 buffer and dispatches through `vpdpbusd` with the standard +128 unsigned-shift trick. Bit-identical to the scalar oracle by construction. Further unpack-in-SIMD optimization is a follow-up; correctness ships first per claude_preflight rule 23.
- **Pack/unpack helpers** are exported (`ternary_pack_row`, `ternary_unpack_row`) so the export pipeline can produce v10 binaries from a QAT'd MEGA checkpoint and the engine can validate round-trip parity.

### addon contract

- **C addon contract** mirrors the python contract from build 3: three methods (`reset`, `observe`, `bias_logits`) plus `destroy` for explicit teardown. Spec at [documentation/addons/c_engine_port.md](../../documentation/addons/c_engine_port.md).
- **Slot-table addon ported** (`veritate_engine/src/addons/slot_table.c`) at python parity: rolling buffer, NUL-byte block, repetition penalty, n-gram block, gender anchors, hand-rolled "named X" scanner, pronoun trie with collision check, name boost at word-start.
- **Per-request addon selection**: the chat_traced stdin header gained an optional 6th token (`addons_csv`). The MRI server passes `?addons=<csv>` from the dashboard's Generation tab through to the C subprocess; the engine swaps the chain on demand.
- **Engine startup env-var path** still works for headless tests: `VERITATE_ADDONS=slot_table veritate.exe chat_traced` runs with the chain loaded before any request comes in.

### v10 model format

- **Header reserved** for MoE: `quant_mode` (INT8/INT4/TERNARY), `n_experts`, `router_topk`. v9 (BOOST) binaries continue to load unchanged (defaults: INT8, 1 expert, top-1).
- **Load path refuses what the engine can't yet run**: any v10 binary with `n_experts > 1` or `quant_mode != INT8` returns a load error with a concrete message pointing at the in-flight MoE forward implementation. This keeps build 4 from silently dropping into half-implemented code paths.

### model smartness meter

- **Reading bins regenerated** from fresh hand-authored sources at `veritate_mri/grade_eval/sources/`.
- **Math** (5 difficulty tiers, 50 problems each) — algorithmic generation: arithmetic → multi-digit → linear algebra → word problems → multi-step.
- **Grammar** (4 pair-types, 50 each) — pairwise NLL preference: subject-verb agreement, articles, tense, word order. Lower mean per-byte NLL on the correct sentence wins.
- **Reasoning** (4 type-tiers, 50 each) — recall, analogy, single-step deduction, multi-step transitive ordering.

## known issues

- MoE forward is not yet implemented in C. v10 binaries with `n_experts > 1` are refused at load with a clear message.

- Ternary AVX-512 path uses scalar unpack in the inner loop. Functional and bit-identical to scalar; perf is bounded by the unpack rather than VNNI throughput. Follow-up: vectorize unpack via a 256-entry LUT and VPSHUFB or via base-3 reciprocal multiplication.
- The MRI server passes the addons CSV per-request through the chat_traced subprocess header. The C subprocess holds the most recently set chain across requests; the very first request after a chain change pays the build cost. Free at process exit only; no leak across long-running sessions.
- ARM64 still not supported. Tracked separately.
