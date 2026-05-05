# Plan -- everything active in this sprint

Single-sprint plan. No future versions. No phases. Every item below is in
flight or in scope right now. Items that depend on other items list those
dependencies inline.

# ------------------------------------------------------------------------------------
# Active blockers (resolve before stacking the rest of the budget)
# ------------------------------------------------------------------------------------

- **Runtime shape refactor.** `V_HIDDEN`, `V_LAYERS`, `V_FFN`, `V_HEADS` are
  preprocessor constants in `engine/src/veritate.h`. That blocks spec decoding
  (target 768/12/3072/12 + draft 256/4/1024/4 cannot coexist), Mamba-2 80M
  drop-in, and any future architecture experiment. Refactor `model_t`,
  `forward*`, `acts_t` / `decode_acts_t` / `verify_acts_t` to take a
  `shape_t` arg; activation pools heap-allocated; numeric kernels already
  shape-agnostic so the lock lives entirely in `model.c` orchestration.
  **Prerequisite for budget steps 3 (spec decoding), 4 (Mamba-2), distillation.**

- **Trace frame size.** Per perf-trace harness (2026-04-29), 78% of the
  per-token wire bytes are float32 attention scores (147 KB / 235 KB). Quantize
  to int8 with per-row scale. Saves ~0.5 ms steady-state wall. **Blocks** the
  user-visible latency target of <1 ms per byte in the MRI browser.
  *(2026-04-29 update: shipped — frame 235 KB → 126 KB, wall 1.85 → 1.69 ms.
  Test exe at $LOCALAPPDATA/veritate/veritate_test.exe, swap when ready.)*

# ------------------------------------------------------------------------------------
# Parallel research tracks (not on the decode budget — separate lanes)
# ------------------------------------------------------------------------------------

- **Multi-module brain** — modular cognitive architecture trained byte-level
  with curriculum mirroring child development. 7 modules: language core,
  workspace, valence head, rule trace, meta-monitor, HDC memory, reality
  monitor. See [docs/MULTIMODULE_BRAIN.md](MULTIMODULE_BRAIN.md).
- **Confidence math** — hard ECE-validated calibrated confidence per token,
  derived from margin + entropy + per-layer lens consistency + residual
  stability. Twin-model branching for fact/fiction disagreement signal. See
  [docs/CONFIDENCE_MATH.md](CONFIDENCE_MATH.md).
- **Curriculum** — Stage A done, B running, C corpus ready, D-H planned. See
  [docs/CURRICULUM_PLAN.md](CURRICULUM_PLAN.md).

# ------------------------------------------------------------------------------------
# Target
# ------------------------------------------------------------------------------------

- 0.03 ms per-token decode at any context length.
- Coherent text generation on the trained 80M TinyStories byte-level model.
- 9800X3D dev box; arm64-mac and pi5-class as cross-platform targets.

Current measured baseline: 0.59 ms decode at pos=10, perplexity 7.88 on
TinyStories val (post-graduation, post-gibberish-fix, per-channel scales,
sparse ffn_down threshold=4).

# ------------------------------------------------------------------------------------
# Decode budget projection
# ------------------------------------------------------------------------------------

| Step | Optimization                                | -delta    | Running |
|------|---------------------------------------------|-----------|---------|
| 0    | Baseline                                     | -         | 0.59 ms |
| 1    | INT4 + QuaRot weight format (composes here)  | x 0.85    | 0.50 ms |
| 2    | Mixture of Depths (40% layer reduction)      | x 0.60    | 0.30 ms |
| 3    | Speculative decoding K=4 acceptance 0.85     | / 1.72    | 0.18 ms |
| 4    | Mamba-2 SSD architecture (O(1) per token)    | x 0.6     | 0.10 ms |
| 5    | BitNet b1.58 ternary weights                 | x 0.3-0.5 | 0.03 ms |

Targets compound multiplicatively. The 0.03 ms moonshot lands when all five
items ship. Intermediate budgets are real exit ramps if any single step
underperforms.

# ------------------------------------------------------------------------------------
# Shipped this sprint (already in engine/src/ or training/)
# ------------------------------------------------------------------------------------

Engine kernels:

- `forward_verify` (M=K decode-shape matmul, 2.13x at K=4)
- Branchless min-heap top-K threshold sampler (43x faster)
- AVX-512 INT4 packed matmul + QuaRot weight format
- Sparse ffn_down kernel with GELU zero-clamp at threshold=4 (default on)
- Per-output-channel weight scales (replaces per-tensor scale)
- C-side per-layer logit lens (precomputed during decode)
- Decision-tracing fields in TFRM frame (decisiveness, dla_picked, dla_argmax,
  argmax_byte) wired into chat_traced_loop and the MRI backend

Training:

- `train.py` weight-export fix: transpose to match `prep_b()` row-major layout,
  shared embed/pos_embed activation scale
- `export_quarot_int4.py` offline rotation + INT4 packing
- Per-channel scale export
- xIELU activation reference implementation (validated 1.3% perplexity
  improvement at 10M params)
- Re-exports of the trained 80M model in v3 (uniform), per-channel, and
  INT4 + QuaRot formats

MRI infrastructure:

- `mri/server/diff.py` C/PyTorch differential trace harness
- `chat_traced` subprocess respawn logic
- `/neuron` endpoint with byte affinity
- Decision-trace and decisiveness panels in the live MRI

# ------------------------------------------------------------------------------------
# In flight (background agents)
# ------------------------------------------------------------------------------------

1. **QAT mode 2 + xIELU into the 80M model.** Aggressive quantization-aware
   training that simulates the int8 inference forward, including per-channel
   scaled requant and the threshold=4 GELU clamp. Composes with the already-
   validated xIELU activation. Target: perplexity within 1.5x of fp32 baseline.
   Closes the residual quantization drift identified in Finding 12.

2. **Speculative decoding end-to-end.** Train a 5M-parameter byte-level draft
   model on TinyStories. Integrate with the shipped `forward_verify` kernel
   in `chat_traced_loop`. Target: 1.5-2x decode throughput at realistic draft
   acceptance.

3. **Mamba-2 SSD prototype.** Build the selective state-space block as an
   alternative `block_t`. Train a 10-20M parameter model on TinyStories,
   compare quality and per-token decode latency to the equivalent transformer.
   Validates the architectural pivot before scaling to 80M.

4. **Decision-tracing fields in the C engine.** TFRM frame v5 protocol bump:
   `decisiveness[V_LAYERS]`, `dla_picked[12]`, `dla_argmax[12]`, `argmax_byte`.
   Includes `byte_direction` precompute (V_LAYERS x V_FFN x V_VOCAB) at model
   load time. Brings the v8 PyTorch backend's decision-trace panels online for
   the C backend.

# ------------------------------------------------------------------------------------
# Pending engineering -- next batch after current agents land
# ------------------------------------------------------------------------------------

5. **Mixture of Depths gate.** Per-token early exit per the lens commitment
   distribution measured in experiment 21. Expected 30-40% realistic layer
   reduction (theoretical ceiling 56%). Train a small gate network that
   predicts per-token exit layer.

6. **BitNet b1.58 ternary weights.** Native ternary training {-1, 0, +1},
   multiplies become sign flips and adds. Composable with Mamba-2 SSD or
   transformer. Per literature: 2.4-6.2x x86 speedup over INT4.

7. **HDC long-term memory layer.** Hyperdimensional encoding of past
   conversation turns (1 KB per turn). Microsecond retrieval over 1000+
   turns. Sidesteps the V_SEQ=256 limit without retraining the transformer.

8. **Analog backend stub.** Noise-injecting `matmul_int8` function pointer
   per the dispatch table. Forward runs end-to-end with deterministic noise
   modeling Mythic-style flash-cell analog matmul. Validates analog
   readiness (already shown viable up to 5% per-matmul noise in exp 14).

9. **Speculative compute during typing.** Forward / prefill begins as soon
   as the user types. KV cache extends per-keystroke; backspace truncates.
   At sentence-period commit, the matching forward is already computed.
   TTFT becomes "look up the matching pre-computed forward."

10. **Per-keystroke speculative branches.** Branch tree of likely user
    intents, pruned each keystroke by which still-match the typed prefix.
    By commit, all surviving branches are pre-computed.

11. **Brain-style architecture: meta-monitor.** Small fast head that scores
    main-model output for confidence, contradiction, OOD input. Triggers
    deferral to tool / search / human when confidence is low.

12. **Sleep-cycle memory consolidation.** Verbatim KV during active session;
    rolling summary during idle; full consolidation at sleep cycle.
    Compressed embeddings store the gist; verbatim is dropped beyond a window.
    Valence drives consolidation priority.

13. **Cross-platform kernel ports.** AVX2 (Intel Mac mini, older PCs),
    NEON SDOT (Apple Silicon, modern Android), NEON-only (Pi 4),
    AMX (Apple M-series stretch). Per the locked function-pointer kernel
    contract in PLATFORMS.md.

14. **Huge-page weight allocation.** `VirtualAlloc + MEM_LARGE_PAGES` for
    2 MB pages. The 80 MB model goes from ~1500 TLB entries to ~40.

15. **NPU port of one matmul.** AMD XDNA SDK on the dev box. Run NPU and
    CPU concurrently; block N on NPU while block N+1's prefill prep runs
    on CPU.

16. **DirectStorage weight streaming.** Stream prepped_b_t from NVMe through
    DMA. No user-space copy. Validates "model in SSD" as the achievable
    approximation of the analog-SSD long game.

17. **Heterogeneous compute scheduler.** Run different layers on different
    silicon (CPU SIMD, NPU, iGPU, DMA engines) concurrently. Use everything
    available; saturate the system.

# ------------------------------------------------------------------------------------
# Quality gate
# ------------------------------------------------------------------------------------

Every speed change preserves bit-match against the scalar oracle (1 LSB int8
tolerance for non-matmul stages). Every quality change is measured against
the C engine `ppl` subcommand on TinyStories val.

The differential trace harness (`mri/server/diff.py`) is the working tool for
the next residual-drift investigation. Item 1 above (QAT mode 2) is the
direct-line response to Finding 12.

# ------------------------------------------------------------------------------------
# What is no longer on the table
# ------------------------------------------------------------------------------------

- *Streaming KV writes (movntdq).* Lost: K/V are read many times after
  write; non-temporal stores are net negative. Documented in exp 01.

- *Fused layernorm into matmul.* Lost: layernorm is 0.8% of layer time,
  not the leverage point. Documented in exp 02.

- *Polynomial GELU in fp32.* Lost: int8<->fp32 conversion erases the speed
  win on the existing pipeline. Documented in exp 12.

- *Per-head KV cache layout reorg.* Deferred: 21% on attention reads, 0.4%
  net. Picked up automatically when MQA/GQA or grouped-query attention
  lands in a future architecture.

- *RWKV-7 architectural pivot.* Lost: 7 input projections per layer is
  strictly slower than transformer's 3 at V_SEQ=256. Documented in
  RWKV_PORT.md. Mamba-2 SSD is the chosen architectural pivot.
