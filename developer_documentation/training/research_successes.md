# research successes

Validated results with the evidence that proved them. Entries are research outcomes, not bug fixes.

## 2026-07-04: composed trunk (E5 hybrid) beats both parents

- Tested: `trunk=hybrid` = patched local attention + constant-state recurrent global mixer on patch slots, muon, canonical 10M shape (15.9M params at ~dense per-byte FLOPs), fineweb_edu, 12000 steps, single delta vs `e2patched` (its attention-global parent) and vs `e1muon` (dense).
- Result: best final val of all arms (0.9707 vs patched 0.9776, recurrent 0.9900, dense 0.9990); 1.70x wall-clock to dense-final quality; 1.15x to patched-final; ahead of patched at 98/120 matched evals; throughput 79.5k tok/s (113 percent of dense). Falsifier cleared on both conditions.
- Meaning: the composition is not additive-only, it stacks: patch-level compute cost, more params per FLOP, AND the global path carries O(1) state (no KV growth on the long-range component). This is the novel trunk the campaign was aiming at; it becomes the default research architecture going forward.
- Run: `models/e5hybrid_10m_qat/`. Single seed; margins over dense are large, margin over patched (0.007 final, 1.15x) is under the 5 percent band: needs a second seed before the beats-patched claim is reported externally (agent_roe seed rule). The beats-dense claim is comfortably outside noise.

## 2026-07-04: constant-state trunk at byte level (E3) = quality parity with attention

- Tested: `trunk=recurrent` (gated linear recurrence, scalar-per-head decay, O(1) state per head) vs dense attention, both muon, canonical 10M shape (recurrent 10.95M, +8.6 percent from the output gate, disclosed), fineweb_edu, 12000 steps.
- Result: per-step quality BEATS attention (final val 0.9900 vs 0.9990; ahead 117/120 matched evals; reaches dense-final at step 8600). Wall-clock: 18 percent slower per step (57.9k vs 70.3k tok/s, unoptimized chunkwise PyTorch), so at equal wall-clock it sits +0.011 behind: inside the +-0.03 parity band. Falsifier (worse by more than 0.03 at equal wall-clock): survived.
- Meaning: attention is not required for byte-level quality at this scale, and the constant-state trunk's decode memory is O(1) in position vs a KV cache growing without bound. This is the conversation-length lever.
- Run: `models/e3recur_10m_qat/`. Single seed; margin under 5 percent, so the parity claim (not the win claim) is the reportable result per agent_roe seed rule.

## 2026-07-04: boundary-patched trunk at byte level (E2) = 1.82x wall-clock savings

- Tested: `trunk=patched` (SpaceByte-style, global blocks on spacelike-boundary slots, seq/4) vs dense, both arms muon, canonical 10M shape, fineweb_edu, 12000 steps. Patched = 15.0M params at roughly dense per-byte FLOPs.
- Result: patched reaches the dense arm's final val (0.9990) at step 8600 / 6679s = **1.82x wall-clock savings**; final val 0.9776 vs 0.9990 at equal steps; ahead at 105/120 matched evals; realized throughput 128 percent of dense. Falsifier (no bpb win at equal wall-clock, or under 70 percent throughput): cleared on both.
- More parameters per FLOP is real at byte level: the patched arm carries 49 percent more params yet steps faster, because global attention/FFN runs on 4x fewer positions.
- Stack so far (measured, 10M): Muon 1.60x x patching 1.82x = ~2.9x wall-clock vs AdamW dense.
- Runs: `models/e1muon_10m_qat/` (baseline), `models/e2patched_10m_qat/`. Single seed each; margins large.

## 2026-07-03: Muon optimizer at byte level (E1) = 1.60x byte savings

- Tested: canonical 10M byte model, fineweb_edu, 12000 steps, identical arms except optimizer. AdamW final val 1.0375; Muon (2D hidden weights, RMS-matched lr, torch native) final val 0.9990, reaching the AdamW-final quality at step 7500 = 1.60x fewer bytes to target. Ahead at 115/120 matched evals from step 500 on; ~3 percent step-time overhead. Falsifier was under 1.15x: cleared.
- Muon lags during warmup (first ~400 steps); judge nothing before step 500.
- Adopted as the default optimizer for subsequent experiment runs (`optimizer=muon` on `/trainers/run`). Single seed; effect size (60 percent) far exceeds the 5 percent multi-seed threshold in agent_roe.
- Runs: `models/e1adamw_10m_qat/`, `models/e1muon_10m_qat/` (train.csv curves).

## 2026-07-02: curated Python byte corpus at scale

- Built: `trainers/corpus/py_code_v1_train.bin` (12.0 GiB) + val (251 MiB) from codeparrot-clean. 1.55M files, every file passes `ast.parse`, exact-deduped, flat uint8 byte stream compatible with existing loaders. 35 of 54 source shards still unconsumed (more available on demand).
- Plus ~200K GPT-4-quality instruction pairs downloaded (Magicoder OSS-Instruct + Evol-Instruct sets, ~580 MB).
- Evidence: build meta in `py_code_v1_train.bin.meta.json`; delimiter and UTF-8 validity spot-verified.

## 2026-07-03: instruction SFT lifts a base model fast (pipeline proof)

- Tested: chat-template SFT of a 0.5B code base on the downloaded instruction sets, prompt-masked loss, MPS bf16.
- Result: HumanEval pass@1 (held-out, unit tests executed) went 0 percent to 15 percent by step 1000 (~16M tokens). Confirms the SFT pipeline, the eval harness, and that data quality is the operative lever at small scale.
- Caveat: base model was subword (BPE), off the byte-level mission; run retired in favor of byte-level work. Checkpoint: `models/qwen25coder_0p5b_sft/`.

## 2026-07-02: measured training throughput ceiling on M3 Ultra

- Measured: 0.5B transformer, bf16, MPS: ~4,858 tok/s eager, ~6,484 tok/s with torch.compile at fixed shapes (bs32/seq512). 1.5B: ~1,947 eager / ~2,187 compiled. MLX is the same tier as torch-MPS (not a lever). GPU is FLOP-bound at ~70 percent of realizable bf16 compute.
- Implication: wall-clock for any plan computes directly from these numbers; framework changes buy nothing, torch.compile buys ~35 percent, fixed shapes are mandatory (see failures 2026-07-03).
