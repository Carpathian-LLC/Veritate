# Experiments tracker

Master hypothesis log for the moonshot push toward 0.03 ms decode +
perfect text generation. Append-only. Status of each idea: pending,
in-flight, won, lost, blocked.

# ------------------------------------------------------------------------------------
# Target
# ------------------------------------------------------------------------------------

- 0.03 ms per-token decode at any context length
- Perfect text generation quality (no gibberish, coherent prose)
- 9800X3D dev box, eventually arm64-mac and pi5-class targets

Current baseline: 1.0 ms decode at pos=10, 1.18 ms at pos=250.
Quality: PyTorch path coherent, C path gibberish (open bug).

Gap: ~33x speed, fix gibberish.

# ------------------------------------------------------------------------------------
# Hypothesis log
# ------------------------------------------------------------------------------------

## H1 -- Streaming KV writes
LOST. K/V are read many times after write. exp 01.

## H2 -- Fused layernorm + matmul
LOST. LN is 0.8% of layer time. exp 02.

## H3 -- Top-K attention at decode
DEFERRED. Attention is 2% of decode at pos=200. Low leverage. exp 03 (analysis).

## H4 -- QuaRot Hadamard rotation for INT4
WON. Synthetic 35% reduction (exp 04). Real-weights ppl +0.45% vs INT8 on
TinyStories 130K val (exp 15). End-to-end engine integration shipped (exp 18):
`training/export_quarot_int4.py` + `engine/kernels/x86_64/matmul_int4.c`,
version-4 .bin format, bit-identical to scalar INT4 oracle.

## H5 -- INT4 packed AVX-512 matmul
WON. AVX-512 cross-lane permute (vpermt2 64-bit qword) interleaves unpacked
nibbles into sequential int8 then VNNI dpbusd accumulates. Bit-match scalar
oracle on ffn_up and ffn_down decode shapes. Decode m=1 latency 0.83 ms vs
0.96 ms INT8 (1.15x faster). exp 05 + 18.

## H6 -- Per-head KV cache layout
SMALL. 21% on attention reads, 0.4% net. Defer to MQA/GQA refactor. exp 06.

## H7 -- Linear attention (Mamba/RWKV-style recurrence)
PRINCIPLE WON. 4.32x speed at T=256, retrain required. exp 07.

## H8 -- Speculative decoding with tiny draft
WON (math + acceptance, 2026-04-28). 3.3m-param byte-level draft trained on
TinyStories at hidden=256 layers=4 ffn=1024 heads=4, 5000 steps, val ppl 2.15
vs target 1.65. Vaswani-style rejection sampling implemented in pytorch with
kv-cached forwards on both models. Observed acceptance: 0.646 at K=4, 0.765
at K=8 (T=0.7, 200 tokens, n=3 trials). Spec output's CE under target sits in
the same regime as baseline output's CE -- distribution correctness confirmed.
Cost-model projection on 9800x3d c engine using exp 11's measured
forward_verify timings: 1.10-1.24x decode tok/s at observed acceptance,
upper bound 1.74x at higher acceptance per exp 11. C-engine wiring deferred
(needs second compile-time model_t for the small-shape draft). exp 08 + 11
+ 25.

## H9 -- Hyperdimensional computing for sequence
NOT THE FIT. Memory primitive at 700x speed, not generative. Becomes a
long-term memory feature alongside the transformer, not a replacement. exp 09.

## H10 -- xIELU activation as drop-in for GELU
WIN. 10.77M-param byte-level TinyStories, 8K steps, identical seed. Final
val loss GELU 0.7499 / ppl 2.117, xIELU 0.7373 / ppl 2.090. xIELU pulls
ahead at step 100 and stays ahead through 7900. Training-time cost is
2.83x per-step (eager bf16 torch.where + expm1); inference cost is
identical via INT8 LUT. Recommend swap on next 80M run. Finding 22.

## H11 -- C engine gibberish quality bug
WON (FIXED 2026-04-28). Two compounding bugs in `training/train.py` `export_to_bin`:
(1) weight transpose mismatch -- PyTorch stores `nn.Linear.weight` as `[N, K]`
but `prep_b()` reads `[K, N]` row-major; every weight matmul read a transposed
matrix. Fix: `np.ascontiguousarray(W.T)` before serialize. (2) embed and
pos_embed quantized at independent scales (55.7 and 489.2) summed as int8 in
the engine -- a meaningless mixed-unit add. Fix: `quantize_embed_at_act_scale`
quantizes both at scale 32. After fix L0 residual_post cos_dist 0.987 -> 0.011
vs PyTorch. Engine math untouched. Diagnosed via `mri/server/diff.py`
differential trace harness. Re-export of trained .bin required (done for
tinystories-80m and tinystories-80m-qat). FINDINGS.md Finding 22.

## H12 -- AVX-512 INT4 kernel with vpermt2b permute
WON (GRADUATED 2026-04-28). Cross-lane unpack via 64-bit permutexvar after
AVX2 lane-local unpacklo/unpackhi. Bit-identical to scalar INT4 oracle on
both ffn_up (k=768, n=3072) and ffn_down (k=3072, n=768) decode shapes.
Lives in `engine/kernels/x86_64/matmul_int4.c::matmul_int4_vnni_prep`.
exp 18.

## H13 -- forward_verify kernel for speculative decoding
WON (GRADUATED 2026-04-28). M=K decode batched matmul. Bit-match scalar
oracle clean (max_lsb=0 at K=1,2,4,8,16 vs K sequential forward_decode).
Single-thread batched matmul wins at K=2..7 (1.93 ms at K=4 vs 4.07 ms
K*decode = 2.10x), MT at K>=8 (3.22 ms at K=16 vs 16.28 ms = 5.05x).
Resulting speculative decoding speedup: 1.72x at K=4 acceptance=0.85,
1.29x at K=4 acceptance=0.70. Unblocks H8. Lives in engine/src/model.c
as forward_verify, declared in engine/src/veritate.h. exp 11. WORKBOOK
2026-04-28 graduation entry.

## H14 -- Real-weights QuaRot validation via Python pipeline
WON (GRADUATED 2026-04-28). Per-head Hadamard size 64 + per-row INT4
quantization. End-to-end perplexity on TinyStories 130K val: INT4-QuaRot
1.6640 vs INT8 1.6565 (+0.45%). Pipeline lives in
`training/export_quarot_int4.py`; emits `data/models/tinystories-80m-quarot-int4/veritate-int4.bin`.
exp 15 + 18.

## H15 -- RWKV-7 port investigation
LOST. Investigated 2026-04-28. exp 10 + docs/RWKV_PORT.md.
Verdict: RWKV-7 is slower than current transformer at V_SEQ=256 because the
time-mix block does 7 input projections per layer vs the transformer's 3
(qkv). Estimated decode 1.6-2.0 ms / token vs 1.0 ms today. The state-size
advantage (2-8x smaller than KV cache) only matters at long context, which
we don't have. Mamba-2 SSD form is the cleaner pivot: 3 input projections,
selective-scan reduces to existing matmul kernel, BitMamba-2 ships INT8
state reference. RWKV-7 not on the active list pending long-context use case or
quality gap at 80M that Mamba-2 cannot close.

## H16 -- BitNet b1.58 ternary weights
PENDING. Native training with {-1, 0, +1} weights. Multiplications become
sign flips and adds. Per literature: 2.4-6.2x x86 speedup over INT4.

## H17 -- Distill 80M -> 40M while retaining quality
PENDING. Distillation experiment: train 40M student against 80M teacher
on TinyStories. Halves decode time if quality holds.

## H18 -- T-MAC LUT-only inference kernel
PENDING. Replace matmul with table lookups. Memory ops only, no multiplies.
Per literature: works for 1-2 bit weights.

## H19 -- Train-time QAT with hooks (in flight, parallel agent)
PENDING. The QAT-hooks agent is instrumenting training to capture
quantization scale evolution. Lets us tune scale schedules for
better INT8 -> INT4 transition.

## H20 -- Polynomial GELU instead of LUT
PENDING. Quick test: 4-op polynomial fit of GELU, all in registers.
Tiny win expected (GELU is < 1% of layer), but tests register-resident
arithmetic vs LUT lookup.

## H21 -- Branchless top-K sampler
WON (GRADUATED 2026-04-28). Min-heap replacement for the selection sort
in sample_token. Bit-exact threshold (heap returns the same Kth-largest
element). 43x faster (~13 us -> < 1 us). ~1.2% of decode time saved. Lives
in engine/src/model.c sample_token. exp 13. WORKBOOK 2026-04-28 entry.

## H22 -- Speculative compute during typing
PENDING. IDEAS.md tier 2.6: forward starts as user types, KV cache extends
per keystroke, backspace truncates. TTFT becomes "look up matching forward."

## H23 -- Mixture of Depths (per-token early exit)
PENDING. ADEPT-style per-token layer skip. Easy tokens exit at L4, hard
tokens run all 12 layers. Quality and speed both depend on the gate
network.

## H24 -- Hymba-style hybrid block (Mamba + sliding-window attention)
PENDING. Highest predicted quality at 80M per RESEARCH.md. Two parallel
heads per layer: Mamba state head + 64-token sliding attention head.
Sum outputs.

## H25 -- Compile-time architecture variants via dispatch
PENDING. Allow different block_t implementations (transformer / mamba /
RWKV / hybrid) selectable at startup. Same dispatch table pattern as
matmul. Lets us A/B architectures at the same model file.

## H26 -- Sparse-aware ffn_down kernel (exp 16 + 17 follow-up)
WON (GRADUATED 2026-04-28; threshold default raised from 0 to 4 in exp 20).
matmul_int8_sparse_decode + ffn_down_decode dispatcher live in
engine/kernels/x86_64/transformer_avx512.c. Pre-scans the post-GELU
activation, dispatches to the sparse path when `n_nz * 2 < V_FFN`, falls
through to dense `matmul_int8_vnni_prep` otherwise. Bit-identical int32
output to the dense path by construction.

Bench (9800X3D, QAT 80M, `bench 50 200`):
- threshold=0: forward_decode p50 0.953 ms, sparse fires 33%, ppl 19.350.
- threshold=4 (NEW DEFAULT): forward_decode p50 0.769 ms (1.24x), sparse
  fires 100%, ppl 18.842 (-2.6% vs thr=0; clamping low-magnitude GELU acts
  as a denoiser on the QAT model's quantization-drifted residual).
- threshold=6: ppl 60.3 (+212%; cliff). Do not raise default further
  without re-validating on the deployed checkpoint.
- threshold=8 (illustration only): forward_decode p50 0.547 ms, ppl 61.5.

Validated on 51 000 byte tokens of TinyStories val. Threshold flag
`VERITATE_GELU_ZERO_THRESH` ships compile-time, now defaulted to 4 in
build.bat. exp 19 (kernel) + exp 20 (threshold sweep).

## H27 -- Per-output-channel weight scales (Finding 12 candidate 1)
WON (GRADUATED 2026-04-28). Replaces single `scale_q24` per matmul with one
per output column. Weight format `VERITATE_MODEL_VERSION_PERCOL = 5` (file-format identifier, not project version).
PPL 17.31 -> 7.88 (-54%, on tinystories-80m base 80M model), decode p50 0.77
-> 0.59 ms (-24%, post-GELU activations are 1.4% nonzero vs 15.6% with
uniform scales because per-col calibration moves more outputs into the
GELU dead zone). L11 residual drift cosine distance 0.58 -> 0.48 (-18%).
Bit-match scalar oracle preserved -- per-col scales apply in the requant
step, the matmul kernel itself is unchanged.

Lives in `engine/src/model.c` (`requant_pb` helper picks per-col vs
uniform), `engine/src/veritate.h` (`prepped_b_t.scale_per_col`),
`engine/kernels/x86_64/matmul_vnni.c` (`prep_b` initializes NULL,
`free_prepped_b` releases), `training/train.py`
(`quantize_int8_per_row`, `export_to_bin_percol`),
`training/ckpt_to_bin.py --per_col`. Uniform-scale path is intact for existing
bins; per-channel detection is by header version.

Does not close the gap to fp32 (1.68 ppl) -- per-channel format still 4.7x above.
Remaining gap requires QAT mode 2 or wider intermediate (Finding 12
candidates 2 and 3). Per-channel is the cheap half. Finding 24.
exp 22.

## H28 -- v8 decision-trace data on the C engine
WON (2026-04-28). Wired `decisiveness`, `dla_picked`, `dla_argmax`,
`argmax_byte` into the C path so the same MRI panels work on both
backends. Engine builds a per-layer int16 `byte_direction` table at
`model_load` (~150 ms one-time, 18.9 MB), `chat_traced_loop` writes
the new fields after `final_act + logits`. TFRM bumped 4 -> 5; frame
+480 bytes (+0.20%). Decode latency unchanged (p50 0.83 ms). Bit-match
scalar oracle preserved -- the new code is a read-only side channel
on already-captured trace data. Top-3 DLA contributors agree with
the PyTorch backend on essentially every high-confidence token; full
top-12 sets agree on >80% of frames (boundary disagreements in the
rank-12 tail). Decisiveness shape matches PyTorch within 5-15% magnitude.

Lives in `engine/src/veritate.h` (`dla_entry_t`, `byte_direction*`,
`VERITATE_TRACE_VERSION = 5`), `engine/src/model.c`
(`byte_direction_build`, `decisiveness_compute`, `dla_top`),
`engine/src/main.c chat_traced_loop` (frame-write changes),
`mri/server/c_engine.py` (numpy DLA dtype + parser),
`mri/server/app.py _build_c_mri_frame` (int -> fp32 display rescale).

Open: int4 path emits zero-filled DLA tables because `prep_b_int4`
does not retain `b_rowmaj`. Saturation and memory.peak_pos still
PyTorch-only. exp 23.

## H29 -- Mamba-2 SSD prototype (PyTorch evidence)
IN-FLIGHT (2026-04-28). Replaces PLAN item 3. PyTorch-only validation; C
port deferred until quality / O(1) decode are confirmed.

Reference block at `training/mamba2_block.py` (selective scan via
log-prefix cumsum + masked exp). Recurrent `step` proven to match the
parallel `forward` to 3.6e-7 max abs diff via
`experiments/26_mamba2_prototype/verify_step_parity.py`. NaN bug fixed
in the upper-triangle of the decay matrix (mask before exp, not after).

Param-matched pair on TinyStories byte-level (~7.5M):
- mamba-2: hidden=384 layers=8 head_dim=64 n_state=64 expand=2 (7.62M)
- transformer: hidden=288 layers=8 ffn=1024 heads=4 (7.52M)

Training: 2500 steps each, B=8 T=256 bf16, AdamW cosine 3e-4 -> 3e-5.
Bench: per-token decode latency at pos=10/100/256, generated samples
on the prompt "Once upon a time", plus state size vs equivalent KV cache.

Output lives in `experiments/26_mamba2_prototype/RESULTS.md` (numbers,
samples, recommendation) plus `data/models/mamba2_test/` and
`data/models/mamba2_test_txfm_baseline/` (checkpoints + sample text).

## H30 -- QAT mode 2: simulate the C engine int8 forward in PyTorch (Finding 12 candidate 2)
WON (2026-04-28). Aggressive QAT that simulates per-channel weight scales,
the int32 -> int8 requant rounding, post-GELU threshold=4 zeroing, and an
int16 residual stream. Warm-started from
`tinystories-80m/checkpoints/step_45000.pt` and fine-tuned 10000 steps
at lr 5e-5 -> 5e-6 cosine, batch 16, bf16. Wall ~26 minutes on RTX 5070.

C-engine ppl 7.88 -> 4.44 (-44%, within 2.6x of fp32 vs 4.7x before).
PyTorch QAT2 forward reaches val ppl 1.64, slightly below fp32 base
ppl 1.68 -- the int8 quantization loss is fully recovered in training.
L11 residual drift (vs fp32 base model.py) cos_dist 0.481 -> 0.241
(-50%, target was 0.30).

Decode latency 0.588 ms -> 0.750 ms (+27%). FFN sparsity dropped from
1.4% to 15.1% non-zero because QAT2 trained the model to USE the FFN
intermediates -- the v5 baseline got accidental speedups from quant-noise
sparsity. Re-sweeping `VERITATE_GELU_ZERO_THRESH` on the QAT2 weights
should recover some of that speed.

C engine chat output is "there was a two two two ..." -- fragments
become real words, sentence-level coherence still broken in the engine.
PyTorch QAT2 forward generates coherent prose: "Once upon a time, there
was a small boy named Joe. He was very lonely in his bedroom...". Both
forwards agree on top-5 logits at the prefill but diverge during
multi-token decode; suspected cause is LN weight fold ordering
(engine quantizes `(x-mean)/std` to int8 then matmul applies `qkv*ln_w`;
QAT2 quantizes `((x-mean)/std)*ln_w` to int8 then matmul uses raw `qkv`).
Fix lives in a QAT2 v3 follow-up.

Pure training-side change. No engine code touched. Bit-match scalar
oracle preserved.

Lives in `training/qat_v2.py`, `training/qat_v2_finetune.py`,
`run_training.py` (`--qat2` flag), `mri/server/diff.py` (TFRM v5 patch).
New model at `data/models/tinystories-80m-v5-qat2/`. Finding 25.
exp 24.

# ------------------------------------------------------------------------------------
# Cross-cutting concerns
# ------------------------------------------------------------------------------------

- "perfect text generation" requires fixing H11 (gibberish bug) before any
  speed claim is meaningful. Quality is the gate.
- The 96 MB L3 budget is the constraint that drives everything. Architectures
  that fit are viable; architectures that don't aren't.
- Bit-match scalar oracle is non-negotiable for kernel changes. Quality
  (perplexity) is the equivalent gate for architecture / quantization changes.
