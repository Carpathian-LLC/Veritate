# failures (kill list)

Falsified approaches. Each entry: the measured result that killed it and the retry condition. An approach here is dead until its retry condition is met. Short entries only.

## relational role binding

**Zero relational role binding at any scale — architectural wall, not data or capacity**
OBJECT-role accuracy 4%/0%/0% at 122M/200M/800M. Confirmed at 10M/26M on purpose-built role data: held-out who 17%/what 0%; more training made controls worse (100%→50%). Fails at 5 sizes × 2 data regimes.
Retry: explicit binding primitive (slot/register layer), or route around it via relations in the index (validated 36%→100%, successes.md). Do not chase data or capacity fixes.

## rag / retrieval

**Model is a good reader but a poor relevance judge**
36% of irrelevant chunks get a confident fabrication. One-chunk-at-a-time + confidence-picking scored worse than stuffing (35% vs 50%); confidence does not track correctness. Relevance lives outside the LM.

**Hard-negative mining does not lift key-head retrieval**
Hard-store recall@1 0.351→0.327 (noise). Ceiling is frozen-trunk feature quality. Retry only on richer trunk features (800M).

**Multi-leaf (top-k) context injection makes grounding worse**
grounded_acc top-1 0.130, top-3 0.120, top-5 0.080. A 200M copy-limited generator cannot disambiguate candidates. Retry with a stronger generator or a re-ranker collapsing k→1.

**Key head overfits toy schemas — resolved as a diversity artifact**
3 schemas: unseen-schema recall@1 0.540→0.176 (below baseline). 11 schemas: 0.332 (1.5x baseline). Real data transfers (successes.md).

**RARS (retrieval folded into O(1) recurrent state) fails exact recall**
K=8 overflow: prefix 0.000 / RARS 0.025; K=64 both 0.000. The state is a gist mechanism, not an exact-recall buffer. Ship top-1 prefix injection. Retry only with an explicitly addressable state.

**Corpus-echo suffix sidecar degrades quality monotonically with echo strength**
alpha 0.0→0.3: nll/byte 0.3797→0.3902 on a real 85M checkpoint; sc6 variant ppl 1.9037→2.3941 on held-out val. Source checkpoint gone; numbers not re-derivable. (experiments/rag + sidecars, 2026-05)

## architecture trainability

**Delta-rule state update (Gated DeltaNet) not trainable on this stack**
Two divergences under two numeric regimes at 10M (fp32 cancellation ~2e17 fixed by block-recursive inverse; second run still NaN'd at step 2094). Retry: capped beta, much lower delta-projection LR, or a fused kernel off MPS.

**Pinned memory register (decay-exempt slots): no recall benefit, real quality tax**
Pure chat: +0.073 val (~7σ), recall 0.00=0.00. Recall mix: quality-neutral but recall 0.338 vs 0.423 baseline. Killed on pre-registered line. Retry: much larger scale or pins trained with an explicit recall loss.

**Selective language modeling (RHO-1) at byte level**
Tail-10 val 1.0638 vs 0.9763 baseline (~8σ worse). Byte-level "easy" tokens carry structure. Retry at patch/subword granularity or anneal-phase only.

**Test-time depth scaling via weight-tied loops does not emerge**
CE by loop count R=1..8: peaks at R=3 (the training-time mean), degrades past it.

**Fast-weight memory does not store retrievable facts**
Recall lift 0.0 at every distance (both reset and carried regimes); carried variant actively hurts at 4096B+ (win rate 0.167). Kept only as a free context extender (val 0.9867 beats dense). Retry: carry horizons at recall distance, ≥170M scale, or retrieval memory.

**Unnormalized fast-weight inner loop breaks at real width**
Stable at hidden=64; NaN in first forward at hidden=320. Fix: unit-normalize k/q, divide inner grads by chunk size, 0.02-std init. Rule: stability-smoke test-time-learning modules at real width.

**TRM/HRM puzzle recursion does not transfer to language modeling**
Literature + ~1M-param port on TinyStories lost to a plain transformer in every configuration.

**Backprop-free learning rules have no LM parity at any scale** (literature verdict)

**Student self-improvement (STaR/BoN) collapses below 1B** (literature: marginal-to-negative at 0.5B). Retry once the student clears nonzero executed-benchmark pass rate.

**Ternary (BitNet) is not a training-cost lever below 3B**
Literature: needs ~2x hidden width below ~3B, QAT adds cost. Measured in-repo on 85M: ternary QAT ppl 1.753 (+13.66% vs fp32 1.542) at 0.1975 bytes/weight. A decode/energy lever, not a training lever. (experiments/v2/ternary_qat)

**INT4 per-tensor PTQ is catastrophic**
+193.5% ppl on a trained 85M. Per-row and QuaRot variants are near-free (successes.md). (experiments/v2/int4_quarot)

**Naive sparse kernels lose to dense by 800-1700x at 87.5% activation sparsity**
dense 1.256 ms/iter vs CSR 1002.7 vs gather 2170.6 on this hardware, despite 8x fewer multiplies in theory. ReLU-sparsity FLOP savings are not realizable here. (experiments/v2/neuron)

**Sparsity regularizers buy little prunable capacity at quality parity — and forced pruning without them is ruinous**
group-lasso 7.6% reduction at ~same val; l1strong 3.7%; baseline forced-50% prune costs val 0.9888→1.3637. The Muon+group-lasso stack (successes.md) prunes 61.2% but its raw pre-prune quality trails plain baseline (1.3023 vs 1.2638). (experiments/v2/neuron + stack)

**Aux-expert sidecar is a wash in-domain and hurts ASCII-art OOD**
base vs +aux val nll Δ+0.0004; ascii OOD nll 5.912→6.040. Small n, low confidence, but no signal to chase. PG19 OOD win is real (successes.md).

**Streaming attention sinks: real 2.7x throughput, degenerate output**
209.6 vs 77.2 B/s, zero extra MPS memory — but output degenerates to a repetition loop at 4096-byte prompts. Both halves count. (experiments/streaming, checkpoint gone)

**Dynamic-shape batching on MPS is a 23x slowdown**
~200 tok/s vs ~4,858 at fixed shapes. All MPS training uses fixed or bucketed shapes.

**torch.compile crashes the hybrid trunk on MPS**
Inductor `aten.convolution_backward` stride assertion (2026-07-13). ~33% win on dense only. Re-benchmark only after a torch/MPS upgrade.

## training recipe kills

**Late-phase recall-SFT at 25% dose collapses out-of-distribution recall (121M)**
Needle recall ~190B: 0.92→0.08. Narrow templated recall data replaces emergent copy-from-context. Same recipe at 10% in-pretrain dose taught genuine retrieval (0.47 vs 0.05). Retry: ~5% in pretrain or surface-diverse needles.

**Identity SFT silently destroyed conversation-copy — a skipped needle gate hid it**
Recall 0.917→0.167 while val IMPROVED (0.647→0.644). Standing rule: never skip a pre-registered falsifier.

**sft_identity_v1 (20% refusal family) teaches the refusal frame, not identity**
2/12 probes, val regression. The refusal family contained the target vocabulary and dominated. Retry: name40/maker40/purpose15/refusal5, dose 18%+.

**High-dose capability SFT destroys retention (two failed attempts before the working recipe)**
dose 0.50 no-replay: facts destroyed, template intrusion. dose 0.35 replay 0.15: form below base. Working recipe: dose 0.15, replay 0.75, many steps (successes.md).

**Narrow-template prose SFT fails the gate and leaks its shape**
6/12 (+33pt, gate ≥40pt) at ~10 templates/family; verbatim echo, shape leakage. 18-22 templates/family cleared it (successes.md).

## compute-wall / from-scratch attempts

**From-scratch 2.5B dense byte coder hits the compute wall**
Val plateau ~1.0 by 205M tokens, zero code ability at ~550 tok/s. Retry: a 10x+ training-efficiency lever at equal quality.

**Cheap byteification of a subword coder does not recover code ability**
Qwen-0.5B → vocab 256, 2000 steps on 10MB distilled Python: HumanEval 0/30 held-out; in-distribution samples looked correct (memorized surface). Rule: only held-out executed benchmarks count.

## process / measurement kills

**"Long context is free via n_chunks" — the flag was never wired; the test measured nothing**
Both bench runs were the same config (22,381 vs 22,367 tok/s). What survives: seq alone costs throughput (2048→22,381, 8192→11,019 tok/s on fortis). Rule: confirm tokens/step arithmetically before trusting any throughput number.

**MoE on MPS: 0.715 of hybrid throughput (kill line 0.70) — sequencing, not kill**
Attributed to missing sparse-routing kernels on MPS, not the idea. Deferred to an 80M A/B with a pre-committed symmetric rule.
