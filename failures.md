# failures (kill list)

Falsified approaches. Each entry: the measured result that killed it and the retry condition. An approach here is dead until its retry condition is met. Short entries only.

## training throughput levers

**Net2Net growth-at-flatten has no trigger on real text: the 3.6x was a saturating-corpus artifact**
Growth only nets compute if the cheap stage stops when its val curve flattens (successes.md 2026-07-25 rule). Built the detector and the widen (`veritate_mri/training/grow.py`, `grow_to_ffn` flag, function-preservation verified) and measured the premise directly: on enwik8 at the 5m shape, **all three completed 10,000-step runs still show 9.2-9.8% relative val gain in their final window**, and none ever flattens. A 1500-step smoke never fires either (10.6% gain). The original 3.6x was measured on a procedurally-generated curriculum corpus that its own entry flags as SATURATING; real-text pretraining at these scales does not stop paying, so there is no moment at which widening is free.
Retry: only where a stage genuinely saturates (a narrow curriculum stage, a drill corpus, or a heavily-repeated small corpus). Do NOT apply to real-text pretraining. The widen and detector are built and tested if a saturating stage ever appears. (2026-08-05)

**Product-key memory is not a training-speed lever**
It is an inference and capacity lever only. Training still reads every addressed slot and the optimizer still updates the table, so capacity costs training time rather than saving it. Measured on M3 Ultra (seq 512, b4, MPS): dense 592M **2542 tok/s** vs PKM 2309M-capacity **983 tok/s**. Any plan that budgets training time from ACTIVE params is wrong for PKM. (2026-08-03)
Retry: only if a sparse-update path lands that makes optimizer cost scale with touched slots rather than table size, and it beats dense at matched capacity.

**Sparse gradients on the PKM value table: no win on high-bandwidth hardware**
At 500M-class capacity (sub_keys 1448 = 2,096,704 slots, 537M value params, b8 seq256) M3 Ultra: dense grad **0.188 s/step**, `F.embedding(sparse=True)` + SparseAdam **0.209 s/step**. Only 4.8% of slots are touched per step, so the sparsity is real; unified memory bandwidth just makes the dense optimizer update cheap enough that sparse tensor construction costs more than it saves. (2026-08-03)
Retry: on a bandwidth-poor box (CPU-only edge training), or once table size exceeds memory and the dense grad allocation itself becomes the constraint.

**`n_chunks` optimizer amortization does not generalize across shapes**
Measured +68% at chat200m's shape. At 593M (24L h1280 ffn5120, seq 1024, b24, bf16, MPS) it is a LOSS: n_chunks 1 **5722 tok/s**, 2 **4924** (0.86x), 4 **3993** (0.70x). Splitting an already-large batch starves the GPU; the win only exists where the per-step batch is below the hardware knee. (2026-08-03)
Retry: re-benchmark per shape before enabling. Never carry the setting across model sizes.

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

**Matching structured layers to dense on PARAMETERS is the wrong control and inverts the result**
Holding parameters equal forces a structured layer to a 5-6x wider hidden dimension, and activation bytes scale with hidden, not with params. Under that control Monarch measured 0.13-0.18x on cardinal and read as dead. The correct control is matched HIDDEN dimension, where activations are identical and the structure shows as a parameter/FLOP cut: the same benchmark then reads 1.35x at H=1024 and 3.15x at H=2304 (ideas.md IDEA 14). Rule: for any layer that changes the params-to-hidden relationship, hold the activation shape fixed and let params vary, never the reverse. (2026-08-05)

**int8 training math does not raise cardinal's compute roofline: no VNNI means the win is bandwidth, not arithmetic**
The `_mm256_cvtepi8_epi16` + `_mm256_madd_epi16` path is not arithmetically faster than `_mm256_fmadd_ps` on AVX2 without VNNI; it moves 4x fewer bytes. Measured on cardinal, single core, fp32 and int8 kernels structurally identical (same transposed-B dot product, scalar parity checked), sweeping matrix size to separate the two effects: cache-resident **N=64 1.175x, N=80 1.305x, N=96 1.528x**; bandwidth-bound **N=256 3.415x, N=1024 2.770x**. Arithmetic-only the format is worth ~1.2-1.5x, and training on cardinal is compute-bound (4.81x thread scaling, successes.md), so it cashes at the low end. Also: comparing a transposed int8 kernel against a row-major fp32 one reports 4.31x, of which **1.93x is memory layout alone** — a fair int8 claim requires identical structure in both arms.
Retry: only on a chip with VNNI (`vpdpbusd`), where the arithmetic ratio is real, or for bandwidth-bound decode where the 4x traffic cut already pays (successes.md PKM kernel). (2026-08-05)

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
Inductor `aten.convolution_backward` stride assertion (2026-07-13). ~33% win on dense only. Re-tested torch 2.12 on 2026-08-03: IDENTICAL crash, same op, same assertion. Still dead whole-model.
Partial workaround measured (successes.md): compiling only `blk.ff` and the local-attention submodules, leaving the recurrent blocks eager, does not crash and buys **1.10x** at 593M. The conv is the only blocker, so a compile-safe short conv would unlock the rest.

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

**Aggregate val bpb does not measure instruction-format adherence (wren_sft, 2026-08-05)**
wren_sft 200m, mix fineweb_edu 0.40 / openwebtext10g 0.30 / chat_5gb 0.15 / wikitext103 0.05 / veritate_v1 0.05 / instruct 0.05 — 70% general web, 20% chat/instruct. Val fell 0.7798 (step 65,000) → 0.7656 (step 95,000), ~30,000 steps. 16-prompt greedy A/B over that same interval: 95k won 2 (factual recall — "Mercury, Venus, Earth."→"Jupiter.", "I can't do that"→"Elephant"), 65k won 3 (format — correct prime definition→nonsense, one-sentence ocean→ramble, "Yes, water is wet"→"No, water is wet"), 11 ties. Arithmetic wrong at both (2+2 → 3 vs 5; Monday→Monday vs Sunday). Val tracks the 70% it is mostly made of; the 20% you actually want is invisible to it. Rule: for any SFT run, the stop/continue signal is the target behavior measured directly or a domain-split val — never aggregate bpb. n=16 greedy, single seed; enough to refuse a stop decision, not enough to rank checkpoints.

**No real eval data on mirach — the suites are smoke fixtures**
`veritate_mri/data/eval/samples/`: mmlu 3 items, hellaswag 2, ifeval 36. Any suite number computed here is noise. CPU eval also runs ~5 min/item (ifeval, 128 max_new), so even the 36 is ~3 h/checkpoint. Blocks every "did the last N steps buy anything" question. Fix before the next run: real suite data on disk + eval on GPU.

**`layers` on a patched-family trunk is not the checkpoint's block count (wren1_0 launch, 2026-08-12)**
`VeritatePatched` builds one flat list of `N_LOCAL_ENC (2) + layers + N_LOCAL_DEC (2)` blocks, so wren_base at `--size 200m` (`layers=16`) wrote 20 `blocks.N.*` entries. A resume that inferred `layers=20` by counting those entries built a 24-block model; `load_resume_state` used `strict=False`, so it loaded the 20 real blocks, left **54,649,152 params at random init**, printed no warning, and would have "SFT"ed a partly-random model to completion. Caught only because the printed param count (325,159,488) did not match the checkpoint (270,510,336). The `shape.layers: 16` in wren_base's config.json was correct all along; the earlier todo calling it a bug was wrong.
Rule: never infer a layer count from `blocks.*` keys without subtracting the trunk's local-block overhead, and never resume on `strict=False` without asserting nothing was stranded. Both now enforced (`trunk_block_overhead`, `load_resume_state(require_complete=True)`, 34 tests). Diff `sum(p.numel())` against the checkpoint before trusting any resume.
**Wren 1.0 chat SFT: a transcript-heavy mix buys turn depth and pays in circular filler (2026-08-16)**
wren_base + 8,000 steps, `loss_mask=assistant`, mix mixed_chat 0.30 / chrg 0.20 / wren_identity 0.18 / hansard 0.12 / scotus 0.08 / ukinquiry 0.07 / veritate_chat 0.05. 35 h. Model deleted 2026-08-17; these are the numbers.

30-prompt greedy A/B vs the base: median reply 96 B -> 276 B, but **repeats a 6-word window 0.06 -> 0.44**, grounded retrieval (answer present in the prompt) **0.62 -> 0.38**, identity 0.00 -> 0.50. Longer and worse. It learned chat cadence and filled it with restatement ("The ocean is a vast and complex ecosystem that is deeply interconnected with the ocean itself." twice in one reply), and began overwriting facts sitting in its own context ("The blue folder holds the contracts" when the prompt said the red one did).

Three causes, all confirmed by the wren1_1 retry that fixed them (successes.md 2026-08-17):
1. **47% of the mix was congressional/parliamentary/court transcript**, a formulaic register. Cutting it to 0.20 took looping to 0.19 and restored grounded retrieval to 0.62.
2. **Identity at dose 0.18 over a 226 KB corpus (~1,250 passes) memorized question phrasings, not facts** -- "Who are you?" perfect, "What platform were you trained on?" returned a paragraph about Ariana Grande. The corpus had no spec family at all; parameters/context/platform only ever appeared inside name and maker replies. Adding 250 spec templates took identity to 0.83 at a THIRD of the dose.
3. `sft_idk` was never in the mix, an authoring miss, so the run said nothing about refusal behaviour.

Rules: cap formal-transcript corpora under 0.20 of a chat SFT mix; they are a turn-DEPTH source, not a voice source. Grade a chat SFT under GREEDY decode -- `writing_health` samples at t=0.7 and reported distinct_4 rising 0.729 -> 0.986 ("repetition collapsing") for the checkpoint that looped 44% greedily; the two metrics disagreed completely and only the greedy one predicted chat quality. For identity, buy paraphrase diversity per fact before buying dose.
