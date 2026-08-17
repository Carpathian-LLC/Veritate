# successes

Validated results with the evidence that proved them. Entries move here from `ideas.md` when an idea clears its falsifier. Short entries only: claim, numbers with units and date, done.

## measurement method

**Eager PyTorch cannot rank decode architectures: ~20 us per op at batch=1 regardless of work**
Measured on cardinal (i7-9700T, 7 threads): `x*2` on a 320-vector 20.3 us; a 320x320 matmul doing 102,400 MACs 38.0 us. 102k multiply-adds cost 17.7 us more than doing nothing, so op COUNT sets batch-1 cost, not FLOPs. Dense FFN wins in PyTorch because it is 3 fat ops; any sparse/gated/conditional design needs more ops and loses regardless of merit. Same product-key-memory algorithm: **4.9x slower in eager PyTorch, 3.90x faster in C** (both arms AVX2 int8). Rank batch-1 architectures in C, never in eager. (2026-08-03)

## architecture and optimizer levers

**Partial torch.compile routes around the hybrid-trunk MPS crash: 1.10x**
Whole-model `torch.compile` still dies on `aten.convolution_backward` at torch 2.12 (failures.md). Compiling only `blk.ff` plus the local-attention submodules and leaving the recurrent blocks eager runs clean: 593M on M3 Ultra b24 bf16 **5603 -> 6138 tok/s**. ff-only is 1.01x (nothing); the local-attention blocks carry the win. Ceiling is low because 20 of 24 blocks are recurrent and stay eager. (2026-08-03)

**593M training-throughput ceiling on M3 Ultra: ~20.5 days at Chinchilla**
Measured, 24L h1280 ffn5120 seq1024 bf16 MPS: eager b24 **5603 tok/s** (24.5 d); + partial compile **5937** (23.1 d); + b=64 **6689** (20.5 d). Total headroom from every throughput lever combined is **1.19x**, and `n_chunks` is a loss at this shape (failures.md). **A sub-week 500M budget is therefore a data-efficiency problem, not a throughput problem: it must come from needing fewer tokens (distillation), not faster steps.** (2026-08-03)


**Product-key memory: 102x capacity at 3.90x faster decode, quality parity**
Threshold-gated PKM vs dense FFN, both arms on the same AVX2 int8 integer-dot matvec + int8 activation quant, cardinal, batch=1, core-pinned: dense **199.6 us/token** (819,200 params) vs PKM **51.2 us/token** (83,886,080 params), 6.49x fewer bytes read. Three-way quality A/B trained end to end on the clamped 800 MHz i7 (5m shape 8L h256 ffn1024, enwik8, AdamW, fp32, qat off, seq 256 b8, 10,000 steps per arm) — final val: dense **1.299641**, top-k PKM **1.294813**, **threshold PKM 1.282350**. Threshold wins at every matched checkpoint (2000/4000/6000/8000/9500/10000), so the SAME variant the fast kernel runs is also the best-scoring one; the trained model and the kernel are one computation. Margin 1.33% on a single seed, so the reportable claim is PARITY, not a win (agent_roe seed rule), though the sweep across all six checkpoints is stronger than an endpoint. Capacity is unexploited at this scale: 12.7x the params only matches, because enwik8 (95 MB) cannot fill it. (2026-08-03)

**Firing threshold beats top-k selection: 15x on the selection stage**
Ranking candidates then taking the best k does not vectorize; a threshold compare does. AVX2 compare+movemask **0.586 us** vs `topk(512, k=32)` **8.778 us**. Sorts were 52% of PKM's int8 cost (topk 66.9 us of 128.0 us total); the scattered gather was only 8%. Forcing the gather fully contiguous gained 7%, so access pattern was never the bottleneck. Threshold PKM int8: 139.3 -> 69.6 us/token. (2026-08-03)

**AdamW over Muon on a clamped CPU: 11x**
cardinal (i7-9700T @ 800 MHz BIOS clamp), 5m shape: Muon **9.5 s/step**, AdamW **0.81 s/step** (2582 tok/s). Muon's Newton-Schulz orthogonalization is a measured win on GPU (entry below) and a heavy tax on a clock-starved CPU. Pick the optimizer per box, not per project. (2026-08-03)

**Muon optimizer: 1.60x byte savings**
AdamW final val 1.0375 vs Muon 0.9990 at 12k steps (10M byte model); Muon reaches AdamW-final quality at step 7500. Default optimizer. (2026-07-03)

**Boundary-patched trunk (SpaceByte-style): 1.82x wall-clock savings**
Patched reaches dense's final val (0.9990) at step 8600. Final val 0.9776 vs dense 0.9990 at equal steps; 49% more params yet 128% of dense throughput.

**Constant-state recurrent trunk: quality parity with attention, O(1) decode state**
Final val 0.9900 vs dense 0.9990 (ahead 117/120 matched evals). 18% slower per step, so equal-wall-clock is parity (+0.011, inside the ±0.03 band). The prize is the fixed decode state, not the curve.

**Composed hybrid trunk (patched + recurrent) beats both parents**
Final val 0.9707 (patched 0.9776, recurrent 0.9900, dense 0.9990); 1.70x wall-clock to dense-final; 113% of dense throughput. Composed stack at 10M: Muon 1.60x × patching 1.82x ≈ 2.9x. Default research architecture. The 1.15x margin over patched-final is inside the 5% noise band; second seed owed.

## model growth and pruning

**Net2Net FFN-widen growth is a real 3.6x step-lever — but only nets compute if the small phase stops at saturation**
Steps-to-target vs from-scratch: 9.0x @val 0.30, 3.6x @0.1214 (900 vs 250 steps). Function-preservation verified (resume loss 0.0917 vs parent 0.0923). Counter-result: at total compute the grown arm cost 1.66x MORE because the cheap stage ran 2000 steps past its ~1250-step saturation. Rule: stop every stage at val-flatten, then widen.

**Muon + group-lasso induces massively prunable dead capacity at near-zero post-prune cost**
stack_win_50m: 61.2% param reduction, val 1.3023→1.3027 (Δ+0.0004) after prune; Muon-only control 16.7% at Δ-0.0001. Quality tax of the regularizer itself is in failures.md. (2026-07, experiments/v2/stack)

## training recipes and dosing

**In-pretrain dosing beats late-phase SFT for teaching a small model a new skill**
Confirmed repeatedly: chat200m (identity + grounded-read + copy dosed from step 0), chat80idk_80m abstention (6% sft_idk dose). Late-phase concentrated dosing destroys conversation-copy (failures.md).

**Low-dose (0.15) + heavy replay (0.75) capability SFT installs format obedience without forgetting**
IFEval 280-item: 24.3%→43.9%, +19.6pt, p=1.0e-6 (wren base vs wren_sft). Retention 6/7. Val drift +0.012. Boundary: format obedience installs at this scale; content accuracy does not (base undertrained at 4.8 tok/param).

**sft_idk 6% in-pretrain dose produces honest abstention**
10M PoC: 75% abstention precision at step 1200. 80M ship model: hit the 90/80 gate at step 9000, natural register retained.

**Layered capability-SFT campaign on chin200m: 4/4 skills pass, one deployable stacked model**
grounded_read 12/12 (+75pt), multiturn 11/12 (+92pt), instruct 8-9/12 (+50-58pt), prose_v2 10/12 (+66pt) after template-diversity fix. Stacked fork (grd+mt+inst @10% each): grd preserved 12/12, mt bleed fixed at 10% dose, inst dropped to 6/12 (wants ~15%). Lessons: template diversity is first-order; skills have different dose-response curves.

## retrieval and external memory

**Zero-training mean-pool retrieval is scale-stable**
recall@1 0.62 / recall@5 0.82, flat across store sizes 1e3→1e5.

**Trained contrastive key head clears the milestone gate**
recall@1 0.40→0.976, recall@5 0.71→0.998 over a 1e5-leaf store. End-to-end top-1 injection: grounded accuracy 0.75 vs bare 0.02.

**Key head generalizes to real content across unseen topics**
33 real topics: unseen-topic recall@1 0.070→0.697 (≈ seen 0.667). Hard mixed distractors: unseen 0.497/0.783 (7x baseline). Resolves the toy-schema overfit in failures.md as a diversity artifact.

**Natural (teacher-written) queries transfer but are ~2x harder than heuristic queries**
Held-out topics recall@1 0.409/0.684 (fineweb), 0.322/0.608 (hard). End-to-end n=100: grounded_acc 0.140 vs bare 0.010 (14x), bottlenecked by retrieval precision.

**Sub-quadratic drill-down retrieval validated toward trillion scale**
IVF drill-down at nprobe=16 scores 3.1% of a flat scan for recall@1 0.960 vs exact 0.976; candidate cost scales √N. FAISS IVF-PQ: 29x faster at N=1e6, 155x at N=1e7; 32B/vector holds recall@1 0.856. Trillion-char index feasible on-box: 500M leaves × 32-64B = 16-32GB RAM.

**Relations stored in the index route around the role-binding wall**
(subject, relation, object) triples + DB-join hop outside the model: multi-hop accuracy 36%→100% on the same 200M reader. Cost moves to offline index build.

**Grounded-copy RAG works from 300M up; the precursor sweep that proved it**
grounded_chunk_800m grounded_copy_acc 0.225 vs base 0.017; grounded_300m 0.237 vs control 0.0; grounded_v2_300m 0.157; "proper"-retrieval 200M arm 0.075 (n=120). Pre-grounded_v3 numbers, disjoint model names. (experiments/v2/rag, 2026-06/07)

**Aux-expert sidecar helps out-of-domain long-form**
PG19 nll 2.198 (trunk alone) → 1.823 (+aux, Δ-0.374); code prompts 3.928→3.098. In-domain it is a wash (failures.md).

## rag-writer architecture

**Grounded single-fact slot copy is near-ceiling at 122M — bigger models buy nothing**
100% @122M, 100% @200M, 97% @800M. Collapses with stuffed context: 1 chunk 100%, 3→60%, 8→30%.

**A ~30-line IDF lexical retriever + a 200M reader solves the K=8 task the model alone cannot**
100% system accuracy vs 40% model-alone; `system_acc ≈ retriever_precision@1 × reader_acc(1 chunk)`. Caveat: synthetic unique entities flatter lexical retrieval; expect 70-90% precision@1 on paraphrased real queries.

**Role-masked (assistant-only) loss preserves capability under SFT**
Opt-in, zero throughput cost. chat800m_v2: grammar 0.72→0.82, reading 0.748→0.762 vs unmasked. Clean same-corpus ablation not yet run.

## quantization and deployment

**Post-hoc INT8 export preserves quality without QAT at these scales**
grd1: 12/12 target-skill greedy replies byte-identical fp16 vs int8, 51% smaller. stack1: 28/36 vs 29/36 fp16. chat800m_v2 990MB, chat_80m 127MB.

**INT4 per-row and QuaRot PTQ are near-free on a trained 85M**
fp32 ppl 1.542 baseline: INT8 PTQ +0.02% ppl, INT4 per-row +0.43%, INT4 QuaRot +0.38%. Per-tensor INT4 is catastrophic (failures.md). (experiments/v2/int4_quarot)

**Batched prefill is a 1.64x cold-TTFB win on linux x86_64 and was shipped disabled**
The engine's batched-prefill path (`hybrid_prefill`, bitwise-identical to sequential) was pinned to width 1 for every arch because it regresses 12x on Apple Silicon. Measured on cardinal, 200m hybrid trunk, cold 320-byte prompts with unique text so nothing hits the state cache: batch 1 **3.46s**, 8 **2.38s**, 32 **2.11s**, 64 **2.12s**. Now per-arch in `c_engine.prefill_batch()`; linux x86_64 gets 32, unmeasured tiers stay at 1.
Cold TTFB is **pure prefill**: fitting TTFB against prompt length over 1-640 bytes gives **-0.03s fixed + 11.9 ms/byte**, so there is no fixed per-request overhead to remove and prefill throughput is the only lever. Even batched, prefill runs at ~34 GF/s against cardinal's 166.7 peak. (2026-08-08)

**Read-ahead plus the engine state cache already cut TTFB 6.3x, and both were on by default**
Simulating a typist (prefix request every 40 bytes) against the 200m hybrid trunk on cardinal: cold paste-and-enter of 320 bytes **3.90s** vs **0.62s** after read-ahead has walked the prefix in. Background cost is 3.8s of prefill spread over the ~64s it takes to type 320 bytes, a ~6% duty cycle. `c_engine.py` sets `VERITATE_STATE_CACHE` on the engine subprocess with `state_cache=True` by default, so the parent process env is empty by design - checking the parent is not a test of whether the cache is on. A hand-set parent value silently WINS over the per-model default (`env.setdefault`) and orphans the warm cache. (2026-08-08)

**wren decode on cardinal (i7-9700T, 800MHz-clamped): ~35 ms/byte, ~1.8s TTFB**
1.5x the ~23ms bandwidth floor; the gap is an ~11-12ms/position serial floor (recurrent stack + rmsnorm), not a defect. The ≤1.3 ms/byte aspiration needs a BIOS unclamp (~4x) or second RAM DIMM (~2x).

**Split-precision split-device training: 58% peak-VRAM drop at QAT-parity convergence**
bf16 master on CPU, INT8 fake-quant copy shipped per forward, STE grad return. Converges within noise of standard QAT on a 25M model; 1B-class training on 12 GB VRAM feasible with grad-ckpt. Cost: per-forward H2D weight traffic. Mechanism lives in `veritate_core/qat.py`.

## chat model milestones

**chat200m chat-phase SFT clears all four pre-registered gates**
Needle copy 1.00/0.83 @190B/475B; bare identity 3/3 name + 3/3 maker; grounded alien-fact read 3/3; empathy register intact.

**chat200m pretrain beats the 80M's finished pretrain before its own is 40% done**
2.005B tokens in 39.9h at 14.1-14.3k tok/s; val 1.70→0.812. At matched tokens, 200M beats 80M inside the 80M's own window.

**chat80m: the first conversing byte model**
Three phases (pretrain 1.695→0.942, midtrain →0.681, SFT →0.647), ~9h GPU, ~470M tokens. Converses; factual recall fails (compute wall).

## corpus and infra benchmarks

**Curated Python byte corpus at scale**
12.0 GiB train / 251 MiB val, 1.55M files, all `ast.parse`-clean, exact-deduped.

**Instruction SFT pipeline proof**
HumanEval pass@1 0%→15% by step 1000 (~16M tokens) on a 0.5B subword base.

**Training throughput ceiling on M3 Ultra**
0.5B bf16: 4,858 tok/s eager, 6,484 compiled (bs32/seq512). 1.5B: 1,947/2,187. MLX ties torch-MPS; GPU is FLOP-bound at ~70% of realizable bf16.

**Fluency chat corpus rebuilt — brevity and length-decay both eliminated (2026-08-10)**
`~/Library/Mobile Documents/com~apple~CloudDocs/Mirach-Corpuses/new_chat_corpus/`, 227.1 MB, 87,828 conversations, 409,328 assistant replies. Median reply **267 B** (was 14 B), **90.4% in the 200-400 B band** (was 0.7%), **0.00% under 150 B** (was 55.8%), turns/conv median 4 / mean 4.66 (was 2.4). Sources: openbmb/UltraChat 118 MB (MIT), ultrachat_200k 51 MB (MIT), Daring-Anteater 43 MB (CC-BY-4.0), hh-rlhf helpful-* dirs only 12 MB (MIT), in-house veritate_sft 7,261 convs, cogito persona 474 convs preserved byte-identical (SHA verified). All commercially clean; NC/share-alike sources (no_robots, Dolly, DailyDialog) rejected on license.
**The load-bearing result is the no-decay curve: t1 259 B → t7 277 B, ratio 1.069, rising monotonically with depth.** The old corpus decayed 284→220→116 B by turn 5, which is what trained fade-out. Method that produced it: truncate long assistant-instruct replies to leading whole sentences (never mid-sentence — a mid-sentence cut is a training example that says "stop mid-thought"), cut at the first list marker, strip disclaimer sentences whole, hard 150 B floor with zero tolerance, within-conversation Jaccard dedup at 0.6, and a turn-desync check that the next user turn share a content word with the *retained* reply.
Rule: build conversations COMPLETE and let the packer cut turn-boundary-aligned windows to the current `seq`; building "for seq 1024" bakes the limit in permanently.
Ceiling, not yet solved: no permissively-licensed source has both depth and length. UltraChat caps at 7 assistant turns, Daring-Anteater at 3, hh-rlhf at 8 (3 convs). No-decay is therefore proven through turn 7 only; sustained 10+ turn conversation needs a generated source or IDEA 7. Spec: `CORPUS_SPEC.md` (session scratchpad, v1 + v2 amendments).

**Wren 1.1 chat SFT: a usable chat model at step 1,250, and the checkpoint ladder that found it (2026-08-17)**
wren1_1 = wren_base + `loss_mask=assistant` on mixed_chat 0.50 / veritate_chat 0.18 / chrg 0.10 / wren_identity 0.06 / sft_idk 0.06 / hansard 0.06 / scotus 0.04. Stopped at step 3,000 of 6,000. **Best checkpoint is step 1,250, not the last one.** 30-prompt greedy A/B (16 format, 8 grounded, 6 identity):

| | wren_base | wren1_0@8000 | wren1_1@1250 |
|---|---|---|---|
| median reply | 96 B | 276 B | **182 B** |
| closes its turn | 0.94 | 0.94 | 0.94 |
| repeats a 6-word window | 0.06 | 0.44 | **0.19** |
| grounded retrieval | 0.62 | 0.38 | **0.62** |
| identity | 0.00 | 0.50 | **0.83** |

What the two fixes bought, against wren1_0's failure: cutting formal transcripts 0.47 -> 0.20 took looping from 0.44 to 0.19 and RESTORED grounded retrieval to the base's 0.62 (wren1_0 had regressed it to 0.38, answering "The blue folder holds the contracts" when the prompt said the red one did). Rebuilding the identity corpus around paraphrase diversity per fact -- a 250-template spec family where v1 had none -- took identity 0.50 -> 0.83 at a THIRD of the dose (0.18 -> 0.06). Every spec question v1 failed now answers exactly and in voice. `sft_idk` at 0.06 produces real refusals ("I don't know that answer, sorry.") instead of confabulation.

**The ladder is the transferable part.** Quality peaked at step 1,250 and degraded monotonically after: looping 0.12 (1,000) -> 0.19 (1,250) -> 0.31 (2,000), grounded 0.50 -> 0.62 -> 0.38, hs_ppl on fixed text rising every one of 12 checkpoints. The last checkpoint was the worst usable one. Never assume the final checkpoint of a chat SFT is the best; score a ladder on a fixed prompt set and pick.

Rules: (1) a chat SFT at this scale peaks in the low thousands of steps, so checkpoint densely (ckpt_every 250) and plan to select, not to run to completion. (2) Watch small-corpus concentration: putting 0.30 of the mix on 10.4 MB (veritate_chat 8.7 MB + sft_idk 1.27 MB + identity 0.43 MB) degraded fixed-text perplexity 7.5x faster than wren1_0's 0.23-on-8.9 MB. (3) 6 prompts cannot rank checkpoints; the rate's standard error is ~+/-0.2. 16+ per category.
