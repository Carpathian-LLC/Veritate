# successes

Validated results with the evidence that proved them. Entries move here from `ideas.md` when an idea clears its falsifier. Short entries only: claim, numbers with units and date, done.

## architecture and optimizer levers

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
