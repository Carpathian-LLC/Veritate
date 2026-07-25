# ideas

Open ideas and active campaigns. Front of the research pipeline. Lowercase per preflight rule 12; root placement is an explicit user instruction (2026-07-07).

## workflow (how an idea leaves this file)

An idea lives here only while it is unproven and open. When work on it finishes, it LEAVES this file:

- Proven (cleared its falsifier, evidence in a real run): cut the entry from here, paste it into `successes.md` with the measured evidence.
- Falsified (hit its kill line) or abandoned: cut the entry from here, paste it into `failures.md` with the measured result and a retry condition.

Nothing stays in both places. `ideas.md` only ever holds live work. `successes.md` and `failures.md` are the permanent ledgers. `research.md` is the index over all three plus the standing research maps and papers.

Rule: an idea is not "done" until it is either in `successes.md` or `failures.md`. No silent deletions.

---

## IDEA 1: frontier-grade chat on this box in 1-2 weeks (the infinite-context chat campaign)

Status: OPEN. Owner: execution model. The constraints below are the standard. Hit them.

### THE STANDARD (what it must do)

A byte-level chat model that: converses at frontier-grade on grounded tasks; holds a trillion-character context it can recall from exactly (IDEA 2); keeps a fixed-size gist of the whole history always on; runs entirely on this one machine at $0; and is trained end to end in 1-2 weeks.

### NON-NEGOTIABLE CONSTRAINTS

1. One machine (M3 Ultra). No cloud, no external compute, $0 marginal cost.
2. Serves ON-DEVICE. A frontier teacher is used OFFLINE to make training data only, never in the inference path.
3. 1-2 week wall clock, end to end.
4. Byte-level (vocab=256). Stay in the Veritate lane. No off-the-shelf 70B.
5. The trillion-character context is delivered by the addressable memory of IDEA 2 (sub-quadratic, on-disk). Not negotiable.

### HOW TRAINING IS SOLVED: make the model smaller

Solve training by shrinking the trained model, DeepSeek-style: capability comes from architecture, sparsity, and distillation, not dense scale.

- The core holds REASONING AND BEHAVIOR, not world knowledge. Knowledge lives in external memory (IDEA 2). Externalizing knowledge is what makes the model small enough to train here in days. This is the key move.
- Small ACTIVE compute, large effective capacity. A dense core at or under ~1.5B active, or a fine-grained MoE with tiny active FLOPs (DeepSeekMoE: many experts, top-k active, large total params held in cheap RAM). Training compute scales with ACTIVE params: keep those small.
- Distillation, not from-scratch knowledge pretraining. An offline teacher generates, a larger teacher judges (validated in-repo: 0 to 15% HumanEval fast, `successes.md` 2026-07-03). The core learns to converse, reason, and QUERY MEMORY.
- The proven efficiency stack rides underneath: Muon 1.60x, patching 1.82x, hybrid O(1)-state trunk (`successes.md` 2026-07-04).

### THE ARCHITECTURE: small distilled core + recurrent working state + trillion-char memory

A SYSTEM with three memory tiers. Intelligence is small-and-distilled; knowledge is external-and-addressable. Each tier is grounded in an in-repo result.

Tier A. Reasoning core (small, distilled). A ~0.5-1.5B active byte hybrid-trunk model. Job: instruction-following, reasoning, tone, and querying/citing memory. Small because knowledge is externalized. Trained by teacher distillation (the teacher is required: label-free self-improvement below 1B collapses, `failures.md` 2026-07-04).

Tier B. The trillion-character context = the learned hierarchical addressable memory of IDEA 2. Byte-native keys learned with the LM, learned drill-down over an index tree, retrieved bytes folded into the recurrent state. The trillion characters, addressable, on the SSD.

Tier C. Working state (cross-turn) = the hybrid recurrent trunk wired into generation. O(1) state per byte (`successes.md` 2026-07-04). E4b proved the recurrent path extends usable context for free and beat dense on val. Wire `forward_streaming` into `/generate` so a session folds into constant-size state instead of being re-fed as text.

### THE NEW PIECE: retrieval folded INTO recurrent state (RARS)

Stream the retrieved bytes through the hybrid recurrent trunk so they compress into the O(1) state before decoding the answer. The trunk ingests arbitrarily many retrieved bytes at constant memory (that is what E3 bought); retrieval picks WHICH slice of the trillion to fold in. The exact-read context is bounded by state size, not by the 1 KB attention window, so k is large and the answer is conditioned on a compressed read of many retrieved chunks. This reuses two measured in-repo results (O(1) state parity, free context extension) and unifies Tier B and Tier C.

### GATES (the standard, as pass/fail)

- RARS gate: retrieval-into-recurrent-state beats BOTH the bare core and vanilla window RAG on needle QA at equal wall-clock. If not, ship the stronger of the two and record why.
- Core gate: the distilled core clears a chat-quality bar (teacher-judged win-rate plus a grounded-QA accuracy floor) above chat80m, at or under 1.5B active, inside the time budget. Measure it and drive it up.
- Separation gate: grounded-QA and open-domain accuracy are reported SEPARATELY, and open-domain is driven UP by growing the memory and distilling more, never by excusing it.

### 2-WEEK EXECUTION SKELETON (for the execution model)

- Days 1-3: Tier B. Build and index a large on-disk corpus; wire retrieval into `/generate`; show grounded QA beats the bare model. Highest value, lowest risk.
- Days 3-6: Tier A. Distill the ~0.5-1.5B core from the offline teacher (SFT plus logit KD where the teacher exposes logits); before/after chat eval vs chat80m.
- Days 5-8: Tier C. Wire hybrid `forward_streaming` cross-turn state into `/generate`; measure session-length coherence at O(1) cost.
- Days 8-12: RARS. Fold retrieved bytes into recurrent state; run the RARS gate vs bare core and vs window RAG.
- Days 12-14: Integrate and eval. BoN plus a small reranker on the core for reasoning; grounded vs open-domain scored separately; write the verdict into `successes.md` or `failures.md` and cut this entry.

### THE DELIVERABLE

A $0 on-device byte chat system that converses at frontier-grade on grounded tasks, recalls any span of a trillion-character context exactly (IDEA 2), keeps a fixed-size gist of the whole history always on, and holds unbounded sessions at constant per-token cost. The core is small because knowledge lives in memory: that is what makes it trainable in 1-2 weeks.

---

## IDEA 2: the trillion-character context mechanism (hierarchical dual memory)

Status: OPEN. Owner: execution model. The engine behind IDEA 1 Tier B. The standard: at generation time the model recalls and conditions on ANY span of a 1e12-byte history, and grounded-QA accuracy does NOT decay as the history grows from 1e9 to 1e12 chars.

### THE DESIGN CONSTRAINT (what the mechanism must be)

The context mechanism is addressable and sub-quadratic. The model conditions on any span of 1e12 chars by either a fixed-size compressed state (gist) or sparse selection of the relevant subset (exact). Both are built below and the model chooses between them. The trillion bytes live, addressably, on SSD. That is the whole requirement.

### Architecture: dual memory with a learned gate

Two memory systems, one small byte core (IDEA 1 Tier A) reads both:

1. COMPRESSIVE STATE (always-on gist of everything).
   - The hybrid trunk's O(1) recurrent state, streamed over the entire history once (offline for a static corpus, online for a session). Folds 1e12 bytes into a fixed-size state that fits in RAM.
   - Gives gist conditioning on the whole trillion at zero per-token retrieval cost. Its job is coverage; exact recall is the addressable memory's job.

2. ADDRESSABLE HIERARCHICAL MEMORY (exact recall on demand).
   - The 1e12 bytes on SSD as ~500M leaf chunks of 2 KB. Each leaf has a learned byte-native key (d=256).
   - A learned index tree: L0 500M leaves, L1 ~1M summary nodes, L2 ~2K super-nodes. Higher levels carry a learned digest key AND a learned compressed summary value of their subtree.
   - Read: the core emits a query; drill-down attends L2 (flat, cheap), picks top-b, descends to their L1 children, picks top-b, descends to leaves, reads the winners' raw bytes, attends over them. Effective conditioning on 1e12 chars in O(branching times depth) = a few thousand candidate scores per read.
   - Backed by IVF-PQ / HNSW ANN: leaf search is milliseconds at 500M vectors (billion-scale ANN is a solved engineering regime).

3. LEARNED GATE. The core learns WHEN to answer from gist vs pay for an exact read, and how many hops to spend. Cheap turns never touch disk; recall-heavy turns drill down.

### Where the invention is (none of this shipped as a trillion-scale byte-native on-device system)

- Byte-native keys learned JOINTLY with the LM (no tokenizer, no frozen embedder): the model shapes its own address space.
- LEARNED hierarchical drill-down trained on retrieval targets (straight-through / distillation), so the tree is navigated by the model, not cosine similarity alone. This is the core research object.
- DUAL memory with a learned gist-vs-exact gate: novel composition.
- Retrieved bytes folded INTO the recurrent state (RARS): the exact-read context is bounded by state size, not the 1 KB attention window, so k can be large.

### Why it trains inside 1-2 weeks (the scale-invariance trick)

The read op sees a FIXED-SIZE candidate set (say 64) regardless of index size, so it is scale-invariant: train query + drill-down + integrate on short sequences (8-32 KB) with constructed (query, gold-span, hard-negatives, answer) triples and teacher-distilled retrieval targets, then deploy the identical op against a trillion-scale external index. The index grows at inference with no retraining (train-small, deploy-huge). The compressive tier trains with the trainer's existing state-carry path.

### On-device feasibility budget (one M3 Ultra, 512 GB UMA)

- Leaf bytes: 1 TB on SSD (needs ~1 TB free disk; flag if absent).
- Keys: 500M times 256-d. fp16 raw = 256 GB (mmap, hot set in RAM); PQ-compressed to ~64 B = 32 GB, fits in RAM. Use IVF-PQ.
- Index build: one embedding pass over 1 TB (core forward at byte throughput; hours to a couple days, one-time, offline) plus FAISS IVF-PQ train/add.
- Query latency: single-digit-to-tens of ms per read at 500M vectors; a turn does a handful of reads. Interactive.
- UMA is the unlock: weights, index working set, and decode buffers share one 512 GB pool, no host-device copy.

### GATES (the standard, as pass/fail)

- Needle-in-a-trillion: plant facts at random byte offsets; measure exact-recall accuracy as the corpus grows 1e9 to 1e11 to 1e12. Recall must stay flat with size and position.
- Monotonicity (the external-memory signature): grounded-QA accuracy must be NON-DECREASING as memory grows 1e9 to 1e12. If more memory does not help, the model is not using it: fix the query/integration path.
- Latency bound: per-read latency stays bounded (target sub-second) as the index grows to 1e12.
- Drill-down value: learned drill-down beats flat IVF top-k on recall@k at equal latency. If not, ship flat ANN and record it.

### THE LEVER TO PUSH (query quality)

The work is the small core's ability to issue good queries and integrate retrieved evidence: retrieval quality tracks query quality. Train it against the gates above and drive it up. Verbatim recall over a trillion is delivered by the storage-and-address design; making the core use it well is the lever.

### Sub-idea (OPEN, 2026-07-13): background "thinking" pass over context memories

Early thought, unshaped. When the model reviews its context memories (the addressable/gist tiers of IDEA 2), run that review as an explicit "thinking" state: the model works on the retrieved context in the background before committing to an answer, rather than folding everything inline in one forward pass. Open questions: how the thinking state is expressed (a separate decode phase over retrieved bytes, spare tokens, or a state-carry pass), how it composes with the trillion-char read path (RARS folds retrieved bytes into recurrent state — does the thinking pass run over that state?), and what triggers it (the learned gist-vs-exact gate could also gate "spend thinking here"). Value if it works: idle/background compute processes memory ahead of the user turn, so recall-heavy turns pay less latency at answer time. No gate or falsifier yet; park until IDEA 2's read path and RARS land.

### Sub-idea (2026-07-21): concrete P1 prototype + gated-slot RARS revival

Research analysis 2026-07-21 (in temp/agent_research_2026_07_21.md if kept) proposes the smallest possible falsifier for the "learned hierarchical drill-down beats flat IVF-PQ" claim: **P1 = 2-level drill-down at 1e7 chars, no RARS, no gist gate**, reusing chin200m + its trained key head. Build a two-level index over 5K leaves; L1 = ~250 super-nodes as centroids of children. Supervised branch-selection with sibling-InfoNCE, ~2K steps, ~1-2 GPU-hours on the 5070. **Kill line: drill-down recall@1 must be within 3 pts of flat top-1 at ≥5× fewer leaves scored.** If it can't match flat at two levels and 1e7 chars, three levels at 1e12 won't work — fall back to flat IVF-PQ + trained key head + top-1 prefix-injection (which is already validated at recall@1 0.70 per external_memory_retrieval.md).

**RARS status (from failures.md 2026-07-13): originally FAILED — at K=8+ retrieved leaves the gold washes out due to GLA decay + fixed capacity.** The honest recommendation is to ship top-1 prefix-injection as the primary path and treat RARS as research. IF pursued anyway, the mechanically defensible variant is a **learned-gated slot overwrite** (not addition): reserve K=4 outer-product rank-1 "retrieval slots" per head on the GLA state, learned per-slot gate g_i ∈ [0,1] decides overwrite. Slots decay OFF the shared α so they persist. Kill line: K=4 slots must beat top-1 prefix-inject by >5 pts on natural queries at 200M scale; else RARS stays dead for good. Cost: ~8 GPU-hours. Only worth if P1 clears AND top-1 prefix-inject with the 800M student doesn't already clear the grounded-QA gates.

Gist tier reality check: use the hybrid trunk's O(1) GLA state directly (already present), don't build a rolling summarization pass. Producer = one streaming pass over history at trunk throughput. Size ~1.2 MB fp16. Falsifier: turn gist OFF and measure grounded-QA on cheap turns; if gist-off is within noise, gist is not carrying signal and its complexity is unjustified.

### Sub-idea (OPEN, 2026-07-13): candidate re-ranker to convert recall@k into effective top-1

The 2026-07-13 falsification (failures.md) showed multi-leaf (top-k) context injection HURTS a small generator: chat200m grounds worse with 3 or 5 leaves than with 1, because it cannot disambiguate among candidates. But retrieval recall@5 (~0.63) is far above recall@1 (~0.37) — the right leaf is usually IN the top-k, the generator just cannot pick it. A re-ranker that scores the top-k candidates with a richer (query, leaf) interaction than pooled-feature cosine (cross-attention scorer / small cross-encoder over the byte core) and injects only the single best leaf could lift effective top-1 toward recall@5 without the multi-leaf confusion. Ceiling ~0.63 vs current 0.37 = up to ~1.7x grounding. Kill line: re-ranked top-1 must beat raw top-1 recall by >5 pts on held-out natural queries. Retry-gated behind the 800M: a stronger generator may ALSO disambiguate multi-leaf directly (re-test eval_teacher_topk first), making the re-ranker redundant — build the re-ranker only if the 800M still needs single-leaf precision. Code seed: `experiments/v2/memory/eval_teacher_topk.py`.

---

## IDEA 3: throughput is the bottleneck — reduce FLOPs per token (Door 1)

Status: OPEN, **HIGH PRIORITY — run the moment the 800M finishes**. Owner: execution model. Opened 2026-07-14.

The framing, established honestly: training throughput = (peak FLOPs x MFU) / (FLOPs per token). There are exactly three doors. On this box (M3 Ultra, no tensor cores) door 3 (peak FLOPs) is capped by the Apple GPU and door 2 (MFU) is already largely tapped (framework is not the lever — MLX ~= torch-MPS; rule 24e speed levers tested every launch). So **the only door that moves here is door 1: fewer FLOPs per token** — which is also the only door that is an invention problem rather than a purchase, and every win transfers to CUDA silicon too. FLOPs/token ~= 6 x active-params, so the levers attack active params or sequence positions processed. Already won: patching (SpaceByte, 1.82x — cuts global-block positions), Muon (1.60x — fewer tokens to target). Already killed: looped/recursive depth (lost to hybrid). The untested door-1 levers below are the campaign.

### T0 (prerequisite): profile before picking a lever

Do NOT choose a lever before measuring the dominant cost. Two passes: (a) analytical FLOP breakdown from the trunk config at 800M (attention vs GLA-scan vs MLP vs optimizer share) — zero GPU, arithmetic only; (b) empirical per-component step-timing on an isolated 10M (not the live 800M). Specific suspicion to settle: is the GLA recurrence running as an efficient chunked-parallel scan on the target device, or a sequential loop leaving MFU on the floor (the scan was the CPU-side cost in the memory-store build, failures.md 2026-07-11)? Output: a chart that names the dominant cost, which SELECTS which of T1-T3 to fund. Kill line for the whole idea: if the profile shows the model is not FLOP-dominated where we think, the lever list is wrong — re-derive.

### T1: hybrid_moe arm (the biggest untested lever — and it is already built)

MoE is wired: `trunk=hybrid_moe`, `MoEFFN` in `veritate_core/model_patched.py` (`global_ffn=="moe"`, `moe_aux_sum`/`moe_expert_share` load-balancing), selectable through `vanilla_trainer.py`. It decouples capacity from throughput: many small experts, top-k active, so a large-CAPACITY model trains at small-ACTIVE-FLOP throughput. Never scaled or measured. Mechanism: run `trunk=hybrid_moe` matched-active-FLOPs against the winning `trunk=hybrid` at 10M, then 80M. Falsifier: at matched active-FLOPs and equal wall-clock, hybrid_moe must beat hybrid on val by >5 pts (agent_roe seed rule: needs a second seed before any external claim). If it holds, MoE is the path to a much larger effective model at fixed throughput — the direct answer to the throughput bottleneck and to "train the largest model we can."

**2026-07-21 readiness assessment: ~75% ready, but with three real blockers before a 24h bench can even start.**
1. **Trainer/exporter architecture mismatch (HIGHEST blocker, ~1-2 days to fix).** `veritate_core/model_moe.py` implements DeepSeekMoE-style: 1 shared expert (`up`/`down`) + 8 routed experts stored as `self.experts[e]` (each = full `FFN`), gate router, top-2 combine. `veritate_mri/training/export.py:215-411` `write_block_moe` expects the legacy MEGA layout: a `router` matrix + `experts_up.{e}` / `experts_down.{e}` per-expert and top-1 combine. Fetch keys mismatch (`ff.experts.{e}.up.weight` in trainer vs `experts_up.{e}.weight` in exporter). Exporter also drops the shared expert entirely. **The trainer's MoE and the exporter's MoE are two different architectures.** Bench can only run in pure PyTorch until this is fixed.
2. **Engine top-K=2 refused (MEDIUM, ~1 day).** `veritate_engine/v1/src/model.c` v11 loader refuses `router_topk > 1`. Trainer uses top_k=2. Options: run bench in PyTorch only (no engine numbers), or set MOE_TOP_K=1 (loses the "fine-grained" quality claim).
3. **No batched MoE prefill (LOW for val-only bench, HIGH for wren decode ms/byte).** Line 763-772 of model.c falls back to sequential per-token decode when n_experts > 1. Fine for a validation-loss bench; fatal for shipping to Cardinal.

**Ready to bench (val-loss only): 24h plan.** Fork config `hybrid_moe_80m_bench` (~85M active, ~250M total with 8+1 quarter-experts on 12 global blocks). Control arm: matched-shape `trunk=hybrid`. Budget 12h/arm at bs=16, seq=1024, 20k steps (~1.3B tokens). Decision gates: step 4000 dead-expert check (any expert share <0.02 or >0.5 → KILL, retry with MOE_BIAS_GAMMA=3e-3 or top_k=1); step 10000 throughput check (MoE tok/s ≥65% of hybrid); step 20000 val gate (MoE bpb beat hybrid by ≥5 pts at equal wall-clock). Pass = build wren-scale MoE plan (300-500M total, 200M active). Fail = move to T2 (dynamic patching).

Compatibility notes: state cache is shared across experts (GLA state lives on the recurrent global mixer, not inside FFN, so no per-expert state needed). But v13 int8 QAT export is NOT compatible today because of the trainer/exporter key mismatch above — needs the same fix.

### T2: dynamic / adaptive patching (the sharpest byte-native lever)

Byte-level pays the sequence tax hardest (~4-5x more positions than subword for the same text), which is exactly why fixed patching already won 1.82x. Extend it: entropy-based dynamic patching (Byte Latent Transformer style) — bigger patches over predictable spans, small patches only where bytes are surprising — so the expensive global blocks fire on far fewer, better-chosen positions. Mechanism: add an adaptive-patch trunk variant (or a patch-size schedule) vs fixed-patch hybrid. Falsifier: >5% tok/s at equal val, or equal tok/s at better val, on matched 10M runs. Highest-leverage because it hits the cost byte-native amplifies.

**Concrete design (2026-07-16, ready to execute).** The MPS fixed-shape kernel-cache rule (preflight 24c: dynamic shapes = 23x slowdown) forbids variable slot counts per batch, so "dynamic" cannot mean "variable S". Instead: keep `slots = seq // PATCH_STRIDE` FIXED and change the SELECTION rule inside `VeritatePatched._boundary_slots()` (`veritate_core/model_patched.py:164`) from "first S boundaries in order" to "TOP-S boundaries by entropy". Bytes still fold into the last slot whose anchor precedes them (unchanged mask/scatter path). Anchor positions are re-sorted ascending after top-k so patch coverage remains contiguous. Fixed shape preserved end-to-end; only the RANKING of which boundary bytes become slot anchors changes. Concretely three source-of-entropy variants to A/B (each is one boolean flag on the trunk config):
- E-static: parameter-free precomputed bigram surprisal from a small in-repo byte-bigram table (fast, no learned params, arch-agnostic — the low-risk baseline).
- E-probe: 1-layer byte-level probe LM (~50k params) whose loss is added to the trainer, entropy = -log p(next_byte | prev). Full BLT approximation.
- E-schedule: purely position/decay-based schedule (no learned entropy at all) as a null-hypothesis control — proves whether the win is from entropy specifically or just from spacing.
Wall-clock: only the boundary-ranking op changes (a top-k over ~seq/2 candidates), so per-step FLOPs are UNCHANGED at same S. The win must come from val, not throughput; the falsifier's "at equal val" clause is the honest check. Rule 34d test surface: `tests/model/test_patched_dynamic_slots.py` — (a) fixed-shape invariance across all three entropy modes; (b) E-static is deterministic byte-for-byte across arches (macOS arm64 / Linux x86 / Windows x86); (c) fallback to first-S ordering when entropy-mode is off is BYTE-IDENTICAL to current `VeritatePatched`. Kill line: no entropy variant beats fixed-patch by >5 pts val on matched 10M, second seed per agent_roe. Retry condition if killed: try at 80M (small-scale ranking noise may swamp signal; BLT's own gains were at ≥400M).

### T3: mixture-of-depths (skip layers on easy tokens)

Not every byte needs every layer (the space after a word is trivial). Learned per-token layer skipping (MoD / early-exit / CALM family) cuts FLOPs/token on the easy majority. Distinct from the killed looped-depth experiment (that ADDED compute per param; this REMOVES it on easy tokens). Mechanism: per-token learned router that skips global blocks below a confidence/importance threshold. Falsifier: matched val at measurably fewer FLOPs/token (>10%) vs hybrid, no quality tax outside the 5% band. Lower priority than T1/T2 (more build, less certain), fund only if the T0 profile shows depth is where the FLOPs sit.

---

## IDEA 4: corpus style is the capability-per-parameter lever (the mixed_code ablation)

Status: OPEN. Owner: execution model. Opened 2026-07-15. Artifacts already built: `veritate_mri/tools/build_code_corpus.py` and the `mixed_code_*` bins in Mirach-Corpuses.

What is already settled externally (do not re-prove): data quality dominates parameter count at small scale. phi-1 (1.3B, "Textbooks Are All You Need") beat models 10x larger on HumanEval using ~7B tokens of textbook-quality filtered plus synthetic code; FineWeb-Edu showed classifier-filtered educational web text outperforms the raw dump it came from; dedup alone measurably improves LMs (Lee et al.); TinyStories showed clean narrow distributions give coherent tiny models. StarCoder's ablations add a caution: near-dedup always helped, but aggressive popularity filtering (GitHub stars) HURT — filters that cut diversity can subtract capability.

What is NOT settled anywhere public: which corpus STYLE (raw files vs cleaned files vs textbook-tier vs Q&A-interleaved) wins for a BYTE-LEVEL model at 200M, and whether the winner is stable enough to set the mix for the largest runs. Nobody publishes byte-level corpus-style ablations; this is cheap to own.

Mechanism: four bins, identical size and seed, one axis changed each (built, see `developer_documentation/corpus/code_corpus.md`): mixed_raw (control: size caps + exact dedup only), mixed_files (full filters + near-dedup, edu score>=3), mixed_edu (same, score>=4 textbook tier), mixed_qa (filtered code 50% + StackOverflow Q&A 50%). Train the same 200M recipe, same steps, on each (GPU box, dashboard launch, `model_type=code`); rank on shared held-out clean val plus code evals.

Falsifier / decision line: a style must beat mixed_raw by >5% val bpb or a clear code-eval margin, second seed per agent_roe before any claim. If no style separates from the control, corpus style is not a lever at this scale: record it in failures.md and spend on architecture (IDEA 3) instead.

Scale-transfer caveat (pre-registered): a 200M winner picks the mix FAMILY, not the final ratios. Ultra-narrow textbook data can cap larger models (diversity starvation: the StarCoder lesson), so re-validate the winning style at 1-3B before committing a farm-scale run to it.

---

## IDEA 6: chat200m Chinchilla-optimal — the full-length training bet before we commit to 1B

Status: OPEN, TRAINING (launched 2026-07-16, RTX 5070, dashboard). Owner: execution model. The proving-ground for whether prior 200-270M runs were under-trained.

### The premise (from the user, 2026-07-16)

Every prior chat200m-class run was well under Chinchilla-optimal: the Mac's chat200m did 2.0B tokens on 270M params (~7 tokens/param), the chat80idk_80m ship did ~1.1B on 121M (~9 tokens/param). Chinchilla-optimal is ~20 tokens/param. The suspicion: the reason a 270M model has never been *fully fluent* on this box is not the architecture — it is under-training. Before committing a 12+ day 1B run, prove or disprove that hypothesis on a 200M-class run we CAN afford (~40 hours wall-clock).

### The run (RUNNING)

- Model: `veritate_200m` trainer, 270,510,336 params (16L h1024 ffn4096, heads 16, seq 1024)
- Stack: hybrid trunk + Muon + bf16 + WSD sqrt decay + `--use_act_ckpt --use_8bit_adam` (the CUDA speed levers the Mac cannot use)
- Corpus mix (clean-data bet, in-pretrain dosing from day 1): fineweb_edu 45%, chat_500mb 20%, wikitext103 12%, code_qa_100mb 6%, sft_idk 6%, mixed_code_edu_200mb 5%, py_code_100mb 3%, agent_150mb 3%
- Bench-picked config on 5070: batch 24, seq 1024, n_chunks 4, 6.1 GB VRAM (11.94 GB total; comfortable) — 15,435 tok/s in bench, 38k tok/s measured actual
- Token budget: 55,000 steps × (24 × 1024 × 4) = **5.4B tokens** (20 tokens/param, Chinchilla-optimal)
- Wall-clock: **~40 hours at measured throughput** (bench was conservative)
- LR: 3e-4 base with 1000-step warmup, WSD sqrt decay to 3e-5 over the last 15% of steps
- Model dir: `models/chin200m/`, checkpoint every 1500 steps, eval every 500

### GATES (the standard, as pass/fail)

- Under-training gate (the whole point): val loss at step 55,000 must be materially lower than chat200m's 0.812 @step 20,400 (successes.md 2026-07-09) at matched EFFECTIVE dataset. If Chinchilla-optimal doesn't produce a meaningfully lower loss floor than the Mac run's 7-tokens/param cutoff, the "prior runs were under-trained" hypothesis is FALSIFIED and the 1B run must budget architecture time, not just token time.
- Fluency gate (the user's real ask): the post-SFT model must clear all four 2026-07-10 gates that chat200m cleared — needle copy 1.00/0.83, bare identity 3/3, grounded read-off-page 3/3, empathy intact — AND additionally not fail obvious open-ended dialogue in the way chat80idk_80m still occasionally does at 121M.
- Abstention gate (IDEA 5 dose confirmed): 6% sft_idk in the pretrain should preserve the 90/80 abstention precision the 80M ship model hit, without a late-phase SFT that would overwrite copy skill (failures.md 2026-07-06/08 lesson).

### Chat SFT phase (queued, launches at pretrain completion)

Recipe cloned from the 2026-07-10 success (successes.md): resume from final pretrain checkpoint, ~4k-8k SFT steps, mix chat_500mb 55% / chat_50mb 15% / sft_idk 8% / grounded (fineweb + wikitext) 22% at base_lr 2e-5 WSD → 2e-6. HARD gate: needle conversation-copy must NOT erode (the failures.md 2026-07-08 lesson).

### The bet + the retry condition

If IDEA 6 clears its gates: this is the recipe we scale up to 1B on the phi-1-clean-data path (successes.md 2026-07-16 estimate: 4-15 days at 5-8B tokens on clean data, 12-30 days at 16B Chinchilla). If IDEA 6 FAILS (under-training was not the bottleneck), it forces IDEA 3's architecture campaign (throughput / MoE / dynamic patching) before we spend GPU-weeks on 1B.

---

## IDEA 8: chin200m layered capability-SFT campaign (grounded_read / multiturn / instruct / prose) + Cardinal QAT

Status: OPEN, IN FLIGHT. Owner: execution model. Opened 2026-07-19. Runs on top of IDEA 6's chin200m base (step 55000, val 0.776).

### The framing (locked 2026-07-19)

Two identity SFT forks on chin200m (`chin200m_ident` 2/12 probes, `chin200m_ident2` 3/12 but with wrong-context bleed — "Call me Veritate" answering garden-tips, "Carpathian." answering ocean prose) proved for the second time (see `failures.md` 2026-07-08 for the first) that a 500-1000 pair narrow-template SFT at 3e-5 cannot rewrite a 40h pretrain's baked argmax path without eroding pretrain skills. The pretrain baked "no fixed name" in as the top-1 next-token continuation — the SFT nudge did not move the argmax, but it DID shift the words "Veritate" and "Carpathian" into unrelated contexts. Classic register-tuning failure.

**Decision.** Identity, empathy, engaged: NOT candidates for SFT. Solve identity in the serve layer with a system prompt ("You are Veritate, made by Carpathian.") the way July chat200m did. Empathy/engaged register are the same class of problem — the pretrain owns them.

**Remaining campaign = SFTs that teach skills the model LACKS**, which do not fight the pretrain because the pretrain has nothing to say there:

1. `sft_grounded_read_v1` — read facts from an in-context passage (base is word-salad on this).
2. `sft_multiturn_v1` — remember what was said 1-2 turns ago (base fails outright).
3. `sft_instruct_v1` — follow format constraints ("one sentence", "list 3 items").
4. `sft_prose_v1` — write more than 5-word fragments.

Followed by (parallel low-priority track):

- QAT int8 fork of the best combined-SFT checkpoint for Cardinal serving. Does not block the 4 SFTs.

### GATES (the standard, as pass/fail)

For each of the 4 capability SFTs, evaluated on a fresh chin200m fork resumed from step 55000 with 2500-4000 SFT steps at base_lr 2e-5 WSD → 2e-6:

- Capability gate: on ≥6 held-out probes for the target skill (invented entities where relevant), post-SFT accuracy MUST beat base chin200m by ≥40 percentage points. If it doesn't beat by 40pt, the skill is not learning — record and re-dose or re-write the corpus.
- Regression gate (the failures.md 2026-07-06/08 lesson, non-negotiable): needle conversation-copy must NOT erode. Score before/after; if the copy skill drops >5pt, ROLL BACK the fork and reduce the SFT dose.
- Wrong-context bleed gate (the 2026-07-19 lesson): a battery of ORTHOGONAL probes (garden tips, ocean prose, a chess opening) must not surface the SFT vocabulary. If bleed shows up, the dose is too high or the mix lacks pretrain-anchor mass.

For QAT: greedy-parity + bpb-delta gate (rule 24, < 0.005 bpb) vs the fp16 fork, AND measured decode ms/byte on the Cardinal i7-9700T target ≤ 1.3 ms/byte at int8.

### Kill lines

- If any 2 of the 4 capability SFTs fail the capability gate with the standard recipe, the whole "layered SFT" thesis at this base scale is falsified: rethink as in-pretrain dosing (repeat IDEA 6 with the SFT stems folded into the pretrain mix) rather than post-hoc fine-tunes.
- If QAT int8 costs > 0.01 bpb (double the rule-24 tolerance) or fails greedy-parity on 3x64 probes, ship fp16 to Cardinal and record the int8 attempt in `failures.md` with the diff.

### Progress ledger (append here as sessions run)

- 2026-07-19: **all 4 capability corpora BUILT + PACKED**, byte-deterministic seed 20260719, generic `veritate_mri/tools/build_sft_corpus.py` (6/6 tests pass). Bins installed to `trainers/corpus/` and desktop staging dirs:
  - `sft_grounded_read_v1`: 2000 pairs / 4 families (single/two/compound/unstated fact, invented entities). Train sha `5998b72c…`, val sha `a238eb4b…`.
  - `sft_multiturn_v1`:     1500 conversations / 6 callback families (pet/name/preference/plan/work/location). Train sha `dcc34251…`, val sha `6f2d071c…`.
  - `sft_instruct_v1`:      1500 pairs / 4 families (one_sentence/three_items/number_only/yes_no). Train sha `12311ef8…`, val sha `ddc46529…`.
  - `sft_prose_v1`:         1000 pairs / 4 families (two_sentence/paragraph/description/explanation). Train sha `265e4a78…`, val sha `79f5cf41…`.
- 2026-07-19: **grd1 fork PASSED all gates → moved to successes.md.** Trained clean (chin200m@55000 → 58500, 2h46m, exit 0, val 0.776 → 0.778 = +0.002 drift). Probes: 12/12 grounded transfer (base 3/12 false-positives; +75pt lift, crushes ≥40pt gate), 4/4 bleed CLEAN (no wrong-context leak into garden/ocean/chess/cooking), regression sweep 6/12 vs base 3/12 (doubled; needle-copy preserved; one expected minor `engaged` over-abstention from `unstated_fact` refusal family). Full evidence in successes.md 2026-07-19.
- 2026-07-20: **mt1 fork PASSED capability + bleed gates, PARTIAL PASS overall → moved to successes.md with honest boundaries.** Trained clean 2h46m, val 0.776 → 0.777. Capability 11/12 callback (+92pt vs base 0-1/12), bleed 3/4 clean (one callback-shape phrase leak "you said …" into garden context), regression sweep 2/12 vs base 3/12 (one-axis drop within noise; the lost axis symptomatic of the same callback pattern leak). Fork usable; combined-stack tuning may need mt dose 10% instead of 15%.
- 2026-07-20: **inst1 fork PASSED all gates cleanly → moved to successes.md.** Trained clean 2h46m, val 0.776 → 0.778. Capability 8-9/12 instruct vs base 2/12 = +50-58pt, bleed 4/4 clean (no instruct-shape overrun of open-ended prompts), regression sweep 4/12 vs base 3/12 (+1 axis, no erosion). Cleanest capability SFT so far — no bleed unlike mt1.
- 2026-07-20: **prose1 initial run FAILED gate → RETRY prose_v2 PASSED and moved to successes.md.** v1 was 6/12 (+33pt, under 40pt gate) due to narrow templates. v2 with broader templates (18-22 structurally distinct per family, top 5-gram frequency dropped 60/300 → 49/1000) hit **10/12 (+66pt)** and the smoking-gun `instruct_single_sentence` prose-bleed FIXED. Full ledger under successes.md 2026-07-20 prose_v2 entry.
- 2026-07-20: **STACKED fork chin200m_stack1 (grd+mt+inst at 10% each) PASSED as combined-capabilities deployable model.** Grounded 12/12, mt 11/12 (bleed FIXED at 10% dose vs solo 15%), inst 6/12 (mild dose-sensitivity drop), 12/12 bleed clean across all 3 probes. successes.md 2026-07-20.
- 2026-07-20: **int8 QAT of stack1 for Cardinal shipping — 28/36 target passes preserved from 29/36 fp16 (-1 refusal flip), 12/12 bleed clean, 51% smaller (541 MB → 277 MB).** Ship-ready. successes.md 2026-07-20.
- **Layered-SFT campaign verdict, FINAL: 4/4 SFT skills passed + STACKED fork + int8 QAT deployable.** grd1 ✅, mt1 ⚠️→stack1 fixed, inst1 ✅, prose_v2 ✅, stack1 ✅, stack1_int8 ✅.
- 2026-07-20: **int8 QAT gate CLEARED via post-hoc export → moved to successes.md.** chin200m_grd1 re-exported int8: 541 MB → 277 MB (51% smaller), all 12 grounded target-skill replies BYTE-IDENTICAL between fp16 (pytorch) and int8 (C engine), 4/4 bleed clean on int8. Post-hoc quantizer is high-fidelity enough that QAT training is unnecessary for a single-family SFT fork. Cardinal wall-clock unmeasured (physical box), but 51% size drop on a bandwidth-bound decode = projected ~2x speedup, inside the 1.3 ms/byte IDEA 8 target envelope. Ship int8 for Cardinal.
- Remaining open axes: (a) prose_v2 retry with broadened corpus (per failures.md 2026-07-20 retry conditions), (b) optional grd1 + inst1 + mt1 stacked SFT (interleave the 3 SFT stems into one fork mix to get all capabilities in one deployed model), (c) measured Cardinal decode ms/byte when the int8 bin ships to the physical box.

### Sub-idea (OPEN, 2026-07-20): int4 QAT on the final stacked model — the next aggressive edge lever

The int8 post-hoc path shipped clean (1 refusal flip out of 36 target probes, 51% size drop). The natural next lever is **int4**, which halves size again (277 MB → ~135 MB) and — since Cardinal decode is RAM-bandwidth-bound (2026-07-13 evening ledger) — projects to another ~2x decode speedup. The trade is quality: post-hoc int4 usually collapses; the QAT training path (`--qat_enabled --quant_mode int4`) is what makes int4 competitive by teaching the model to keep argmax stable under 4-bit rounding.

Test plan (deferred until after IDEA 8 shipping is stable):
- Fork the winning stacked model (grd+mt+inst+prose_v2 stack at whatever dose recipe wins).
- Run QAT training with `qat_enabled=true, quant_mode="int4"` for ~500-1500 steps at a very low LR (5e-6 or so) to fine-tune the weight distribution for int4 quantization without overwriting the learned capabilities.
- Gate: run the 3-suite target probe (grd + mt + inst + prose_v2) plus bleed. Pass = target regressions bounded to ≤2 pt drop per axis and bleed 12/12 clean. Fail = ship int8.
- Also test: post-hoc int4 (no QAT) as a control, so we can measure how much of the win is from QAT vs the quantizer itself.
- If int4 QAT holds, ternary is the aggressive stretch (each weight ∈ {-1,0,+1}, ~4x smaller than int8). Ternary only ever works with QAT and even then usually costs real quality.

Kill line: if int4 QAT drops target-probe pass count by >4 pt total across the 4 suites, or introduces new bleed, int4 is not worth it — ship int8 and pursue kv-cache compression (IDEA 7 Track B) for further Cardinal speedup instead.

Retry condition: revisit when a stacked model at 500M+ params exists (small models are more brittle under aggressive quantization; a bigger model has more redundancy to absorb the noise).
- Trainer arg gotcha caught + fixed for the next Claude: `--resume` is a STRING (the model dir name), not a boolean; passing `true` fails with exit 2 "expected one argument". Payload shape saved in the resume trail below.

**2026-07-21 research: int4 QAT NO-GO on wren (defer per line 288 stands).** Three independent hard blockers:
1. **No v13-hybrid int4 export path exists.** `veritate_mri/training/export.py:41` `HYBRID_DTYPES = {"fp32", "fp16", "int8"}` — no int4 entry. `veritate_engine/v1/src/hybrid.c:1177` explicitly refuses anything but INT8. The v11 unified format has `VERITATE_QUANT_INT4=1` reserved but `model.c:1896-1903` errors "only INT8 (0) and TERNARY (2) are wired; INT4 via unified format is reserved." Adding this path is ~200-400 LOC across export.py + hybrid.c + a new AVX2 int4 kernel.
2. **No AVX2 int4 kernel exists — Cardinal is AVX2-only.** The existing `veritate_engine/v1/kernels/x86_64/matmul_int4.c` is VNNI (AVX-512). Cardinal's i7-9700T Coffee Lake has no AVX-512. Without an AVX2 int4 kernel, decode falls back to `matmul_int4_scalar_prep` — likely SLOWER than the current int8 AVX2 path despite the 2× smaller footprint. Shipping int4 to Cardinal without first writing the AVX2 kernel makes wren MEASURABLY WORSE, not better.
3. **Kill-line credibly at risk on the 200M/undertrained base.** Int8 already cost 1 refusal→hallucination flip on `grounded_unstated_refusal`. Int4 is ~16× coarser step than int8, and wren is ~1000-4000× under-trained vs SoTA small models per [[project_arch_strategy_2026_07_20]] — the "bigger model absorbs noise" assumption the kill-line relies on doesn't hold. Post-hoc int4 envelope: 5-10 pass drops + new bleeds; QAT int4 envelope: 2-5 drops. Real risk of breaching the >4 kill-line.

Bandwidth math also disappoints: measured int8 wren = 35 ms/byte = 1.5× the 23 ms bandwidth floor (there's an ~11-12 ms/pos SERIAL floor from recurrent stack + rmsnorm). Int4 halves weight traffic (277→135 MB), so bandwidth floor drops to ~11 ms/byte but the serial floor stays. Realistic int4 projection: **~22-25 ms/byte, ~1.4× speedup over int8**, not 2×. And only if the AVX2 kernel gets written; scalar fallback goes the other way.

**Recommendation:** ship int8 stack1 (wren, done), defer int4 as the current retry condition already prescribes. Pre-work if user forces the issue: write AVX2 int4 kernel first, microbench standalone vs int8 AVX2, only then consider export + QAT. If int4 AVX2 doesn't beat int8 AVX2 by ≥1.3× at same tensor shape, int4 is dead-on-arrival regardless of quality.

### Resume trail (for the next Claude if this one dies)

- Base model: `models/chin200m/` (step 55000, val 0.776, hybrid+Muon+bf16, veritate_200m trainer).
- Prior failed forks: `models/chin200m_ident/`, `models/chin200m_ident2/` — do NOT continue-fork from these; always re-fork from `chin200m` step 55000 for a clean baseline.
- Fork endpoint: `POST /models/fork` with `{source: "chin200m", step: 55000, name: "chin200m_grd1"}`.
- Train endpoint: `POST /trainers/run` with `{id: "veritate_200m", args: {name: "<forkname>", resume: "<forkname>", corpus: "...", total_steps: <base_step + sft_steps>, base_lr: 2e-5, warmup_steps: 200, lr_schedule: "wsd", wsd_decay_frac: 0.5, wsd_decay_kind: "sqrt", min_lr: 2e-6, ...}}`. **CRITICAL:** `resume` is a STRING (the model dir name), not a boolean; passing `true` fails with exit 2. Mirror `chin200m_ident2/config.json:training_args` for the trainer-invariant fields, only change name/resume/corpus/base_lr/warmup/total_steps.
- Rules: (24a) all training via dashboard, no CLI trainer launches. `feedback_no_parallel_training`: never fire a second trainer while one is live. `feedback_resource_warning`: warn on RAM/VRAM before any 1B+ run.
- Probes: model after `probe_chin200m_base.json` shape at `temp/probe_chin200m_{grd1,mt1,inst1,prose1}.json`. Include the wrong-context bleed battery (garden, ocean, chess) in every probe.

---

## IDEA 7: unbounded talk length, and reinventing the KV cache for edge decode

Status: OPEN. Owner: execution model. Opened 2026-07-17. Companion to IDEA 2: as the trillion-char context mechanism grows the model's memory, this grows how long it can SPEAK and WORK in one sitting, and how cheaply.

### The premise (from the user, 2026-07-17)

The model must keep talking as long as the thought needs, and keep working as long as the task needs. Today three layers bound a reply, and only one is a knob: (a) the chat route caps a turn at MAX_NEW=256 bytes (raisable to 4096 per request, already implemented); (b) turn-stop markers already end a reply the moment the model closes its turn, but the current models do not reliably emit ChatML `<|im_end|>` until the ChatML SFT redo lands, so the cap bites mid-sentence; (c) the HARD wall: seq=1024. Local-attention KV caches are sized [seq, H] and decode refuses pos >= seq (engine v13 and the PyTorch path alike). Prompt + template + reply share ~1 KB. No knob moves layer (c); it is a training + architecture + engine problem, and it is the whole idea.

The architectural opening: in the hybrid trunk only the 4 local attention blocks (2 enc + 2 dec) carry seq-bound KV. The 12 global GLA blocks are ALREADY O(1)-state and position-unbounded (~28 MB total state at h=768, constant forever). We are 4 blocks away from a model that can decode indefinitely.

### Track A: unbounded talk (the capability)

Mechanism, in order of increasing ambition, each independently testable at 10M scale first (agent_roe seed rule applies):
1. Rolling-window KV for the local blocks: ring-buffer the [seq, H] caches so pos wraps instead of refusing. Local blocks were only ever attending nearby bytes; the question is purely positional encoding. Learned pos_emb breaks on wrap, so test (i) wrap-modulo pos_emb, (ii) attention-sink variant (pin the first k positions, roll the rest, StreamingLLM-style), (iii) retrain-with-relative-pos for the 4 local blocks only.
2. state_carry=chunks (already in the trainer, STATE_CARRY_TRUNKS) trained so GLA state legitimately carries across window boundaries: the model learns that context beyond the window lives in recurrent state + (IDEA 2) external memory, not in KV.
3. Falsifier for A: a rolling-window model must generate 8x seq (8192 bytes) with no val/bpb cliff at the wrap point and no coherence collapse vs the same checkpoint truncated at 1024. If quality cliffs at the boundary regardless of variant, unbounded decode requires retraining at longer seq (record the cost curve), not cache surgery: log it and fall back to seq-growth SFT.

### Track B: reinvent the KV cache for edge (the efficiency)

Cardinal-class reality (measured this repo): i7-9700T decodes at ~12 GB/s RAM bandwidth, int8 kernels already at the bandwidth ceiling, 16 GB RAM ceiling found at 13.9 GB for an 800M trainable. KV is fp32 in v13 today. The research menu, cheapest first, every step gated by the v13 parity discipline (greedy-transcript parity + bpb gate < 0.005, scalar-reference bitwise rule 24):
1. KV dtype: fp16 then int8 per-row KV (halve, then quarter, cache traffic and footprint). Attention is ~2% of decode compute but KV reads are pure bandwidth, and bandwidth IS the edge ceiling.
2. Ring + sink eviction (from Track A) caps KV footprint at a constant regardless of talk length: constant-memory decode on a 16 GB box.
3. KV compression research (low-rank / clustered / shared-head KV, H2O-style heavy-hitter eviction): only if 1-2 leave a measured gap. Measure first (rule 102): profile KV bytes/byte-decoded on cardinal before funding this.
4. Falsifier for B: each step ships only if greedy parity holds on 3 fixed prompts x 64 bytes AND p50 ms/byte on cardinal improves or footprint drops >= 2x with bpb delta < 0.005. A lever that saves memory but breaks the parity gate is dead.

### The tie (why this is one idea)

IDEA 2 gives the model a place to REMEMBER beyond the window; IDEA 7 gives it a mouth that never has to stop mid-thought and a cache that fits the $300 box. Ship order: ChatML SFT first (stop markers make the cap a safety net, already planned), then Track B step 1 (pure engine win, no retraining), then Track A variant (ii), then the rest as measurements dictate.
