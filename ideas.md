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
