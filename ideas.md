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

### Sub-idea (OPEN, 2026-07-13): candidate re-ranker to convert recall@k into effective top-1

The 2026-07-13 falsification (failures.md) showed multi-leaf (top-k) context injection HURTS a small generator: chat200m grounds worse with 3 or 5 leaves than with 1, because it cannot disambiguate among candidates. But retrieval recall@5 (~0.63) is far above recall@1 (~0.37) — the right leaf is usually IN the top-k, the generator just cannot pick it. A re-ranker that scores the top-k candidates with a richer (query, leaf) interaction than pooled-feature cosine (cross-attention scorer / small cross-encoder over the byte core) and injects only the single best leaf could lift effective top-1 toward recall@5 without the multi-leaf confusion. Ceiling ~0.63 vs current 0.37 = up to ~1.7x grounding. Kill line: re-ranked top-1 must beat raw top-1 recall by >5 pts on held-out natural queries. Retry-gated behind the 800M: a stronger generator may ALSO disambiguate multi-leaf directly (re-test eval_teacher_topk first), making the re-ranker redundant — build the re-ranker only if the 800M still needs single-leaf precision. Code seed: `experiments/v2/memory/eval_teacher_topk.py`.
