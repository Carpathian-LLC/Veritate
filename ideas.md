# ideas

Open ideas and active campaigns. Mechanism + falsifier only. Ideas graduate to `successes.md` or `failures.md` when their falsifier resolves.

## frontier chat system

**IDEA 1 — small distilled core + trillion-char external memory + O(1) recurrent working state**
Core ≤1.5B active distilled from a frozen teacher; knowledge externalized into IDEA 2's memory; the hybrid trunk's recurrent state as cross-turn working memory. Falsifiers: beats bare core and window-RAG on needle QA at equal wall-clock; distilled core clears a chat bar over chat80m; grounded vs open-domain accuracy reported separately.

**IDEA 2 — hierarchical dual memory (compressive gist + addressable exact recall) at trillion-char scale**
Byte-native keys learned with the LM, hierarchical drill-down index for exact recall, recurrent state as always-on gist. Falsifiers: recall flat as the corpus grows 1e9→1e12; grounded-QA non-decreasing with memory growth; learned drill-down beats flat IVF at equal latency, else ship flat ANN.
- P1 prototype: 2-level drill-down at 1e7 chars. Kill line: recall@1 within 3pt of flat top-1 at ≥5x fewer leaves scored.
- Gated-slot RARS revival: K=4 learned-gated overwrite slots. Kill line: beat top-1 prefix-inject by >5pt on natural queries at 200M, else RARS stays dead.
- Re-ranker: rescue the recall@5 0.63 / recall@1 0.37 gap. Kill line: re-ranked top-1 beats raw top-1 by >5pt on held-out natural queries.
- Background "thinking" pass over retrieved context before answering — unshaped, parked.

## throughput and flops

**IDEA 3 — cut FLOPs/token; the only door open on this box (no tensor cores)**
T0: profile the dominant cost first. T1 (MoE, `trunk=hybrid_moe`, wired): matched-active-FLOPs must beat hybrid by >5pt val at equal wall-clock; blockers: trainer/exporter mismatch, engine refuses top-k=2, no batched MoE prefill. T2 (entropy-based dynamic patching): >5% tok/s at equal val on matched 10M runs. T3 (mixture-of-depths): matched val at >10% fewer FLOPs/token — lowest priority. T4 (patching scale curve at 121.8M, run before T2). Decision: ≥1.82x holds → fund T2; 1.2-1.82x → restate headline; <1.2x → correct the composed-stack claim.

## corpus style

**IDEA 4 — which byte corpus style wins: raw / filtered / textbook / Q&A-interleaved**
Four matched bins, same 200M recipe, rank on held-out val + code evals. Falsifier: a style must beat mixed_raw by >5% val bpb or a clear code-eval margin, second seed required. A 200M winner picks the family, not final ratios — revalidate at 1-3B before farm scale.

## chinchilla-scale bet

**IDEA 6 — is under-training, not architecture, why no 200-270M run has been fully fluent?**
chin200m (270.5M) to 20 tok/param (5.4B tokens) vs prior 7-9. Falsifier: val at step 55,000 materially below chat200m's 0.812 at matched effective data. chin200m@55000 is already the SFT-campaign base — outcome may just need formal closeout.

## unbounded context

**IDEA 7 — unbounded talk length + a KV cache that fits a $300 edge box**
Track A: rolling-window KV for the 4 local-attention blocks + `state_carry=chunks`. Falsifier: generate 8x seq with no bpb cliff at the wrap point. Track B: KV fp16→int8, ring+sink eviction. Falsifier: greedy parity holds AND p50 ms/byte improves or footprint drops ≥2x with bpb delta <0.005.

## green-ai product

**IDEA 8 — smallest strong conversationalist, all facts external; attack the role-binding wall directly**
Bind-then-read: a discrete slot/register layer between retrieval and the LM body, supervised on grounded_v3. Kill line: lifts OBJECT-role accuracy off the ~0% floor without breaking slot copy.
Product P1 (committed): on-device private-document assistant. Falsifier: retrieval gates re-run with paraphrased queries sharing no token with the gold chunk.
Product P2: local-first with cloud fallback, router gated on retrieval score, never LM self-judgment. Falsifier: escalation precision/recall on a mixed set.
Below 122M is untested — minimum viable reader size unknown.

## developmental training

**IDEA 9 remnant — value/aversion module**
A small separate network learns "avoid this output" and modulates decoding. No falsifier defined. Parked.

## speculative decode

**IDEA 10 — byte n-gram speculative decode: the only lossless lever left on bandwidth-bound hardware**
Draft K bytes by longest-suffix match against prompt+generated text, verify in one batched weight stream, accept the longest matching prefix (greedy output byte-identical). Non-boundary decode already at 83% of cardinal's bandwidth ceiling. Gates: mean accepted length ≥2.5 → ship; 1.5-2.5 → ship only if ms/byte improves; <1.5 → failures.md, retry once an MTP-head draft source exists. Report accept length by traffic type, not just the mean.
