# Ideas

Strategy and vision doc. Where the project is heading and why. Distinct from:
- `RESEARCH.md` — cited literature.
- `WORKBOOK.md` — chronological build log with numbers.
- `LEARNINGS.md` — cross-platform notes.
- `PLAN.md` — single-sprint plan with concrete decode-budget projections.

**Note (2026-04-28): all sections below are active scope.** Version markers
(v3.5, v4, v5, v6.1, etc.) and Tier numbering used throughout this document
are historical narrative, not a deferral schedule. Every idea here is either
shipped (see PLAN.md graduation list), in flight (see PLAN.md in-flight list),
or queued for the current sprint. Nothing is on a "future version." If an
idea proves wrong it gets a LOST entry in EXPERIMENTS_TRACKER.md; if it
proves right it ships.

This file remains the design-space record. New ideas land here first. PLAN.md
tracks what is currently in motion. EXPERIMENTS_TRACKER.md is the hypothesis
log with status.

# ------------------------------------------------------------------------------------
# The vision (long-term goal)
# ------------------------------------------------------------------------------------

Build an LLM that *feels* sub-millisecond — not one that literally runs full inference in <1 ms.

A real LLM forward pass cannot be made sub-ms by physics. But a human eye sees nothing under ~150 ms, and streaming hides everything after the first token. So the real bar is:

**TTFT (time to first token) under ~100 ms after the user commits the prompt.**

And "commit" can happen *before* the period — at the moment the sentence is 80% likely complete.

Concrete vision pieces:
1. **Speculative compute during typing.** Forward / prefill begins as soon as the user starts typing. Every keystroke extends in-flight work. Backspace cancels.
2. **Period as commit signal.** At sentence end, fan out into sub-agents that work in parallel behind the UI while the first response streams.
3. **Heterogeneous, unconventional hardware.** Use every available silicon — CPU SIMD, NPU, iGPU, DMA engines, eventually PIM-style "compute where data lives" (Mythic-style flash-cell analog, HBM-PIM). Long-game floated: model in the SSD, using analog gates for math.
4. **Constraints are the design tool.** Insane goals kept rooted in reality push the innovation. No softening.

# ------------------------------------------------------------------------------------
# The pivot point — Strategy A vs Strategy B
# ------------------------------------------------------------------------------------

Two fundamentally different ways to make the response feel sub-millisecond:

### Strategy A — make the forward faster
Physics has the steering wheel. v1–v2 took the matmul from 746 ms to 0.358 ms (1024³ INT8). Further wins exist but the slope is flattening.

### Strategy B — run the forward when the user isn't waiting
The user types for 1–3 seconds before committing. That's 1000–3000 ms of free compute on the table.

### The math that makes the pivot obvious

> At 300 CPM (medium typing speed), there are **200 ms between keystrokes**.
> A full v3.1 forward is **7.57 ms**.
> So between any two keystrokes, you can run **~26 full forwards**.

That number is the unlock. The forward does not need to be faster — we need to credibly use those 26 forwards per keystroke.

What 26 forwards per keystroke could do:
- Forward on the literal text-so-far.
- Forward on top-K sentence completions ranked by a small prior.
- Forward on the most likely auto-corrected variants.
- Forward on different intent classifications (question / command / continuation).
- Maintain N speculative branches; prune the ones disagreeing with each new keystroke.

By the time the user hits Enter, the answer for whichever branch matches reality is already computed. TTFT becomes "look up the matching forward" — basically zero.

### What this reframes

Under Strategy A: 7.57 ms is a number to drive down with kernel work.
Under Strategy B: 7.57 ms is a *budget*. As long as the forward fits comfortably inside the inter-keystroke window with headroom for branching, every further millisecond saved buys *more speculative branches*, not faster response.

Different optimization function.

Under A you fight for SIMD on softmax. Under B you spend that engineering on cancellation, branch scheduling, and KV cache management — those determine how many useful forwards fan out per typing window.

The project has been Strategy A through v1–v3. The vision is Strategy B. We're not abandoning A — every ms saved compounds — but B is the bigger lever, and **it changes which problems are worth solving next**.

# ------------------------------------------------------------------------------------
# Where the 7.57 ms goes (working theory, not yet measured)
# ------------------------------------------------------------------------------------

Per layer (×4 = whole forward):

| Stage | Likely cost | Note |
|---|---|---|
| 4 matmuls | ~0.8–1.5 ms | Mostly thread-pool sync overhead |
| 2 LayerNorms | ~0.1 ms | scalar fp32 stats, sqrtf per row |
| Attention scalar loops (Q·K^T, score·V) | ~0.4 ms | pure scalar, ~32K dot products per layer |
| Softmax | ~0.2 ms | expf per element |
| GELU | ~0.4 ms | tanhf per element, 64×1024 |
| Requant + residual | ~0.1 ms | scalar int32→int8 |

**The matmul number is the surprising one.** A 1024³ matmul takes 0.358 ms. The matmuls inside the forward are far smaller (qkv = 64×256×768 ≈ 12M ops vs 2.1B). Pure compute should make these microseconds, not hundreds of microseconds. Suspect: thread pool spreads work across 16 logical cores, each thread gets ~4 rows, finishes in nanoseconds, sits waiting. Plus per-call SetEvent + WaitForMultipleObjects overhead × 16 calls per forward.

**Hot path for tiny matmuls is thread sync, not VNNI.** Likely fix: single-thread for tiny matmuls, multi-thread for big ones, threshold tuned.

Validate before believing this. That's tier-1.

# ------------------------------------------------------------------------------------
# Experiment roadmap — tiered by what teaches us most per unit time
# ------------------------------------------------------------------------------------

### Tier 1 — establish the metrics

1. **Latency-while-typing harness.** Simulate keystrokes at 300–600 CPM. After each keystroke, fire a partial forward. Measure: how much prefill finishes before the period? Plot TTFT vs typing speed. *Defines the entire vision's success criterion.*
2. **Cancellation cost.** How long to abort an in-flight forward and restart. If <10 ms, speculate aggressively. If high, that becomes the next optimization target.
3. **L3 residency profiler.** AMD uProf or similar. Profile L1/L2/L3 miss rates. The 6 MB model should live in L3 permanently. If activation buffers are evicting weight cache lines, free latency to recover.
4. **Per-stage forward profiler.** Confirm the table above with real numbers.

### Tier 2 — first real efficiency wins

5. **Speculative decoding with INT4 draft model.** A 1M-param INT4 draft proposes 4–8 tokens; INT8 model verifies in one batched forward. Realistic 2–5× decode speedup, exact same outputs. Makes INT4 a *latency primitive*, not just a v4 quantization step.
6. **Streaming prefill (v3.X).** Forward runs as user types, KV cache extends per-keystroke, backspace truncates. Prefill complete when user hits period. TTFT = decode-first-token only.
7. **Scalar attention → VNNI.** The Q·K^T and score·V loops are scalar today. Probably 0.5–1 ms saved.
8. **Thread-pool threshold tuning.** Single-thread small matmuls, multi-thread big ones. Likely the largest forward-time win still on the table from Strategy A.

### Tier 3 — heterogeneous hardware

9. **Huge-page weight allocation.** VirtualAlloc + MEM_LARGE_PAGES → 2 MB pages. 6 MB model = 3 TLB entries instead of 1500.
10. **NPU port of one matmul.** AMD XDNA SDK. Run NPU and CPU concurrently — block N on NPU while block N+1's prefill prep runs on CPU. First "use all the silicon" demo.
11. **DirectStorage weight streaming.** Stream prepped_b_t from NVMe through DMA, no user-space copy. Validates "model in SSD" as minimum-data-motion. The achievable approximation of the analog-SSD dream.

### Tier 4 — the crazy stuff

12. **Speculative tree, pruned by keystrokes.** Tree of N likely user intents. Each keystroke prunes branches that disagree. By the period, all surviving branches pre-computed. Wasted compute is fine — trade joules for perception.
13. **Per-keystroke compute exploitation.** At 600 CPM, 100 ms between keystrokes. Forward = 7.57 ms → ~13 free forwards per keystroke. Most useful use? Probably top-K likely auto-completions scored against the actual next char when it arrives.
14. **Analog backend stub.** A `matmul_int8` function pointer that injects deterministic noise modeling Mythic-style flash-cell analog matmul. Forward runs end-to-end. Studies model resilience to analog noise *before* analog hardware exists.

### Order to run them

Tier 1 first, in full. Tier 2 then 3 mostly in parallel. Tier 4 once Tier 1 has shown the typing-window math is real.

Each experiment lands in `WORKBOOK.md` with numbers.

# ------------------------------------------------------------------------------------
# Commitment-level reasoning (streaming input)
# ------------------------------------------------------------------------------------

The model does different work at different user-commitment levels. Most "wasted" speculative compute isn't waste — KV cache is incremental, only backspaced suffixes are lost.

- **Mid-fragment** — prefill KV cache, classify intent (question / command / code).
- **Comma / pause** — pre-warm relevant context (RAG lookup, tool prep).
- **Period** — commit, start drafting.
- **Enter** — stream first token.

Semantic fragments are the harder case ("the auth thing... actually no... what was that bug"). The model prefills all three, but needs the period to disambiguate which fragment is the real intent. Pauses and punctuation are the user's commitment signal.

# ------------------------------------------------------------------------------------
# Brevity as a latency lever
# ------------------------------------------------------------------------------------

Default to short responses; long-form is opt-in. Most AI overexplains. Fewer tokens out = less compute = lower TTFT and lower total latency. Direct multiplier on every other optimization.

- Default mode: terse, conversational.
- Long mode: triggered by "explain in detail," "go deep," "show me the full reasoning."
- Bonus: matches how humans actually want to interact most of the time.

# ------------------------------------------------------------------------------------
# Idle pre-thinking
# ------------------------------------------------------------------------------------

Between turns, the model keeps reasoning on the conversation. When the user types again, it's already ahead.

What it does in idle:
- Keeps KV cache warm and resident in L3.
- Runs speculative branches: "what's the likely next question?"
- Pre-generates candidate follow-ups, summaries, "did you mean" options.
- Pre-warms tool/RAG context for likely follow-ups.

Workstation idle compute is effectively free. Battery devices need a budget. The principle: the model is never *not* thinking — it's always slightly ahead of the user.

# ------------------------------------------------------------------------------------
# Brain-style architecture — specialists, router, memory hierarchy
# ------------------------------------------------------------------------------------

The brain is many narrow specialists running concurrently, not one big model. Mapped to Veritate:

**Split by cognitive function, not by user task.** Wrong: "code model + chat model + tool model" — that splits by app, brittle. Right: language / reasoning / memory / output / metacognition — like the brain. *Specialty* (coder vs novelist) is the trained content riding on shared substrate. Same wiring, different knowledge. That's also why polymathy is possible.

- **Functional layers** — language, reasoning, memory, output. Shared substrate.
- **Specialty** — what content the layers were trained on. Coder vs novelist vs analyst.
- **Router** picks which specialist activates per input. Beyond MoE — MoE picks experts but runs them in one synchronous forward. Goal here: experts run on different cores/silicon *concurrently*, results merged.
- **Working memory** = hot KV cache in L3.
- **Long-term memory** = compressed embeddings on disk, pulled when relevant.

### Metacognition — the missing primitive

Self-awareness here doesn't mean "the model has feelings about itself." It's a **meta-monitor**: a small fast head that watches the main model's outputs and scores them for confidence, contradiction, out-of-distribution input. Brain analog: prefrontal cortex regulating the rest.

Practical use: when the meta-monitor flags low confidence, defer — to a tool, to search, to a human. Most current LLMs don't know when they're guessing. That's the gap.

How to build: open question. Worth thinking about.

# ------------------------------------------------------------------------------------
# Sleep cycles and memory consolidation
# ------------------------------------------------------------------------------------

Brain analogy: hippocampus holds today verbatim. Sleep replays it to the neocortex, which stores a compressed gist. Verbatim trace is dropped. Yesterday is sharper than last Tuesday because last Tuesday has been consolidated more times.

Veritate cycle:

- **Active session** — full KV cache, verbatim.
- **Idle** — light consolidation, rolling summary.
- **Sleep cycle** (≤8h max, sooner if idle long enough) — full pass: compress the day into dense embeddings, update long-term store, drop verbatim beyond a window.
- **Next morning** — working memory empty, long-term enriched.

### Valence as the salience signal

What gets consolidated harder is what carried emotional weight — repeated topics, user reactions, explicit marks. Without valence the model compresses uniformly and loses what mattered. Valence drives the priority queue of consolidation.

### "Better through usage" — two flavors

1. **Memory growth only** — long-term store gets richer over time; weights never change. Safe, fast, no drift. Start here.
2. **Weight updates** — nightly LoRA adapters trained on the day's data. Real learning, but risks catastrophic forgetting. Add only after #1 is stable.

# ------------------------------------------------------------------------------------
# Hardware reality column — separating dream from achievable
# ------------------------------------------------------------------------------------

**Dream:** SSD analog gates for math. Model lives in the storage substrate; reads are computations.

**Reality:** Consumer SSD controllers don't expose analog cells. The closest real-world analog is Mythic AI — custom ASIC, not a repurposed consumer SSD. PIM (HBM-PIM, UPMEM, IBM PCM) is research/lab.

**Achievable approximation on commodity hardware:**
- DirectStorage + huge-page mmap + DMA — model lives on NVMe, streams to L3, never round-trips DRAM. Same principle (minimize data motion), commodity substrate.
- 96 MB L3 on the Ryzen 9800X3D dev box is the unfair advantage right now. Whole 4-layer model is ~6 MB. Fits 16× over. Lean into this.
- Modern Ryzen has XDNA NPU (7040+ series; 9800X3D does not). Intel chips have NPU 4.0. iGPU on most chips. Heterogeneous schedule: embed on NPU, attention on AVX-512, FFN on iGPU. All running simultaneously.
- DMA engines move memory without using CPU cores.

The principle is **compute lives where data lives**. Substrate is TBD; the architectural commitment is what carries forward.

# ------------------------------------------------------------------------------------
# v3.5+ implementation sketches
# ------------------------------------------------------------------------------------

These translate the strategy above into concrete sub-versions that can be
specified, built, and tested.

## v3.5 — streaming prefill (per-keystroke KV extension)

The C engine already has the primitive. `forward_decode` extends the KV
cache by exactly one token at position `cache->len`. Streaming prefill is
just calling `forward_decode` once per keystroke instead of inside a
generation loop.

Cost per keystroke at stretch shapes (V_SEQ=256, 80M model): ~2 ms decode.
Human typing cadence at 300 cpm = 200 ms between strokes. We're 100× under
budget. Speculative-extending past the cursor (predict next char, decode,
discard if wrong) is feasible without straining the budget.

Sub-versioning:
- v3.5.0 — append-only streaming + backspace rewind (`cache->len -= 1`).
- v3.5.1 — branch-on-edit (cache as a tree; mid-line edits spawn child
  branches matching the new text).
- v3.5.2 — speculative extension (predict next char, prefill, discard on
  miss). Requires a trained model with non-trivial predictive power.

## v3.6 — sentence-boundary parallel decoders

At a period or commit signal, snapshot the cache (`memcpy(cache_b, cache_a,
sizeof(kv_cache_t))`, ~5 µs) and spawn N decode workers, each running a
different (seed, temperature) tuple. Stream the first one to the user; rank
the rest by a fast scorer (next-token probability sum or a small judge),
and either present alternatives or kill them.

This is the "fan out at the period" piece of the long-term vision.

## v4 — INT4 weight quantization

Q4_K_M format from llama.cpp: 32 INT4 values per block, one fp16 scale per
block. Halves weights on disk (85 MB → ~42 MB) without materially harming
quality given a calibrated PTQ recipe.

Kernel options on x86 (no native INT4 dot product):
- **Unpack + VPDPBUSD.** Per inner-block load 32 INT4 nibbles, expand to
  INT8 with 2× shifts + mask, then run our 4×4 register tile. ~50 % of
  INT8 throughput.
- **VPSHUFB lookup table.** Pack pairs of INT4 nibbles into a 16-byte LUT
  addressable by the unpacked nibble. Higher peak; more memory traffic.

Path 1 first. Path 2 only if the gap to peak is meaningful.

## v5 — moonshots

**Mamba/SSM backend.** Replace attention with a selective state-space
recurrence. Per-token cost O(1) in context length. Hand-fused INT8 SSM
kernel for AVX-512 + NEON would be genuinely undone work — almost no
project has done a fully hand-rolled INT8 SSM inference engine.

**Mamba + valence channel.** Per the user's framing: structure the SSM
state into a fast-decay factual subspace and a slow-decay valence subspace.
Auxiliary loss from a sentiment classifier localizes affective signal to
the valence partition. At inference, that channel is readable and steerable.
Research bet, not engineering. Adopt only if vanilla Mamba lands.

**JIT runtime kernel emission.** Today the matmul shapes are compile-time
`#define`s. Next step is JIT-emitting kernels at startup once shapes are
known from the loaded model, baking into a writable-exec page. AsmJit-style.
~1 ms compile cost, recovered on every subsequent inference. Estimated
1.5–3× over generic templated kernels on hot paths because pre-compiled
kernels are generic; JIT kernels are shape-specialized.

**Analog backend.** Mythic AI flash-cell array, Lightmatter photonics,
EnCharge switched-capacitor. Veritate's `matmul_int8` function pointer is
already shaped to swap backends — `matmul_int8 = matmul_mythic` is one line.
The driver work depends on hardware availability; deferred until dev-kit
access.

# ------------------------------------------------------------------------------------
# v6 — research moonshots (post-v5)
# ------------------------------------------------------------------------------------

Three pieces of the user's vision that go beyond standard inference-engine
work. Each is real research, ranging from "a couple weeks of focused work"
to "multi-year program."

## v6.1 — Synthetic data via teacher distillation

**What it is in plain language:** instead of training Veritate on text we
scraped from somewhere, we have a *bigger smarter model* (the "teacher")
generate millions of training examples for us, then train Veritate (the
"student") on those. The teacher could be a local Ollama model, or a
cloud LM via API. This is called *knowledge distillation* in the
literature — Stanford Alpaca, OpenAssistant, Microsoft Phi, and DeepSeek's
recent reasoning models all use variants of this technique.

**Why it's interesting for Veritate:**
- We control the training corpus 100%. Vocabulary, style, quality, content
  domain — all chosen, not scraped.
- We can target specific capabilities (story-writing, dialogue, code,
  reasoning) by changing what we ask the teacher to produce.
- Bypasses the "we don't have the GPU compute to train on a trillion
  tokens" constraint by training on a smaller curated synthetic set
  rather than the entire internet.
- The synthetic-data pipeline is custom code — no PyTorch dataset loaders,
  no HuggingFace dependencies. Pure HTTP calls + JSON.

**Tractability today:** ~1-2 days for a basic pipeline. Generate
~100k stories from Ollama qwen3-coder:30b on the local box (free,
already running). Train Veritate on the result. Compare to TinyStories-
trained baseline.

## v6.2 — Project MRI — the glass model

> *"Model the human brain and the different regions, but if there's a way for
> us to figure this out, it's our moonshot."* — user, 2026-04-27

### What "MRI" means here

In medicine, an MRI lets a doctor see *inside a living brain* without cutting
it open — every region of the brain shows up as colored intensities, and you
can watch them change as the patient thinks about different things. **Project
MRI is the same thing for Veritate.** Every internal computation, every
activation, every neuron value sits in plain memory in our hand-coded C
runtime. Capture them while a forward pass runs. Render them in a browser.
Click on a token in the output and see exactly which neurons fired in which
order to produce it.

### Term-defines for this section

- **Activation:** the numerical output of one layer's computation, fed into
  the next layer. The "values flowing through" the network.
- **Neuron:** for transformers, usually means *one dimension of the FFN's
  hidden activation*. So a 3072-wide FFN block has 3072 neurons. A neuron
  "fires" when its value is large.
- **Attention head:** a sub-circuit inside attention. We have 12 layers ×
  12 heads = 144 distinct heads. Each one has learned to do something
  specific (track one type of pattern across the input).
- **Mechanistic interpretability:** research field that reverse-engineers
  trained neural networks at the level of individual neurons and circuits.
  Anthropic's interpretability team is the leading group; they've named
  hundreds of specific circuits in real LLMs.



**What it is in plain language (carried over from earlier sketch):** look
inside a neural network and see *why* it predicted what it did, at the level
of individual neurons firing and information flowing between layers.
Reverse-engineering of trained weights.

The Veritate inference engine has a unique property nobody else has:
*every single matmul, every activation, every attention score is a small
INT-typed number sitting in a deterministic memory layout in our hand-coded
runtime.* We can tap any of them at any time without performance impact
because we wrote every line.

### Why Veritate is uniquely positioned

Every other interpretability tool today (TransformerLens, Anthropic's tools,
Neuronpedia) wraps PyTorch and pays a heavy runtime cost to capture
activations — often 10-100× slowdown. Our C engine is the opposite:

- Every activation, every score is plain memory in deterministic layout.
- Capturing them is `memcpy`.
- Cost when off: literally zero (the trace pointer is null, no branches).
- Cost when on: one extra write per layer, ~50 µs total per forward pass.
- The format is ours; we don't fight someone else's data structures.

### Brain-region analogy (real, not metaphor)

Transformer blocks organize into rough functional regions, the same way
brain lobes do. For our 12-layer model:

| layers | analogous brain region | what they tend to learn |
| --- | --- | --- |
| 0-3   | sensory cortex      | bytes, n-grams, basic patterns          |
| 4-8   | association cortex  | combine patterns into concepts (words, syntax, semantics) |
| 9-11  | prefrontal cortex   | task-specific output (next-token in *this* context) |

Each of the 12 × 12 = 144 attention heads is a specialized circuit.
Anthropic's published interpretability papers have *named* hundreds of
these in real LLMs ("induction head," "name-tracker," "previous-token
copy," etc.). For our model that vocabulary doesn't exist yet — we'd be
writing it.

### Sub-versioning — Project MRI as a roadmap

**v6.2.0 — C-side trace capture (foundation).** Add `forward_with_trace()`
next to existing `forward()`. Same math, dumps every activation to a flat
binary trace file. Captures: per-layer pre-residual + post-residual
INT16 stream, per-layer FFN neuron firings (the 3072-dim "neurons"),
per-head attention scores. ~80 lines of C plus a `trace_t` struct.

**v6.2.1 — binary trace format spec.** "VRMR" magic + version + per-tensor
metadata (name, shape, dtype, byte offset/length) + raw bytes. Designed
for streaming — viewer can mmap and seek.

**v6.2.2 — HTML/JS viewer (no dependencies).** Plain HTML + Canvas + a
single JS file. Loads a trace file via fetch. Renders:
- Heatmap of activation magnitude per (layer, position)
- Attention pattern matrix per head — which output positions attended to
  which input positions
- Top-K firing FFN neurons per output token
- Click any output token → trace cascade highlights every activation that
  contributed (gradient-attribution style)

**v6.2.3 — Region labels.** Color each layer with a label
(sensory/association/prefrontal). Static per-model for now; learnable
clustering later.

**v6.2.4 — Probing.** Curated test inputs across categories ("animals,"
"colors," "questions"). For each: capture trace, find neurons that fire
ONLY for that category. Those are our project's first *named* neurons.
Publishable artifact.

### Bonus use case discovered during v3.4.5

The MRI viewer is *also* the right debugging tool for the C-vs-PyTorch
divergence we hit in v3.4.5. Run the same prompt through PyTorch and through
the C engine, capture both traces, render them side-by-side in the viewer.
The first layer where the heatmaps diverge is where the quantization error
compounds. **Same tool, two purposes** — interpretability for users, debugger
for us. Worth doing.

### Sizing for v6.2.0

For our V_SEQ=256 / V_HIDDEN=768 / V_FFN=3072 / V_LAYERS=12 / V_HEADS=12
shapes, one full trace is roughly:
- Residual stream (pre + post per layer): ~9 MB INT16
- FFN neurons per layer: ~9 MB INT8
- Attention scores per head: ~38 MB FP32 (the chonky part)
- **Total full trace: ~56 MB** per forward pass

For the v6.2.0 minimal version skip attention scores → ~18 MB per trace.
Manageable on disk and in browser memory.

### Time estimate

- v6.2.0 (C trace capture): **half a day**
- v6.2.1 (format spec): trivial
- v6.2.2 (viewer): **2-3 days** of HTML/JS
- v6.2.3 (region labels): trivial
- v6.2.4 (probing artifact): **1-2 days**
- **Total v1: ~1 week of focused work**

Genuinely no one has done this for a hand-coded INT8 inference engine.
Publishable. Could be Veritate's defining feature.

**What "glass model" looks like concretely:**
- Add a `forward_with_trace(model, cache, tokens, trace)` function to
  the C engine. Same forward pass, but writes activation snapshots at
  every layer to a `trace` buffer. Off by default, no perf hit.
- Render the trace in a small JS+HTML viewer: per-layer activation
  heatmaps, attention head matrices for each input position, residual-
  stream magnitude over depth, top contributing neurons per output token.
- Interactive: click on an output token, see which input tokens it
  attended to most heavily, which layers fired most, which neurons
  contributed most.

This is *mechanistic interpretability tooling for a hand-coded INT8
inference engine.* Anthropic's interpretability team does this for big
LMs in PyTorch with research-grade tooling. Nobody has done it for an
ASIC-style INT8 engine. Genuinely novel territory.

**Tractability today:** ~3-5 days for a basic v1. The C-side tap is
straightforward. The JS viewer is the bulk of the work. v2 with circuit
analysis (which combinations of attention heads encode which concepts)
is a longer-running research thread.

## v6.2.5 — Brain formation timeline ("the film of learning")

**What it is in plain language:** during a training run we save checkpoints
every ~10K steps. Each one is a snapshot of the model's brain at that moment
in its development. Today we throw most of these away after training. The
moonshot: turn them into a *film of the model learning*. A scrub-able
timeline. Watch the brain organize itself as the user drags the playhead.

### What's on screen

Single page, three stacked panels, one timeline at the bottom:

- **Top — generation.** Same prompt fed to every checkpoint. Watch the
  output mutate from random binary garbage → printable letters → real words
  → grammatical sentences → coherent story. Like time-lapse of a child
  learning to talk.
- **Middle — neuron firing heatmap.** Activation magnitudes per
  (layer, neuron) at each checkpoint. Watch random noise crystallize into
  structured firing patterns. Specific neurons "wake up" at specific steps.
- **Bottom — attention patterns.** All 144 attention heads. Watch them go
  from uniform soup → diagonal (look-at-self) → structured patterns
  (track quotes, track subject, track prior token).
- **Auto-detected events.** Algorithmic pass over consecutive checkpoints
  flags moments like "Neuron 4231 in layer 7 acquired a function at
  step 32K — now fires only on quoted dialogue." Generated by comparing
  each checkpoint's neuron firing pattern to the next; flagging neurons
  that abruptly become selective.

### Why this is novel

Mechanistic-interpretability researchers (Anthropic, Olsson et al.) probe
checkpoints across training and publish *charts* of metrics — induction-
head emergence curves, in-context-learning phase transitions. Nobody has
shipped the actual *film* — the brain itself, rendered, scrubbable. And
nobody has done it for a hand-coded INT8 inference engine where every
activation is plain memory.

Project MRI's "glass model" claim earns its keep here: the same trace
infrastructure built for v6.2.0 just runs N times.

### Tractability

Trivial extension of v6.2 once a training run finishes. Pseudocode:

```
for ckpt in sorted(checkpoints):
    model = load(ckpt)
    trace = forward_with_trace(model, prompt)
    write_trace(f"frame_{ckpt.step}.bin")
generate timeline.html that picks frames by playhead position
```

Real cost is the multi-frame viewer, not the C side.

### Phase-transition detection (stretch)

Loss curves are smooth. Capability emergence often isn't — Olsson showed
in-context learning appears in a "step jump" inside a few hundred training
steps. Our viewer can *see* these in the brain rather than infer them from
loss. Flag any neuron / attention head whose activation distribution
changes by >Nσ between consecutive checkpoints. The list is the
phase-transition log of the model's life.

### Time estimate

- Sweep + per-frame trace generation: half day
- Multi-frame timeline viewer: 1-2 days
- Phase-transition detector: 1 day
- **Total: ~3 days of focused work** post-training

Could become Veritate's most-shared artifact. Hard to look away from.

## v6.3 — Strategic position on "everything custom"

**The honest framing:** Veritate's distinctive value is the *runtime*
(the .exe that runs the trained model on a user's machine) — hand-coded
C + assembly with our own kernels, our own KV cache, our own dispatch.
That's where the unfair-advantage hardware constraints live and where
nobody else competes hard.

The *training* phase (which produces the .bin file we then ship) is
offline, GPU-bound, compute-limited, and shows up nowhere in any user-
facing artifact. Using PyTorch for training is using a tool to make a
file. The file is the product.

Replacing PyTorch wholesale would consume years and add zero value at
inference. Worse: it would force us to also reimplement autograd,
distributed-training infrastructure, kernel libraries — all of which
exist and work and aren't the point.

**The reframe:** *everything custom at inference, strategic at training.*
- Inference: 100% ours. Hand-coded down to the AVX-512 intrinsics. We
  own every byte at runtime.
- Training: use the right tool. PyTorch for the gradient stuff, our own
  scripts for everything else (data prep, calibration, exports).
- Tooling around the model (interpretability viewer, eval harness,
  synthetic-data pipeline): 100% ours. This is where "no existing tools"
  pays off — nobody has built these things specifically for hand-coded
  INT8 inference engines.

The constraint we should hold: *anything that ships in the binary is
ours. Anything that produces the binary is whatever's most efficient.*

# ------------------------------------------------------------------------------------
# v6.4 — Incremental-computation moonshots ("don't recompute, update")
# ------------------------------------------------------------------------------------

> Brainstorm session, 2026-04-28. Started from a question: "what's the
> algorithm where you compute from a checkpoint plus the new direction
> instead of recomputing?" Answer: incremental computing (data-systems
> term) and Kalman filter (maps/navigation term) are the two faces of
> the same idea. This section asks: where in AI inference are we still
> recomputing what we could be updating?

## The abstract framing

Almost every speedup we've shipped is an instance of *don't recompute
what you already know*:

| layer of recomputation eliminated | mechanism | win |
|---|---|---|
| matmul column-pack per call | `prep_b()` separation | 3× |
| thread creation per call | persistent pool | 1.3× |
| attention K/V across decode tokens | KV cache | 57× |
| GELU per-element transcendentals | 256-byte LUT | 96× |

The win is always huge because *the universe of "things we recompute
that we don't have to" is enormous and barely audited.* Each item above
was found by paying attention to one specific pattern. The list below
asks: what other patterns are we still blind to?

## Where computational waste likely still lives

**1. Layernorm stats.** Every layer's LN computes mean and variance over
V_HIDDEN elements. During decode, the residual stream at most positions
doesn't change between tokens — only the new position's row is fresh.
For cached positions whose residuals are reused across blocks, the LN
stats are also reusable. Today we don't cache them. Likely 13 ms in the
prefill profile is partially redundant work. Cost to verify: half a day
of profiling.

**2. Softmax over stable distributions.** Attention scores at position
N+1 *for cached past positions* depend on the new query but the same
old keys. We recompute the full softmax. If the score distribution is
peaky (which it usually is past the first few layers), we could approximate
with a top-K + tail mass and update incrementally as new positions arrive.
Tradeoff: numerical drift if approximation compounds.

**3. Activation cache across multi-turn chat.** Conversation turn N+1
shares 90%+ of its prompt KVs verbatim with turn N. vLLM's prefix-caching
catches the K/V layer. Could we go *deeper* — cache mid-block FFN
activations across turns whose preceding token sequence is identical?
The tradeoff is memory: full activation cache is 18 MB per turn vs ~5 MB
for K/V. Worth it if matching prefixes are common.

**4. Layer skipping for "easy" tokens.** Early-exit research (DeeBERT,
CALM) shows many tokens converge in the first few layers. Detection is
the hard part. Cheap detector: confidence margin in argmax over current
layer's projected logits. If margin > τ at layer K, skip K+1..L. Ours
is a fixed-architecture model, so we can hard-code which token classes
benefit (whitespace, punctuation, stop-words).

**5. Delta residual streams.** During decode, the INT16 residual stream
changes by a small amount per token at most positions. We currently
store the full magnitude. INT4 deltas on top of an INT16 base, refreshed
every K tokens, would 4× the residual memory bandwidth (currently 8% of
the 167 ms forward). Risk: delta-encoding bugs are nasty.

**6. Per-token deterministic output cache.** The model is deterministic
at temp=0. (prompt, position, model-weights-hash) → fixed output token.
For repeated prompts (system prompts, common phrases), we could cache the
*final outputs* and skip inference entirely on cache hits. This is what
LLM gateways (Portkey, Helicone) do at the request level; we could push
it down to the per-token level for in-flight prompts.

## The architectural moonshot — Mamba / state-space models

The cleanest-ever incremental-computation answer to LLMs.

**What it is.** A state-space model (SSM) replaces attention's
"recompute over all past tokens" with a fixed-size hidden state vector
that updates once per token: `h_{t+1} = A·h_t + B·x_t`, output `y_t =
C·h_t`. The matrices A, B, C are learned. The state `h_t` is a Kalman-
filter-style "compressed checkpoint of the past."

**Why it matters for Veritate.**
- Per-token compute is **O(1) in context length**. 8K context costs the
  same as 256 context. Today's transformer decode at 1.46 ms scales
  with sequence length; SSM decode is constant.
- The state is small (`d_state ~ 16` per channel). Fits in registers.
  The hot path is a single matmul + state update — *exactly* what our
  AVX-512 VNNI kernels are tuned for.
- INT8-friendly. The state-update math is structurally identical to
  what we already do, just with a different connectivity pattern.
- No KV cache management. No attention-window decisions. Memory is
  fixed and tiny.

**What it costs.**
- New training pipeline (PyTorch supports Mamba; not a wholesale rewrite).
- New kernel: SSM scan (selective scan algorithm). Not a matmul, but
  reduces to one with the right reshaping.
- Quality on long-context tasks: SSMs lag transformers on tasks
  requiring arbitrary lookback (associative recall, multi-hop QA).
  Hybrid architectures (Jamba, Zamba) interleave SSM with sparse
  attention to recover this.

**Estimated payoff.** 0.1 ms decode at any context length is plausible.
Veritate would be *the* hand-coded INT8 SSM inference engine — a
category nobody else occupies. Same niche advantage as the MRI work.

## The "Kalman as primitive" framing

Generalizing further: every place in inference where we have *a prior
estimate + new information* is a Kalman-update opportunity.

- **Streaming prefill** (already on roadmap as v3.5). Each keystroke is
  a measurement updating the prompt's KV state. Backspace = state
  rollback. Frame it explicitly as a state estimator and the
  divergence/branch problem becomes tractable.
- **Speculative decoding** (already on roadmap). Draft model = cheap
  predictor; full model = measurement; verification = Kalman update on
  the token-distribution estimate. Standard technique, but the
  Kalman framing makes the *uncertainty* track naturally — could
  drive how aggressively to speculate.
- **Idle pre-thinking.** During between-turn idle, the model
  speculatively generates likely user follow-ups. Each new keystroke
  is a measurement updating which speculative branch is alive.
  Particle-filter framing: maintain N branches with weights, update
  weights per keystroke, prune low-weight branches.
- **RAG retrieval.** Old retrieved docs + new query → updated relevance
  estimate. Today most RAG re-runs the full retrieval per query. A
  Kalman-style update over a sliding context window is plausible.

## The wildest moonshot — continuous-state inference

**Premise.** Tokens are discrete, but the underlying semantic state is
continuous. What if generation happened in latent space (continuous
hidden vectors) and only discretized at the output boundary?

This is COCONUT (Meta, 2024) territory. Veritate-specific implications:
- Decode becomes a continuous trajectory, not a token sequence. Each
  "step" updates a hidden vector incrementally. Pure Kalman.
- Sampling temperature becomes a noise injection on the trajectory.
- Multi-branch speculation = multi-trajectory ensemble. Pruning happens
  by trajectory divergence.
- Output discretization happens once, at the end of a "thought," not
  per token.

This is research, not engineering. Cited because the abstract framing
of this whole section points at it: *AI inference today is heavily
discretized when the underlying math wants to be continuous.* Continuous
inference is the limit case of incremental computation.

## Ranking — what to look at first

1. **Profile where the 167 ms actually goes** post-AVX-512 attention.
   Attention is still 82% of forward; LN is 8%. Confirm before guessing.
   Half a day.
2. **Multi-turn activation cache.** Likely highest payoff per engineering
   week for chat use cases. 1-2 weeks.
3. **Layer skipping for easy tokens.** Lower-bound 1.3× decode speedup,
   upper-bound 2-3× depending on detector quality. 1 week to prototype.
4. **Mamba prototype in Python.** Train a 10M-param Mamba on TinyStories.
   Validate quality is in range. Decide if v5 architecture pivot is real.
   1-2 weeks.
5. **Kalman-framed streaming prefill.** Already on the v3.5 roadmap;
   formalize the state-estimator framing first to make speculative
   branching tractable. 1 week.

The list is unranked beyond #1; profile before committing to anything below.

# ------------------------------------------------------------------------------------
# v6.5 — CPU-native moonshot ("the GPU is solving the wrong problem")
# ------------------------------------------------------------------------------------

> Brainstorm continuation, 2026-04-28. Premise: we're writing for CPU and
> assume hardware keeps improving along that axis. Push that assumption
> hard — what is the CPU *insanely* good at that we're under-exploiting,
> and how do we design end-to-end around its strengths instead of
> treating it as "GPU but smaller"?

## The reframing — autoregressive decode is serial; CPU is built for serial

GPUs were designed for embarrassingly-parallel work — graphics shading,
massive batched training. For *prefill* (compute every position
simultaneously) they win by 100×. For *decode* (token N+1 depends on
token N's full forward result) the parallelism they offer can't be
used. They hide latency by issuing many independent warps; an
autoregressive dependent chain has only one warp to issue.

CPUs are designed for exactly this regime: deep speculative pipelines,
out-of-order execution across dependent ops, branch prediction that
runs ~200 instructions ahead, cache hierarchies tuned for irregular
access. *Decode is the CPU's home turf.* The industry uses GPU for it
out of inertia — the model trained on GPU, the inference fits on the
same GPU, nobody re-evaluated the hardware fit per-stage.

This reframes the project from "hand-coded inference engine targeting
CPU" to *the CPU-optimal LLM stack — not CPU-as-fallback, CPU-as-right-
answer*. The pitch: *every modern AI box has a CPU. On small/medium
models at batch=1, CPU is faster than GPU because the GPU is solving
the wrong problem.*

## What CPU is insanely good at (the under-exploited list)

| capability | order-of-magnitude advantage vs GPU | Veritate exploitation status |
|---|---|---|
| function-call / kernel-launch latency | 1-2 ns vs 5 µs (~3000×) | partial — we have one big binary |
| L3 cache size & bandwidth at random access | 96 MB / ~250 GB/s vs ~6 MB / GPU L2 | partial — model fits but we don't pin |
| branch prediction on data-dependent control | near-perfect on predictable patterns | not exploited |
| out-of-order execution across dep chains | 200-instruction window | not exploited (matmul is naive serial) |
| pointer chasing / irregular access | prefetcher + cache + speculation | not used |
| AVX-512 mask-register conditional SIMD | per-element skip at zero branch cost | not used |
| variable-time per-instruction execution | ~5 GHz peak with elastic IPC | not used |
| OS / memory / I/O / DMA integration | direct memory-map, kernel bypass | not used |
| predictable latency | no driver, no JIT, no kernel jitter | implicit win, not designed for |
| hardware-prefetch-friendly access patterns | learn-and-fetch ~64 cycles ahead | not designed for |

Every row that says "not exploited" is a moonshot.

## The candidate exploits, ranked by gut-belief in payoff

### 1. JIT-specialize kernels to the actual loaded model

At `model_load()` time, emit assembly with this model's actual constants
baked in as immediates. Today our matmul kernels are generic across any
INT8 weights. With JIT we can:

- **Constant-fold zero weights.** Sparse-trained model with 25% zeros →
  25% fewer multiplies, encoded as missing instructions.
- **Inline scales.** `scale_q24` becomes a compile-time literal in the
  emitted code; saves a load per requantize.
- **Specialize tile shape per matrix.** qkv (64×768×768) and ffn_up
  (256×768×3072) want different tile dimensions. Today we use one
  generic kernel; JIT lets each matmul have its own.
- **Bake the model shape into immediates.** No more loop-bound checks;
  unrolled exactly to V_HIDDEN=768.

Cost: ~1 ms emit at startup using AsmJit/Xbyak-style runtime codegen.
Recovered every inference thereafter.

Estimated 2-3× over current kernels on the FFN matmuls. Compounds with
everything else.

### 2. L3-resident model with explicit cache control

Today we trust the cache replacement policy. The 9800X3D has CLDEMOTE
(demote a line to L3 without eviction) and CLWB (write-back without
flush). With explicit control:

- Pin all weights to L3 ways (or use way-partitioning if available).
- Demote activation lines after use to free L1/L2 for next layer.
- Prefetch next-layer weights with PREFETCHT0/T1 timed to compute.

Result: weights never leave L3, never touch DRAM after load.
Bandwidth-bound inference becomes L3-bound (~250 GB/s) instead of
DRAM-bound (~80 GB/s). 3× memory-bandwidth headroom.

Has hardware-availability gotchas (way-pinning is platform-dependent;
fallback is "design access pattern so the LRU naturally keeps weights
hot"). Probably 1.5-2× wall time reduction on memory-bound matmuls.

### 3. Software pipelining across layers

Today: layer N matmul → layer N LN → layer N+1 matmul. Linear chain.
The matmul uses INT8 SIMD execution units; LN uses scalar FP32 units.
*These are different physical execution ports.* CPU OoO can overlap
them if we structure code so the data dependencies allow.

Specifically: while layer N's matmul is finishing the last few output
rows, layer N+1's LN-stats compute (mean/variance) can start on the
already-finished rows. Restructure forward() so the compute of stage
N+1 begins on the partial outputs of stage N.

Risk: cache thrashing if layer N+1 inputs evict layer N outputs.
Calibrate per-stage tile size so both fit in L1.

Estimated 10-15% wall time. Smaller than #1/#2 individually but
compounds with them.

### 4. Massive precomputed LUTs

The 96 MB L3 holds an *enormous* precomputed surface. We've used 256
bytes of it (GELU). What else is tabulatable?

- **Softmax exp() LUT for INT8 inputs.** 256 entries × 4 bytes = 1 KB.
  Replaces `expf` in the hot path.
- **64 KB INT8×INT8 product LUT.** Every possible byte-pair product
  precomputed. Cache-resident in L1. Replaces some scalar multiply
  paths in attention scoring.
- **First-layer activation cache for common tokens.** Top 1000 byte
  bigrams × layer-0 output (768 bytes each) = 768 KB. Skip layer 0
  entirely on cache hits. Free for repeated-prompt tokens (stop words,
  punctuation).
- **Per-head common attention pattern cache.** 144 heads × N common
  prompt prefixes. If you've seen "Once upon a time, " before, its
  attention pattern at every head is precomputed.

Each is small in code; total potentially saves tens of milliseconds.
The cost is the precompute step and the cache pressure of competing
LUTs. Tier them by hit rate.

### 5. AVX-512 mask registers for structured sparsity

If training imposes 4:1 structured sparsity (every 4 weights, exactly
1 is forced to zero), we can use `k`-masks to skip zero multiplies
*at the SIMD level with zero branch cost*. The ZMM operates on 64
INT8s; the matching `k` mask says "lanes 0,4,8,... are zero, skip
them." VPDPBUSD respects the mask.

25% FFN compute reduction at ~0% quality cost (per the literature on
structured sparsity in well-trained models). Compounds with INT4 (v4)
for combined 8× FFN compute reduction.

### 6. Speculative twin decode on a sibling SMT core

The 9800X3D has 8 physical cores × 2 SMT = 16 logical. Decode is
serial on one core, leaving 15 idle during the chat decode loop.

Run two decodes in parallel:
- Core A: real decode of token N+1.
- Core B: speculative decode of token N+2 *assuming* token N+1 will be
  the argmax of the current token N's distribution.

If correct (which it is most of the time at temp=0): 2 tokens per
token-of-latency.

If wrong: roll back core B's KV cache delta, no quality loss, no
worse than today.

SMT siblings share L1 → free data sharing on the cache writes.

Estimated 1.6× decode throughput on confident sequences. Stacks on
top of speculative decoding (v4) — speculative decoding speeds up
*one* generation; this parallelizes the generation itself.

### 7. Hardware-prefetch-aware weight layout

The CPU's hardware prefetcher detects strided access patterns and
pulls cache lines ahead of time. Today we rely on it but don't
design for it. Specifically:

- Weight tiles are stored in matmul order, but accessed in "for j: for
  i" loops which hit the tiles non-strided.
- Bias and scale arrays are stored separately, requiring multiple
  prefetch streams.

Restructure: store each weight matrix as a sequence of
"(weights, scales, biases)" interleaved tiles in the exact order they're
read. The prefetcher then pulls each tile as a single stream, no
prefetch-hint instructions needed.

~5-10% win on memory-bound layers, free on compute-bound. Combines with #2.

### 8. Variable-time per-token inference budget

Tokens vary in difficulty. "the " after a noun is easy; the first
token of a new paragraph is hard. CPU can spend variable cycles per
token; GPU enforces uniform across the warp.

Heuristic: confidence margin in the layer-K logits. If margin > τ,
exit early. Calibrate τ per layer using held-out val data.

For our 12-layer model, average exit at layer 7 → ~40% reduction in
average decode time. Worst case (tough token) goes through all 12 as
today.

This is layer skipping (already in v6.4), framed as the CPU advantage:
GPUs can't do this without warp divergence kills.

### 9. Branch-prediction-aware sampling

Modern CPU branch predictors track ~10K branches with neural-network-
derived heuristics. They are, in effect, *small models predicting
which way control will flow*.

For greedy sampling, the branch "is token X the argmax?" is
data-dependent and predictable when the model is confident. The
predictor will learn the pattern and speculatively run the next-token
forward path — overlapping with the actual logit computation.

Effectively: the CPU is a small auxiliary speculative-decoding model
*for free*. We just have to structure the sampling code to expose
predictable branches.

Hard to measure independently; falls out of writing the sampler in
predictor-friendly form. ~5% effect, free.

### 10. CLDEMOTE-driven cache flow

CLDEMOTE moves a line from L1/L2 to L3 without evicting. Write a
result, demote it, free the L1 slot for the next compute. Today we
write activations and let LRU figure it out — sometimes those
activations are immediately re-read by the next layer (good, stay in
L1), sometimes they're not read for many layers (should be demoted,
free the slot).

Profile-guided cache management. Annotate each store with where it
should live. The CPU has the instructions; nobody uses them.

## The compound case

None of these are 10× wins individually. But:

- **JIT specialization (2-3×)**
- **L3 pinning (1.5-2×)**
- **Software pipelining (1.1-1.15×)**
- **Variable-time decode (1.4×)**
- **Twin decode (1.6× on confident sequences)**

Multiplicatively: 2.5 × 1.7 × 1.1 × 1.4 × 1.6 ≈ **10×**. Even if half
of these fail or compete with each other, the realistic compound win is
3-5× *on top of v3.4.5's current numbers*. That puts the 80M model's
167 ms forward at 30-50 ms, and the 1.46 ms decode at 0.3-0.5 ms.

Combined with Mamba (v6.4) — which is constant per-token-cost — the
0.1 ms decode hard target becomes plausible without distillation.

## What this changes about the project's external pitch

Veritate stops being "we hand-wrote a transformer in C." That's a
craft project. It becomes:

> *The optimal CPU stack for autoregressive LLM decode. Designed end-to-
> end around what the CPU is actually best at. No GPU required. No CUDA.
> No driver. Sub-millisecond per token on a $400 chip.*

That's a category — not a project. Nobody is competing here because
the assumption "GPU = inference" is uninspected. The same way Carmack
proved CPUs could do graphics, then Sweeney proved they couldn't, then
GPUs ate graphics — there's a swing back coming for inference at small
batch, and we'd be ahead of it.

## Order to attempt

1. **Profile first** (still gating, per v6.4). 167 ms breakdown post-
   AVX-512 attention. Without this we're guessing at multipliers.
2. **JIT specialization (#1).** Highest single win, lowest external
   risk. ~2 weeks.
3. **L3 cache control (#2).** Platform-dependent but universal benefit.
   ~1 week of profiling + tuning.
4. **Variable-time decode (#8).** Calibration-bound, lowest engineering
   risk. ~1 week.
5. **Twin decode (#6).** Architectural change but the SMT cores are
   sitting idle. ~2 weeks.
6. Everything else falls out of having #1-#5 in place.

The whole compound stack: ~2 months of focused work to ship the
"CPU-native" pitch with hard numbers behind it.

# ------------------------------------------------------------------------------------
# Project MRI v7 — QAT-aware learning view
# ------------------------------------------------------------------------------------

## The gap

Project MRI's Learning tab today shows the same panels as the Generation tab,
just replayed across saved checkpoints of an FP32 training run. That answers
"is this model getting more coherent?" but not "is this model surviving the
INT8 squeeze?" — which is the question the platform actually exists to answer
("INT8 first, analog-ready" per CLAUDE.md).

The QAT machinery already exists in `training/qat.py` (fake-quant STE, three
modes — off / weights-only / weights+activations+residual) and
`training/qat_finetune.py` saves checkpoints with `qat: True`. Nothing in the
MRI captures that signal during training, so the Learning tab can't render it.

## Hooks to capture during training

Per QAT step, log to a sidecar CSV (e.g. `docs/qat_train.csv`) so the live
training tab and offline replay can both read it:

- **Per-layer saturation rate** — fraction of values clipping at ±127 after
  fake-quant. Most direct "where does this layer break?" signal.
- **Per-layer fake-quant scale** — the per-tensor max-abs scale for weights
  and (in mode 2) activations + residual. When scales settle, the layer has
  found its INT8 budget; when they keep rising, the layer is fighting the
  representation.
- **FP32 vs fake-quant logit KL** on a small fixed eval batch (every
  N steps). A single number per checkpoint summarizing total quantization
  cost.
- **Killed neurons** — per layer, count of neurons whose post-GELU activation
  is non-trivial in FP32 but flatlines after fake-quant. Direct
  interpretability story; later seeds the INT4 / distillation work.

These should also be captured into the per-checkpoint timeline JSON (one
saturation map, one scale array, one KL value, one killed-neuron list) by
extending `mri/probes/timeline_probe.py`.

## Panels to render in the Learning tab

Add as a new section below the existing checkpoint scrubber. The existing
panels stay — they answer "did the model learn?". The new ones answer "did
it survive quantization?":

1. **Saturation map.** 12-row strip, brightness = % saturated for that layer
   at this checkpoint. Same shape language as the FFN brain. Scrub → watch
   layers go from "hot under quantization" to "settled".
2. **Quant scale evolution.** Line chart per layer over training steps,
   weights and activations on separate axes. Want them to flatten.
3. **FP32 vs quant divergence.** Single line: KL on the eval batch over
   training steps. Watch it drop.
4. **Neuron-kill bar.** Per layer, count of dead-after-quant neurons. Hover
   shows neuron IDs (clickable into the existing neuron modal).

## Multi-model selector (the "view more models" piece)

Today the Learning tab loads a single `data/timeline/timeline.json`. To compare
runs (FP32-base vs QAT-1-finetune vs QAT-2-finetune; or the same recipe at
different sizes) we need:

- One timeline directory per model: `data/timeline/<model_name>/timeline.json`
  + per-step JSON files. Existing single-file layout migrates by moving into
  `data/timeline/default/`.
- Server endpoint `/timelines` that scans `data/timeline/` and returns
  `[{name, n_checkpoints, precision, params}]`.
- Dropdown in the Learning tab. Swapping models reloads `timeline.json` for
  the chosen run; ckpt scrubber + token scrubber re-bind to the new data.

Stretch (worth scoping but not blocking): **side-by-side mode.** Pick two
runs, see them in two columns of panels. Direct visual answer to "what does
turning on QAT mode 2 actually do to layer 7?"

## Bounded next step

Don't do the whole thing at once. Wire one metric end-to-end first to prove
the data path:

1. ✅ **Saturation map shipped.** Captured offline in `mri/server/brain.py`
   from existing FFN hooks (no training changes needed) and rendered as a
   12-row strip in the Learning tab. Re-running `mri/probes/timeline_probe.py`
   populates per-checkpoint data; the panel falls back to a "data not present"
   message on older runs.
2. ✅ **FP32 vs quant logit KL shipped.** `Brain.compute_quant_kl()` snapshots
   weights, fake-quants them per-tensor to INT8, runs a second forward, and
   returns KL bits between FP32 and quantized next-byte distributions
   (weights restored after). Captured per checkpoint by `timeline_probe.py`
   and stored as `checkpoints[i].quant_kl_bits`. Frontend renders a line
   chart in the Learning tab between "output evolution" and "checkpoint
   scrubber" — cool blue dots for FP32 checkpoints, warm orange for QAT,
   neon green for the currently-selected one. Click any point to jump to
   that checkpoint.
3. **Quant scale evolution per layer.** Per-layer max-abs across training.
   Companion chart to KL.
4. **Killed neurons.** Per layer, count of neurons strongly active in FP32
   that flatline post-quant.
5. ✅ **Multi-stage comparison v0 shipped.** `timeline_probe.py` now accepts
   `--checkpoints_dirs DIR1 DIR2 ...` and builds one unified timeline across
   stages. Each checkpoint gets `stage` (= dir basename), `effective_step`
   (= `step` + warm-start origin parsed from the parent ckpt's args), and
   `warm_start_step`. Frontend "output evolution" panel renders as a grid
   with one column per stage, rows ordered by `effective_step` so FP32 and
   QAT trajectories share a vertical time axis (empty cells in stages that
   don't have a ckpt at that step). KL chart x-axis uses `effective_step`
   so the line spans both stages on one timeline. Frame cache is keyed by
   `(stage, step)` so two checkpoints with the same step number from
   different stages don't collide. Run with:
   `py mri/probes/timeline_probe.py --checkpoints_dirs data/checkpoints data/checkpoints_qat_v2`
6. **Side-by-side panel inspection (v1).** Today selecting a checkpoint
   shows its panels alone. v1: pick two checkpoints (one per stage) and
   render their FFN brain / saturation / attention / etc. in mirrored
   columns so the QAT effect on each layer is directly visible. Requires
   loading two per-checkpoint JSONs at once and a "compare" toggle in the
   header.

This sequence keeps every step shippable and skippable.

## Scope alignment

This is the bridge from v3 (FP32 PyTorch) to v4 (INT4) on the platform
roadmap. Without QAT visibility, we are quantization-blind — we'd ship v4
on faith. With it, every brittle layer is named before it ships.

The "INT8 first, analog-ready" principle in CLAUDE.md is the whole point:
the MRI exists to make that transition legible, not just to pretty-print
FP32 generation. This is the MRI doing its job.

# ------------------------------------------------------------------------------------
# Project MRI v8 — decision tracing & neuron personality
# ------------------------------------------------------------------------------------

## The gap

By v7 the Learning tab can show *what's happening* to a model under QAT
(saturation, KL trajectory). It still can't answer the more interesting
interpretability question: **why did the model pick byte X over byte Y?**
Or: **what specifically did this neuron learn from training?** The existing
panels (FFN brain, top-firing neurons, candidates, logit lens) describe
state but never decompose it.

## What the math gives us, for free

The whole transformer ends with `logits = final_residual @ embed.T`. The
final residual is exactly the embedded prompt + a sum of every
`block_output - block_input` along the way. So every byte's logit is a sum
of contributions you can attribute *to the specific neuron that produced
each piece*. This is **direct logit attribution (DLA)** and it is exact for
this architecture — no approximations, no SAEs needed for v1.

Per FFN neuron `(L, n)`, the contribution to byte `b`'s logit is exactly:

```
activation[L, n] * (ffn_down[:, n] · embed[b, :])
```

The right-hand factor is constant per checkpoint — precompute it once into
a `byte_direction` table of shape `(layers, ffn_dim, vocab)`, ~38 MB at
fp32. After that, every per-frame attribution is a single elementwise
multiply per neuron.

## Shipped (v8.0)

1. ✅ **`byte_direction` precompute** in `Brain.__init__`
   ([mri/server/brain.py](../mri/server/brain.py)). At server start, we compute
   `(W_down.T @ W_E.T)` per layer. Cached in `self.byte_direction`. Also
   `self.W_E_T` for projecting per-layer residual deltas to logit deltas.

2. ✅ **`Brain.neuron_byte_affinity(layer, neuron_id)`** returns the top
   positive and top negative bytes that neuron writes toward when it
   fires — its "personality." Surfaced via the `/neuron/<L>/<n>` endpoint
   (`affinity` field).

3. ✅ **Per-frame DLA (`Brain._dla_top`)** — at every yielded token, for
   the picked byte AND the argmax byte, returns the top 12 (layer, neuron)
   contributors with their activation, weight, and contribution. Fields
   `dla_picked` / `dla_argmax` on the frame.

4. ✅ **Per-layer decisiveness** — for each layer, projects its
   `(block_out - block_in)` through the unembedding to get a per-byte
   logit-delta vector, reports `max_abs / mean_abs`. Field `decisiveness`
   on the frame. High = layer committed to a direction; low = layer split
   the vote.

5. ✅ **`argmax_byte`** also added to the frame so the frontend can show
   "what the model expected" alongside "what got sampled."

6. ✅ **Memory probe positional context.** `Brain.build_memory_from_corpus`
   now stores `peak_pos` per (neuron, story) — the byte index where that
   neuron's activation peaked in that story. Schema:
   `{text, score, peak_pos}`. The neuron modal renders the ±12-byte window
   around `peak_pos` as a green-highlighted span so you can read what
   actually triggered the firing.

7. ✅ **Frontend panels** (Generation + Learning):
   - **decision trace** — two side-by-side tables. Left: top DLA
     contributors to the byte that was sampled. Right: top contributors to
     the byte the model expected (argmax). When the two lists differ, the
     gap is the *surprise* explanation. Click any row → existing neuron
     modal (now enhanced).
   - **per-layer decisiveness** — 12-bar chart, region-tinted, shows where
     the model committed vs stalled.
   - Enhanced **neuron modal** — byte affinity chip rows ("votes for"
     vs "votes against") + memory stories with peak position highlighted.

8. ✅ **Learning tab clickability.** `cFfnL` and `cTopL` are now click-to-
   modal too, matching the Generation tab. Same `showNeuronModal()` flow.

## Tradeoffs to know

DLA is a *direct* attribution: it accounts only for first-order writes
to the residual. Second-order paths (neuron A fires, which makes neuron B
fire one layer later, which then writes to the byte direction) are not
unwound. For full causal accounting you'd need **path patching** or
**activation patching** — extra forward passes per probed component.

For the model size we have (80M, 12 layers, 768 hidden), DLA is the right
v1: cheap, exact for the dominant terms, and answers ~80% of "why this
byte" questions in practice. v2 can layer on patching for the top-3
attributions when the user clicks "trace deeper."

## Shipped (v8.1) — neuron circuit view

The "isolated neuron" problem: a single neuron's activation magnitude or
byte affinity, alone, doesn't tell you what role it plays. To make it
meaningful you need to show its *neighbors* — what fed it, what reads it.
Done end-to-end:

1. ✅ **`Brain.neuron_predecessors(L, n)`** — for the most recent forward
   pass, finds the top earlier-layer FFN neurons whose
   `activation × write-direction` contributed most to neuron `(L, n)`'s
   pre-activation. Dynamic, per-frame: "for *this* token, who made me
   fire?"
2. ✅ **`Brain.neuron_successors(L, n)`** — static analysis from weights
   only: top later-layer FFN neurons whose read-direction aligns with this
   neuron's write-direction. "Who is wired to listen to me?"
3. ✅ **`Brain.neuron_stats(L, n)`** — current activation + probe-max +
   percentile, so the modal can show "currently firing at 23% of probe
   peak" instead of just the raw 0.42.
4. ✅ **`/neuron/<L>/<n>` endpoint** — returns `{stories, affinity,
   predecessors, successors, stats}`.
5. ✅ **Modal**: region description blurb at top, activation-context box
   with plain-language hint, "fed by" and "feeds into" chip rows
   (clickable to navigate the circuit), byte affinity, then memory
   stories with peak position. Each section gets a one-line caption
   explaining what it is and why it matters.
6. ✅ **Chart descriptions**: per-layer decisiveness, residual stream
   depth, per-layer contribution all rewritten with concrete plain-language
   readings. Decisiveness chart now has region band labels (SENSORY /
   ASSOCIATION / OUTPUT) on the bars.
7. ✅ **Region legend** on FFN brain panels (Generation + Learning) so the
   sensory / association / output meanings are always visible without
   expanding the explainer.

The circuit view is what flips a neuron from "an opaque number" into "a
node with upstream contributors and downstream listeners." Click any chip
in fed-by or feeds-into to walk the circuit.

## Not yet shipped (v8.2+)

- **Per-attention-head DLA.** Same idea, harder bookkeeping (need to split
  `W_O` per head). Equally valuable for "why did the model attend to byte
  position 12 instead of 18."
- **Click a candidate, see who voted for it.** Right now we attribute only
  the picked byte and the argmax byte. Generalizing to any byte the user
  clicks is one DLA call.
- **Click a memory story, jump to that part of the training corpus.**
  Today we show the story text + peak position, but there's no link back
  to the corpus offset.
- **Activation patching.** "What if neuron (L, n) had been silent? Show
  me how the byte distribution would have changed." One additional forward
  pass per probed component. Realistic for top-3 contributors only.

## How to hook in

See [docs/reference/BRAIN_HOOKS.md](reference/BRAIN_HOOKS.md#decision-tracing-fields)
for the full list of frame fields, the `/neuron/<L>/<n>` endpoint shape,
and the `byte_direction` table semantics.

# ------------------------------------------------------------------------------------
# Probe-driven curriculum — closing the loop between eval and training
# ------------------------------------------------------------------------------------

The grade probe and concept probe (`training/checkpoint_probe.py::dump_grades`,
`dump_concepts`) already produce a live read of the model's weakness:
per-grade-band perplexity in `grades_step_<N>.json`, per-concept surprise in
`concepts_step_<N>.json`. The MRI dashboard renders both. Today this signal
is purely diagnostic — it does not influence the next batch the trainer sees.

That's the gap. The probes know which categories are STRUGGLING; the trainer
keeps sampling uniformly anyway. Closing this loop turns the probe stack
from instrumentation into a control system.

## Three escalation levers

Each lever uses the same probe JSONs, just at different points in the
training pipeline. Build them in order; each is independently shippable.

### Lever 1 — Probe-driven data mixer (v1, smallest viable)

After every probe run, compute per-category sample weights for the next
training shard. Bucket the existing corpus by category (cheap regex/keyword
match against the 8 category word lists in [`mri/static/conversation.html`'s
`CONCEPT_GROUPS`](../../mri/static/conversation.html), promoted to a shared
constants file). Reweight the data loader's bucket sampler proportional to
each category's mean surprise. STRUGGLING clusters get more samples;
MASTERED clusters get downweighted (not zeroed — keep some flow to prevent
drift).

No model changes. No loss changes. Pure sampler-side intervention.

### Lever 2 — Auxiliary probe loss

Add a small auxiliary loss term: at each training step, run the worst-K
concept probes through the model and add their mean surprise (scaled, e.g.
0.05 * main_loss) to the gradient. Forces the model to directly improve
the metrics the dashboard tracks. Cheap (~50 extra forward tokens per
step) but louder — the gradient now has an explicit "fix the probes"
signal.

### Lever 3 — Conditioned generation (the lever already in the spec)

Tag training data with a `<grade=K>` prefix token and train the model to
condition on it. At inference, prefix with the desired grade and the model
generates at that level. This is the long-form plan in
[`docs/notes/GRADING_SCALE.md`](../notes/GRADING_SCALE.md#three-eval-suites-per-grade)
under Suite C. Biggest lever, biggest scope: requires per-sample grade
labels, larger corpus pre-processing pass, and Stage D+ to be realistic.

## v1 schema — `mix_weights_step_<N>.json`

Sits next to the existing probe JSONs in `data/models/<run>/`:

```json
{
  "step": 14000,
  "based_on": { "grades_step": 14000, "concepts_step": 14000 },
  "weights": {
    "objects":    0.7,
    "emotions":   0.9,
    "family":     0.7,
    "colors":     1.0,
    "attributes": 0.8,
    "actions":    1.1,
    "math":       2.0,
    "meta":       1.4
  },
  "rationale": "math STRUGGLING (avg 4.2 bits), meta high (avg 2.6 bits)",
  "controller_version": "v1"
}
```

Weights normalize so the total expected sample budget per shard is
unchanged. The `rationale` field is a human-readable diagnostic, written
straight to the MRI's new "data mix this shard" panel.

Old shards never need this file. Trainer falls back to uniform sampling
when missing.

## Where it plugs in

| Layer            | New code                                                   | Existing code touched                                  |
| ---------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| Probe            | `training/probe_mixer.py::compute_mix_weights(probe_dir)`  | none                                                   |
| Trainer          | `training/qat_v2.py` data loader: read latest mix_weights at shard boundary, update bucket sampler weights | one block, ~15 lines                |
| Corpus prep      | `scripts/categorize_corpus.py` — one-shot tagger that writes `data/corpus/<corpus>_buckets.json` mapping doc-id → category | none                                |
| Probe orchestrator | `training/checkpoint_probe.py::dump_concepts` calls `compute_mix_weights` after writing concepts JSON | one line          |
| MRI panel        | new "data mix this shard" classroom panel in conversation.html | one new render fn, ~50 lines        |

That's it. Five touch points; one new module; no schema changes to
existing probe outputs; trainer change is additive. The total v1 surface
is small enough that the controller can be feature-flagged off by simply
not writing `mix_weights_step_*.json` files.

## Goodhart guard — split the probe set

Standing risk: a probe-driven controller will overfit the probe corpus.
The model "improves" on the dashboard because it has effectively trained
on the test.

Mitigation: split the 50 canonical concepts in
[`training/checkpoint_probe.py::CONCEPTS`](../../training/checkpoint_probe.py)
into two disjoint halves at probe time. The "control" half feeds the
mixer. The "honest" half is dashboard-only and never feeds back. The MRI
shows both side-by-side; if they diverge sharply, the controller is
gaming the probes.

Same idea for grade bands — keep one held-out grade-eval bin per band
that the mixer never sees. ~4 KB per band, cheap.

## Multi-brain extension path

The 8 categories in `CONCEPT_GROUPS` already map to the planned modules
in [MULTIMODULE_BRAIN.md](../MULTIMODULE_BRAIN.md):

| Category   | Maps to module                |
| ---------- | ----------------------------- |
| emotions   | Valence Head (#3)             |
| math       | Rule Trace (#4)               |
| meta       | Workspace (#2), Reality Monitor (#7) |
| family     | Workspace                     |
| objects, attributes, colors, actions | Language Core (#1) |

So once the modules exist, the same `mix_weights_step_<N>.json` flips
from "weight the data loader" to "select which module's parameters get
gradient updates this batch." Same input, new consumer. The schema
generalizes by adding an optional `module_routing` field; old
consumers ignore it.

In other words: ship the v1 mixer on the monolith now. When multi-brain
modules land, the controller is already wired and tested — only the
sampler-side action changes.

## v1 build order

1. ✏️ **`training/probe_mixer.py`** — pure function, takes a probe dir
   path, returns a weights dict. Unit-testable in isolation.
2. ✏️ **`scripts/categorize_corpus.py`** — one-shot, dumps doc-id →
   category JSON for the existing TinyStories shards. Cheap regex pass;
   any sample matching no category falls into a `general` bucket.
3. ✏️ **Trainer data loader update** — read latest mix_weights every
   N steps; reweight the bucket sampler. ~15 lines in
   [`training/qat_v2.py`](../../training/qat_v2.py).
4. ✏️ **Probe orchestrator update** — `dump_concepts` calls
   `compute_mix_weights` after writing its JSON. Single line.
5. ✏️ **CONCEPTS split** for the Goodhart guard. Two halves, label them
   `control` / `honest` in the JSON.
6. ✏️ **MRI dashboard panel** — "data mix this shard" reads
   `mix_weights_step_*.json`, shows current weights, intervention
   sparkline, and the rationale string. Confirms the controller isn't
   doing something stupid.

Each step is shippable on its own — the mixer is dormant until the
trainer hooks it up, the trainer is harmless without weights files,
the panel renders nothing useful without writes happening. So the order
above can be reshuffled to whatever lands first.

## Open question — should the mixer affect Stage D specifically?

Stage D (Q&A and arithmetic) is the most likely target for v1 because
the math/meta categories are the worst-performing on current dashboard
reads, and Stage D's purpose is exactly those categories. Running the
mixer on Stage D from day one gives a clean A/B: mixer-on Stage D vs.
the historical Stage A/B/C runs done without it.

Risk: Stage D is also the stage with the least mature corpus. If the
mixer over-weights one bucket and that bucket has corpus issues, we get
a worse model and a confused signal. Worth running both arms.

# ------------------------------------------------------------------------------------
# Concept-to-neuron mapping — making the brain regions literal
# ------------------------------------------------------------------------------------

The MRI shows what each top-K firing neuron does. The concepts panel
shows which 50 concepts the model has formed. These two views are
disconnected: clicking the concept "math/equals" tells you the model is
struggling, but not *where* in the network "equals" lives. Clicking
neuron (L7, n=237) tells you it fires on certain bytes, but not whether
that lines up with any concept the dashboard tracks.

Closing this gap turns the existing region labels (L0-L3 sense, L4-L8
association, L9-L11 output) from a generic taxonomy into a
concept-by-concept atlas: "colors live in L4-L5", "math in L7-L9", etc.
The mapping is what makes "we have a glass-box brain" literal rather
than aspirational.

## What we already capture

`training/checkpoint_probe.py::_capture()` hooks FFN activations during
the canonical-prompt forward pass; `dump_probe` writes top-K activations
per layer to `probe_step_<N>.json`. The plumbing for capturing
per-position activations is solved.

`_concept_surprise_bits()` runs the model on each (preamble, target)
pair and returns the surprise score, but it does NOT capture activations.
That's the missing link.

## v1 — concept-coding neurons

Extend `_concept_surprise_bits()` to also capture top-K FFN activations
per layer at the last preamble position (i.e., the position where the
model is about to predict the target). These are the neurons that "set
up" the concept's output. Add to the existing concepts schema:

```json
{
  "step": 14000,
  "concepts": {
    "cat": {
      "surprise_bits": 1.5279,
      "top_neurons": [
        {"layer": 4, "id": 237, "v": 1.42},
        {"layer": 5, "id": 1023, "v": 1.18},
        {"layer": 7, "id": 588, "v": 0.95}
      ]
    }
  }
}
```

Backwards-compatible add: dashboards that don't read `top_neurons` are
unaffected. Size cost ~10 KB per checkpoint, negligible.

## v2 — direct logit attribution per concept

The MRI already runs DLA (direct logit attribution) for the canonical
prompt's chosen byte. Run the same DLA on each concept's target byte.
This gives causal contribution scores: "neuron (L9, n=512) added +0.42
log-prob to the byte 'c' in 'cat'." More concept-specific than raw
activation because it isolates neurons that *voted for the right
answer*, not just neurons that fire on the surrounding context.

Both v1 and v2 are useful — they answer different questions. Activation
asks "what fires when this concept appears?" DLA asks "what caused this
concept's output?" Most concept-coding neurons show up in both lists.
The ones that appear only in DLA are the "decisive" neurons; the ones
only in activation are context-detectors.

## Reverse lookup: neuron-to-concept

For each (layer, neuron_id) that appears in any concept's `top_neurons`
list, record the concepts. The neuron biography panel already exists in
the MRI; it gains a new line: "responds most to: cat, dog, fish (concept
probes)." This makes neuron interpretation grounded in vocabulary the
user can read, not just byte glyphs.

## Brain-region atlas

Aggregate per-category across layers. For each of the 8 concept
categories (objects, emotions, family, colors, attributes, actions,
math, meta), compute the mean activation per layer averaged over the
category's concepts. Render as a category &times; layer heatmap. The
result tells the story at a glance: "math concentrates in L7-L9",
"colors are L4", "story-meta spans L8-L11". This is the literal
brain-region atlas the project's name implies.

Two views unlock from this:

1. **Click a concept &rarr; highlight its layer signature.** Bars across
   L0-L11 showing where in the stack that concept lives. Hover any layer
   to see the top contributing neurons.
2. **Click a region &rarr; see what concepts it codes.** Pick L7, see a
   ranked list: math (avg act 1.8), meta (1.4), actions (0.9). Tells you
   what the region's specialization is.

## Multi-brain alignment

The category &rarr; region mapping is also the **bootstrap signal** for
the multi-module brain in [MULTIMODULE_BRAIN.md](MULTIMODULE_BRAIN.md).
If math currently lives in L7-L9 of the monolith, that's evidence the
Rule Trace module should attach its read/write taps at L7-L9. The
existing 80M is doing the architectural search by itself; the atlas is
how we read what it found.

## Hook impact (and why this is documented before built)

This change touches `training/checkpoint_probe.py` &mdash; specifically
`_concept_surprise_bits()`. That code is part of the active training
pipeline; QAT2 / curriculum-D runs are currently writing concept JSONs.
Modifying the probe code mid-run is fine in principle (the schema add
is backwards-compatible), but the new field will be absent for any
checkpoints written before the change. Two safe paths:

1. **Land alongside the next training run**, so the new field is present
   from step 0 of that run.
2. **Backfill the in-flight run** with a one-shot
   `scripts/backfill_concept_neurons.py` that loads each saved
   checkpoint, runs the 50 concept probes with neuron capture, and
   writes a separate `concept_neurons_step_<N>.json` (no overwrite of
   existing files). Dashboard merges both.

Both work; (2) is the lighter touch when training is already in flight.

## v1 build order

1. ✏️ Extend `_concept_surprise_bits()` to optionally capture activations
   and return top-K firers per layer alongside surprise.
2. ✏️ Update `dump_concepts()` to call the extended version and include
   `top_neurons` in the JSON.
3. ✏️ Dashboard: when a concept row is clicked (concepts panel), also
   render its layer signature bar and the top-3 neurons per relevant
   layer with click-through to the neuron biography.
4. ✏️ Per-category heatmap panel ("regional atlas") in the Learning tab.
5. ✏️ Optional: backfill script for already-trained checkpoints.

Steps 1-2 are ~30 lines of Python in one file. Step 3 is ~80 lines of
JS. Step 4 is ~60 lines of JS. Total v1 is small.
