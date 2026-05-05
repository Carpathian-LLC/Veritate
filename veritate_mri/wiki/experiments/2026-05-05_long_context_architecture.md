---
title: How a byte-level model remembers full chapters
date: 2026-05-05
tags: [context, memory, architecture, rope, hdc, workspace, sleep, theoretical]
summary: Four-layer theoretical plan for giving a byte-level model long-range memory. None of these layers are built yet.
---

> **Status: purely theoretical.** Nothing on this page is implemented in the current engine or training pipeline. This is the design space for what long-context memory could look like in Veritate, written down so the plan is legible. The 1B PoC currently in training is the language core only. Every mechanism below is a future experiment, not a shipped feature.

A byte-level model fills its attention window 4-5x faster than a BPE model. To make it remember "chapter 1 in chapter 12" we stack four mechanisms. None alone is enough; together they cover short, medium, and long range.

## layer 1: bigger raw attention window

Cheapest, quadratic in cost. Currently `seq=256` on the 80M PoC, planned 1024 on the next language core. Beyond that we swap learned positional embeddings for **RoPE** (rotary position embedding).

RoPE encodes position as a rotation applied to query and key vectors during attention, instead of a learned lookup table. Two wins:

- The model never has to learn a new row for a new position. Position is a formula, not a parameter.
- We can train at 8k and run at 16-32k by extending the rotation. Quality degrades past the training length but does not fall off a cliff.

Gets us tens of thousands of bytes. A novel still does not fit.

## layer 2: workspace (the carry)

A 512-dim vector that travels with the model token-by-token. Modules read and write it at chosen layers. Think of it as the model's scratchpad for "what is this story about right now."

Mechanics:
- At layer 6, modules project the residual stream into a small write into the workspace.
- At layer 12, modules read from the workspace via cross-attention.
- Auxiliary loss: workspace at time `t` should be predictive of byte `t+1` via a tiny linear probe. That loss is what teaches it to summarize narrative state instead of being noise.

This is a learned, continuously-updated summary embedding. It does not remember chapter 1 verbatim. It remembers the gist of chapter 1 in 512 floats, and the gist rides along into chapter 12.

Spec at [docs/plans/multimodule_brain.md:46-59](../../../docs/plans/multimodule_brain.md#L46).

## layer 3: HDC memory (long-term store)

Hyperdimensional Computing. The actual long-context play.

How it works:

1. At the end of every "turn" (chapter, paragraph, conversation segment, configurable), take the final hidden state and a hash of the prompt. Encode them as a 10,000-dim binary vector. Store on disk. ~1 KB per turn.
2. At the start of every new turn, encode the new prompt the same way. XOR-distance against every stored vector. Top-K closest matches retrieved in microseconds. HDC is purpose-built for this: sub-millisecond similarity search over millions of entries, no GPU, no float math.
3. The retrieved past-turn workspace states get injected into layer 1 of the residual stream as extra tokens for that forward pass.

So the model sees `[retrieved gist of chapter 1] + [retrieved gist of chapter 4] + [current chapter 12 text]` all in the same attention window. Relevant past chapters are physically present during the forward pass but stored externally between passes.

This is RAG with two twists:

- The "documents" being retrieved are the model's own past hidden states, not raw text. Compressed semantic memory, not verbatim.
- The retrieval index is an HDC binary store, not a vector DB. ~1 KB per memory. Stays inside the constraint #3 envelope: lives next to the checkpoint, not inside it.

Spec at [docs/plans/multimodule_brain.md:111-120](../../../docs/plans/multimodule_brain.md#L111).

## layer 4: sleep cycles (consolidation)

The cleanup process, offline.

Brain analogy: hippocampus holds today verbatim, sleep replays it to neocortex, neocortex stores a compressed gist, verbatim trace is dropped. Yesterday is sharper than last Tuesday because last Tuesday has been consolidated more times.

Veritate version:

- **Active session:** full KV cache, verbatim.
- **Idle:** light rolling summary.
- **Sleep (≤8h or sooner if idle):** background pass that compresses the day's HDC entries into denser, more abstract entries. Verbatim beyond a window gets dropped. Valence drives priority. Emotionally-weighted moments consolidate harder; trivia decays faster.

Offline. Does not affect the 0.1 ms / byte decode budget.

Spec at [docs/plans/IDEAS.md:311-325](../../../docs/plans/IDEAS.md#L311).

## putting it together: chapter 12 of a novel

1. Tokenize the prompt to bytes, encode as HDC vector, query the store. Top-3 prior-chapter workspace gists come back. (microseconds)
2. Inject those 3 gist vectors into layer 1 alongside the chapter 12 text. (free, small fixed cost)
3. Forward pass runs over `[3 gists + N bytes of current chapter]` inside the RoPE-extended attention window. Workspace carries narrative state forward token-by-token.
4. At end of chapter 12, encode the new final state as another HDC vector. Add to store.
5. Tonight during sleep, consolidate the day's entries. Old chapters compress harder.

The model never has the whole novel in attention. It has the current chapter plus three relevant compressed memories of past chapters. Coherence over a novel comes from the layered system, not from one giant window.

## status

- RoPE: small upgrade, not yet done.
- Workspace: planned in [multimodule_brain.md](../../../docs/plans/multimodule_brain.md), not built.
- HDC memory: planned, not built.
- Sleep cycles: vision-stage in [IDEAS.md](../../../docs/plans/IDEAS.md).

The 1B PoC currently in training is the language core only: layer 1 of the stack. The other layers come after it converges.
