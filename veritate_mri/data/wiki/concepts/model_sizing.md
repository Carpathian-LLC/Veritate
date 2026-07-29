---
title: sizing a training run
date: 2026-07-28
tags: [concepts, training, basics]
summary: How much text a Veritate model needs, and why counting bytes as tokens undertrains a run by roughly four times.
---

# sizing a training run

Every published scaling law tells you how much text a model needs in **tokens**. Veritate models are byte-level, so the number the trainer counts is **bytes**. Those are not the same unit, and treating them as the same is the single easiest way to undertrain a model without noticing.

## the conversion

One English word-piece is about four and a half bytes. Measured on the actual Veritate corpora with a modern 151k-vocabulary tokenizer, sampling 8 MB from the middle of each corpus:

| corpus type | bytes per token |
| --- | --- |
| prose, size-weighted | **4.55** |
| source code | **4.12** |

So the conversion in both directions is:

```
tokens = bytes / 4.55
bytes  = tokens * 4.55
```

Use 4.55 for prose and 4.12 for a code-heavy mix. Do not use 4, which is a common rule of thumb and understates how much text you need.

## how much text a model needs

The Chinchilla result puts the compute-optimal point at **20 tokens per parameter**. That is the point where you stop getting efficient returns on loss. It is *not* the point where a model becomes a good conversationalist: modern small chat models are trained far past it, often to several hundred tokens per parameter.

Both targets, expressed in the bytes a Veritate corpus actually has to contain:

| parameters | Chinchilla (20/param) | conversational (200/param) |
| --- | --- | --- |
| 50M | 4.6 GB | 46 GB |
| 100M | 9.1 GB | 91 GB |
| 200M | 18 GB | 182 GB |
| 270M | 25 GB | 246 GB |
| 500M | 46 GB | 455 GB |
| 800M | 73 GB | 728 GB |
| 1B | 91 GB | 910 GB |

## the trap

A run's total byte budget is:

```
bytes = batch_size * seq * n_chunks * total_steps
```

That number looks like a token count and is not one. A 270M model trained on 5.9 billion bytes has seen 1.3 billion tokens, which is **4.8 tokens per parameter** — under a quarter of Chinchilla, even though 5.9 billion is a bigger-looking number than the 5.4 billion tokens Chinchilla asks for.

The symptom of an undertrained byte model is specific and easy to misread: **the output has the right shape and invented words in it.** Asked to name four instruments it will return exactly four comma-separated items, one of which is not a word. That is not a formatting problem and no amount of instruction tuning fixes it. The model has not read enough English to lock in the lexicon, and it is spelling by guess.

## checking a run before you launch it

Convert first, then decide.

1. Compute `bytes = batch_size * seq * n_chunks * total_steps`.
2. Divide by 4.55 to get tokens.
3. Divide by the parameter count to get tokens per parameter.
4. Compare against the table above.

If the answer is below 20, the run is undertrained no matter what the description says. Either raise `total_steps`, or pick a smaller model.

That last option is the one worth taking seriously. On a fixed corpus, a **fully trained small model beats a partly trained large one** at conversation. The parameter count should be chosen after the data budget is known, not before.
