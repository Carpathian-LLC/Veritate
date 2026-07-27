# Why long conversations get expensive, and what a constant-state model changes

How attention remembers, why that memory grows with every byte produced, and what the hybrid trunk gives up to keep its own memory fixed.

## Abstract

Every other paper in this programme assumes the reader already knows what a key-value cache is and why a growing one hurts. This paper does not assume it. The constant-state property is the reason the hybrid trunk was chosen over a plain transformer, it is the reason the serving engine was written by hand, and it is the property that the long-context work exists to test. A reader who does not know what a growing cache costs has no way to judge whether any of that was worth doing.

So this paper explains the mechanism from the beginning: what attention is for, what it stores while producing text, why that store cannot be avoided in a transformer, and what happens to memory and speed as a conversation lengthens. It then states what the alternative buys and what the alternative costs, using figures already measured and recorded elsewhere in this programme rather than new ones. No experiment in this paper is new. The contribution is the explanation and the assembly: one place where the mechanism, the arithmetic, and the measured consequences sit together.

The short version is that a transformer's memory of a conversation grows by a fixed amount per byte, forever, and both its memory footprint and its per-byte generation cost climb with it. A constant-state model holds one fixed-size summary instead. The summary never grows, and the thousandth byte costs what the first one did. The price is that a fixed-size summary is lossy, which is a limitation this programme has measured directly rather than argued around: recall of a planted fact falls from 0.92 to 0.25 within a few hundred bytes and reaches 0.00 past the local attention window.

## What attention is for

A language model performs one operation repeatedly: given everything written so far, guess what comes next. The difficulty is that the information needed for the guess can sit anywhere in what came before.

Consider a sentence where the model must complete the final word:

> Sarah put the keys on the table. An hour later, she picked up the ...

A person answers "keys" without effort. Doing so requires reaching back eleven words to one specific earlier word, and ignoring the others. "Table" is nearer and wrong. "Sarah" is the grammatical subject and wrong. The correct target sits at no fixed distance and holds no fixed grammatical role, which is why a rule such as "look at the previous three words" cannot work. Sometimes the answer is one word back and sometimes it is fifty thousand bytes back.

Attention is one mechanism for solving that. The recurrent state described later is another. They solve the same problem with opposite cost structures, and the difference between those cost structures is the subject of this paper.

## Query, key, and value

Attention works by having every position describe itself for the benefit of positions that come later.

As the model reads each position, it computes three vectors from that position's representation. In the reference implementation all three come from a single projection, `qkv`, which produces them together and then splits them apart.

The **key** is the description a position advertises about itself, written for whoever searches later. The **value** is what that position hands over if it is selected. The **query** is the question the current position is asking of everything before it.

The three are separate projections because the three jobs differ. What a position advertises to searchers is not what it contributes when found, and neither is the question it is itself asking. In the example above, the position holding "keys" advertises something like "a small portable object," carries the meaning of keys, and simultaneously asks its own question about what follows it. One representation, three roles, three vectors.

The mechanism at a given position is then short. Take that position's query. Compare it against the key of every position that came before, producing one relevance score per earlier position. Normalise those scores into weights that sum to one. Output the weighted blend of the corresponding values.

Two properties of this follow immediately, and both matter later.

The first is that attention has no notion of distance. Position 500 reaches position 1 exactly as easily as it reaches position 499. There is no decay and no locality. Every past position remains individually addressable, at full fidelity, for as long as it is kept.

The second is that nothing here was designed by hand. The model was never told what "relevant" means. It learned, over billions of bytes, what description to advertise and what question to ask. Training is the process that makes those three projections useful.

## Why the cache exists

The cost structure changes completely between training and generation, and the cache is what that change produces.

During training the model sees a whole sequence at once. Every position computes its query, key, and value simultaneously, and every comparison happens as one matrix multiplication. That is expensive in a specific way: comparing every position against every earlier position means the work grows with the square of the sequence length. It is also the shape of computation a GPU handles well, so the cost is tolerable at a fixed sequence length.

Generation inverts the situation. The model produces one byte, appends it to what exists, and feeds the result back to produce the next. Byte by byte, in strict order, with no way to parallelise across positions that have not been written yet.

Here the observation that produces the cache: when the model generates byte 501, the keys and values for bytes 1 through 500 are bit-for-bit identical to what they were when it generated byte 500.

This holds because a position's key and value depend only on that position and what preceded it. Appending byte 501 changes nothing about what byte 12 advertises or what it carries. Recomputing all 500 keys and values from scratch for every byte produced is therefore repeated work, and the repetition compounds: doing it for every generated byte turns a linear cost into a quadratic one.

The fix is to compute each position's key and value once, when the position is first seen, and store them. Every subsequent byte reads the store instead of rebuilding it. That store is the key-value cache, and the name is exactly what it holds.

The asymmetry in the name is worth stating, because it is a common point of confusion. Queries are not cached. A query is used once, at the moment its own position is computed, to interrogate everything before it. Once that comparison is done the query has no further use, and no future position ever consults it. Keys and values are consulted by every future position for the remainder of the conversation. Keys and values are the archive; queries are consumed on use.

Nothing about this is exotic. Every transformer in production does it, because the alternative is quadratic waste. The cache is the correct engineering answer to the mechanism attention specifies.

## What the cache costs

The cache is correct, and it has one property that cannot be engineered away: it grows by one entry per position, without bound, for as long as the conversation continues.

The arithmetic is straightforward from a model's shape. Take the shape of chat200m, the 270,510,336-parameter model in this programme: 16 layers, hidden size 1024, weights stored in fp16 at two bytes per number. A plain transformer of that shape stores, for every byte of context, one key and one value of 1024 numbers in each of 16 layers. That is 2 times 1024 times 2 bytes, which is 4 KB per layer, and 64 KB once all 16 layers are counted.

Sixty-four kilobytes per byte of context. Because the model is byte-level, one byte of context is approximately one character of conversation. Following that figure out:

- At 1,024 bytes of context, the sequence length these models train at, the cache is 64 MB.
- At roughly 8,300 bytes, about two pages of text, the cache reaches 541 MB, which is the size of the entire chat200m weight file. Past that point the conversation outweighs the model.
- At 100,000 bytes, a short book, the cache is 6.4 GB.
- At 1,000,000 bytes, the cache is 64 GB.

This is arithmetic from a published shape rather than a measurement of a model built here, since the model built here does not carry an unbounded cache. It is the cost that was avoided, stated in the units the avoidance is measured in.

Size alone understates the problem, because generation on this class of hardware is limited by memory bandwidth rather than by arithmetic. The serving work measured the decode matvec streaming weights at about 80 GB/s on a single fp32 core, close to the machine's single-core memory ceiling. When the bottleneck is how fast numbers can be moved out of memory, the cost of a data structure is set by how many bytes must be streamed, and every generated byte requires comparing the new query against every cached key.

At 100,000 bytes of context, streaming a 6.4 GB cache at that rate costs roughly 80 milliseconds of pure memory traffic for each byte produced, on top of the weights. At a million bytes it is close to a second per byte, assuming the cache fits in memory at all. The model does not fail at some threshold. It degrades continuously, getting slower with every byte it writes, and the degradation is worst precisely when the conversation has become long enough to be worth continuing.

This is the cost that makes long context expensive to serve everywhere, independent of who is serving it. It is a property of the mechanism, not of any particular implementation.

## What a constant state does instead

The recurrent alternative changes what is stored rather than how it is stored.

Instead of retaining every position's key and value, each head keeps one fixed-size matrix. Every byte updates that matrix by folding in the current byte's contribution and multiplying the existing contents by a learned decay, so recent information is written in while older information fades at a rate the model learned rather than one chosen by hand. The engine enforces the update in place, with no allocation inside the decode loop, which is how the property survives the trip from training code to serving code.

The size of that matrix is a constant of the model. It does not depend on position, on conversation length, or on anything the user does. A conversation that has run for a thousand bytes uses precisely the same state as one that has run for ten, and the same as one that has run for ten million.

The hybrid trunk is not free of attention, and the distinction matters for anyone reading the footprint honestly. It keeps local attention on every byte, and that component does carry a cache. The difference is that the local cache is capped by the fixed sequence length rather than by the length of the exchange, so it reaches a ceiling and stops. Everything recurrent is flat from the first byte. The total is bounded, which is the property that matters, rather than zero, which would be a stronger claim than the architecture supports.

Measured on the served model at hidden size 768 and sequence length 1024, the complete decode state is about 28 MB in fp32. That total is the sum of three bounded pieces: four attention caches, each capped by the sequence length; twelve recurrent state matrices, each a fixed block that does not grow; and the convolution rings. None of the three grows with how long the conversation runs.

Twenty-eight megabytes, fixed, against a figure that passes 6 GB somewhere inside a single long document.

## What the measurements say

Every number below is recorded elsewhere in this programme and is repeated here so the tradeoff can be read in one place.

**Training quality is close to a wash.** On the shared 10M-parameter rig, the constant-state arm finished at 0.9900 bits per byte against the attention baseline's 0.9990, and led at 117 of 120 matched evaluation points. It also ran 18 percent slower per step on an unoptimised implementation, which places it at parity rather than ahead once the comparison is made at equal wall-clock. The reportable result was parity, and it was reported that way. The constant-state design was adopted despite winning nothing on the training curve, because the decode property was the point.

**The composed architecture did win.** The hybrid trunk, which runs constant-state recurrence on a patched stream, posted 0.9707 bits per byte, the best of every arm tested, ahead of both the patched parent at 0.9776 and the recurrent parent at 0.9900. The margin over the patched parent sits inside the noise band this programme requires for external claims and carries a second-seed caveat.

**Serving is fast and its footprint is fixed.** The hand-written engine runs the architecture at roughly one millisecond per byte on a single M3 Ultra core, produces output identical letter for letter to the reference implementation, and holds decode state at about 28 MB regardless of conversation length. The per-byte cost splits cleanly: an ordinary byte costs about 56 million operations, and a boundary byte costs about 244 million, because the twelve recurrent global blocks fire only at boundaries.

**The fixed state forgets, and the curve is steep.** Measured on the chat80m checkpoint with streaming carry enabled, recall of a planted fact is 0.92 at a distance of about 190 bytes, 0.25 at about 480 bytes, and 0.00 at 2k, 8k, and 32k alike. The flat zero past the local window is not slow decay but a pathway carrying nothing, and the collapse inside the window is a separate failure from the collapse outside it.

**Compressing retrieved text back into the state does not work.** The direct attempt to give the fixed state exact recall, by retrieving the needed span and folding it into the recurrent state, was falsified. The state saturates and cannot hold verbatim spans. Exact recall has to come from an addressable read, not from compressing that read back into constant memory.

**Reading from an external store does work.** A 270M model conditioning on a 51-million-byte external store, roughly 50,000 times its own window, recalls a planted fact at 0.97 and answers from it at 0.75, against 0.02 with no retrieval. On natural free-form questions rather than templated ones the numbers are considerably lower, with top-1 retrieval at about 0.32 and end-to-end grounded answering at 0.14 against a bare rate of 0.01, and that ceiling is set by retrieval precision rather than by the model.

## What it means

### For a general reader

A standard chatbot keeps a running transcript of everything it has read, in a form only it can use, and consults the whole transcript every time it produces a single character. The transcript never stops growing. That is why long conversations cost more to run than short ones, and why they slow down as they go.

The architecture described here keeps a fixed-size notepad instead. Each new character is folded onto the notepad and the older writing fades slightly to make room. The notepad is the same size after a million characters as after ten, so the millionth character costs what the first one did.

The tradeoff is honest and it is not small. A transcript preserves everything exactly. A notepad of fixed size has to discard, and what it discards is chosen by the model rather than by anyone who can inspect the choice. Broad meaning survives that compression. Exact wording frequently does not, which is why a fact planted a few hundred characters back is recalled about a quarter of the time and one planted two thousand characters back is not recalled at all.

The response to that limitation is to stop asking the notepad to do a job it cannot do. When something has to come back word for word from far away, it is stored outside the model and looked up, which is measured to work over stores of 51 million bytes.

### For the machine-learning reader

Three points carry the weight.

First, the cache is not an implementation detail that a better runtime could remove. It is what the mechanism specifies. Attention defines the output at a position as a function of every earlier position's key and value, so those keys and values must exist somewhere when the position is computed. Caching them is the cheap way to satisfy that requirement, and recomputing them is the expensive way. The growth is in the definition.

Second, the constant-state trunk was adopted on a training result of parity, not victory. That ordering is the point worth taking from this programme rather than the loss numbers. A lever that buys nothing on the training curve and changes the deployment cost structure is still worth adopting, provided the deployment claim is stated as a structural property and the quality cost is measured rather than assumed. The quality cost here was measured, and it is the forgetting curve.

Third, the two failure modes visible in that curve are distinct and imply different fixes. Recall collapsing from 0.92 to 0.25 inside the local attention window, where the fact remains directly visible, points at training pressure: ordinary conversational data never demands long-range exact recall, so the ability is never trained. Recall sitting at exactly 0.00 across 2k, 8k, and 32k, with no decay gradient between them, points at mechanism: the pathway carries nothing across window boundaries. Neither has been closed.

## Honest limitations

This paper introduces no new measurement, and it should not be read as evidence for anything. It is an explanation of a mechanism and a gathering of results reported in full elsewhere, with the sources named in the evidence trail.

The cache figures for a plain transformer are arithmetic derived from a model shape rather than a measurement of a system built here. They are checkable from the shape and they are stated as the cost avoided, not as the cost paid.

The constant-state advantage is a claim about memory and about the cost of producing each byte. It is not a claim about quality, and this programme has not shown that a fixed-size state holds quality at long context. The measured position is the opposite: the state forgets steeply, no mechanism tested so far has flattened that curve, and the retrieval path that works around the limitation is bound by retrieval precision on natural questions, which is currently about 0.32 for top-1.

The single-machine compute wall recorded in the efficiency work is untouched by anything here. A bounded decode footprint makes a small model cheap to serve for a long conversation. It does not make a small model competitive with a large one.

## Evidence trail

- Architecture, the E3 and E5 arms, and the training numbers: [The composed efficiency stack](https://carpathian.ai/publications/composed-efficiency-stack); ledger entries in `successes.md` dated 2026-07-03 and 2026-07-04.
- Reference attention implementation and the cached decode path: `veritate_core/model.py` (`CausalSelfAttention`, `kv_cache_patch_attn`). Constant-state update: `veritate_core/model_recurrent.py`.
- Decode footprint, the 28 MB figure, the per-byte operation counts, and the memory-bandwidth measurement: [Serving a research architecture on a CPU](https://carpathian.ai/publications/cpu-serving-engine).
- The degradation curve and the two hypotheses it separates: [Long-context memory for constant-state models](https://carpathian.ai/publications/long-context-memory).
- The external-store results and the falsification of folding retrieved bytes into the state: [External addressable memory](https://carpathian.ai/publications/external-memory-retrieval); ledger entries in `successes.md` dated 2026-07-11 through 2026-07-13 and in `failures.md` dated 2026-07-13.
