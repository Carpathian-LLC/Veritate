# Smallest chatting model: 3 neuron experiments

Goal: smallest int8 chatting model + test 3 neuron ideas on one 50M ReLU base.

## Status (2026-06-02)
- [running] 50M ReLU base, 8000 steps (~7h left). 1.0B tokens.
- [queued]  auto-pipeline runs after base: int8, chat SFT, int8-chat, A, B, C, then measures.
- [done]    dashboard: "neuron balance (100% use)" training option; prune panel in Models tab.
- [done]    articles for LinkedIn -> research_articles/ (6 files).

A = prune dead neurons. B = L1 sparsify. C = neuron balance (100% use, new).

## RESULTS (done 2026-06-03). val = nats/byte, lower better.

Base 50M ReLU: val 0.989, 87.5% of neurons idle per token, 5.4% fully dead.

- int8 base: val 0.987 = basically FREE. (trained-in int8 works great)
- int8 chat: talks in full sentences, right chat format, but facts wrong (too small + tiny chat data). "chat shaped, not chat smart."
- A prune: a good model has almost no dead neurons -> auto-prune removes 0%.
  Forcing 33% smaller costs quality (0.99 -> 1.36). No free lunch.
- B L1 (more dead neurons): 87.5% -> 88.1% idle, same quality. GOOD direction.
- C balance (100% use): killed ALL dead neurons (5.4% -> 0%), same quality...
  but no quality gain AND it undoes the sparsity we want. So: it works, but
  it's the wrong goal. Sparsity wins, not 100% use.

Bottom line: all 3 tested. int8 = free. Sparsity (B) helps. Pruning (A) needs
sparsity first. 100% use (C) is a dead end for efficiency.

Also fixed a real bug: the model loader ignored ReLU and loaded every model as
GeLU (would mis-run any non-GeLU model, including in chat). Fixed + tested.

## ROUND 2 (3 follow-up tests, done 2026-06-03)

1. L1 -> prune (make it smaller for free): only ~3.7% smaller for free. Strong L1
   makes more zeros PER TOKEN but does NOT kill whole neurons, so little to prune.
   To really shrink you need "structured" sparsity (kill whole neurons), not this.
2. Sparse kernel (turn 87.5% zeros into speed): 8x fewer math ops are POSSIBLE,
   but 0x realized on this Mac - there is no kernel that skips zeros (the sparse
   paths were 800-1700x SLOWER). Needs a custom kernel. Dead end for now.
3. More chat data (2.3M -> 3.8M) + longer SFT: NO improvement. Still fluent but
   makes up facts. A 50M model is too small to be factual; more data won't fix it.
   Need a bigger model (200M+) or RAG for facts.

Net round 2: all 3 hit honest walls. Real next levers = structured sparsity (not
L1), a custom sparse kernel, and a bigger student (not more data).
