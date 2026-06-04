# Smallest chatting model: 3 neuron experiments

Goal: smallest int8 chatting model + test 3 neuron ideas on one 50M ReLU base.

## Status (2026-06-02)
- [running] 50M ReLU base, 8000 steps (~8h). 1.0B tokens (Chinchilla-optimal).
- [next] int8 (QAT) + chat SFT
- [next] A: prune dead neurons
- [next] B: L1 sparsify (more dead neurons, faster)
- [next] C: neuron balance (force 100% neuron use) <- new
- [done] dashboard: balance shows up as a training option; prune button.
- [done] articles for LinkedIn -> research_articles/

## Results (filled in as they finish)
- base val loss: TBD
- int8 chat works?: TBD
- A prune: TBD
- B L1: TBD
- C balance: TBD

Heads up: balance and prune are opposite ideas. Prune = remove dead neurons
(smaller). Balance = make all neurons used (no dead). We test both.
