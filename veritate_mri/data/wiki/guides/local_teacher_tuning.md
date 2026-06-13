---
title: "Tuning a local teacher (Ollama) for synthetic data"
date: 2026-06-12
tags: [teacher, synth, ollama, performance, troubleshooting]
summary: Synthetic data generation sends many requests in parallel to your teacher model. On a local server (Ollama, LM Studio, llama.cpp) throughput and reliability depend on two settings, the server's parallelism and the dashboard's concurrency. This page explains how to set both, and how to free GPU memory from the dashboard.
---

Synthetic corpus generation fans out many requests to the teacher model at once. When the teacher is a **local** server, two settings decide whether that fan-out is fast, slow, or fails:

1. **The server's parallelism** (how many requests it runs at the same time).
2. **The dashboard's teacher concurrency** (how many requests it sends at the same time).

If the dashboard sends far more than the server can run in parallel, the extra requests queue. A deep queue is slow, and a request that waits too long is dropped. The two numbers should roughly match.

## The fast path (recommended)

For a small teacher (7B to 32B) on a machine with plenty of memory:

1. Set the server to run several requests in parallel.
2. Set the dashboard teacher concurrency to the same number.

A value of **8 to 16** is a good starting point for a 7B to 13B model. Larger models (70B and up) decode slower and use more memory per parallel request, so use **2 to 4**.

## Setting Ollama parallelism

Ollama runs one request at a time unless you tell it otherwise, via the `OLLAMA_NUM_PARALLEL` environment variable. It must be set **before** Ollama starts, so set it then restart Ollama.

**macOS (Ollama app):**

```
launchctl setenv OLLAMA_NUM_PARALLEL 8
```

Then quit and reopen the Ollama app.

**macOS / Linux (running `ollama serve` yourself):**

```
OLLAMA_NUM_PARALLEL=8 ollama serve
```

**Linux (systemd service):**

```
sudo systemctl edit ollama
```

Add under `[Service]`:

```
Environment="OLLAMA_NUM_PARALLEL=8"
```

Then `sudo systemctl restart ollama`.

**Windows:** set `OLLAMA_NUM_PARALLEL` as a user environment variable, then restart Ollama from the tray.

Memory note: each parallel slot holds its own context (KV cache), so parallelism costs memory on top of the model weights. A 7B at 8-way parallel adds only a few GB; a 70B at 8-way adds much more. If the machine runs low on memory, lower the number.

Apple Silicon note: decoding is memory-bandwidth bound, so throughput climbs with parallelism but flattens out. Past roughly 8 parallel requests the gains are small, so there is little reason to go higher than the dashboard concurrency you actually use.

## Setting dashboard concurrency

In **Settings - Teacher Model**, set **max concurrency** to match `OLLAMA_NUM_PARALLEL`. Local providers default to a safe low value so generation works out of the box on an untuned server; raise it once you have set parallelism.

## Freeing GPU memory

Ollama keeps a model resident in memory after use so the next request is fast. To reclaim that memory without leaving the dashboard, use the **Free GPU memory** button next to Start Synthesis. It unloads every model the local server is holding. Stopping a synth job also unloads that job's teacher model.

## Reading the results

- **completed** - a sample passed the quality gates and was written.
- **skipped** - a near-duplicate of a sample already kept (deduplicated).
- **filtered by quality gate: reply length ...** - the model's reply was shorter or longer than the accepted range, so it was not kept. This is a filter, not a server error. If long replies are being dropped and you want them, raise the teacher max tokens and the max-chars gate.
- **timed out** - the request waited longer than the timeout, almost always because concurrency is far above the server's parallelism. Lower concurrency or raise `OLLAMA_NUM_PARALLEL`.

If a job sees a long run of failures it stops itself and logs the dominant reason, so a misconfiguration fails fast and clearly instead of grinding.
