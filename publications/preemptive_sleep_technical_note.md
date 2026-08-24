# Technical note: preemptive sleep — background self-improvement that costs a served request nothing

*Sam Malkasian, Carpathian LLC — 2026-08-23*

Result: on a deliberately weak box — an Intel i7-9700T, 8 cores, no hyperthreading, BIOS-locked to 800 MHz — a 270M byte-level model can train on its own conversations while it serves them, with **no measurable cost to the people talking to it**. Unyielding background consolidation costs a served request 2.5–3× throughput and roughly 200× first-byte latency. Suspending the training child for the duration of each request removes that cost entirely: serving under an active training run measures the same as serving on an idle box. Caveats up front: one box, one model scale, one architecture family, greedy decode, single-seed timings; the quality side of the loop (does the model actually get better) was validated separately in the E4 campaign and is not re-established here.

## The problem

A model that learns from its own experience has to train somewhere. On a workstation with a spare GPU this is uninteresting — training and serving use different silicon. On the hardware most people actually own, they use the same silicon, and consolidation is not a background task in any meaningful sense: it is a foreground task wearing a hat.

The measurement, on the box above, serving a 64-byte reply:

| condition | first byte | throughput |
|---|---|---|
| idle box | 13 ms | 18.3 ms/byte |
| consolidating, unyielding | 2,856–3,113 ms | 44.6–57.7 ms/byte |

The trainer child took 7 of 8 cores. A three-second wait before the first character is not a degraded experience, it is a broken one. The usual answers — nice the trainer, cap its threads, wait for an idle window — are all partial. Niceness and thread caps still leave the model streaming weights through a shared memory bus. Idle windows mean the model only improves on a box nobody is using, which on a single-user machine means it improves rarely and unpredictably.

## The mechanism

Two pieces, both small.

**A serving beacon.** Every generation the box serves already passes through one wrapper — the one that records the exchange to the experience log, on both backends and both routes. That wrapper marks a counter up on entry and down on exit. Counting rather than flagging matters: with two overlapping requests, the first to finish must not declare the box quiet.

**Suspend, not throttle.** When a generation starts, the sleep controller sends `SIGSTOP` to the training child; after a quiet window (default 5 s) it sends `SIGCONT`. The dependency runs runtime → consumer: the beacon exposes a hook, the controller registers on it, and the inference layer never imports the training layer.

The choice of `SIGSTOP` over renicing or thread-capping is the whole trick, and the reason is that **a suspend loses no work**. The process keeps its memory, its optimizer state, its position in the step. A step interrupted fifty times still costs exactly the CPU time it would have cost uninterrupted; only wall time stretches. That inverts the usual design pressure. The intuition on weak hardware is to shrink the step so an interruption wastes less — but if interruptions waste nothing, the right step is the *largest* one that fits, because larger steps amortize the fixed per-step optimizer cost (Muon's Newton-Schulz runs on every 2D weight every step regardless of batch) over more tokens.

## The measurement

Serving under an active consolidation run, same box and model:

| condition | first byte | throughput |
|---|---|---|
| idle box | 13 ms | 18.3 ms/byte |
| consolidating, preemptive | 12–23 ms | 17.9–18.2 ms/byte |

First-byte figures are steady state. The first request after an idle gap costs more in both conditions (295 ms idle, 143 ms preemptive) as caches warm; the comparison above is like for like.

Verified at the OS level rather than inferred from latency. Sampling the child's process state every 150 ms across a request gives:

```
RNl RNl TNl TNl TNl TNl TNl TNl TNl TNl TNl TNl TNl TNl ...
```

`R` → `T` the moment the request arrives, for its whole duration.

One bug worth recording, because it is the kind that measures as working and is not. The controller's watcher chose its poll interval *before* sleeping, so a suspend landing mid-period left the child stopped for up to a full watch period after the box went quiet. On a box receiving a message every 30 s the run would have been suspended essentially forever — visibly "working" in every latency measurement, while making no progress at all. The fix is to decouple the cadences: poll for the resume check on a short interval, run the full pass on the long one.

## What we found while looking

Isolating the contention cost required attributing every millisecond of decode, which turned up three results worth more than the original question.

**Decode on this trunk is bimodal, and the split is a word boundary.** The model file carries a 256-entry `boundary[]` table. When the byte fed into a step is a boundary byte — whitespace, punctuation — that step runs the GLA recurrent global-block stack, which every other step skips. Prefill amortizes one weight stream over every boundary byte in a chunk; single-byte decode cannot. So **every word-initial byte pays a full global-block weight stream**. Measured over 128 greedy bytes: non-boundary p50 **10.07 ms**, boundary p50 **50.11 ms**, 24 of 127 bytes, **54% of total decode time**. The two classes spell themselves out. Fast bytes: `here re any ifferent ays o earn istory.` Slow bytes: `a m d w t l h O w i t s h t a s p`. The sentence is "There are many different ways to learn history. One way is to study history through a scholarly perspective."

**The serving stack is not the bottleneck, and it is not close.** The instinct on seeing 18.3 ms/byte end-to-end against a 9.9 ms/byte forward pass is to blame the Python, the JSON, the per-byte SSE frame. Engine-direct instrumentation puts the entire Python pipeline — frame read, struct parse, event dict, `json.dumps`, SSE write, socket — at **0.02 ms/byte: 2 ms out of 2,198, or 0.1%**. An A/B against the non-streaming path measures the same 18.6–21.5 ms/byte, confirming it. Coalescing SSE frames, the obvious optimization, would have bought nothing. The gap was entirely the boundary steps above.

**Quantization attacks the boundary step directly, because that step is bandwidth.** Identical weights, 128 greedy bytes: fp16 **15.17 ms/byte** vs int8 **10.42 ms/byte**, a **1.46×** speedup; end to end, int8 plus eight pinned engine workers took a reply from 17.5 to 10.2 ms/byte, **1.72×**. The win concentrates exactly where the theory says it should — the boundary step drops 41.34 → 30.13 ms. It is not free, though: we had assumed int8 was greedy-byte-identical at this scale, and it is not. Exporting fp16 and int8 from one checkpoint and decoding greedily, 1 of 5 replies matched; the rest diverged mid-reply, which is what a single flipped argmax does under greedy decoding. The speedup is real, the free-lunch framing was not, and shipping int8 needs a quality evaluation rather than a parity check.

**Thread auto-calibration can stop a rung early and look like a hardware cap.** The engine sizes its worker pool by climbing a 1, 2, 4, … ladder while each rung beats the previous by at least 13%, timing *non-boundary* steps only. On this box 4 threads measure 16.58 ms/byte and 8 threads measure 14.85 — a 10.4% gain the knee rejects. The visible symptom was a box that never used more than half its cores while generating, which reads as a hardware limit and is not one. The pick is also unstable: two models on the same box cached 8 and 4 respectively, because the test compares only against the immediately previous rung and run-to-run noise straddles the threshold. A knee measured per step class, rather than one fixed constant timed on the cheap class, is the open follow-up.

## Why this matters beyond one box

The standard framing for continual learning is a scheduling problem: find the idle window, take the GPU, give it back. That framing fails on the hardware most models will actually run on, because the window never reliably arrives.

Preemption replaces the scheduling problem with a priority problem. The model does not need an idle box; it needs to be interruptible within milliseconds and to lose nothing when interrupted. Both fall out of the OS for free, and the result is that consolidation can run *continuously* — in the gaps between keystrokes, during a long read, overnight — instead of waiting for a window a single-user machine rarely provides.

The honest limits: this is one 8-core CPU box at a locked clock, one model family, and it measures latency rather than learning. It says background consolidation can be made free to the user. It does not say the consolidation is good, and a preempted run on a busy box still takes far longer in wall time than the same run on an idle one — the CPU time is conserved, not reduced.

## Reproducing

The controls are `sleep_preempt` (on by default), `sleep_resume_s` (quiet window before resuming), `sleep_reserve_cores` (held back from the trainer's BLAS budget, covering the window before a suspend lands), and `sleep_nice`. Suspend/resume and niceness go through psutil, so the mechanism is the same on Linux, macOS, and Windows, where psutil maps niceness onto priority classes. Without psutil the run proceeds unyielding and says so in the log.
