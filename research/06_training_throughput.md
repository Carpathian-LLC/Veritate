# Amortizing the optimizer step: pre-launch throughput tuning for the 270M pretrain

How an affine, fixed-cost-per-update model of training time turned a budgeted four-to-five-day
run into a measured two-day run, with no loss of quality per unit of text read.

## Abstract

Training a neural network is a loop: read a batch of text, compute how the weights should
change, apply that change, repeat. The step that applies the change (the "optimizer step")
carries a cost that does not depend on how much text the batch contained. Updating a million
weights costs the same whether the gradient behind that update was computed from a thousand
characters or a hundred thousand. If each optimizer step is made to process more text, that
fixed cost is spread over more useful work, and the machine spends a larger fraction of its time
computing the thing we actually want (better weights) rather than on the per-step bookkeeping
that surrounds it. This is amortization, the same idea that makes a bulk purchase cheaper per
unit than a single one.

Before committing a single M3 Ultra to a multi-day pretraining run of a 270-million-parameter
model, we measured the exact shape of this trade-off on the real model and chose the operating
point deliberately rather than by intuition or by scaling a guess from a smaller run. The method
is a two-parameter cost model, fit to a sweep of configurations, that predicts a run's speed from
one number: how many tokens each optimizer step processes. The run came in at roughly twice the
throughput a naive parameter-scaling budget predicted (about two days rather than four to five),
and a token-matched comparison against an earlier, smaller model shows that the added speed cost
nothing in quality per unit of text read. This note gives the cost model and its derivation, the
measurements behind it, and the honest reason the fastest available setting was not the one we
chose: past a point, buying more speed means buying it with weight updates, and weight updates
are themselves a quality variable.

Terms are defined at first use. "Throughput" is tokens processed per second; a "token" here is
one byte, since the model is byte-level with a vocabulary of 256, so token counts and character
counts coincide. "Validation loss" is measured in bits per byte, lower being better; it is the
model's compressed cost, in bits, to predict each held-out byte. "bf16" is the 16-bit
brain-float format used for the forward and backward compute. "Muon" and "AdamW" are two
optimizers, the rules for turning a gradient into a weight update. "Latency" is the wall-clock
time one optimizer step takes; "step time" is the same quantity, which we write as t_step.

## What we did

### The measurement principle

The governing principle is a refusal: do not extrapolate throughput from a differently sized
model, a different precision, or a different optimizer, and then budget days of compute against
the guess. Parameter count, sequence length, chunking, numeric format, and optimizer each move
the timing, and they do not move it independently or linearly. So we measure on the exact
architecture, shape, numeric precision, and optimizer that will run, before the run starts.

This is affordable because of a fact that is easy to overlook: latency does not depend on the
values in the weights. A dense floating-point multiply-accumulate takes the same time regardless
of whether its operands are trained values or random noise; the shapes of the tensors, the
dtype, and the graph of operations fix the compute, and the numbers flowing through it do not.
(bf16 autocast keeps the arithmetic away from the denormal regime where value-dependent slowdowns
can appear, so the invariance holds in practice as well as in principle.) A random-weights copy
of the real model is therefore a faithful stand-in for timing: it exercises precisely the kernels
the trained model will, at precisely the cost, without spending a single real training token to
find out. We can characterize a two-day run in minutes.

### The model under test

The model is the `veritate_200m` manifest at the hybrid trunk, paper 1's winning architecture
([the composed efficiency stack](01_efficiency_stack.md)): 16 recurrent global blocks wrapped in
2 local-attention encoder and 2 local-attention decoder blocks (20 blocks total), hidden size
1024, feed-forward 4096, 16 heads, sequence length 1024. At the hybrid trunk this manifest
instantiates 270,510,336 parameters. The manifest's nominal "202M" is a dense estimate; the
hybrid trunk carries about 34 percent more, which is why we name the object by its true parameter
count throughout and treat "270M" as the model under test. All timing was on the MPS backend
under bf16 autocast, forward plus backward, reported as the median of the timed steps after a
warmup, with the GPU confirmed otherwise idle so no other work contended for it.

### The three levers, and why the chunk knob is the interesting one

Three knobs set the number of tokens processed per optimizer step, which we call T:

    T  =  B * S * K

where B is the batch size (sequences per step), S is the sequence length (tokens per sequence),
and K is the number of chunks per step (`n_chunks`). All three raise T, but they pay for it
differently, and the differences are the whole design space.

Batch B is the obvious lever, and the expensive one: each of the B sequences in a step needs its
own forward activations held in memory for the backward pass, so activation memory grows roughly
linearly in B. Raising B raises T but pushes against the memory ceiling.

Sequence length S trades context for tokens per step: a longer S lets the model attend across
more text at once, but it changes what the model is learning (its effective context window), so
S is not a free throughput dial. It is an architecture decision wearing a throughput costume, and
in our sweep it bought little (about 4.7 percent, detailed below).

The chunk knob K is the interesting one, and it is nearly free. With K greater than one, each
step streams K consecutive length-S chunks through the constant-state recurrent trunk. The trunk
is recurrent: it maintains a fixed-size state that summarizes everything seen so far, and that
state is carried across the chunks, so the hidden state at the end of chunk i initializes chunk
i+1. A step with K chunks therefore lets each row see an effective sequence of length S*K, stitched
together through the carried state. The key move is what happens to the gradient: it is truncated
to a fixed window of `bptt_window` chunks (here 2). Only the last `bptt_window` chunks retain
their activations for the backward pass; the earlier chunks are run forward to advance the state
but their activations are discarded and no gradient flows back through them. This is truncated
backpropagation through time (TBPTT) over a chunked sequence, the standard device for training a
recurrent model on sequences longer than one can afford to differentiate end to end.

The consequence is the reason chunks are cheap. Activation memory, the dominant cost of the
backward pass, is bounded by `bptt_window` chunks and is therefore independent of K. Raising K
from 1 to 4 quadruples the tokens per step while the retained activations still cover only 2
chunks. K buys tokens per step at almost no incremental memory. Batch buys them at a linear
memory cost; sequence length buys them by changing the model's context; chunks buy them for
free, up to the compute-bound ceiling we derive next. This asymmetry is what makes the tuning
tractable: there is a lever that raises T without spending the two resources (memory and context)
that the other levers spend.

### The affine cost model, derived

We then posited the simplest cost model that could fit the sweep, an affine one:

    t_step  =  c0  +  c1 * T

The step time decomposes into two kinds of work. One kind scales with the number of tokens
processed: the forward and backward compute is a fixed amount of matmul per token, repeated T
times, so it contributes a term linear in T. The other kind is paid once per optimizer step
regardless of how many tokens the step saw: the weight update touches the same fixed set of
weights every step, and the dispatch overhead of assembling a step (kernel launches, the reduction
that forms the gradient, the Python that drives it all) is likewise per-step, not per-token. So it
contributes a constant term. An affine model is exactly the minimal model that carries both a
token-linear term and a token-independent term, and no more. We could add a T-squared term to
capture, say, memory-bandwidth saturation at large T, but the linear fit holds across the whole
measured range to within about 2 percent, so no higher-order term is warranted; the data do not
ask for one.

The two coefficients have direct physical readings:

- c1 is the marginal cost of one token's forward-plus-backward compute: the time to push one more
  token through the model and pull its gradient back. On a compute-bound accelerator this is
  realized at a roughly constant rate, so the marginal time per token is a constant of the model
  and the chip. It is measured in microseconds per token.
- c0 is the fixed per-step overhead: everything paid once per optimizer step irrespective of T,
  dominated by the weight update itself. Because the number of weights does not depend on T, the
  update cost does not either. It is measured in seconds per step.

Throughput follows immediately:

    R  =  T / t_step  =  T / (c0 + c1 * T)  =  1 / (c1 + c0 / T)

Two properties of R matter. It is monotonically increasing in T (its derivative with respect to T
is c0 / (c0 + c1*T)^2, which is positive whenever c0 is), so more tokens per step is always at
least weakly faster. And it approaches a ceiling: as T grows, the term c0/T vanishes and R
approaches 1/c1 from below, never reaching it. The fixed cost c0 is amortized over ever more
tokens but never to zero. The sweep below fits c0 and c1 and locates the launched run on this
curve.

## What we measured

The sweep varies one lever at a time, then pushes each to its practical limit, then fits the
affine model and checks it on a held-out point. Every figure below is a direct measurement in the
conditions described above (MPS, bf16 autocast, forward plus backward, median over timed steps,
GPU idle). Nothing is extrapolated or invented.

### The batch scan (fixed chunks K = 2, Muon), the first lever

Holding K = 2 and sweeping batch size B, throughput climbs with T while step time and peak memory
climb more slowly:

- B = 4: T = 8,192 tokens/step, throughput 3,885 tok/s, step 2,108 ms, peak memory 3.6 GB.
- B = 8: T = 16,384, throughput 6,429 tok/s, step 2,549 ms, peak memory 4.1 GB.
- B = 12: T = 24,576, throughput 8,253 tok/s, step 2,978 ms, peak memory 3.9 GB.
- B = 16: T = 32,768, throughput 9,709 tok/s, step 3,375 ms, peak memory 3.7 GB.
- B = 24: T = 49,152, throughput 11,504 tok/s, step 4,273 ms, peak memory 4.3 GB.
- B = 32: T = 65,536, throughput 12,887 tok/s, step 5,086 ms, peak memory 4.0 GB.

The signature of amortization is visible directly in these rows. From B = 4 to B = 32 the tokens
per step rise eightfold (8,192 to 65,536) but the step time only about 2.4-fold (2,108 ms to
5,086 ms), because most of the extra tokens ride on compute the small step was already paying c0
to launch. Throughput therefore rises from 3,885 to 12,887 tok/s, roughly 3.3-fold, and the rise
is decelerating: each doubling of B adds less throughput than the last, exactly the diminishing
return the affine model predicts as c0/T shrinks. Peak memory stays in a narrow band (3.6 to 4.3
GB) because the recurrent trunk's dominant memory is its state and the bounded activation window,
not the batch, so B is not memory-bound here; it is amortization-bound.

### The chunk scan (fixed batch B = 24, Muon), the cheap lever

Holding B = 24 and sweeping the chunk count K exercises the free lever:

- K = 1: T = 24,576, throughput 8,445 tok/s, step 2,910 ms, peak memory 4.4 GB.
- K = 2: T = 49,152, throughput 11,504 tok/s, step 4,273 ms, peak memory 4.3 GB.
- K = 4: T = 98,304, throughput 14,150 tok/s, step 6,947 ms, peak memory 4.2 GB.

The B = 24, K = 2 row is shared with the batch scan (11,504 tok/s, 4,273 ms, 4.3 GB) and serves
as the anchor tying the two scans together. Raising K from 1 to 4 lifted throughput 68 percent
(8,445 to 14,150 tok/s) while peak memory actually fell slightly, from 4.4 to 4.2 GB. This is the
truncated-BPTT result made concrete: quadrupling the tokens per step cost nothing in memory
because only the `bptt_window` = 2 chunks are ever retained for the backward pass, so the memory
footprint is set by the window, not by K. Chunks are the throughput lever that does not push on
memory at all.

### Pushing the levers to their limits, and the two binary levers

Beyond the one-lever-at-a-time scans, we measured each lever near its practical limit and the two
on/off choices (optimizer, activation checkpointing):

- B32 K4 Muon: throughput 15,290 tok/s, step 8,573 ms, peak memory 3.6 GB. Faster than B24 K4 by
  raising batch on top of full chunking.
- B48 K4 Muon: throughput 16,063 tok/s, step 12,240 ms, peak memory 4.3 GB. The fastest Muon
  point measured, reached by pushing batch further still.
- B24 K4 AdamW: throughput 16,421 tok/s, step 5,986 ms, peak memory 5.8 GB. The fastest point in
  the grid, but on the other optimizer, and at a higher memory cost.
- B24 K2 Muon with activation checkpointing: throughput 9,603 tok/s, step 5,119 ms, peak memory
  5.8 GB. Checkpointing recomputes activations to save memory but here neither saved memory nor
  helped speed, so it was not used.
- B24 K4 at sequence 512 (2,048 tokens per row): throughput 12,148 tok/s, step 4,046 ms, peak
  memory 3.7 GB. Halving the sequence and letting chunking make up the row length.

### Fitting the affine model

The affine model fits the whole grid to within about 2 percent on held-out configurations, with

    c1 = 54.5 microseconds per token       (the marginal forward-plus-backward compute)
    c0 = 1.6 s for Muon,  0.7 s for AdamW  (the fixed per-step overhead)

As a check on a point the fit can be read against, the chosen configuration B24 K4 predicts

    t_step = 1.6 + 54.5e-6 * 98,304 = 6.96 s

against a measured 6.95 s, and a predicted throughput of 14,128 tok/s against a measured 14,150.
The prediction lands within a fifth of a percent of the measurement, which is the whole claim of
the method: a two-parameter model, fit once, forecasts a multi-day run's speed before it starts.

The ceiling implied by c1 is

    1 / 54.5 microseconds = 18,350 tok/s

This is the compute-bound limit no scheduling choice can beat on this model and this chip. It is
not a number any batch, chunk, or optimizer setting can improve, because it is what one token's
compute costs; the levers only govern how closely a run approaches it by amortizing c0 away.

### The launched configuration

The launched configuration is B = 24, S = 1024, K = 4: T = 98,304 tokens per optimizer step,
14,150 tok/s, which is 77 percent of the 18,350 tok/s ceiling. At that rate the run reaches its
2.006 billion token budget in 20,400 steps and about two days of wall clock. The live run has
held the predicted speed: sampled at step 6,200, throughput reads 14,100 to 14,230 tok/s even
with unrelated evaluation jobs sharing the machine, because those jobs contend for CPU cores
while the trainer owns the GPU, so they do not slow the step.

### Why not the throughput maximum

The obvious objection is that we left speed on the table. B48 K4 is measurably faster (16,063
tok/s against the chosen 14,150), and B24 K4 on AdamW is faster still (16,421). We rejected both,
because the number of optimizer steps is itself a quality variable, and past a point buying
throughput means selling weight updates.

The arithmetic is direct. Throughput sets the wall clock for a fixed token budget, but the number
of optimizer steps is the budget divided by T, and T is what the fast configurations raise.
Doubling the batch to B48 doubles T, so at the fixed 2-billion-token budget B48 K4 would reach the
end in only about 10,200 steps, roughly half the weight updates of the chosen point. The chosen
B24 K4 keeps roughly 20,000 updates. This matters because a weight update is where learning
actually happens: too few updates per token means the optimizer takes fewer, larger, coarser
steps toward the solution, and the gradient noise that a smaller effective batch supplies (noise
that aids exploration and generalization) is averaged away. Chasing the throughput ceiling by
enlarging T therefore trades update count for speed, and too few updates per token degrades
convergence. B24 K4 is the point that keeps the update budget near 20,000 while still sitting at
77 percent of the compute-bound ceiling: near the fast end of the curve, but not so far along it
that the run starves itself of updates. The throughput maximum is the wrong objective; the right
one is the fastest point that still spends enough updates.

### Did the speed cost quality per token

The remaining worry is subtler: even at a healthy update count, could the amortization itself
have degraded learning? A large effective batch can wash out the gradient signal, and carrying
recurrent state across chunks could in principle teach the model something harmful. The
token-matched test is designed to catch exactly this. Both the 270M run and the earlier 80M run
write validation loss on the same benchmark, so we can compare them at matched token counts, in
bits per byte (lower is better):

- At about 200M tokens seen: the 80M reads 1.085, the 270M reads 1.044.
- At about 395M tokens seen: the 80M reads 0.993, the 270M reads 0.987.
- At about 590M tokens seen: the 80M reads 0.954, the 270M reads 0.937.

The 270M run is ahead at every matched point and widening, and it earns that lead under a harder
handicap. Only 37 percent of the 270M's training diet overlaps the validation corpus, against the
80M's 68 percent, so a smaller share of what the larger model reads resembles what it is graded
on. A model with less overlap should, all else equal, look worse at matched tokens, not better;
the 270M is winning while carrying the heavier handicap, so the measured gap understates its true
advantage rather than flattering it. The logic of the test is the direction of the failure it
would have produced: had the amortization tuning introduced an optimization pathology (an
effective batch large enough to wash out the gradient signal, or a state-carry across chunks that
hurt learning), it would show here as a validation curve at or below the smaller model at matched
tokens. It does not. The fast configuration did not break learning.

## What it means

### Amdahl's law for the training step

The cost model is Amdahl's law for the training step. c0 is the serial fraction, the part of a
step that must be paid once per update and cannot be amortized within that step. The token
compute is the parallel fraction. T is the only lever against the serial cost, and as T grows the
serial fraction's share of the step time, c0 / (c0 + c1*T), shrinks toward zero while the achievable
speedup saturates toward the compute-bound ceiling 1/c1. This is precisely the shape Amdahl's law
predicts: a fixed serial cost caps the speedup, and past the knee of the curve further investment
buys diminishing returns. Reading a multi-day run's speed off this curve before launch, rather
than discovering it after, is the whole method. The curve is cheap to measure (a random-weights
copy, minutes of timing) and it turns a budgeting guess into a forecast good to a fraction of a
percent.

Three consequences are worth stating plainly.

### The chunk knob is the cheapest throughput there is

The chunk knob is the cheapest throughput available, because it raises T without spending either
of the two resources the other levers spend. It does not raise memory, which is bounded by the
truncation window `bptt_window` and is independent of K, and it does not change the model's
context length, which stays at S. Batch raises memory; sequence length trades context for
tokens per step and here bought only 4.7 percent; chunks buy throughput almost for free until the
ceiling. Where a run has a constant-state recurrent trunk to carry state across chunks, chunking
is the first lever to reach for and the last to run out.

### Throughput is a position on a curve, not a property of parameter count

The 270M model is not the 2.2-times-slower object that naive parameter scaling from the 80M
predicts, and the reason is the curve, not the hardware. The earlier 80M ran at T = 24,576 (B12,
K2), a poorly amortized point far down the slope where c0/T is large and throughput sits well
below the ceiling. The 270M runs at T = 98,304, four times higher and much nearer the ceiling,
where the same fixed cost is spread over four times the tokens. The larger model is heavier per
token (a larger c1) yet runs at a throughput that scaling by parameter count alone would not
predict, because throughput is set by where a run sits on its amortization curve at least as much
as by its parameter count. The same chunk lever, applied to the 80M, would have moved it up its
own curve and sped it up too. The lesson generalizes: before comparing the throughput of two
models, ask where each sits on its curve, because a poorly amortized large model can out-throughput
a poorly amortized small one, and the difference is scheduling, not size.

### The optimizer choice is a fixed-cost story that pays back in tokens

The optimizer choice is a fixed-cost story. Muon's larger c0 (1.6 s against AdamW's 0.7 s) is the
Newton-Schulz orthogonalization it applies to the 2D weight matrices each step, a matrix operation
whose cost depends on the weight-matrix dimensions and not on the batch, so it is a pure c0 term
that amortizes into the ceiling like any other fixed cost as T grows. This is why Muon's penalty
shrinks with amortization: at large T the orthogonalization is a smaller and smaller share of the
step. Muon is 16 percent slower in raw wall clock at K = 4, but it reaches equal validation loss
in about 1.6 times fewer tokens (measured at 10M, paper 1). Netting the two effects (1.6 times
fewer tokens to a target, each token 1.16 times slower to process) leaves Muon reaching that target
in roughly 0.73 of AdamW's wall clock, a 1.4-times win, which is why it was kept even though AdamW
posts the higher raw throughput in the grid. Raw throughput is the wrong scoreboard for an
optimizer; wall clock to a quality target is the right one, and on that scoreboard the slower-per-step
optimizer wins.

### Honest limitations

The throughput numbers are direct measurements and the affine fit is tight (within about 2 percent
on held-out configurations, and within a fifth of a percent on the chosen point), so the timing
claims stand on measurement. The weaker claims should be named as such.

The optimizer quality trade rests on a single-seed 10M-scale result assumed to hold at 270M. That
the 1.6-times token advantage persists at 27 times the scale is an assumption, not a measurement
at this scale; a single seed cannot rule out that the advantage narrows or that the number moves
with model size.

The quality-per-token comparison is one run per model with different corpus mixes. It confounds
size with mix, so it validates the specific claim it was built to test, that the fast
configuration did not break learning, and does not establish any scaling law between the two
models. The handicap direction (the 270M winning under lower val-mix overlap) makes the
qualitative "no pathology" conclusion robust, but it does not turn one run per model into a
controlled experiment.

Latency was measured on random weights (values do not affect timing, as argued above) on the M3
Ultra specifically. Other hardware sits on the same affine curve with its own constants c0 and c1;
the shape transfers, the numbers do not, and any port would remeasure both before budgeting.

Finally, the ceiling 1/c1 = 18,350 tok/s is a property of this model's per-token FLOPs and this
chip's realized bandwidth. It is the wall this tuning approaches, not one it moves. Amortization
buys the distance from the current operating point up to that wall; it does not, and cannot, push
the wall back. Moving the wall is a different problem (fewer FLOPs per token, or more realized
bandwidth), addressed elsewhere in this collection and not by scheduling.

## Evidence trail

- Sweep grid, raw numbers, and the launch decision: `worklog.md`, section dated 2026-07-08
  (chat200m launch), and the 2026-07-07 v13 round 2 section for the sibling latency work.
- Optimizer byte-efficiency (Muon 1.6x at 10M): paper 1
  ([the composed efficiency stack](01_efficiency_stack.md)); `successes.md`, entry dated
  2026-07-03 (E1).
- The hybrid trunk being timed: paper 1; `successes.md`, entry dated 2026-07-04 (E5).
- Live throughput and validation columns: `models/chat200m_200m/train.csv`.
- Token-matched 80M baseline: `models/chat80m_80m/train.csv`; `successes.md`, entry dated
  2026-07-05.
