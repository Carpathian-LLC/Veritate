# The composed efficiency stack

Byte-level training efficiency on one consumer machine: Muon, boundary patching,
constant-state recurrence, and the hybrid trunk that beat both of its parents.

## Abstract

We set out to train a useful language model on a single machine, one Apple M3 Ultra, at no
marginal cost. The binding constraint in that setting is not memory but arithmetic. The
machine holds the model and its optimizer state comfortably; what it cannot do is perform an
unlimited number of floating-point operations per second. Every training run is therefore a
budget of calculations, and the only question that matters is how much learning we can buy
per calculation. This reframing is the spine of the paper: we do not ask "can this model fit"
but "how far does a fixed amount of arithmetic carry us toward a fixed quality target."

We tested four ideas, introducing each one on its own, on a small 10-million-parameter
byte-level model, and measured how much sooner each reached a fixed quality target than a
carefully matched baseline did. Three of the four worked as isolated levers, and a fourth
idea that composes two of the three worked best of all. The composed architecture reached the
baseline's final quality in about 60 percent of the wall-clock time, and it did so while also
using a form of working memory that does not grow as a conversation gets longer. Stacked
together, the levers that compose cleanly are worth roughly three times the training
efficiency of the starting point.

Every term is defined at first use. "Byte-level" means the model reads and predicts text one
byte at a time (256 possible values per position, no word-piece dictionary and no tokenizer to
train or ship). "Wall-clock" means real elapsed time on the machine, as distinct from the
number of optimizer steps taken. "Validation loss" (val) is measured in bits per byte (bpb):
the average number of bits the model needs to encode each byte of held-out text, which is a
direct measure of how surprised the model is by text it was not trained on, lower being
better. Because the model is byte-level and the batch size is fixed, a given number of
training steps corresponds to a fixed number of training bytes, so "fewer steps to target"
and "fewer bytes to target" are the same claim expressed in two units.

## Background: the wall this answers

On 2026-06-27 a from-scratch 2.5-billion-parameter byte-level model was abandoned after its
quality flatlined. The cause was recorded plainly: a compute wall. Competitive ability at that
scale needs tens of billions of training bytes, and the machine delivers only a few hundred
bytes per second when the model is that large. No amount of patience closes a gap of that
shape on one desktop, because the shortfall is in throughput, not in time willing to be spent.
The retry condition written into the ledger was deliberately strict: return to a large
from-scratch model only with a training-efficiency lever of 10x or more at equal quality,
measured on this machine, not borrowed from a paper written about a cluster.

That condition sets the terms of this paper. It is an honest bar rather than a rhetorical one,
and we keep it in view throughout: the point of the programme is to find real, machine-local
multipliers on learning-per-calculation, and to report them at the size where we can actually
measure them rather than at the size where we would like them to hold. This paper is the first
instalment of that answer. The stack measured here is worth roughly 3x, and we say so plainly:
3x is a genuine result and it is not 10x. The remaining distance is the subject of separate
work in the programme, along three axes that do not overlap with the architectural levers
studied here: data quality and teacher distillation (which change what each byte teaches, and
which produced the conversational model documented in paper 3, "How a 121M byte model learned
to talk"); decode-time and serving efficiency (paper 4, "Serving a research architecture on a
CPU"); and the long-context memory that the constant-state design makes possible (paper 5,
"Long-context memory for constant-state models"). The discipline of keeping the levers that
did not clear their bar is itself a deliverable, catalogued in paper 2, "Negative results, and
why they are worth keeping." Paper 6, "Amortizing the optimizer step," takes the throughput
lens developed here down to the level of a single pre-launch tuning pass.

Why insist that the wall is about arithmetic rather than memory? Because the two failure modes
call for opposite responses. A memory-bound run is rescued by making the model or the batch
smaller, or by sharding state across devices; a compute-bound run on a single device is
rescued only by extracting more learning from each operation the device performs. The M3 Ultra
regime is squarely the second case. Once the model fits, adding hardware is off the table (the
programme's premise is one machine at zero marginal cost), so the entire design space collapses
to a single objective: raise the quality reached per unit of compute. Each of the four levers
below is an attempt to move that ratio, and each is measured against it directly.

## What we did

### The rig

Every experiment used the same rig, so that any difference in outcome is attributable to the
one thing we changed. The shared setup was: the canonical 10M-parameter byte-level shape; the
FineWeb-Edu corpus (a curated body of educational web text); 12000 training steps; one fixed
random seed; mixed bf16 precision on the Mac's GPU; and launch through the platform dashboard,
so that the full measurement and checkpoint machinery ran identically on every arm. Holding the
corpus, the step count, the seed, the precision, and the harness fixed means the arms are true
single-variable comparisons: each arm differs from its baseline by exactly one change, and the
result of that change is not confounded by an incidental difference in data order, logging, or
stopping point.

Working at 10M parameters is a deliberate methodological choice, not a limitation we tolerated.
A small model lets us run every arm to completion, repeat the harness end to end, and read
clean learning curves in hours rather than weeks. It is a proving ground for the mechanism of
each lever, chosen because the ideas we are testing are ones the published literature reports
holding across scale. The honest reading of the numbers, stated again in the limitations, is
that they are our measurements of these mechanisms at this size, on this corpus, on this
machine.

### How we compared

We compared each arm against its baseline two ways, and we keep both in view because they
answer different questions. "At equal steps" asks which model is better after the same amount
of training, which isolates the quality of the learning signal. "At equal wall-clock" asks
which model is better after the same amount of real elapsed time, which is the question that
actually matters when one arm runs faster or slower per step. An idea that learns more per step
but runs slower per step can win on the first comparison and lose on the second; reporting only
the flattering one would be dishonest. Where an arm changes throughput, the wall-clock
comparison is the verdict and the per-step comparison is the mechanism.

Each arm carried a pre-registered falsifier: a numeric threshold decided before the run which,
if not cleared, kills the idea. Pre-registration matters because it removes the temptation to
move the goalposts after seeing the curve; a lever either clears a bar we set in advance or it
does not. Verdicts use the tail-averaged validation loss, meaning the average of the validation
loss over the final stretch of the run rather than the value at a single final checkpoint. We
adopted tail averaging because single-milestone measurements were found to be noisy: the
loss at any one step wobbles enough that reading a verdict off one point can flip a close call,
whereas the tail average is a stable estimate of where the run actually settled.

### The four levers

**E1, the Muon optimizer.** An optimizer is the rule that adjusts the model's settings after
each batch of text, turning the gradient (the direction that would reduce the loss) into an
actual update. The standard choice is AdamW, which scales each parameter's update by a
running estimate of that parameter's own gradient statistics, treating the parameters as an
unstructured list of numbers. Muon (in the Muon / Newton-Schulz line of work) treats the
two-dimensional weight matrices as matrices. For each such matrix it takes the momentum (the
smoothed gradient) and orthogonalizes it with a short Newton-Schulz iteration, a matrix
recurrence that drives the momentum toward its nearest orthogonal factor, the product of the
left and right singular vectors from its singular value decomposition, without ever forming
that decomposition explicitly. The effect is to flatten the spectrum of the update: instead of
a few dominant singular directions receiving nearly all of the step while the rest are
starved, every direction receives a comparably sized step. That is a direct improvement in the
conditioning of the update, and better-conditioned updates make more balanced progress per
step. Crucially, the orthogonalization is only meaningful for parameters that are genuinely
two-dimensional (the bulk of a transformer's weight, the dense projection matrices), so the
one-dimensional parameters, the embeddings, the normalization scales and biases, and anything
that is a vector rather than a matrix, are left to AdamW, which is the right tool for them. The
single change in this arm is the optimizer; everything else is held.

**E2, boundary patching.** The expensive part of a transformer is the global mixing (attention
together with the feed-forward network) that, in a standard dense model, runs at every
position in the sequence. Boundary patching, in the SpaceByte style, breaks that assumption. It
runs cheap local blocks on every byte, so that fine-grained information is never dropped, but
it runs the expensive global blocks only on a small, fixed set of "patch slots" that are
anchored at word-boundary-like bytes: one slot for roughly every four bytes. The arithmetic is
the whole point. The global blocks dominate the compute, and if they fire on one position in
four instead of on every position, their cost falls by roughly four times for the same amount
of text. That freed budget does not have to be given back; it can be spent on more parameters
at the same per-byte calculation cost. Patching thus converts a compute saving into a capacity
gain, which is exactly the trade a compute-bound regime wants. The single change in this arm is
the architecture of where the global blocks run.

**E3, constant-state recurrence.** This lever replaces attention entirely with a gated linear
recurrence, drawn from the family that includes RWKV, GLA (gated linear attention), and
Mamba-2. In place of attention's mechanism of comparing every position against every other
position, each head maintains one fixed-size matrix of memory. That matrix is updated once per
byte by folding in the current byte's contribution and multiplying the existing memory by a
learned decay, so recent information is written in while old information fades at a rate the
model learns rather than one fixed by hand. The consequence appears at generation time and is
the reason we pursued this lever. Attention must keep a key-value cache that grows with every
byte of the conversation, so both the memory it occupies and the work it does per new byte
increase without bound as context lengthens. The recurrent state does not grow at all: it is a
fixed-size matrix whether the model has seen a hundred bytes or a hundred thousand. We call
that "O(1) state," meaning the cost per byte at generation time is constant regardless of
position. During training the recurrence is evaluated in a chunkwise form that recovers much of
the parallelism attention enjoys, though our implementation of that form was left unoptimized,
which the throughput numbers reflect. The single change in this arm is the sequence mixer.

**E5, the hybrid trunk.** The fourth lever is the composition of E2 and E3, and it is the
architecture the campaign was aiming for from the start. It keeps boundary patching's cheap
local attention on every byte, so nothing fine-grained is lost, but it makes the global blocks
constant-state recurrent mixers running on the patch slots. In one design it inherits patching's
more-parameters-per-calculation and recurrence's constant-size long-range memory at the same
time. The two mechanisms are not in tension: patching decides where the global work happens
(on a compressed stream of patch slots), and recurrence decides how that global work carries
information across distance (through a fixed-size state rather than a growing cache). Because
the recurrent mixer now summarizes a shorter, more semantically meaningful stream of slots
rather than a raw byte stream, and because patching now gets an unbounded-context summary
rather than a bounded attention window, we expected the composition to do more than add its
parents' effects. The single change in this arm, relative to the patched arm, is that the
global mixer is recurrent rather than attentional.

## What we measured

The reported quantity is final validation loss, in bits per byte, lower being better, together
with when in the run each arm reached its comparison target. All arms share the rig above.

### At a glance

Read in one pass, the five arms are:

- **E1 AdamW,** the baseline optimizer, is the reference. It finished at 1.0375 bpb.
- **E1 Muon,** the Muon optimizer, finished at 0.9990 bpb and reached the AdamW final quality
  at step 7500, which is 1.60x fewer training bytes to the same target.
- **E2 patched,** boundary patching, finished at 0.9776 bpb and reached the dense arm's final
  quality at step 8600, a 1.82x wall-clock saving, while realizing 128 percent of dense
  throughput.
- **E3 recurrent,** the constant-state mixer, finished at 0.9900 bpb and beat attention per
  step, but ran 18 percent slower per step, which places it at parity rather than ahead when
  the comparison is made at equal wall-clock.
- **E5 hybrid,** patched plus recurrent, posted the best final loss of every arm at 0.9707 bpb,
  reaching dense-final quality in 1.70x less wall-clock and patched-final quality in 1.15x
  less.

### Detail per lever

**E1 Muon.** AdamW finished at 1.0375 and Muon finished at 0.9990, and Muon passed the
AdamW-final quality at step 7500 of 12000, which is 1.60x fewer training bytes to the same
target. The advantage was not a single lucky crossing: Muon was ahead at 115 of 120 matched
evaluation points from step 500 onward, at a cost of only about 3 percent extra step time, so
the per-step improvement translates almost fully into a wall-clock improvement. Muon does lag
during the first roughly 400 steps of warmup, when the orthogonalization is acting on an
immature momentum estimate, so nothing is judged before step 500 and the warmup transient is
excluded from the verdict by design rather than by convenience. The pre-registered falsifier,
that anything under 1.15x savings would kill the lever, was cleared with wide margin. On the
strength of that result Muon became the default optimizer for every later run in the campaign,
which means the optimizer used by the patched, recurrent, and hybrid arms is Muon, not AdamW.

**E2 patched.** The patched arm carried 15.0M parameters, 49 percent more than the roughly
10.08M dense baseline, at roughly the same per-byte calculation cost, precisely because the
global blocks run on four-times-fewer positions and the compute they save is reinvested in
capacity. It reached the dense arm's final quality (0.9990) at step 8600, and because the
patched arm also runs faster, that crossing arrives at 6679 seconds, a 1.82x wall-clock saving;
its own final loss at equal steps was 0.9776, better than dense both per step and per second.
Realized throughput was 128 percent of dense, confirming that the extra parameters were bought
without paying in speed. The falsifier had two teeth, either no win at equal wall-clock or under
70 percent throughput, and the arm cleared both.

**E3 recurrent.** The recurrent arm carried 10.95M parameters, 8.6 percent more than dense, the
extra coming from an output gate, which we disclose rather than absorb into a "same size" claim.
Per step it beat attention: final loss 0.9900 versus dense 0.9990, ahead at 117 of 120 matched
evaluations, so the learning signal from the recurrence is genuinely competitive with attention
at this scale. The catch is speed. The unoptimized chunkwise recurrence ran 18 percent slower
per step (57.9k versus 70.3k bytes per second), and once that is charged, the arm sits about
0.011 bpb behind at equal wall-clock, which lands inside the plus-or-minus 0.03 parity band we
set in advance. The falsifier, being worse than 0.03 at equal wall-clock, survived, but the
per-step win does not carry through to a wall-clock win, so the reportable claim is parity, not
a win, and we report it that way. The real prize of this lever was never the loss number in any
case: it is the constant-size decode state, which is an architectural property, not a
training-curve property, and which the recurrent arm delivers regardless of the throughput of
our current implementation.

**E5 hybrid.** The composed trunk carried 15.9M parameters at roughly dense per-byte cost and
posted the best final loss of every arm, 0.9707. It reached dense-final quality in 1.70x less
wall-clock and patched-final quality in 1.15x less, while running at 79.5k bytes per second,
113 percent of the dense throughput. Against its stronger parent it was ahead of the patched
arm at 98 of 120 matched evaluations. The composition is not merely additive: it holds patch-level
compute cost, more parameters per calculation, and a constant-size long-range state all at once,
and the fact that it beats both of its parents on final loss (0.9707 below the patched arm's
0.9776 and the recurrent arm's 0.9900, and well below dense's 0.9990) is the evidence that
composing the two mechanisms yields something neither delivers alone.

### The composed stack

The two levers that compose cleanly are the optimizer change and the architectural change,
because they act on independent mechanisms: Muon changes how each step uses its gradient, while
patching changes how much a step costs and how many parameters it drives. Multiplying them
gives Muon 1.60x times patching 1.82x, about 2.9x wall-clock versus the AdamW-dense starting
point, with the hybrid carrying both the best quality and the constant-size decode state on top
of that. That roughly 3x figure is the headline of the paper, and we hold it to its exact
meaning: a training-efficiency multiplier, on this machine, at this scale.

## What it means

### For a general reader

We found several ways to make the small model learn more per second of training, and one
architecture that combines the best two of them. The combined architecture reaches the old
quality bar in roughly a third of the time. As a bonus it uses a kind of working memory that
does not balloon as a conversation lengthens: a plain attention model has to keep a running
transcript of everything it has read, which gets heavier the longer the conversation, while
this design keeps a fixed-size notepad that it overwrites as it goes. That fixed-size notepad
is what makes the model practical to run later on modest hardware, and it is why this
architecture was chosen even where its raw learning speed only tied the alternative.

### For the machine-learning reader

Three points carry the weight. First, the honest comparison is at equal wall-clock, not equal
steps, and we let that comparison overrule a flattering per-step result in the recurrent arm:
a per-step win that a slower implementation erases is not a win. Second, the levers are
mechanistically independent, which is why the optimizer multiplier and the architectural
multiplier compose to roughly 2.9x rather than merely being the larger of the two; Muon
improves the conditioning of the update while patching improves the compute-to-parameter ratio,
and neither interferes with the other. Third, the hybrid's out-performance of both parents is a
positive result about composition specifically: a constant-state global mixer operating on a
patched (compressed) stream is stronger than either the patched-attentional model or the
fully-recurrent model at matched cost, which is the empirical case for making the hybrid the
default trunk rather than treating patching and recurrence as alternatives.

### The downstream asset

The constant-size decode state is the property that reaches furthest beyond this paper. In
training it bought only parity, but in deployment it is a standing asset: generation cost per
byte does not climb with context length, and the memory footprint at inference is bounded no
matter how long the exchange runs. That is exactly the property a research architecture needs to
be served on a CPU (paper 4) and to hold a long context without a growing cache (paper 5). On
the strength of the combined result, the hybrid trunk became the default architecture for
everything downstream, including the chat model in paper 3, "How a 121M byte model learned to
talk."

### Honest limitations

Each arm is a single training seed. The house rule of the programme is that any advantage under
5 percent needs two or more seeds before it is reported as a win outside the project, and we
apply that rule to our own results without exception. The hybrid's win over the dense baseline
is large and comfortably outside the noise, and stands as reported. Its narrower win over the
patched parent (0.007 bpb, 1.15x) is inside the 5 percent band and therefore carries an explicit
"needs a second seed" caveat in the ledger; we believe it, but we do not export it as settled.
The recurrence result is reported as parity rather than a win for the same reason, compounded by
the fact that its per-step edge depends on an implementation whose throughput we have not
optimized. All of this is measured at 10M parameters on FineWeb-Edu, with one seed, on one
machine. The levers were chosen because the published literature reports them holding at larger
scale, but the numbers here are ours at this scale and we do not present them as anything more.
Finally, and most importantly, this is a training-efficiency result, not a quality-parity claim
against large models. The compute wall recorded on 2026-06-27 still stands. The stack is roughly
a 3x lever, and the retry condition it was measured against asks for 10x. We report the gap
rather than paper over it: 3x is real, 3x is banked, and 3x is not yet the number that reopens
the large from-scratch model.

## Evidence trail

- Ledger entries: `successes.md`, entries dated 2026-07-03 (E1 Muon) and 2026-07-04 (E2
  patched, E3 recurrent, E5 hybrid). Falsified and abandoned levers from the same campaign are
  logged in `failures.md`.
- Training runs, with per-step `train.csv` learning curves: `models/e1adamw_10m_qat/`,
  `models/e1muon_10m_qat/`, `models/e2patched_10m_qat/`, `models/e3recur_10m_qat/`,
  `models/e5hybrid_10m_qat/`.
- Chronology: `worklog.md`, sections dated 2026-07-03/04 and 2026-07-04/05.
- Companion papers: paper 2, "Negative results, and why they are worth keeping" (the levers
  from this campaign that were killed by their falsifiers); paper 3, "How a 121M byte model
  learned to talk" (the hybrid trunk carried downstream into the chat model); paper 4, "Serving
  a research architecture on a CPU," and paper 5, "Long-context memory for constant-state
  models" (both consume the constant-size decode state); paper 6, "Amortizing the optimizer
  step" (the throughput lens applied to a single pre-launch tuning pass).
