# Serving a research architecture on a CPU

The v13 engine format: constant-size decode state, byte-for-byte parity to the reference,
backwards compatibility by construction, and about one millisecond per byte on a CPU core.

## Abstract

A model that trains inside a research framework is not the same thing as a model that serves quickly
on a plain CPU. Training frameworks are built to be flexible and differentiable; a serving engine
is built to be fast, exact, and predictable in its memory use. The two goals pull in different
directions, and the gap between them is where most of the latency, and most of the surprises, live.

Our chat model makes that gap wider than usual, because it does not use a standard transformer. It
uses the hybrid trunk from paper 1 (the composed efficiency stack), which interleaves ordinary
attention with a constant-state recurrent mixer and short convolutions. A stock transformer runtime
has no vocabulary for the recurrent state update or for the convolution rings, so running the model
on one would mean either rewriting it into something it is not, or bolting foreign operations onto a
runtime whose entire decode path assumes a growing attention cache. We chose neither. We wrote a
small, hand-tuned C engine that runs the architecture directly, exactly as trained.

The result runs at roughly one millisecond per byte on a single M3 Ultra core, produces
letter-for-letter identical output to the slow reference implementation, and keeps every earlier
model format working without any change to their code paths. Crucially, its memory footprint while
generating is fixed regardless of how long the conversation grows, which is the entire reason the
constant-state architecture was worth building in the first place: a serving engine that let the
footprint grow with the conversation would throw that property away. This paper describes the on-disk
format, the design rule that keeps it backwards-compatible, the three-layer exactness discipline that
makes its speed trustworthy rather than merely fast, the measured latencies including an honest
low-power estimate, and two instructive bugs the exactness discipline flushed out.

Terms are defined at first use. "Decode" is generating text one byte at a time. "Parity" means two
implementations produce the same bytes. "bpb" is bits per byte, the quality measure, lower being
better, and it is the natural unit for a byte-level model that has no word-piece dictionary. "fp32"
and "fp16" are 32-bit and 16-bit floating point; "int8" is 8-bit integers. "QAT" is
quantization-aware training, where a model is trained with the eventual integer arithmetic simulated
in the loop so that it learns to tolerate the rounding. "SIMD" is single-instruction-multiple-data,
the vector arithmetic a CPU core uses to multiply many numbers at once; "NEON" is the ARM SIMD
instruction set on Apple silicon. A "matvec" is a matrix-vector multiply, the dominant operation in
decoding one byte.

## What we did

### The v13 format and the strict-additivity rule

The engine is a C program, `veritate_engine/v1`, with a new module, `src/hybrid.c`, dedicated to the
hybrid trunk, and a new on-disk format, version 13 of the `.bin` model file. The governing design
rule was strict additivity. v13 is not a modification of the existing loader; it is a new version
byte that carries its own loader and its own forward path, both selected at load time by reading that
version byte and dispatching accordingly. The canonical formats, v3 through v12, are left untouched:
their loaders, their kernels, and their decode loops are exactly the code they were before, reached
by exactly the same dispatch they were reached by before. Nothing about serving an older model
changes, because none of the code that serves an older model changes.

This is a stronger guarantee than "we tried not to break anything." Backwards compatibility here is a
property of the structure, not of our care: because the old paths are physically the same code and
the new path is reached only when the version byte says v13, an older model cannot take the new path
and cannot be affected by it. The cost of that guarantee is a small amount of duplication at the
dispatch boundary, which we consider cheap insurance for a format that has to keep a dozen prior
model generations loadable indefinitely.

Writing a bespoke engine at all, rather than adapting a general-purpose transformer runtime, was a
deliberate consequence of the architecture. The hybrid trunk's recurrent global blocks maintain a
running state that is updated once per byte, and its convolutional components carry short ring
buffers of recent activations. Neither of those has an analogue in a transformer runtime, whose
decode loop is organized entirely around appending a key and value to a cache and attending back over
it. Forcing the hybrid model into that mold would have meant approximating operations the model was
trained with, and any approximation at serve time that differs from train time is exactly the kind of
silent quality regression this project exists to avoid. A direct engine is more work up front and far
less risk afterward.

### One numeric decision: non-QAT checkpoints force fp32 compute

We made one decision about numbers that shaped everything downstream. The hybrid checkpoints were not
quantization-aware-trained. The older canonical formats lean on an aggressive integer pipeline that
is quality-safe only because those models were trained to expect it; the weights learned to live with
the rounding. The hybrid weights did not, so pushing them through the same integer arithmetic would
degrade quality in ways the model never had a chance to compensate for. v13 therefore computes in
fp32 end to end. That is a compute decision, not a storage decision: weights on disk may be stored as
fp32 or as fp16, and fp16 weights are converted exactly to fp32 at load and then used in fp32
arithmetic. Storage in fp16 halves the file and the memory bandwidth without touching the arithmetic
the model actually sees. This separation, exact fp16 storage feeding fp32 compute, is what lets the
shipping default be small on disk while remaining bit-honest to the reference.

### The constant-state contract, enforced in code

The architecture's headline property, that decode memory does not grow with conversation length, is
only real if the engine refuses to let it grow. We enforced the contract in the code rather than
hoping the data structures happened to respect it. Each recurrent head keeps a fixed [d, d] state
matrix. Every byte updates that matrix by a rank-1 outer product and reads from it; the update writes
in place. No memory is allocated inside the decode loop. The per-byte state size is therefore a
constant of the model, not a function of position, and a conversation that runs for a thousand bytes
uses precisely the same state as one that runs for ten.

This is the sharp contrast with attention, whose key-value cache grows by one entry per byte
generated, so that a transformer's decode footprint climbs with the length of the exchange. The
hybrid trunk still contains a bounded amount of attention, and that part does carry a cache, but the
cache is capped by the fixed sequence length rather than by the open-ended conversation. Everything
recurrent is flat. Serving the model on an engine that quietly reintroduced growth, by reallocating
state buffers or by accumulating history, would have silently converted a constant-state model back
into a growing-state one; enforcing no-allocation in the loop is how we keep the architecture's
promise intact from training all the way to serving.

### The three-layer exactness discipline

Speed is only worth having if the fast answer is the right answer, so the engine is built on three
nested layers of exactness, each verified against the one below it.

The first layer is the scalar C path. It is the reference: plain, unvectorized arithmetic that we
treat as the definition of what the engine should compute. It is not the fastest code, and it is not
meant to be; it is meant to be obviously correct and to serve as the ground truth for everything
above it.

The second layer is the SIMD kernels, the NEON vector code that actually delivers the speed. The
requirement on these kernels is not "close to" the scalar path but bitwise-identical to it. Floating
point does not associate, so the order in which partial sums are accumulated changes the result in
the last bits; a vectorized matvec that sums in a different order from the scalar loop would produce
answers that are almost the same and therefore worse than useless, because "almost the same" is
exactly what hides a bug. We pin the arithmetic two ways. The matvec is structured around 16 partial
sums so that the vector and scalar accumulation orders coincide, and the code is compiled with
`-ffp-contract=off` to stop the compiler from fusing a multiply and an add into a single
fused-multiply-add, which rounds once instead of twice and would break the bitwise match. For the
fp16-to-fp32 convert routine we did not sample or spot-check: fp16 has only 65536 possible bit
patterns, so we ran an exhaustive check over all 65536 values and confirmed the C converter matches
the reference on every single one. For a 16-bit type, exhaustive is not a heroic effort; it is simply
the correct test, and it turns a probabilistic claim into a proof.

The third layer is parity to the PyTorch reference: the whole engine's greedy output must match the
research implementation byte for byte. This is where the definition of "match" has to be chosen with
care, and choosing it correctly is itself a result. The recurrence is computed one way at training
time, in parallel over chunks, and another way at decode time, sequentially one byte at a time. The
two are algebraically identical, but in fp32 they round differently, because they accumulate the same
mathematics in a different order. Demanding bit-identical raw scores between them would be demanding
something false. What actually matters for greedy decoding is which byte wins the argmax, and that is
where we define parity: at the level of the chosen byte, not the raw score. We then verified that the
decision was never close enough for the rounding to matter, which is the subject of the parity
measurement below.

## What we measured

### Quality gates

We ran the quality battery on the chat80m step-48000 checkpoint, against the PyTorch fp32 reference
on the same prompts. Parity is greedy decode over 3 chat prompts of 64 bytes each, 192 bytes in
total. bpb is measured over 20 chunks of 1024 bytes on held-out FineWeb-Edu and chat_v1. Three
storage modes were evaluated.

The fp32 mode ships a 487.0 MB file and computes in fp32. It achieved parity 192/192, and its bpb was
1.4447 on FineWeb-Edu and 0.9899 on chat_v1, against the PyTorch reference figures of 1.44466 and
0.98989 respectively: identical to four decimals.

The fp16 mode ships a 243.5 MB file, half the size, and computes in fp32 after an exact convert at
load. It also achieved parity 192/192, and its bpb was identical to four decimals to the fp32 mode
above. Because it is exactly as accurate as fp32 storage while occupying half the disk and half the
weight bandwidth, this is the shipping default.

The int8 per-channel mode would ship a 122 MB file and compute with an int8 dot product followed by
an fp32 requantization. In PyTorch simulation its quality passes: bpb rose by only +0.0006 on
FineWeb-Edu and +0.0009 on chat_v1, and 2 of the 3 parity prompts remained byte-identical to fp32.
Its C kernels are not yet written, so it does not ship today. It is the documented next lever,
offering roughly 2x the bandwidth advantage over fp16, which matters most on the bandwidth-poor parts
discussed below.

### Byte-parity discipline

The C engine matched the PyTorch greedy reference on all 192 bytes, in both the fp32 and fp16 storage
modes. As noted above, this is not automatic, because the training-time recurrence is computed in
parallel chunks and the decode-time recurrence is computed sequentially, and the two are algebraically
identical but round differently in fp32. Parity is therefore defined at the greedy-argmax level, which
byte wins, not at the bit level of the raw scores. The reassuring measurement is the margin: the
smallest winning margin observed across all 192 decisions was 0.0089, and it never flipped the choice.
In other words, even the closest call the model made was decided by a gap comfortably larger than the
last-bit rounding difference between the parallel and sequential recurrences, so no amount of legal
reordering in the arithmetic could have changed the byte that was emitted. That is what makes
argmax-level parity the right and sufficient definition here rather than a convenient weakening of it.

### Backwards compatibility by construction

We tested the strict-additivity claim directly. A committed tiny v9 fixture, with a recorded greedy
transcript, produces byte-identical output before and after the v13 changes. We then went further and
tested that the compiler and linker themselves could not perturb an older model: a build from the
pristine prior commit and a build with the new code agree byte for byte on that fixture across all six
build variants (with and without link-time optimization, and across different link orders), with zero
failures over 20 stress iterations and a clean AddressSanitizer run. The point of exercising six
variants and repeating them under stress is that a heap bug in the new code could, in principle, leak
into an old path through the linker's layout choices even when the source of the old path is untouched;
demonstrating stability across link orders and optimization settings is what turns "the old code did
not change" into "the old model's output did not change."

### Latencies

All latency measurements are on the M3 Ultra, taken over 64-byte greedy generations from chat prompts.
Two numbers characterize each configuration. The p50 is the median byte, which is a non-boundary byte
and therefore the cheaper case. The p95 is a boundary byte, the more expensive case, where the 12
recurrent global blocks fire. The six configurations were as follows.

In fp16 with 4 threads, the default configuration, the median byte took 1.08 to 1.12 ms, the p95
byte took 4.6 ms, and throughput was 520 to 600 tokens per second. In fp16 with a single thread, the
median rose to 1.7 to 1.8 ms, the p95 to 6.8 to 7.4 ms, and throughput fell to 340 to 380 tokens per
second. In fp32 with 4 threads, the median was 1.5 ms, the p95 6.4 ms, and throughput 390 to 430
tokens per second. In fp32 with a single thread, the median was 2.0 to 2.1 ms, the p95 8.2 ms, and
throughput 290 to 330 tokens per second.

Two further configurations pin the work to the CPU's efficiency cores as a low-power proxy. In fp16
with 4 threads pinned to efficiency cores, the median byte took 5.0 to 5.1 ms, the p95 byte 20 to 23
ms, and throughput was 114 to 137 tokens per second. In fp16 with a single efficiency-core thread,
the median was 7.0 to 7.5 ms, the p95 28 to 31 ms, and throughput 83 to 89 tokens per second.

Separately, a full-context perplexity decode, running out to position 1024, ran at p50 2.7 ms and p99
8.9 ms in fp16 with 4 threads. The efficiency-core-pinned rows are an honest low-power proxy: they
stand in for a Cortex-A76-class mobile CPU at roughly 5 to 7 ms per byte today, and they are a
software-throttled estimate on this machine, not a measurement taken on mobile silicon. On such a
target the validated int8 path is the remaining roughly 2x lever, which is why it is worth writing
even though the M3 Ultra does not need it.

### The memory-bandwidth story

The reason fp16 helps at all, and the reason int8 would help more, is bandwidth rather than
arithmetic. Decoding one byte is dominated by the matvec, and the matvec's cost is set by how fast the
core can stream the weight matrix from memory, not by how fast it can multiply. We measured the matvec
streaming weights at about 80 GB/s per fp32 single core, which is close to the machine's single-core
memory ceiling. When the compute is already pinned against the memory wall, the way to go faster is to
move fewer bytes per weight, and that is exactly what narrower storage buys: fp16 halves the bytes
moved per weight relative to fp32, and its advantage grows on bandwidth-poor parts where the ceiling
is lower. int8 would halve the traffic again. This is the mechanism behind the int8 lever: it is not
about doing cheaper multiplies, it is about feeding the same multiplier from a narrower, faster stream.

### Constant-state footprint and the compute split

At hidden size 768 and sequence length 1024, the total decode state is about 28 MB in fp32. That
figure is the sum of three bounded pieces: four attention caches, each bounded by the sequence length
rather than by the conversation length; twelve recurrent state matrices, each a fixed [d, d] block
that does not grow at all; and the convolution rings. None of these three grows with how long the user
keeps talking, which is the constant-state guarantee made concrete in megabytes.

The per-byte compute divides cleanly into two cases, and this split is what produces the gap between
the p50 and p95 latencies above. An ordinary byte costs about 56 million operations. A boundary byte
costs about 244 million operations, because that is when the recurrent global blocks run: all 12 fire
at a boundary and stay quiet otherwise. On top of both cases sits an attention term that grows
linearly, but only up to the fixed sequence cap, after which it too is bounded. So the engine spends
most bytes in the cheap regime and pays the recurrent cost only at boundaries, which is precisely why
the median latency reflects the ordinary byte and the tail latency reflects the boundary byte.

### Two pre-existing bugs surfaced and fixed

The exactness discipline earns its keep by catching bugs that would otherwise degrade output silently,
and it caught two.

The first was a latent memory-corruption bug. A dense-path routine hardcoded a 64-wide attention head
and corrupted the heap on any other head size. For a long time it wore a convincing disguise: it
presented as build-order nondeterminism, output that changed depending on how the binary happened to
be linked, which is the kind of symptom that invites endless speculation about compilers and
optimization levels. AddressSanitizer cut through the disguise and root-caused it to the write past the
64-element head, which is why the backwards-compatibility battery above runs under AddressSanitizer
rather than trusting output comparison alone. Dense loads now refuse a non-64 head size outright,
turning a silent corruption into a loud, early failure.

The second was a protocol bug in the traced chat path. The traced chat protocol was flattening prompt
newlines to spaces, so that every reply produced through the C backend with a multi-line chat template
was quietly degraded: the model was being fed a subtly different prompt from the one the template
described, and there was no crash to announce it, only slightly worse answers. Once newlines are
escaped end to end, through the protocol and back, the multi-line template reproduces the PyTorch
reference byte for byte. Both bugs share a lesson worth stating plainly: a serving engine that is only
approximately faithful will hide exactly these failures inside the noise, and only a byte-for-byte
parity standard makes them visible.

## What it means

For a general reader: the fast engine gives the same answers as the slow scientific one, letter for
letter, but in about a third of a second instead of one letter per second, and it will happily keep
running the older models too, without any risk of the new work disturbing them. Its memory use while
chatting is fixed, so a long conversation does not slowly consume the machine, which is the practical
payoff of the whole constant-state design: you can leave the conversation running without watching the
memory climb.

The exactness matters more than it might sound. It is easy to make a model faster if you are willing to
let it be a little different; the hard and useful thing is to make it faster while proving it is not
different at all. That proof is what lets us treat the engine's speed as free rather than as a
trade-off, and it is what let two real bugs, one that corrupted memory and one that degraded every
multi-line reply, surface as visible failures instead of hiding as slightly worse output.

Honest limitations, stated plainly. Only fp32 and fp16 storage ship today; the int8 mode is
quality-validated in PyTorch simulation but its C kernels are not yet written, so its numbers are a
promise backed by a simulation, not a shipping measurement. The traced telemetry mode allocates about
900 MB at the full sequence length, which is fine on a development box but not on a low-power target,
so low-power serving uses the untraced path. The per-layer telemetry for the recurrent global blocks
is shallow by design, because slot-level state does not fit the dense per-layer frame; the model's
quality and speed are unaffected by that shallowness, which is a reporting limitation and not a
computational one. And all latencies are on the M3 Ultra: the low-power numbers are a software-pinned
proxy for mobile-class silicon, not a measurement taken on it. None of these figures is extrapolated
or invented; where a number stands in for a target we do not own, we have said so.

## Evidence trail

- Build chronology and the five staged gates: `worklog.md`, section dated 2026-07-07
  ("v13 hybrid engine build"), and the same-day platform-integration section.
- Serving status and the int8 lever: `worklog.md`, section dated 2026-07-07 ("v13 round 2").
- Tests: `tests/engine/test_v13_compat.py` (canonical fixture regression),
  `tests/export/test_export_v13.py` (exporter round-trip).
- Parity smoke: `SMOKE_RESULTS/v13_hybrid_parity_smoke.py`.
- Checkpoint served: `models/chat80m_80m/` step 51000 (fp16 bin, 243.5 MB; the quality-gate battery
  above ran on step 48000, the bin was re-exported after the identity round moved the flagship to 51000).
- Architecture served: the hybrid trunk of paper 1 (the composed efficiency stack); the constant-state
  decode contract it depends on is the subject of paper 5 (long-context memory for constant-state models).
