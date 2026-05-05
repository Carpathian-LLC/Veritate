# How we built a glass-box INT8 transformer that crosses PyTorch and C

A practical explainer of the moving parts: why training in PyTorch and inferring
in hand-written C produces bitwise-comparable output, why naive quantization
breaks, what QAT does, and what the diff harness is for.

## The two halves

| Half        | What it runs                                  | Where it lives       |
| ----------- | --------------------------------------------- | -------------------- |
| Training    | A PyTorch transformer, fp32 + bfloat16        | `training/model.py`  |
| Inference   | A hand-coded INT8 transformer in C + asm      | `engine/src/model.c` |
| The bridge  | A `.bin` file (raw INT8 weights + scales)     | `data/models/<name>/veritate.bin` |

Both halves share the same architecture: byte-level vocab (256), hidden 768,
12 layers, 12 heads, 3072 FFN, seq 256. Same causal self-attention. Same
GELU FFN. Same tied embedding (LM head weight = input embedding weight).

The PyTorch half is for learning. The C half is for serving. They don't share
a single line of code — but at the math level they're the same model.

## How a weight gets from PyTorch to C

`training/train.py::export_to_bin` writes:

1. **Header**: `VRTE` magic, version, shape constants (vocab, hidden, layers, ffn, heads, seq).
2. **Embed + pos_embed**: INT8 tables, quantized at activation scale (32 ints per fp unit).
3. **Per layer (12 times)**: ln1_w (INT8), qkv (INT8 + per-row scale), out_proj, ln2_w, ffn_up, ffn_down.

The C engine reads the same layout in `engine/src/model.c::model_load`, then runs
`prep_b()` which transposes the weight matrix and pads it for AVX-512 VNNI.
Loaded once at startup. Reused on every token.

## Why INT8 works at all

A trained weight is a small float — say `0.0312`. We pick a scale (e.g. 32) and
store `round(0.0312 * 32) = 1` as one signed byte. The network is overdetermined
enough that "this weight means roughly 0.03" is enough information to keep
predictions correct, **if** every weight, activation, and bias agrees on the
scale convention. The C engine's whole forward pass is built around one convention:
INT8 activations with scale=32, INT16 residual stream, weight matmul outputs that
get re-quantized back to INT8.

## Why naive INT8 silently breaks

Train an fp32 model, then quantize at the end ("PTQ" — post-training quantization).
The network was never told its weights would be rounded. Three weights that meant
`0.04`, `0.05`, and `0.06` are now all `1` or all `2` after the round. Three
values collapsed to two. The matmul outputs drift, layer by layer. By layer 6,
the residual stream has wandered far enough that the model talks gibberish.

This is exactly what happened to us early on. The C engine produced text that
looked like keyboard mashing while the PyTorch fp32 model wrote coherent
TinyStories prose. Same weights. Same architecture. Output: incomparable.

## QAT — train the network to expect rounding

Quantization-Aware Training puts the rounding **inside the training forward
pass**. Every weight multiplication becomes:

    fake_quant(W) @ fake_quant(activation)

where `fake_quant` rounds the input to its nearest INT8 representation, runs
the multiply in fp32, and treats the round as if it were the identity for
gradient (straight-through estimator). The network sees noisy weights every
forward pass and learns to be robust to that exact noise. By the end of
training, the rounded weights produce roughly the same output as the simulated
quant.

## QAT v2 — simulate the C engine bit-for-bit

Generic QAT isn't enough. Different choices of scale, bias placement, and
activation quant give different rounding behavior. We wrote
`training/qat_v2.py` to simulate **the engine's exact INT8 forward** in
PyTorch:

- Per-row weight quantization (each output channel of every weight matrix gets
  its own scale, matching the v5 per-channel format the engine reads).
- Activation scale = 32 throughout.
- Sparse FFN with `threshold=4` post-GELU clamp (the engine zeroes any neuron
  whose post-GELU activation is below 4, which gives ~50% sparsity for free).
- INT16 residual stream (engine accumulates inter-layer additions in 16 bits to
  avoid INT8 saturation).

Then we fine-tune from a fp32 checkpoint into the QAT2 simulated forward. The
network adapts. After training, PyTorch's QAT2 forward and the C engine's
forward agree within 1 LSB on the same weights.

## The diff harness — finding bugs that hide between two languages

Same weights. Same input. Two implementations. Output differs by 0.1%. Where
is the bug?

`mri/server/diff.py` is a differential trace harness: run a single prompt
through both backends, capture the residual stream after every layer, compute
cosine distance per layer. The first layer where cos_dist jumps is where the
bug hides. Layers above are clean; layers below have already been corrupted.

This caught two compounding bugs in `train.py::export_to_bin`:

1. **Weight transpose mismatch.** PyTorch stores linears as `[out, in]`. The
   engine's `prep_b()` reads `[in, out]`. Without `np.ascontiguousarray(W.T)`
   on export, every weight was written shape-wise correct but transposed.
2. **Embed / pos_embed scale mismatch.** Embed quantized at scale 55.7. Pos_embed
   at scale 489.2. Both summed into an INT8 — but the int representations meant
   different fp values. Fix: both share scale 32 (`quantize_embed_at_act_scale`).

After both fixes, layer-0 cos_dist went from 0.987 (basically random) to 0.011.
The engine started writing real prose for the first time in the project's
history.

The diff harness is the single most valuable debugging tool we built. Any
divergence between PyTorch and C is now bisectable in minutes.

## What "glass-box" means

The engine emits a 235 KB binary frame per generated token, containing:

- Per-layer pre/post residual stream (INT16).
- Per-layer FFN neuron activations (INT8) — which neurons fired.
- Per-layer attention scores at the current position (per head).
- Per-layer "lens logits" — what would the model say if we stopped here?
- Decision-tracing fields: per-layer decisiveness, DLA (Direct Logit
  Attribution) top-8 — which neurons most pushed toward the chosen byte.

The MRI web panel reads those frames live and shows them. We can replay any
generation step-by-step, see which layer made the choice, which neurons fired,
which competing bytes were close runners-up. The visibility is what makes this
project useful as a research tool, not just a fast inference engine.

## What just happened (the lm_head fix)

The benchmark showed `forward_decode` at 1.4 ms median. The browser saw 4 ms
wall-clock per byte. The gap was overhead.

Looked at `chat_traced_loop`. Found two scalar matmuls running back-to-back —
one in the main loop, one inside `sample_token`. Each one was a 256 × 768
INT8 dot product done with a plain C `for` loop. Each one was ~0.5 ms because
the CPU's wide INT8 multiply instruction (`vpdpbusd`, does 32 multiplies per
clock) was sitting unused. We were doing **the same matmul twice**, both
slowly.

Fix: pack the embed table once at model load (`lm_head_build` calls `prep_b`),
expose a single `sample_token_ext` that fills logits + argmax + sample in one
call, delete the duplicate scalar block. Now there's one VNNI matmul per token
instead of two scalar ones. ~1 ms saved per token. Decode kernel floor drops
from ~1.4 ms to ~0.5 ms.

## What's still on the table

| Win                           | Effort   | Expected speedup     |
| ----------------------------- | -------- | -------------------- |
| Spec decoding C-wiring        | 1-2 days | 1.7× (draft model already trained) |
| MoD gate (per-token early exit) | 1 week | 1.6× (40% layers skipped) |
| BitNet b1.58 ternary          | 3 weeks  | 1.5-2.0× (multiplies → adds) |
| Mamba-2 80M scale             | 4 weeks  | O(1) decode at any context |
| xIELU LUT (DONE)              | shipped  | 1-3% quality bump, free |

Stack them and we land at the 0.03 ms per-byte target. Each piece is
independent. The hard part — making PyTorch and C agree on what the model
*means* — is already done.
