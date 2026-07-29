# model_memory (surprise-gated memory trunk)

## what it is

`veritate_core/model_memory.py::VeritateMemory`. Canonical dense trunk plus one parallel neural-memory branch at mid-depth (Titans MAG class). The memory is a 2-layer MLP whose weights are fast weights, updated inside the forward pass by the gradient of its own key-to-value recall loss (the surprise signal) with learned inner lr, momentum, and decay (forgetting). Knowledge is written into memory during use, not by corpus gradient descent; the context window is not the persistence mechanism, the memory is.

## how it works

- `NeuralMemory` (`model_memory.py:50`): slow params are q/k/v/o projections, an output gate, the fast-weight init, and three learned scalars (inner lr, momentum, decay). Fast weights are per-sequence state, not parameters; they never enter the checkpoint.
- Per 64-byte chunk: read first (`_read` with pre-chunk state), then write: closed-form gradient of the recall loss through the tiny MLP (verified exact against autograd: rel err 1.8e-07 fp32, 7e-16 fp64), momentum + decay applied, all inside the outer autograd graph so training shapes the write rule itself (TTT-style).
- Read-before-write per chunk gives chunk-granular causality (verified bitwise: perturbing byte p leaves all prior chunks and same-chunk-earlier bytes unchanged).
- `forward()` resets memory per call (safe default for dumps/eval). `forward_carry()` + `carry_memory()`/`reset_memory()` expose persistence across windows. `veritate_trainer.chunked_step` detects `forward_carry` and trains with memory CARRIED across the contiguous chunks of each step (reset per step, state detached at every backward boundary): the loss then rewards reading memory beyond the attention window. This is the E4b regime; the E4 per-window-reset regime failed its recall falsifier (see failures ledger 2026-07-04).
- Full dump suite writes (13 artifacts verified). Selected per run via the `trunk` reserved flag, value `memory`.

## dependencies

`veritate_core/model.py` (Block, RMSNorm, QuantLinear), `veritate_core/qat.py`.

## pitfalls

- The write signal at init is intentionally small (gated, sigmoid inner lr); the whole bet is that training amplifies it. The knowledge-injection eval is the decider, not train loss.
- Fast weights are state on the module: any second backward through a stale state is invalid; `forward()` resets per call, `carry_memory()` detaches.
- Not `.bin`-exportable; test-time weight updates cannot run in the C engine. A frozen-ternary-trunk + updatable-memory split is the plausible engine path.
- ~2 percent more params than dense at the same manifest shape; disclose in A/Bs.
