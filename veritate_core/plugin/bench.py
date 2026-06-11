# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Empirical training-memory + throughput benchmark. Replaces the analytic estimate's
#   guesswork (which undershoots) with measured numbers: ramps the batch size on the
#   real model until the device runs out of memory, recording the high-water memory
#   and tok/s at each rung. The largest rung that fits is the ceiling.
# - Uses synthetic random byte batches (shapes are all that drive memory/throughput),
#   its own throwaway AdamW, and never saves: no checkpoint, no real weights touched.
#   A trainer invokes run() with its already-built model so MoE/variant footprints are
#   measured for real instead of approximated.
# - Forward return is (logits, loss, ...): loss is index 1 across every variant (the
#   MoE trunk adds an aux term at index 2). The benchmark backprops the index-1 loss; that
#   allocates the full grad + optimizer footprint, which is what the ceiling needs.
# veritate_core/plugin/bench.py
# ------------------------------------------------------------------------------------
# Imports

import time

from veritate_core.plugin import oom_recovery

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_BATCH_RAMP = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
WARMUP_STEPS = 2
TIMED_STEPS  = 3
PROBE_LR     = 1e-4
GB           = 1024 ** 3

# ------------------------------------------------------------------------------------
# Functions


def _device_high_water(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()
        return torch.mps.driver_allocated_memory()
    if device == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    return 0


def _reset_high_water(device):
    import torch
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _free(device):
    import torch
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _step(model, opt, batch, seq, vocab, device):
    import torch
    toks = torch.randint(0, vocab, (batch, seq), device=device)
    tgts = torch.randint(0, vocab, (batch, seq), device=device)
    opt.zero_grad(set_to_none=True)
    out = model(toks, tgts)
    loss = out[1] if isinstance(out, (tuple, list)) else out
    loss.backward()
    opt.step()
    del toks, tgts, loss


def _measure_batch(model, opt, batch, seq, vocab, device):
    """Run warmup + timed steps at one batch size. Returns (mem_bytes, tok_per_s)."""
    import torch
    for _ in range(WARMUP_STEPS):
        _step(model, opt, batch, seq, vocab, device)
    _reset_high_water(device)
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        _step(model, opt, batch, seq, vocab, device)
    if device in ("mps", "cuda"):
        getattr(torch, device).synchronize()
    elapsed = time.perf_counter() - start
    tok_per_s = (batch * seq * TIMED_STEPS) / elapsed if elapsed > 0 else 0.0
    return _device_high_water(device), tok_per_s


def run(model, device, seq, vocab, batch_ramp=DEFAULT_BATCH_RAMP, on_progress=None):
    """Ramp batch size on `model` until OOM; return the measured memory ceiling and
    throughput. `on_progress(str)` receives human-readable lines as it runs (the modal
    renders these). Returns a result dict with max_batch, mem_ceiling_gb, tok_per_s,
    and the full ramp. Mutates throwaway weights only; saves nothing."""
    import torch
    emit = on_progress or (lambda _line: None)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=PROBE_LR)

    emit("detecting RAM ceiling...")
    ramp = []
    for batch in batch_ramp:
        try:
            mem, tok_per_s = _measure_batch(model, opt, batch, seq, vocab, device)
        except RuntimeError as exc:
            if not oom_recovery.is_oom_error(exc):
                raise
            emit(f"batch {batch}: out of memory (ceiling found)")
            _free(device)
            break
        ramp.append({"batch": batch, "mem_gb": mem / GB, "tok_per_s": tok_per_s})
        emit(f"batch {batch}: {mem / GB:.1f} GB, {tok_per_s:,.0f} tok/s")
        _free(device)

    top = ramp[-1] if ramp else None
    result = {
        "device": device,
        "seq": seq,
        "max_batch": top["batch"] if top else 0,
        "mem_ceiling_gb": top["mem_gb"] if top else 0.0,
        "tok_per_s": top["tok_per_s"] if top else 0.0,
        "ramp": ramp,
    }
    if top:
        emit(f"ceiling: batch {top['batch']} at {top['mem_gb']:.1f} GB, "
             f"{top['tok_per_s']:,.0f} tok/s")
    return result
