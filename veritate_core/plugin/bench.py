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

from veritate_core.plugin import mem_executor, mem_planner, oom_recovery

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_BATCH_RAMP = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
WARMUP_STEPS = 2
TIMED_STEPS  = 3
PROBE_LR     = 1e-4
PROBE_BETAS  = (0.9, 0.95)
PROBE_EPS    = 1e-6
PROBE_WD     = 0.0
GB           = 1024 ** 3
# On unified memory an over-budget allocation is SIGKILLed by the OS, not raised as a
# catchable error, so the ramp must stop on a measured budget rather than wait for OOM.
BUDGET_FRACTION = mem_planner.USABLE_FRACTION
# Backend tensor-size limits (not OOM): a rung whose tensors exceed what the backend can
# address bounds the ramp exactly like OOM, so it must stop the sweep, never crash the run.
# e.g. MPS: "MPSGaph does not support tensor dims larger than INT_MAX"; "Invalid buffer size".
SIZE_LIMIT_MARKERS = ("int_max", "invalid buffer size", "tensor dims larger")

# ------------------------------------------------------------------------------------
# Functions


def _is_size_limit_error(exc):
    msg = str(exc).lower()
    return any(m in msg for m in SIZE_LIMIT_MARKERS)


def _device_high_water(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()
        return torch.mps.driver_allocated_memory()
    if device == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    from veritate_core.plugin import hardware
    return hardware.process_peak_rss_bytes()


def _reset_high_water(device):
    import torch
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _memory_budget(device):
    """Usable training memory in bytes for the ramp's stop condition. Unified
    memory and cpu use total RAM (cpu compares this process's peak RSS against it,
    which holds when the box is dedicated to the run); cuda uses VRAM."""
    import torch
    if device == "cuda":
        return int(torch.cuda.get_device_properties(0).total_memory * BUDGET_FRACTION)
    from veritate_core.plugin import hardware
    return int(hardware.unified_memory_bytes() * BUDGET_FRACTION)


def _memory_kind(device):
    """Human label for the memory pool the budget represents. Rule 34c: never
    say a generic 'RAM' when it's actually VRAM or unified memory: the user
    needs to know which arch is being measured. Returns (kind, ceiling_label)
    used in the emit lines. cpu -> physical RAM, cuda -> VRAM (per-GPU),
    mps/other unified backends -> unified memory."""
    if device == "cuda":
        return ("VRAM", "VRAM ceiling")
    if device == "mps":
        return ("unified memory", "unified-memory ceiling")
    return ("physical RAM", "physical-RAM ceiling")


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


def _bucket_gb(plan):
    if plan is None:
        return {}
    return {"required_gb": plan.required_bytes / GB, "budget_gb": plan.budget_bytes / GB,
            "params_gb": plan.params_bytes / GB, "grads_gb": plan.grads_bytes / GB,
            "optimizer_gb": plan.optimizer_bytes / GB}


def plan_result(plan, device, seq):
    """Result dict for a size that cannot fit even at the planner's lowest tier:
    weights+grads alone exceed the budget, so paging the optimizer does not help.
    The trainer emits this instead of building the model (which would OOM/SIGKILL)."""
    return {"device": device, "seq": seq, "fits": False, "tier": plan.tier,
            "max_batch": 0, "mem_ceiling_gb": 0.0, "tok_per_s": 0.0, "ramp": [],
            **_bucket_gb(plan)}


def run(model, device, seq, vocab, batch_ramp=DEFAULT_BATCH_RAMP, on_progress=None, plan=None):
    """Ramp batch size on `model` until OOM; return the measured memory ceiling and
    throughput. When `plan` is an optimizer-offload tier the probe optimizer is the
    NVMe-paged AdamW, so the measured tok/s reflects the real paged regime, not a
    RAM-only fantasy. `on_progress(str)` receives human-readable lines as it runs.
    Mutates throwaway weights + a throwaway optimizer-state dir only; saves nothing."""
    import torch
    emit = on_progress or (lambda _line: None)
    model.train()
    if plan is not None and plan.tier in mem_executor.OFFLOAD_TIERS:
        opt = mem_executor.make_optimizer(model.parameters(), plan, lr=PROBE_LR,
                                          betas=PROBE_BETAS, eps=PROBE_EPS, weight_decay=PROBE_WD)
        emit(f"optimizer paged to NVMe (tier {plan.tier}); step time is disk-bound")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=PROBE_LR)

    budget = _memory_budget(device)
    kind, ceiling_label = _memory_kind(device)
    if budget:
        emit(f"device: {device} | memory budget: {budget / GB:.0f} GB {kind} "
             f"(ramp stops here to avoid an OS kill)")
    # Warn only in the actual failure case: box has an NVIDIA GPU AND torch
    # is CPU-only (cuda_available False). If the user intentionally picked
    # CPU via _device_override for a dual-device auto-tune, that's expected
    # and doesn't deserve a warning: torch CAN reach the GPU, we're just
    # measuring CPU on purpose for comparison.
    if device == "cpu":
        try:
            import torch as _torch
            cuda_ok = bool(_torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        if not cuda_ok:
            try:
                from veritate_core.plugin import deps as _deps
                if _deps.has_nvidia_gpu():
                    emit("WARNING: NVIDIA GPU detected on this box but PyTorch is CPU-only, "
                         "so this benchmark measures the CPU + physical RAM, not the GPU + VRAM. "
                         "Restart the dashboard to install the CUDA torch build, then re-run.")
            except Exception:
                pass
    emit(f"detecting {ceiling_label}...")
    ramp = []
    last_mem = None
    for batch in batch_ramp:
        # Stop BEFORE attempting a rung whose projected footprint exceeds the budget.
        # On unified/cpu memory the over-budget allocation is SIGKILLed (uncatchable),
        # so waiting for a completed rung to cross the line is not enough: a single
        # ramp jump can leap the whole [budget, kill] gap. Project the next rung's peak
        # linearly from the last measured rung (conservative: baseline scales too) and
        # stop if it clears the budget. A kill loses the whole result.
        if budget and last_mem is not None:
            prev_batch = ramp[-1]["batch"]
            projected = last_mem * batch / prev_batch
            if projected >= budget:
                emit(f"batch {batch}: projected ~{projected / GB:.1f} GB {kind} exceeds the "
                     f"{budget / GB:.0f} GB {kind} budget; stopping at batch {prev_batch} "
                     f"({ceiling_label} found)")
                break
        try:
            mem, tok_per_s = _measure_batch(model, opt, batch, seq, vocab, device)
        except RuntimeError as exc:
            _free(device)
            if oom_recovery.is_oom_error(exc):
                emit(f"batch {batch}: out of memory (ceiling found)")
                break
            if _is_size_limit_error(exc):
                emit(f"batch {batch}: exceeds the backend tensor-size limit (ceiling found)")
                break
            # Any other failure once a rung has already fit means the ramp found the
            # ceiling; only re-raise if even the first rung fails (a real model bug).
            if ramp:
                emit(f"batch {batch}: failed ({type(exc).__name__}: {exc}); "
                     f"stopping at batch {ramp[-1]['batch']} (ceiling found)")
                break
            raise
        ramp.append({"batch": batch, "mem_gb": mem / GB, "tok_per_s": tok_per_s})
        emit(f"batch {batch}: {mem / GB:.1f} GB, {tok_per_s:,.0f} tok/s")
        last_mem = mem
        _free(device)

    if hasattr(opt, "close"):
        opt.close()

    top = ramp[-1] if ramp else None
    result = {
        "device": device,
        "seq": seq,
        "fits": True,
        "tier": plan.tier if plan is not None else mem_planner.TIER_NONE,
        "max_batch": top["batch"] if top else 0,
        "mem_ceiling_gb": top["mem_gb"] if top else 0.0,
        "tok_per_s": top["tok_per_s"] if top else 0.0,
        "ramp": ramp,
        **_bucket_gb(plan),
    }
    if top:
        emit(f"{ceiling_label}: batch {top['batch']} at {top['mem_gb']:.1f} GB {kind}, "
             f"{top['tok_per_s']:,.0f} tok/s ({device})")
    return result
