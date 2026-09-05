# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Single source of truth for compute-device + core-count detection. Replaces the
#   per-trainer pick_device() copies that lacked the arm64 MPS guard. Arch/OS strings
#   come from readers.paths (the canonical normalizer); this only adds the torch-aware
#   device ladder and physical-core probe. Trainers reach this via veritate_core.plugin.
# veritate_core/plugin/hardware.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

from readers import paths

# ------------------------------------------------------------------------------------
# Constants

DEVICE_ENV   = "VERITATE_DEVICE"
VALID_FORCED = ("cuda", "mps", "cpu")
KIB          = 1024

# Autocast choices a trainer may ask for. `auto` is the half precision the device
# measures fastest (half_precision_probe); the probe is a square matmul, sized so it
# takes well under a second on any GPU and is cached per process.
PRECISION_CHOICES  = ("auto", "fp16", "bf16", "fp32")
# 2048 is the smallest square that ranks the dtypes the same way 4096 does on an M2
# (1024 is launch-bound and ranked them wrong in one of two trials); ~0.3 s for all three.
HALF_PROBE_N       = 2048
HALF_PROBE_REPS    = 6
HALF_PROBE_WARM_N  = 512       # one untimed matmul first, so the first dtype does not pay the cold start
HALF_PROBE_MARGIN  = 1.10      # fp16 must beat bf16 by this much to displace the upstream default
_HALF_PROBE_CACHE  = {}        # device -> (dtype or None, {"fp16": tflops, "bf16": ..., "fp32": ...})

# CPU inference thread autotune. Batch-1 byte decode is latency-bound: on some
# cores more threads help (bandwidth-rich), on others OpenMP spin-waits at the
# per-token barriers and fewer win. Rather than a per-machine constant, the box
# measures itself once and picks the SMALLEST thread count within INFER_TIE_MARGIN
# of the fastest (frugal: on a flat curve fewer threads is just as quick and frees
# cores; on a spin-wait curve it lands on the real minimum). INFER_THREADS_ENV
# forces a value (CI / debugging).
INFER_THREADS_ENV     = "VERITATE_INFER_THREADS"
INFER_PROBE_WARMUP    = 6
INFER_PROBE_ITERS     = 30
INFER_PROBE_REPEATS   = 2      # min over repeats per count, to shrug off scheduler noise
INFER_PROBE_MAX_REPS  = 24     # cap the per-step layer reps so a deep model still probes fast
INFER_TIE_MARGIN      = 0.08   # a wider pool must be >8% faster than the frugal pick to win
INFER_SHAPE_BUCKET    = 256    # cache key granularity over hidden/ffn, so near-shapes share a probe
_INFER_THREADS_CACHE  = {}     # (arch, cores, hidden_bucket, ffn_bucket) -> measured threads

# ------------------------------------------------------------------------------------
# Functions


def mps_supported():
    import torch
    return bool(getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                and paths.current_arch() == paths.ARCH_ARM64)


def cuda_supported():
    import torch
    return bool(torch.cuda.is_available())


def pick_device(requested="auto"):
    """Resolve a torch device string. `requested` is a CLI value; when "auto",
    the dashboard's VERITATE_DEVICE env override is consulted before auto-detect.
    MPS is arm64-guarded: Intel Macs report mps available but crash mid-step, so
    they fall through to cpu. A forced device that isn't available falls back to
    auto-detect with a warning rather than raising, so no machine is left unable
    to train."""
    req = (requested or "auto").strip().lower()
    if req == "auto":
        forced = (os.environ.get(DEVICE_ENV) or "auto").strip().lower()
        if forced in VALID_FORCED:
            req = forced
    if req == "cuda" and cuda_supported():
        return "cuda"
    if req == "mps" and mps_supported():
        return "mps"
    if req == "cpu":
        return "cpu"
    if req in ("cuda", "mps"):
        from runtime import logs as logmod
        logmod.warn("hardware", f"device {req!r} requested but unavailable; auto-detecting")
    if cuda_supported():
        return "cuda"
    if mps_supported():
        return "mps"
    return "cpu"


def bf16_supported(device):
    """True when `device` runs bf16 autocast at all. CUDA consults torch; MPS
    supports bf16 autocast (not always quickly: see half_precision_probe); CPU is
    False because torch CPU autocast bf16 is emulated (slower than fp32 even with
    AVX512-BF16) and doubles activation bytes on weak boxes."""
    import torch
    if device == "cuda":
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    return device == "mps"


def _gemm_tflops(device, dtype, n=HALF_PROBE_N, reps=HALF_PROBE_REPS):
    import time

    import torch
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    sync = torch.mps.synchronize if device == "mps" else (torch.cuda.synchronize if device == "cuda" else None)
    for _ in range(2):
        a @ b
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    if sync:
        sync()
    return 2.0 * n ** 3 * reps / max(1e-9, time.perf_counter() - t0) / 1e12


def half_precision_probe(device):
    """Which half precision this GPU is fast at, MEASURED once per process: a square
    matmul in fp16, bf16 and fp32. Apple's M1/M2 GPUs run bf16 at roughly half the fp16
    rate (M2, 2026-09-05: fp16 3.2 TFLOPS, fp32 2.8, bf16 1.5), so the upstream default
    of bf16 costs a picture model a quarter of its step there. Returns
    (dtype, {"fp16": tflops, "bf16": tflops, "fp32": tflops}); dtype is None where
    autocast is not worth running (CPU), and bf16 without a measurement where the
    device is not present (an mps question asked on a Linux box)."""
    import torch
    if device in _HALF_PROBE_CACHE:
        return _HALF_PROBE_CACHE[device]
    if device == "cpu":
        result = (None, {})
    elif device == "cuda":
        result = (torch.bfloat16 if bf16_supported(device) else torch.float16, {})
    elif device == "mps" and not mps_supported():
        result = (torch.bfloat16, {})
    else:
        rates = {}
        try:
            _gemm_tflops(device, torch.float32, n=HALF_PROBE_WARM_N, reps=1)     # pay the cold start once
        except RuntimeError:
            pass
        for label, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
            try:
                rates[label] = round(_gemm_tflops(device, dtype), 2)
            except RuntimeError:
                rates[label] = 0.0
        pick = torch.float16 if rates["fp16"] >= rates["bf16"] * HALF_PROBE_MARGIN else torch.bfloat16
        result = (pick, rates)
    _HALF_PROBE_CACHE[device] = result
    return result


def precision_label(amp_dtype):
    """The name a run log and config record for an autocast dtype."""
    import torch
    if amp_dtype is None:
        return "fp32"
    return {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(amp_dtype, str(amp_dtype).split(".")[-1])


def resolve_precision(requested, device):
    """Autocast dtype for `device` given a CLI precision string: `fp32` (None), `bf16`,
    `fp16`, or `auto`, which is the half precision this device MEASURES fastest
    (half_precision_probe). Downgrades a half precision to fp32 (None) when the device
    cannot run it, so no machine runs an emulated path; warns once about bf16 on a GPU
    that is measured slow at it. Returns a torch dtype for autocast, or None for fp32."""
    import torch
    want = (requested or "").strip().lower()
    if want not in PRECISION_CHOICES or want == "fp32":
        return None
    from runtime import logs as logmod
    if device == "cpu":
        if want != "fp32":
            logmod.warn("hardware", f"{want} requested but cpu has no half-precision acceleration: running fp32")
        return None
    if want == "auto":
        return half_precision_probe(device)[0]
    if want == "fp16":
        return torch.float16
    if not bf16_supported(device):
        logmod.warn("hardware", f"bf16 requested but {device} has no bf16 acceleration: running fp32")
        return None
    fast, rates = half_precision_probe(device)
    if fast is torch.float16 and rates:
        logmod.warn("hardware", "bf16 requested but this GPU measures fp16 at " + str(rates.get("fp16"))
                    + " TFLOPS against bf16 at " + str(rates.get("bf16")) + ": precision=auto or fp16 is faster")
    return torch.bfloat16


def physical_cores():
    try:
        import psutil
        n = int(psutil.cpu_count(logical=False) or 0)
        if n > 0:
            return n
    except (ImportError, ValueError):
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _thread_ladder(max_threads):
    """Thread counts to probe: 1, 2, 4, ... up to max_threads (inclusive)."""
    out, n = [], 1
    while n < max_threads:
        out.append(n)
        n *= 2
    out.append(max(1, int(max_threads)))
    return sorted(set(out))


def _probe_infer_threads(hidden, ffn, layers, max_threads):
    """Time a representative batch-1 decode step (the hidden/ffn GEMVs one token
    runs) at each ladder count, then return the SMALLEST count whose latency is
    within INFER_TIE_MARGIN of the fastest. Frugal by construction: a flat
    (bandwidth-rich) curve settles low with no speed loss; a spin-wait curve lands
    on its real minimum. Each count is timed INFER_PROBE_REPEATS times, min kept."""
    import time

    import torch
    reps = max(1, min(int(layers), INFER_PROBE_MAX_REPS))
    prev = torch.get_num_threads()
    try:
        x  = torch.randn(1, hidden)
        w1 = torch.randn(hidden, 3 * hidden)
        w2 = torch.randn(hidden, hidden)
        w3 = torch.randn(hidden, ffn)
        w4 = torch.randn(ffn, hidden)

        def step():
            with torch.no_grad():
                for _ in range(reps):
                    h = (x @ w1)[:, :hidden] @ w2
                    torch.relu((x + h) @ w3) @ w4

        times = {}
        for n in _thread_ladder(max_threads):
            torch.set_num_threads(n)
            for _ in range(INFER_PROBE_WARMUP):
                step()
            best = None
            for _ in range(INFER_PROBE_REPEATS):
                t0 = time.perf_counter()
                for _ in range(INFER_PROBE_ITERS):
                    step()
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            times[n] = best
        floor = min(times.values())
        for n in sorted(times):
            if times[n] <= floor * (1.0 + INFER_TIE_MARGIN):
                return n
        return min(times, key=times.get)
    finally:
        torch.set_num_threads(prev)


def optimal_infer_threads(device, hidden, ffn, layers, max_threads):
    """CPU inference thread count for THIS box, measured once and cached. On an
    accelerator (mps/cuda) the forward runs off-core, so the full count is kept
    with no probe. On CPU the box self-calibrates via _probe_infer_threads: no
    per-machine constant. INFER_THREADS_ENV (int) overrides everything."""
    override = (os.environ.get(INFER_THREADS_ENV, "") or "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    max_threads = max(1, int(max_threads))
    if str(device) != "cpu" or max_threads == 1:
        return max_threads
    key = (paths.current_arch(), physical_cores(),
           int(hidden) // INFER_SHAPE_BUCKET, int(ffn) // INFER_SHAPE_BUCKET)
    hit = _INFER_THREADS_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        n = _probe_infer_threads(int(hidden), int(ffn), int(layers), max_threads)
    except Exception:
        n = max_threads
    _INFER_THREADS_CACHE[key] = n
    return n


def unified_memory_bytes():
    """Total addressable RAM in bytes. On Apple Silicon this is the unified pool
    the GPU and CPU share, so it doubles as the training-memory ceiling. On
    discrete-GPU hosts it is system RAM, not VRAM; mem_planner treats it as the
    unified budget only when the device is mps."""
    try:
        import psutil
        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except (ImportError, ValueError):
        pass
    page = os.sysconf("SC_PAGE_SIZE")
    count = os.sysconf("SC_PHYS_PAGES")
    return int(page) * int(count)


def available_memory_bytes():
    """RAM free for a new process right now, in bytes: the pool less every other
    tenant (served engines, a neighbour's download, a suspended sleep child that
    keeps its RSS). Falls back to the total where the probe is unavailable."""
    try:
        import psutil
        avail = int(psutil.virtual_memory().available)
        if avail > 0:
            return avail
    except (ImportError, ValueError):
        pass
    return unified_memory_bytes()


def process_peak_rss_bytes():
    """Peak resident memory of this process in bytes: the high-water mark since
    start. Monotonic, no OS reset exists. Windows exposes peak_wset via psutil;
    POSIX uses resource.ru_maxrss (bytes on darwin, KiB on linux). Used as the
    cpu memory-ceiling probe where no device allocator counter exists."""
    if sys.platform == "win32":
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is None:
            peak = getattr(info, "rss", 0)
        return int(peak)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * KIB
