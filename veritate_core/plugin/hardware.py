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
    """True when `device` has real bf16 acceleration. CUDA consults torch; MPS
    supports bf16 autocast; CPU is False because torch CPU autocast bf16 is
    emulated (slower than fp32 even with AVX512-BF16) and doubles activation
    bytes on weak boxes."""
    import torch
    if device == "cuda":
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if device == "mps":
        return True
    return False


def resolve_precision(requested, device):
    """Autocast dtype for `device` given a CLI precision string. Downgrades bf16
    to fp32 (None) when the device lacks bf16 acceleration, so no machine runs
    the emulated bf16 path. Returns a torch dtype for autocast, or None for
    fp32. Logs one line on downgrade."""
    import torch
    if (requested or "").strip().lower() != "bf16":
        return None
    if bf16_supported(device):
        return torch.bfloat16
    from runtime import logs as logmod
    logmod.warn("hardware", f"bf16 requested but {device} has no bf16 acceleration: running fp32")
    return None


def physical_cores():
    try:
        import psutil
        n = int(psutil.cpu_count(logical=False) or 0)
        if n > 0:
            return n
    except (ImportError, ValueError):
        pass
    return max(1, (os.cpu_count() or 2) // 2)


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


def process_peak_rss_bytes():
    """Peak resident memory of this process in bytes: the high-water mark since
    start. Monotonic, no OS reset exists. ru_maxrss units differ by platform:
    bytes on darwin, KiB on linux. Used as the cpu memory-ceiling probe where no
    device allocator counter exists."""
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * KIB
