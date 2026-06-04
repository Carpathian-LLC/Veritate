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

from readers import paths

# ------------------------------------------------------------------------------------
# Constants

DEVICE_ENV   = "VERITATE_DEVICE"
VALID_FORCED = ("cuda", "mps", "cpu")

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
    they fall through to cpu regardless of the requested/preference value."""
    import torch
    req = (requested or "auto").strip().lower()
    if req == "auto":
        forced = (os.environ.get(DEVICE_ENV) or "auto").strip().lower()
        if forced in VALID_FORCED:
            req = forced
    if req == "cuda":
        if not cuda_supported():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return "cuda"
    if req == "mps":
        if not mps_supported():
            raise RuntimeError("MPS requested but unavailable (needs Apple Silicon + torch MPS)")
        return "mps"
    if req == "cpu":
        return "cpu"
    if cuda_supported():
        return "cuda"
    if mps_supported():
        return "mps"
    return "cpu"


def physical_cores():
    try:
        import psutil
        n = int(psutil.cpu_count(logical=False) or 0)
        if n > 0:
            return n
    except (ImportError, ValueError):
        pass
    return max(1, (os.cpu_count() or 2) // 2)
