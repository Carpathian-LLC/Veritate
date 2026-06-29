# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - local box detection. produces a Capabilities record from real hardware probes.
# - reuses runtime/sys_metrics readings where available; falls back to platform
#   primitives otherwise. all probes are best-effort, none must succeed.
# - node_id is persisted at data/mesh_node_id so a box keeps the same identity
#   across restarts (the hub uses it to match heartbeats to registrations).
# veritate_mesh/capabilities.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import platform
import socket
import uuid

from .protocol import Capabilities, PROTOCOL_VERSION

# ------------------------------------------------------------------------------------
# Constants

BYTES_PER_GB = 1024 ** 3

# ------------------------------------------------------------------------------------
# Functions

def _node_id_path() -> str:
    from readers.paths import REPO_ROOT
    return os.path.join(REPO_ROOT, "data", "mesh_node_id")


def _load_or_create_node_id() -> str:
    p = _node_id_path()
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                v = f.read().strip()
            if v:
                return v
        except OSError:
            pass
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new_id = str(uuid.uuid4())
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(new_id)
    except OSError:
        pass
    return new_id


def _cpu_cores() -> int:
    try:
        import psutil as _ps
        n = _ps.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    return max(1, int(os.cpu_count() or 1))


def _ram_gb() -> float:
    try:
        import psutil as _ps
        return round(_ps.virtual_memory().total / BYTES_PER_GB, 2)
    except ImportError:
        return 0.0


def _gpu_info() -> tuple:
    """best-effort. no hard deps."""
    try:
        import torch
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            return (round(props.total_memory / BYTES_PER_GB, 2), name, "cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return (0.0, "Apple Silicon", "mps")
    except ImportError:
        pass
    except Exception:
        pass
    return (0.0, "", "none")


def _veritate_build() -> int:
    try:
        from readers.paths import REPO_ROOT
        p = os.path.join(REPO_ROOT, "versions.json")
        with open(p, "r", encoding="utf-8") as f:
            return int((json.load(f) or {}).get("build") or 0)
    except (OSError, ValueError, KeyError):
        return 0


def detect() -> Capabilities:
    """probe the local box. cheap; safe to call on each registration."""
    vram_gb, gpu_name, gpu_backend = _gpu_info()
    sys_lower = platform.system().lower()
    if sys_lower.startswith("win"):
        os_name = "windows"
    elif sys_lower == "darwin":
        os_name = "darwin"
    else:
        os_name = "linux"
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine or "unknown"
    return Capabilities(
        node_id          = _load_or_create_node_id(),
        hostname         = socket.gethostname() or "unknown",
        os_name          = os_name,
        arch             = arch,
        cpu_cores        = _cpu_cores(),
        ram_gb           = _ram_gb(),
        vram_gb          = vram_gb,
        gpu_name         = gpu_name,
        gpu_backend      = gpu_backend,
        veritate_build   = _veritate_build(),
        protocol_version = PROTOCOL_VERSION,
    )
