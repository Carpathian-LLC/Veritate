# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - hardware.py is the single source of truth for device + core detection. These
#   assert the arm64 MPS guard and the device ladder contract that the per-trainer
#   pick_device() copies used to each reimplement (and mostly get wrong).
# tests/plugin_contract/test_hardware.py
# ------------------------------------------------------------------------------------
# Imports

import pytest

# ------------------------------------------------------------------------------------
# Functions

def test_hardware_is_exported():
    """veritate_core.plugin.hardware is part of the plugin surface."""
    from veritate_core.plugin import hardware
    for fn in ("pick_device", "mps_supported", "cuda_supported", "physical_cores"):
        assert hasattr(hardware, fn), f"hardware.{fn} missing from the shared surface"


def test_pick_device_cpu_always_available():
    """pick_device('cpu') returns cpu on every host."""
    from veritate_core.plugin import hardware
    assert hardware.pick_device("cpu") == "cpu"


def test_pick_device_auto_returns_valid():
    """pick_device('auto') resolves to one of the three known backends."""
    from veritate_core.plugin import hardware
    assert hardware.pick_device("auto") in ("cuda", "mps", "cpu")


def test_pick_device_unavailable_backend_raises():
    """Requesting a backend the host lacks raises rather than silently falling back."""
    from veritate_core.plugin import hardware
    if not hardware.cuda_supported():
        with pytest.raises(RuntimeError):
            hardware.pick_device("cuda")
    if not hardware.mps_supported():
        with pytest.raises(RuntimeError):
            hardware.pick_device("mps")


def test_mps_requires_arm64():
    """mps_supported() is False unless the arch is arm64 (Intel-Mac MPS guard)."""
    from veritate_core.plugin import hardware, paths
    if paths.current_arch() != paths.ARCH_ARM64:
        assert hardware.mps_supported() is False


def test_physical_cores_positive():
    """physical_cores() returns a positive int on any host."""
    from veritate_core.plugin import hardware
    n = hardware.physical_cores()
    assert isinstance(n, int) and n >= 1
