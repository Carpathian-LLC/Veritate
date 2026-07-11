# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for the pytorch inference device selection (_pick_infer_device).
#   a fake torch stubs mps availability so nothing touches a real accelerator.
# tests/mri/test_infer_device.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from inference.backends.pytorch import _pick_infer_device

# ------------------------------------------------------------------------------------
# Constants


class _FakeMps:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeBackends:
    def __init__(self, mps):
        if mps is not None:
            self.mps = mps


class _FakeTorch:
    """Minimal torch stub: device() echoes the string, backends.mps mimics presence."""
    def __init__(self, mps_available):
        self.backends = _FakeBackends(_FakeMps(mps_available) if mps_available is not None else None)

    def device(self, name):
        return name


# ------------------------------------------------------------------------------------
# Functions

def test_cpu_pref_forces_cpu_even_with_mps(monkeypatch):
    """VERITATE_INFER_DEVICE=cpu returns cpu even when mps is available."""
    monkeypatch.setenv("VERITATE_INFER_DEVICE", "cpu")
    assert _pick_infer_device(_FakeTorch(True)) == "cpu"


def test_mps_pref_takes_mps_when_available(monkeypatch):
    """VERITATE_INFER_DEVICE=mps returns mps when available."""
    monkeypatch.setenv("VERITATE_INFER_DEVICE", "mps")
    assert _pick_infer_device(_FakeTorch(True)) == "mps"


def test_mps_pref_falls_back_to_cpu_when_unavailable(monkeypatch):
    """VERITATE_INFER_DEVICE=mps falls back to cpu when mps is absent."""
    monkeypatch.setenv("VERITATE_INFER_DEVICE", "mps")
    assert _pick_infer_device(_FakeTorch(False)) == "cpu"


def test_auto_takes_mps_when_available(monkeypatch):
    """auto (default) takes mps when the platform offers it."""
    monkeypatch.setenv("VERITATE_INFER_DEVICE", "auto")
    assert _pick_infer_device(_FakeTorch(True)) == "mps"


def test_auto_falls_back_to_cpu_without_mps(monkeypatch):
    """auto falls back to cpu when there is no mps backend at all."""
    monkeypatch.setenv("VERITATE_INFER_DEVICE", "auto")
    assert _pick_infer_device(_FakeTorch(None)) == "cpu"


def test_unset_defaults_to_auto(monkeypatch):
    """An unset preference behaves as auto (mps when available)."""
    monkeypatch.delenv("VERITATE_INFER_DEVICE", raising=False)
    assert _pick_infer_device(_FakeTorch(True)) == "mps"
