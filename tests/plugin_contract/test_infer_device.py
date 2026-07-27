# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for the pytorch inference device selection (_pick_infer_device) and
#   the CPU inference thread cap (_infer_threads). a fake torch stubs mps
#   availability so nothing touches a real accelerator.
# tests/plugin_contract/test_infer_device.py
# ------------------------------------------------------------------------------------
# Imports:



from inference.backends.pytorch import _pick_infer_device

from veritate_core.plugin import hardware

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
    import runtime.settings as _s
    monkeypatch.setattr(_s, "get", lambda: {"device_preference": "auto"})
    assert _pick_infer_device(_FakeTorch(True)) == "mps"


def test_settings_cuda_pref_used_when_env_unset(monkeypatch):
    """When the env var is empty, device_preference from settings is honored."""
    monkeypatch.delenv("VERITATE_INFER_DEVICE", raising=False)
    import runtime.settings as _s
    monkeypatch.setattr(_s, "get", lambda: {"device_preference": "cuda"})

    class _FakeCuda:
        def is_available(self):
            return True

    class _FakeTorchCuda(_FakeTorch):
        def __init__(self, mps_available):
            super().__init__(mps_available)
            self.cuda = _FakeCuda()

    assert _pick_infer_device(_FakeTorchCuda(False)) == "cuda"


def test_auto_prefers_cuda_over_mps(monkeypatch):
    """auto picks cuda when both cuda and mps are available."""
    monkeypatch.setenv("VERITATE_INFER_DEVICE", "auto")

    class _FakeCuda:
        def is_available(self):
            return True

    class _FakeTorchCuda(_FakeTorch):
        def __init__(self, mps_available):
            super().__init__(mps_available)
            self.cuda = _FakeCuda()

    assert _pick_infer_device(_FakeTorchCuda(True)) == "cuda"


def test_cuda_pref_falls_back_to_cpu_when_unavailable(monkeypatch):
    """VERITATE_INFER_DEVICE=cuda falls back to cpu when cuda is absent."""
    monkeypatch.setenv("VERITATE_INFER_DEVICE", "cuda")
    assert _pick_infer_device(_FakeTorch(True)) == "cpu"


def test_thread_ladder_powers_of_two_plus_max():
    """The probe ladder is 1,2,4,... up to and including max_threads."""
    assert hardware._thread_ladder(8) == [1, 2, 4, 8]
    assert hardware._thread_ladder(6) == [1, 2, 4, 6]
    assert hardware._thread_ladder(1) == [1]


def test_infer_threads_accelerator_skips_probe(monkeypatch):
    """A non-cpu device keeps the full count and never runs the probe."""
    monkeypatch.delenv("VERITATE_INFER_THREADS", raising=False)
    monkeypatch.setattr(hardware, "_probe_infer_threads",
                        lambda *a: (_ for _ in ()).throw(AssertionError("probed on accelerator")))
    assert hardware.optimal_infer_threads("mps", 768, 3072, 12, 16) == 16


def test_infer_threads_single_core_skips_probe(monkeypatch):
    """A one-core budget returns 1 without probing."""
    monkeypatch.delenv("VERITATE_INFER_THREADS", raising=False)
    monkeypatch.setattr(hardware, "_probe_infer_threads",
                        lambda *a: (_ for _ in ()).throw(AssertionError("probed at max=1")))
    assert hardware.optimal_infer_threads("cpu", 768, 3072, 12, 1) == 1


def test_infer_threads_env_override_wins(monkeypatch):
    """VERITATE_INFER_THREADS overrides the measured value on any device."""
    monkeypatch.setenv("VERITATE_INFER_THREADS", "2")
    assert hardware.optimal_infer_threads("cpu", 768, 3072, 12, 16) == 2
    assert hardware.optimal_infer_threads("mps", 768, 3072, 12, 16) == 2


def test_infer_threads_cpu_measures_then_caches(monkeypatch):
    """CPU self-calibrates via the probe, then serves the cached value for the shape."""
    monkeypatch.delenv("VERITATE_INFER_THREADS", raising=False)
    hardware._INFER_THREADS_CACHE.clear()
    calls = []
    monkeypatch.setattr(hardware, "_probe_infer_threads",
                        lambda h, f, l, m: (calls.append((h, f, l, m)), 3)[1])
    assert hardware.optimal_infer_threads("cpu", 768, 3072, 12, 16) == 3
    assert hardware.optimal_infer_threads("cpu", 768, 3072, 12, 16) == 3   # cache hit
    assert len(calls) == 1
