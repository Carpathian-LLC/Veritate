# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sysprobe measures the box Auto tune then sizes a run against, so its numbers end up
#   deciding batch and precision. Pinned here: the disk probe cleans up after itself and
#   never writes into the repo by default, a GPU that fails to probe does not take the
#   whole probe down with it, a box with no accelerator still returns a full payload, the
#   RAM probe degrades to nulls without psutil, and the shape every consumer reads.
#   The real disk and CPU work is slow, so it is stubbed except in one marked-slow test.
# tests/plugin_contract/test_sysprobe.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import tempfile

import pytest

from veritate_core.plugin import sysprobe

# ------------------------------------------------------------------------------------
# Constants

DISK = {"path": "/tmp", "seq_write_mb_s": 900.0, "rand_write_mb_s": 12.0, "free_gb": 100.0}
CPU = {"physical_cores": 8, "brand": "i7", "arch": "x86_64",
       "matmul_gflops": 120.0, "copy_gb_s": 9.0}
RAM = {"total_gb": 23.0, "available_gb": 14.0, "swap_total_gb": 2.0, "swap_used_gb": 0.0}

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(sysprobe, "_probe_disk", lambda d: dict(DISK, path=d))
    monkeypatch.setattr(sysprobe, "_probe_cpu", lambda: dict(CPU))
    monkeypatch.setattr(sysprobe, "_probe_ram", lambda: dict(RAM))


def test_the_payload_carries_every_section(stubbed, monkeypatch):
    """bench and the planner read these keys; a missing one is a KeyError at tune time."""
    monkeypatch.setattr(sysprobe, "_torch_devices", list)
    out = sysprobe.run()
    assert set(out) == {"captured_at", "elapsed_s", "os", "arch", "disk", "cpu", "gpus", "ram"}
    assert out["cpu"]["physical_cores"] == 8 and out["gpus"] == []


def test_a_cpu_only_box_says_so(stubbed, monkeypatch):
    """Not an error: cardinal has no accelerator and must still tune."""
    monkeypatch.setattr(sysprobe, "_torch_devices", list)
    lines = []
    sysprobe.run(on_progress=lines.append)
    assert any("no torch accelerator" in ln for ln in lines)


def test_one_failing_gpu_does_not_take_the_probe_down(stubbed, monkeypatch):
    """A dual-card rig with one bad device must still report the good one."""
    monkeypatch.setattr(sysprobe, "_torch_devices",
                        lambda: [("cuda", 0, "good"), ("cuda", 1, "bad")])

    def _gpu(kind, index, name):
        if name == "bad":
            raise RuntimeError("device lost")
        return {"device": f"{kind}:{index}", "name": name, "matmul_tflops": 30.0,
                "vram_total_gb": 24.0}

    monkeypatch.setattr(sysprobe, "_probe_gpu", _gpu)
    lines = []
    out = sysprobe.run(on_progress=lines.append)
    assert [g["name"] for g in out["gpus"]] == ["good"]
    assert any("probe failed" in ln and "device lost" in ln for ln in lines)


def test_progress_is_optional(stubbed, monkeypatch):
    """The route calls it without a callback."""
    monkeypatch.setattr(sysprobe, "_torch_devices", list)
    assert sysprobe.run()["elapsed_s"] >= 0


def test_the_disk_probe_defaults_outside_the_repo(monkeypatch):
    """It writes 64 MB; the default must never land in the working tree."""
    seen = []

    def _disk(d):
        seen.append(d)
        return dict(DISK, path=d)

    monkeypatch.setattr(sysprobe, "_probe_disk", _disk)
    monkeypatch.setattr(sysprobe, "_probe_cpu", lambda: dict(CPU))
    monkeypatch.setattr(sysprobe, "_probe_ram", lambda: dict(RAM))
    monkeypatch.setattr(sysprobe, "_torch_devices", list)
    sysprobe.run()
    assert seen == [tempfile.gettempdir()]


def test_ram_reads_as_nulls_without_psutil(monkeypatch):
    """psutil is optional; the probe reports unknown rather than refusing to run."""
    import builtins
    real = builtins.__import__

    def _no_psutil(name, *a, **kw):
        if name == "psutil":
            raise ImportError("no psutil")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    assert sysprobe._probe_ram() == {"total_gb": None, "available_gb": None,
                                     "swap_total_gb": None, "swap_used_gb": None}


@pytest.mark.slow
def test_the_disk_probe_measures_and_cleans_up(tmp_path):
    """It writes real files; leaving them behind fills a training box over time."""
    out = sysprobe._probe_disk(str(tmp_path))
    assert out["seq_write_mb_s"] > 0 and out["free_gb"] > 0
    assert os.listdir(tmp_path) == []
