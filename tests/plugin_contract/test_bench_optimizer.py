# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The bench sizes batch and reports the throughput a launch is planned against
#   (preflight 24e). It probed with AdamW while every real run trains on Muon, so it
#   over-reported: 18,588 tok/s predicted against 12,120 measured on a 270M hybrid.
#   Newton-Schulz runs on every 2D weight every step and has to be in the probe.
# tests/plugin_contract/test_bench_optimizer.py
# ------------------------------------------------------------------------------------
# Imports:

import torch

from veritate_core.plugin import bench

# ------------------------------------------------------------------------------------
# Constants

HIDDEN = 32

# ------------------------------------------------------------------------------------
# Functions


class _Tiny(torch.nn.Module):
    """Two 2D weights (Muon's territory) plus a 1D bias (AdamW's)."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(HIDDEN, HIDDEN)
        self.b = torch.nn.Linear(HIDDEN, HIDDEN)


def test_probe_muon_builds_a_stepping_optimizer():
    """Guards the probe-args shim: build_muon reads trainer arg names off it."""
    m = _Tiny()
    opt = bench._probe_muon(m)
    out = m.b(m.a(torch.randn(2, HIDDEN))).sum()
    out.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_probe_args_carry_every_field_build_muon_reads():
    for field in ("base_lr", "weight_decay", "beta1", "beta2", "use_8bit_adam"):
        assert hasattr(bench._ProbeArgs, field), field


def test_probe_args_never_request_8bit_adam():
    """8-bit AdamW needs CUDA and silently falls back; a probe must not depend on it."""
    assert bench._ProbeArgs.use_8bit_adam is False


def test_muon_is_matched_case_insensitively():
    assert bench.PROBE_MUON == "muon"
    assert "MUON".lower() == bench.PROBE_MUON


def test_best_batch_is_the_throughput_peak_not_the_memory_ceiling():
    """Throughput is not monotonic in batch: under Muon this box peaks at 32, still
    fits 64, and collapses there. Launching at the ceiling is a 12x mistake."""
    ramp = [{"batch": 16, "mem_gb": 45.8, "tok_per_s": 11236.0},
            {"batch": 32, "mem_gb": 77.4, "tok_per_s": 13207.0},
            {"batch": 48, "mem_gb": 131.7, "tok_per_s": 12121.0},
            {"batch": 64, "mem_gb": 166.1, "tok_per_s": 1040.0}]
    best = max(ramp, key=lambda r: r["tok_per_s"])
    top = ramp[-1]
    assert best["batch"] == 32
    assert top["batch"] == 64
    assert best["tok_per_s"] / top["tok_per_s"] > 12


class _Clock:
    """Deterministic stand-in for the ramp's wall clock."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def test_ramp_stops_before_a_rung_the_time_budget_cannot_pay_for(monkeypatch):
    """A slow box reports what it measured instead of running to the memory ceiling."""
    clock = _Clock()
    monkeypatch.setattr(bench.time, "monotonic", clock.monotonic)

    def fake_measure(model, opt, batch, *a, **kw):
        clock.t += float(batch)          # step time scales with batch
        return 1024, 100.0 * batch       # tiny footprint: the memory guard never fires

    monkeypatch.setattr(bench, "_measure_batch", fake_measure)
    res = bench.run(_Tiny(), "cpu", seq=8, vocab=256, batch_ramp=(1, 2, 4, 8, 16), budget_s=10.0)
    # 1 -> t=1; 2 projected 2 (total 3); 4 projected 4 (total 7); 8 projected 8 -> 15 > 10.
    assert [r["batch"] for r in res["ramp"]] == [1, 2, 4]
    assert res["time_capped"] is True
    assert res["best_batch"] == 4


def test_a_ramp_that_completes_is_not_flagged_time_capped(monkeypatch):
    """The flag distinguishes a measured ceiling from a budget cutoff."""
    clock = _Clock()
    monkeypatch.setattr(bench.time, "monotonic", clock.monotonic)

    def fake_measure(model, opt, batch, *a, **kw):
        clock.t += 0.001
        return 1024, 100.0 * batch

    monkeypatch.setattr(bench, "_measure_batch", fake_measure)
    res = bench.run(_Tiny(), "cpu", seq=8, vocab=256, batch_ramp=(1, 2), budget_s=10.0)
    assert res["time_capped"] is False
    assert [r["batch"] for r in res["ramp"]] == [1, 2]
