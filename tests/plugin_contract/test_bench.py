# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - bench.run ramps batch size on a real model and reports the measured memory ceiling
#   and throughput. CPU has no device OOM, so the happy-path test walks the full ramp
#   and checks the contract: a result for every rung, positive throughput, progress
#   lines emitted. The OOM-ceiling path is device-specific and exercised on mps.
# tests/plugin_contract/test_bench.py
# ------------------------------------------------------------------------------------
# Imports

import pytest
import torch

from veritate_core import model as vmodel
from veritate_core.plugin import bench
from veritate_core.plugin import mem_planner as mp
from veritate_core.plugin import paged_optimizer

# ------------------------------------------------------------------------------------
# Constants

SMALL = dict(vocab=vmodel.VOCAB_BYTE_LEVEL, hidden=64, layers=2, ffn=128,
             heads=4, seq=16)
RAMP = (1, 2, 4)

# ------------------------------------------------------------------------------------
# Functions


def _model():
    torch.manual_seed(0)
    return vmodel.Veritate(**SMALL)


def test_bench_is_exported():
    """bench is on the plugin surface with run()."""
    from veritate_core.plugin import bench as b
    assert hasattr(b, "run")


def test_bench_returns_result_for_each_rung_on_cpu():
    """On CPU (no OOM) the ramp completes and reports every batch size."""
    lines = []
    result = bench.run(_model(), "cpu", SMALL["seq"], SMALL["vocab"],
                       batch_ramp=RAMP, on_progress=lines.append)
    assert result["max_batch"] == RAMP[-1]
    assert [r["batch"] for r in result["ramp"]] == list(RAMP)
    assert lines, "progress lines must be emitted for the modal"


def test_bench_throughput_is_positive():
    """Measured tok/s at the ceiling is positive."""
    result = bench.run(_model(), "cpu", SMALL["seq"], SMALL["vocab"], batch_ramp=RAMP)
    assert result["tok_per_s"] > 0
    assert result["device"] == "cpu"


def test_bench_saves_nothing(tmp_path):
    """bench writes no files (no checkpoint, no real weights touched)."""
    before = set(tmp_path.iterdir())
    bench.run(_model(), "cpu", SMALL["seq"], SMALL["vocab"], batch_ramp=(1, 2))
    assert set(tmp_path.iterdir()) == before


def test_bench_paged_plan_uses_offload_and_reports_tier():
    """A page-tier plan makes bench probe with the NVMe-paged optimizer and tag it."""
    plan = mp.MemoryPlan(mp.TIER_PAGE, True, 8 << 30, 4 << 30, 1 << 30, 1 << 30, 0, 1 << 20)
    seen = {"paged": False}
    real_make = bench.mem_executor.make_optimizer

    def _spy(*a, **k):
        opt = real_make(*a, **k)
        seen["paged"] = isinstance(opt, paged_optimizer.PagedAdamW)
        return opt

    bench.mem_executor.make_optimizer = _spy
    try:
        result = bench.run(_model(), "cpu", SMALL["seq"], SMALL["vocab"],
                           plan=plan, batch_ramp=(1, 2))
    finally:
        bench.mem_executor.make_optimizer = real_make
    assert seen["paged"] is True
    assert result["fits"] is True
    assert result["tier"] == mp.TIER_PAGE
    assert result["budget_gb"] == pytest.approx(8.0)


def test_plan_result_for_infeasible_size():
    """plan_result reports an unfittable size as fits=False with the floor numbers."""
    plan = mp.MemoryPlan(mp.TIER_INFEASIBLE, False, 218 << 30, 1610 << 30,
                         400 << 30, 400 << 30, 810 << 30, 1 << 20)
    res = bench.plan_result(plan, "mps", 1024)
    assert res["fits"] is False
    assert res["max_batch"] == 0
    assert res["ramp"] == []
    assert res["required_gb"] == pytest.approx(1610.0)
    assert res["budget_gb"] == pytest.approx(218.0)


def test_ramp_stops_at_memory_budget(monkeypatch):
    """The ramp stops before attempting the rung after the one that hit the budget,
    so an over-budget allocation is never attempted (it would SIGKILL on unified mem)."""
    mems = {1: 200, 2: 205, 4: 210, 8: 235, 16: 280}  # GB, grows with batch
    monkeypatch.setattr(bench, "_measure_batch",
                        lambda model, opt, batch, seq, vocab, device: (mems[batch] * bench.GB, 100.0 * batch))
    monkeypatch.setattr(bench, "_memory_budget", lambda d: 217 * bench.GB)
    monkeypatch.setattr(bench, "_free", lambda d: None)
    result = bench.run(_model(), "mps", SMALL["seq"], SMALL["vocab"], batch_ramp=(1, 2, 4, 8, 16))
    # batch 8 (235 GB) reaches the 217 GB budget, so batch 16 is never attempted.
    assert [r["batch"] for r in result["ramp"]] == [1, 2, 4, 8]
    assert result["max_batch"] == 8


@pytest.mark.slow
def test_bench_finds_ceiling_on_mps():
    """On mps the ramp hits a real OOM ceiling and reports a finite max_batch."""
    from veritate_core.plugin import hardware
    if not hardware.mps_supported():
        pytest.skip("needs mps")
    big = dict(vocab=vmodel.VOCAB_BYTE_LEVEL, hidden=256, layers=4, ffn=1024,
               heads=8, seq=128)
    torch.manual_seed(0)
    m = vmodel.Veritate(**big).to("mps")
    result = bench.run(m, "mps", big["seq"], big["vocab"])
    assert result["max_batch"] >= 1
    assert result["mem_ceiling_gb"] > 0
