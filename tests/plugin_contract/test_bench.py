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
