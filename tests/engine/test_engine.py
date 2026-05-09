# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Engine smoke tests. Builds nothing; runs the existing binary if present
#   and skips otherwise. The default no-arg invocation is the engine's
#   built-in self-test (kernel verify, tokenizer round-trip).
# - Per-model load tests are gated on @pytest.mark.slow because they can
#   take several seconds per model and need real .bin files in models/.
# tests/engine/test_engine.py
# ------------------------------------------------------------------------------------
# Imports

import os
import subprocess

import pytest

# ------------------------------------------------------------------------------------
# Constants

ENGINE_TIMEOUT_SECS = 30

# ------------------------------------------------------------------------------------
# Functions

def test_engine_self_test_runs_and_exits_zero(engine_binary_or_skip):
    """Engine no-arg invocation runs the built-in kernel + tokenizer self-test
    and exits 0. Catches kernel regressions immediately."""
    r = subprocess.run([engine_binary_or_skip], capture_output=True, text=True,
                       timeout=ENGINE_TIMEOUT_SECS)
    assert r.returncode == 0, f"engine self-test exited {r.returncode}\nstderr: {r.stderr}"


def test_engine_self_test_reports_verify_ok(engine_binary_or_skip):
    """Self-test output must contain 'verify OK' lines (scalar-vs-SIMD bitwise
    parity). A regression here means a kernel diverged from its scalar oracle."""
    r = subprocess.run([engine_binary_or_skip], capture_output=True, text=True,
                       timeout=ENGINE_TIMEOUT_SECS)
    assert "verify OK" in r.stdout, "engine self-test should report 'verify OK'\n" + r.stdout[:500]


def test_engine_reports_cpu_features(engine_binary_or_skip):
    """Self-test prints detected CPU and SIMD features. Confirms dispatch
    table is wired."""
    r = subprocess.run([engine_binary_or_skip], capture_output=True, text=True,
                       timeout=ENGINE_TIMEOUT_SECS)
    out = r.stdout
    assert "cpu:" in out,        "engine should print detected cpu line"
    assert "features:" in out,   "engine should print detected features line"
    assert "dispatch:" in out,   "engine should print kernel dispatch line"


def test_engine_rejects_nonexistent_model_path(engine_binary_or_skip, tmp_path):
    """When VERITATE_MODEL_PATH points at nothing, the engine must NOT crash
    silently. Either it falls back to the random-init self-test (exit 0) or
    reports an explicit error (non-zero with a stderr message)."""
    bogus = str(tmp_path / "does_not_exist.bin")
    env = {**os.environ, "VERITATE_MODEL_PATH": bogus}
    r = subprocess.run([engine_binary_or_skip], capture_output=True, text=True,
                       timeout=ENGINE_TIMEOUT_SECS, env=env)
    # Either path is acceptable; what we want to forbid is a segfault (returncode = -11
    # or similar negative on POSIX). Anything zero or positive is a clean exit.
    assert r.returncode >= 0, f"engine crashed (signal): rc={r.returncode}\nstderr: {r.stderr[:500]}"


@pytest.mark.slow
def test_engine_loads_each_model_bin_or_rejects_cleanly(engine_binary_or_skip, model_bins):
    """For each .bin in models/, the engine must either load it (exit 0) or
    refuse with a structured error message. A segfault or hang fails the test.

    This is `@pytest.mark.slow` because it touches every committed model
    artifact. Run with `pytest -m slow` to include it."""
    if not model_bins:
        pytest.skip("no .bin files in models/")

    known_clean_rejections = (
        "act_boost",                     # non-QAT BOOST guard
        "magic version mismatch",
        "unknown version",
        "model_load:",
        "shape mismatch",
        "vocab",
        "version",
    )

    for bin_path in model_bins:
        env = {**os.environ, "VERITATE_MODEL_PATH": bin_path}
        # Use bench mode (1 forward + 1 decode), bounded by timeout.
        r = subprocess.run([engine_binary_or_skip, "bench", "1", "1"],
                           capture_output=True, text=True,
                           timeout=ENGINE_TIMEOUT_SECS, env=env)
        if r.returncode == 0:
            continue
        assert r.returncode > 0, \
            f"{bin_path}: engine crashed (signal {r.returncode})\nstderr: {r.stderr[:500]}"
        clean = any(s in r.stderr for s in known_clean_rejections)
        assert clean, \
            f"{bin_path}: engine refused with unknown error\nstderr: {r.stderr[:500]}"
