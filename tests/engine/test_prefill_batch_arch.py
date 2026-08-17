# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Batched-prefill width is per-arch: measured tiers get their measured value,
#   every other tier stays sequential. Guards against one arch's win being
#   extrapolated onto another (apple silicon regresses 12x when batched).
# tests/engine/test_prefill_batch_arch.py
# ------------------------------------------------------------------------------------
# Imports:

from unittest import mock

from inference.backends import c_engine

# ------------------------------------------------------------------------------------
# Functions


def _batch_on(platform_name, machine):
    with mock.patch.object(c_engine.sys, "platform", platform_name), \
         mock.patch.object(c_engine.platform, "machine", return_value=machine):
        return c_engine.prefill_batch()


def test_linux_x86_uses_its_measured_batch_width():
    """The tier with a measurement gets it; 32 is the measured plateau on cardinal."""
    assert _batch_on("linux", "x86_64") == 32


def test_apple_silicon_stays_sequential():
    """Batching costs 14.4s vs 1.15s on M3 Ultra, so darwin/arm64 must not batch."""
    assert _batch_on("darwin", "arm64") == c_engine.PREFILL_BATCH_DEFAULT == 1


def test_windows_and_unmeasured_tiers_stay_sequential():
    """No measurement means no batching: never extrapolate another arch's win."""
    assert _batch_on("win32", "amd64") == 1
    assert _batch_on("linux", "aarch64") == 1
    assert _batch_on("darwin", "x86_64") == 1


def test_machine_case_is_normalized():
    """platform.machine() casing varies by OS; the lookup must not depend on it."""
    assert _batch_on("linux", "X86_64") == 32


def test_width_never_exceeds_the_engine_scratch_bound():
    """V_PREFILL_BMAX is 64 in veritate.h; a larger value would clamp silently."""
    assert all(1 <= w <= 64 for w in c_engine.PREFILL_BATCH_BY_ARCH.values())
