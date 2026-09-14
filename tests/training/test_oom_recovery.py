# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - OOM recovery is backend-agnostic text matching plus a bounded re-exec: the marker
#   list catches CUDA / MPS / CPU exhaustion and nothing else, the fallback depth comes
#   from the environment and stops at MAX_FALLBACKS, and the re-exec forces the safe
#   flags on (dropping their --no- forms) with the depth counter advanced.
# tests/training/test_oom_recovery.py
# ------------------------------------------------------------------------------------
# Imports:

import sys

import pytest

from veritate_core.plugin import oom_recovery as oom

# ------------------------------------------------------------------------------------
# Functions


@pytest.mark.parametrize("msg", ["CUDA error: out of memory", "MPS backend out of memory (MPS allocated: 9 GB)",
                                 "RuntimeError: could not allocate 4096 bytes", "alloc failed"])
def test_memory_exhaustion_is_recognised_across_backends(msg):
    assert oom.is_oom_error(RuntimeError(msg))


def test_other_errors_are_not_oom():
    assert not oom.is_oom_error(ValueError("shape mismatch"))


def test_fallback_depth_reads_the_environment_and_is_bounded(monkeypatch):
    monkeypatch.delenv(oom.COUNTER_ENV, raising=False)
    assert oom.fallback_count() == 0 and oom.should_recover()
    monkeypatch.setenv(oom.COUNTER_ENV, "garbage")
    assert oom.fallback_count() == 0
    monkeypatch.setenv(oom.COUNTER_ENV, str(oom.MAX_FALLBACKS))
    assert not oom.should_recover()


def test_reexec_forces_the_safe_flags_on_and_advances_the_counter(monkeypatch):
    seen = {}
    monkeypatch.setattr(oom.os, "execve", lambda exe, argv, env: seen.update(exe=exe, argv=argv, env=env))
    monkeypatch.setattr(sys, "argv", ["trainer.py", "--size", "20m", "--no-use_act_ckpt", "--no-use_8bit_adam=1"])
    monkeypatch.setenv(oom.COUNTER_ENV, "1")
    oom.reexec_with_flags(["use_act_ckpt", "use_8bit_adam"])
    assert seen["exe"] == sys.executable
    assert seen["argv"] == [sys.executable, "trainer.py", "--size", "20m", "--use_act_ckpt", "--use_8bit_adam"]
    assert seen["env"][oom.COUNTER_ENV] == "2"
