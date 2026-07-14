# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - regression for the v13 persistent prompt/state cache. drives the built engine's
#   chat_greedy A/B harness (same pattern as test_v13_compat) with VERITATE_STATE_CACHE
#   set: a warm restore must decode byte-identically to a cold prefill, an extended
#   prompt must hit the base snapshot, a different model must not cross-hit, and the
#   size cap must hold. skips when the engine binary is absent.
# tests/engine/test_state_cache.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from readers import paths

# ------------------------------------------------------------------------------------
# Constants

FIXTURES    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
V13_FIXTURE = os.path.join(FIXTURES, "hybrid_v13_fixture.bin")
PROMPT      = b"The quick brown fox jumps over"
SUFFIX      = b" the lazy dog"
BUDGET      = "8"
CAP_MB      = "1"
CAP_BYTES   = 1024 * 1024
EVICT_TURNS = 40

# ------------------------------------------------------------------------------------
# Functions


def _engine():
    exe = paths.engine_binary_path()
    if not os.path.isfile(exe):
        pytest.skip(f"engine binary not built: {exe}")
    return exe


def _greedy(exe, bin_path, prompt, cache_dir=None, cap_mb=None):
    env = dict(os.environ, VERITATE_MODEL_PATH=bin_path)
    if cache_dir is not None:
        env["VERITATE_STATE_CACHE"] = cache_dir
        env["VERITATE_STATE_CACHE_LOG"] = "1"
    else:
        env.pop("VERITATE_STATE_CACHE", None)
    if cap_mb is not None:
        env["VERITATE_STATE_CACHE_MB"] = cap_mb
    p = subprocess.run([exe, "chat_greedy", BUDGET], input=prompt,
                       capture_output=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout, p.stderr.decode(errors="replace")


def _dir_bytes(d):
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))


def test_restore_matches_cold(tmp_path):
    """Warm restore of a cached prompt greedy-decodes byte-identically to the cold prefill."""
    exe = _engine()
    cache = str(tmp_path / "cache")
    cold, _ = _greedy(exe, V13_FIXTURE, PROMPT, cache_dir=cache)
    warm, warm_err = _greedy(exe, V13_FIXTURE, PROMPT, cache_dir=cache)
    assert warm == cold
    assert "restored L=" in warm_err


def test_extend_prefix_matches_cold(tmp_path):
    """A prompt extending a cached prefix decodes identically to the same prompt cold."""
    exe = _engine()
    cache = str(tmp_path / "cache")
    _greedy(exe, V13_FIXTURE, PROMPT, cache_dir=cache)
    warm, warm_err = _greedy(exe, V13_FIXTURE, PROMPT + SUFFIX, cache_dir=cache)
    cold, _ = _greedy(exe, V13_FIXTURE, PROMPT + SUFFIX, cache_dir=None)
    assert warm == cold
    assert "restored L=" in warm_err


def test_model_change_invalidates(tmp_path):
    """A different model bin does not cross-hit another model's cached snapshot."""
    exe = _engine()
    cache = str(tmp_path / "cache")
    bin_a = str(tmp_path / "a.bin")
    bin_b = str(tmp_path / "b.bin")
    shutil.copy(V13_FIXTURE, bin_a)
    shutil.copy(V13_FIXTURE, bin_b)
    _greedy(exe, bin_a, PROMPT, cache_dir=cache)
    _, err_b = _greedy(exe, bin_b, PROMPT, cache_dir=cache)
    assert "restored L=" not in err_b
    assert "miss" in err_b


def test_eviction_caps_dir(tmp_path):
    """Storing many prefixes under a tiny cap keeps the cache dir under the byte cap."""
    exe = _engine()
    cache = str(tmp_path / "cache")
    prompts = b"\n".join(b"cache eviction probe line number %02d pad" % i
                         for i in range(EVICT_TURNS))
    _greedy(exe, V13_FIXTURE, prompts, cache_dir=cache, cap_mb=CAP_MB)
    assert _dir_bytes(cache) <= CAP_BYTES
