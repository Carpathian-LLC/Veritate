# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - regression for v13 batched prefill (VERITATE_PREFILL_BATCH). drives the built
#   engine's chat_greedy A/B harness (same pattern as test_v13_compat): a batched
#   prefill must greedy-decode byte-identically to the sequential per-byte path,
#   the scalar matmul fold must match on its own, and batching must compose with
#   the persistent state cache. skips when the engine binary is absent.
# tests/engine/test_prefill_batch.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
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
# prompt >= the batch width so remaining positions clear the batch gate and the
# chunk loop runs a full 32-wide chunk plus a partial tail.
PROMPT      = b"The quick brown fox jumps over the lazy dog today"
BUDGET      = "32"
BATCH       = "32"
# the batched matmul j-splits only for n*k*B >= HYBRID_MT_MIN_WORK (2^18); this
# shape's qkv (3*384 x 384) at B=32 clears it, so row-split actually runs.
MT_SHAPE    = {"vocab": 256, "hidden": 384, "global": 3, "ffn": 1536, "heads": 6, "seq": 64}
MT_NAME     = "hybrid_prefill_mt_fp16"
MT_STEP     = 10
MT_THREADS  = "4"
I8_SHAPE    = {"vocab": 256, "hidden": 32, "global": 3, "ffn": 64, "heads": 4, "seq": 64}
I8_NAME     = "hybrid_prefill_i8"

# ------------------------------------------------------------------------------------
# Functions


def _engine():
    exe = paths.engine_binary_path()
    if not os.path.isfile(exe):
        pytest.skip(f"engine binary not built: {exe}")
    return exe


def _greedy(exe, prompt, batch=None, scalar=False, cache_dir=None,
            model_path=V13_FIXTURE, threads=None):
    env = dict(os.environ, VERITATE_MODEL_PATH=model_path)
    if batch is not None:
        env["VERITATE_PREFILL_BATCH"] = batch
    else:
        env.pop("VERITATE_PREFILL_BATCH", None)
    if scalar:
        env["VERITATE_HYBRID_SCALAR"] = "1"
    if threads is not None:
        env["VERITATE_HYBRID_THREADS"] = str(threads)
    if cache_dir is not None:
        env["VERITATE_STATE_CACHE"] = cache_dir
    else:
        env.pop("VERITATE_STATE_CACHE", None)
    p = subprocess.run([exe, "chat_greedy", BUDGET], input=prompt,
                       capture_output=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_batched_prefill_matches_sequential():
    """Batched prefill greedy-decodes byte-identically to the sequential per-byte path."""
    exe = _engine()
    assert _greedy(exe, PROMPT, batch=BATCH) == _greedy(exe, PROMPT, batch=None)


def test_batched_scalar_matches_sequential():
    """Batched scalar matmul fold matches the sequential scalar matvec byte-for-byte."""
    exe = _engine()
    assert (_greedy(exe, PROMPT, batch=BATCH, scalar=True)
            == _greedy(exe, PROMPT, batch=None, scalar=True))


def test_batched_composes_with_state_cache(tmp_path):
    """Batched prefill with a cold then warm state cache matches the sequential no-cache run."""
    exe = _engine()
    cache = str(tmp_path / "cache")
    reference = _greedy(exe, PROMPT, batch=None)
    cold = _greedy(exe, PROMPT, batch=BATCH, cache_dir=cache)
    warm = _greedy(exe, PROMPT, batch=BATCH, cache_dir=cache)
    assert cold == reference
    assert warm == reference


def _traced(exe, prompt, batch=None, model_path=V13_FIXTURE):
    env = dict(os.environ, VERITATE_MODEL_PATH=model_path)
    if batch is not None:
        env["VERITATE_PREFILL_BATCH"] = batch
    else:
        env.pop("VERITATE_PREFILL_BATCH", None)
    env.pop("VERITATE_STATE_CACHE", None)
    header = b"0 1 32 -1 -1 - 0 0 0 1 0\n"
    p = subprocess.run([exe, "chat_traced"], input=header + prompt + b"\n",
                       capture_output=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_traced_batched_matches_sequential():
    """Traced prefill with batching emits byte-identical MRI frames to the sequential traced path."""
    exe = _engine()
    assert _traced(exe, PROMPT, batch=BATCH) == _traced(exe, PROMPT, batch=None)


def _export_fixture(tmp_path, monkeypatch, shape, name, dtype):
    torch = pytest.importorskip("torch")
    from training import export
    from veritate_core.model_patched import VeritatePatched
    s = shape
    torch.manual_seed(0)
    model = VeritatePatched(s["vocab"], s["hidden"], s["global"], s["ffn"], s["heads"],
                            s["seq"], global_mixer="recurrent", state_rule="gla")
    mdir = tmp_path / name / "checkpoints"
    mdir.mkdir(parents=True)
    torch.save({"model": model.state_dict()}, mdir / f"step_{MT_STEP}.pt")
    cfg = {"shape": {k: s[k] for k in ("vocab", "hidden", "ffn", "heads", "seq")} | {"layers": s["global"]},
           "training_args": {"trunk": "hybrid", "state_rule": "gla"}}
    with open(tmp_path / name / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    return export.export_checkpoint(name, MT_STEP, dtype=dtype)["path"]


def test_batched_threaded_matches_single_thread(tmp_path, monkeypatch):
    """Batched prefill j-split threaded matmul greedy-decodes byte-identically to single-thread."""
    exe = _engine()
    bin_path = _export_fixture(tmp_path, monkeypatch, MT_SHAPE, MT_NAME, "fp16")
    assert (_greedy(exe, PROMPT, batch=BATCH, model_path=bin_path, threads=MT_THREADS)
            == _greedy(exe, PROMPT, batch=BATCH, model_path=bin_path, threads="1"))


def test_batched_i8_matches_sequential(tmp_path, monkeypatch):
    """int8 batched prefill (hoisted activation quant) greedy-decodes byte-identically to sequential."""
    exe = _engine()
    bin_path = _export_fixture(tmp_path, monkeypatch, I8_SHAPE, I8_NAME, "int8")
    assert (_greedy(exe, PROMPT, batch=BATCH, model_path=bin_path)
            == _greedy(exe, PROMPT, batch=None, model_path=bin_path))
