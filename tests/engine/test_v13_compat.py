# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - canonical-compat regression for the v13 engine changes: a v9 fixture must
#   greedy-decode byte-identically to the golden transcript recorded from the
#   pre-v13 engine build. plus a v13 fixture load + generate smoke.
# - requires the built engine binary; skips when absent.
# tests/engine/test_v13_compat.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import subprocess

import pytest
from readers import paths

# ------------------------------------------------------------------------------------
# Constants

FIXTURES     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
V9_FIXTURE   = os.path.join(FIXTURES, "canon_v9_fixture.bin")
V9_GOLDEN    = os.path.join(FIXTURES, "canon_v9_greedy_golden.bin")
V13_FIXTURE  = os.path.join(FIXTURES, "hybrid_v13_fixture.bin")
PROMPTS      = b"Hello world\nabc def\nOnce upon a time\n"
BUDGET       = "32"
N_TURNS      = 3
I8_SHAPE     = {"vocab": 256, "hidden": 32, "global": 3, "ffn": 64, "heads": 4, "seq": 64}
I8_NAME      = "tiny_hybrid_i8"
I8_STEP      = 10
# threading engages only for matvecs >= HYBRID_MT_MIN_WORK (2^18); this shape's
# qkv (3*384^2) and ff (1536*384) clear it so row-split actually runs.
MT_SHAPE     = {"vocab": 256, "hidden": 384, "global": 3, "ffn": 1536, "heads": 6, "seq": 64}
MT_NAME      = "hybrid_mt_fp16"
MT_THREADS   = 8

# ------------------------------------------------------------------------------------
# Functions


def _engine():
    exe = paths.engine_binary_path()
    if not os.path.isfile(exe):
        pytest.skip(f"engine binary not built: {exe}")
    return exe


def _greedy(exe, bin_path, scalar=False, threads=None):
    env = dict(os.environ, VERITATE_MODEL_PATH=bin_path)
    if scalar:
        env["VERITATE_HYBRID_SCALAR"] = "1"
    if threads is not None:
        env["VERITATE_HYBRID_THREADS"] = str(threads)
    p = subprocess.run([exe, "chat_greedy", BUDGET], input=PROMPTS,
                       capture_output=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_v9_greedy_matches_golden():
    """v9 fixture greedy transcript is byte-identical to the pre-v13 recording."""
    with open(V9_GOLDEN, "rb") as f:
        golden = f.read()
    assert _greedy(_engine(), V9_FIXTURE) == golden


def test_v13_fixture_loads_and_generates():
    """v13 fixture loads and emits the full greedy budget for every turn."""
    out = _greedy(_engine(), V13_FIXTURE)
    assert len(out) == N_TURNS * (int(BUDGET) + 1)


def test_v13_simd_matches_scalar():
    """v13 hybrid SIMD matvec (neon/avx2) greedy-decodes byte-identically to scalar."""
    exe = _engine()
    assert _greedy(exe, V13_FIXTURE) == _greedy(exe, V13_FIXTURE, scalar=True)


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
    torch.save({"model": model.state_dict()}, mdir / f"step_{I8_STEP}.pt")
    cfg = {"shape": {k: s[k] for k in ("vocab", "hidden", "ffn", "heads", "seq")} | {"layers": s["global"]},
           "training_args": {"trunk": "hybrid", "state_rule": "gla"}}
    with open(tmp_path / name / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path))
    return export.export_checkpoint(name, I8_STEP, dtype=dtype)["path"]


def test_v13_int8_simd_matches_scalar(tmp_path, monkeypatch):
    """v13 int8 hybrid SIMD matvec (avx2/sdot) greedy-decodes byte-identically to scalar."""
    exe = _engine()
    bin_path = _export_fixture(tmp_path, monkeypatch, I8_SHAPE, I8_NAME, "int8")
    assert _greedy(exe, bin_path) == _greedy(exe, bin_path, scalar=True)


def test_v13_threaded_matches_single_thread(tmp_path, monkeypatch):
    """v13 row-split threaded matvec greedy-decodes byte-identically to VERITATE_HYBRID_THREADS=1."""
    exe = _engine()
    bin_path = _export_fixture(tmp_path, monkeypatch, MT_SHAPE, MT_NAME, "fp16")
    assert _greedy(exe, bin_path, threads=MT_THREADS) == _greedy(exe, bin_path, threads=1)


def test_v13_auto_matches_single_thread(tmp_path, monkeypatch):
    """v13 auto-calibrated thread count greedy-decodes byte-identically to VERITATE_HYBRID_THREADS=1."""
    exe = _engine()
    monkeypatch.delenv("VERITATE_HYBRID_THREADS", raising=False)
    bin_path = _export_fixture(tmp_path, monkeypatch, MT_SHAPE, MT_NAME, "fp16")
    assert _greedy(exe, bin_path, threads=None) == _greedy(exe, bin_path, threads=1)
