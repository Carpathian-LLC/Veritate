# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - greedy-decode parity for the engine: the canonical int8 fixture must match
#   the golden transcript byte for byte, and the hybrid-trunk fixture must match
#   itself across the scalar, SIMD, and threaded matvec paths.
# - requires the built engine binary; skips when absent.
# tests/engine/test_decode_parity.py
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
CANON_FIXTURE  = os.path.join(FIXTURES, "canonical_fixture.bin")
CANON_GOLDEN   = os.path.join(FIXTURES, "canonical_greedy_golden.bin")
HYBRID_FIXTURE = os.path.join(FIXTURES, "hybrid_fixture.bin")
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


def _greedy(exe, bin_path, scalar=False, threads=None, boundary_threads=None):
    env = dict(os.environ, VERITATE_MODEL_PATH=bin_path)
    if scalar:
        env["VERITATE_HYBRID_SCALAR"] = "1"
    if threads is not None:
        env["VERITATE_HYBRID_THREADS"] = str(threads)
    if boundary_threads is not None:
        env["VERITATE_HYBRID_BOUNDARY_THREADS"] = str(boundary_threads)
    p = subprocess.run([exe, "chat_greedy", BUDGET], input=PROMPTS,
                       capture_output=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr.decode(errors="replace")
    return p.stdout


def test_canonical_greedy_matches_golden():
    """Canonical int8 fixture greedy transcript is byte-identical to the golden recording."""
    with open(CANON_GOLDEN, "rb") as f:
        golden = f.read()
    assert _greedy(_engine(), CANON_FIXTURE) == golden


def test_hybrid_fixture_loads_and_generates():
    """Hybrid-trunk fixture loads and emits the full greedy budget for every turn."""
    out = _greedy(_engine(), HYBRID_FIXTURE)
    assert len(out) == N_TURNS * (int(BUDGET) + 1)


def test_hybrid_simd_matches_scalar():
    """Hybrid fp32 SIMD matvec (neon/avx2) greedy-decodes byte-identically to scalar."""
    exe = _engine()
    assert _greedy(exe, HYBRID_FIXTURE) == _greedy(exe, HYBRID_FIXTURE, scalar=True)


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


def test_hybrid_int8_simd_matches_scalar(tmp_path, monkeypatch):
    """Hybrid int8 SIMD matvec (avx2/sdot) greedy-decodes byte-identically to scalar."""
    exe = _engine()
    bin_path = _export_fixture(tmp_path, monkeypatch, I8_SHAPE, I8_NAME, "int8")
    assert _greedy(exe, bin_path) == _greedy(exe, bin_path, scalar=True)


def test_hybrid_threaded_matches_single_thread(tmp_path, monkeypatch):
    """Row-split threaded matvec greedy-decodes byte-identically to VERITATE_HYBRID_THREADS=1."""
    exe = _engine()
    bin_path = _export_fixture(tmp_path, monkeypatch, MT_SHAPE, MT_NAME, "fp16")
    assert _greedy(exe, bin_path, threads=MT_THREADS) == _greedy(exe, bin_path, threads=1)


def test_boundary_class_thread_switch_does_not_change_output(tmp_path, monkeypatch):
    """Decode is bimodal, so the two step classes get their own worker counts and
    the count changes partway through a step. Row-split parity has to survive that
    switch, not just a fixed count."""
    exe = _engine()
    monkeypatch.delenv("VERITATE_HYBRID_THREADS", raising=False)
    bin_path = _export_fixture(tmp_path, monkeypatch, MT_SHAPE, MT_NAME, "fp16")
    split = _greedy(exe, bin_path, threads=1, boundary_threads=MT_THREADS)
    assert split == _greedy(exe, bin_path, threads=1)
    assert split == _greedy(exe, bin_path, threads=MT_THREADS, boundary_threads=1)


def test_hybrid_auto_matches_single_thread(tmp_path, monkeypatch):
    """Auto-calibrated thread count greedy-decodes byte-identically to VERITATE_HYBRID_THREADS=1."""
    exe = _engine()
    monkeypatch.delenv("VERITATE_HYBRID_THREADS", raising=False)
    bin_path = _export_fixture(tmp_path, monkeypatch, MT_SHAPE, MT_NAME, "fp16")
    assert _greedy(exe, bin_path, threads=None) == _greedy(exe, bin_path, threads=1)
