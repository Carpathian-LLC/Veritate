# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - mix_planner.compose(): materializing a plan into ONE unified corpus.
# - the properties that matter: byte counts match the plan, train and val never
#   share a byte, sources interleave instead of forming blocks, and a seed makes
#   the whole thing reproducible.
# tests/corpus/test_mix_compose.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
from training import mix_planner

# ------------------------------------------------------------------------------------
# Constants

CHUNK = 1024

# ------------------------------------------------------------------------------------
# Functions


def _corpus(tmp_path, monkeypatch, sizes):
    """Write fake corpora whose bytes identify their source: stem 'a' is all
    b'a', stem 'b' is all b'b'. Origin is then readable straight off the output."""
    monkeypatch.setattr(mix_planner.paths, "CORPUS_ROOT", str(tmp_path))
    for stem, size in sizes.items():
        with open(os.path.join(tmp_path, f"{stem}_train.bin"), "wb") as f:
            f.write(stem[0].encode() * size)
    monkeypatch.setattr(mix_planner.corpus_reader, "resolve_paths",
                        lambda stem: (os.path.join(tmp_path, f"{stem}_train.bin"), None))


def _plan(rows, spec="x"):
    return {"spec": spec, "warnings": [], "sources": [
        {"stem": s, "weight": w, "bytes_available": avail, "bytes_drawn": drawn,
         "epochs": drawn / avail} for s, w, avail, drawn in rows]}


def test_compose_writes_the_planned_byte_counts(tmp_path, monkeypatch):
    _corpus(tmp_path, monkeypatch, {"a": 40000, "b": 20000})
    out = mix_planner.compose("uni", _plan([("a", 0.6, 40000, 12000),
                                            ("b", 0.4, 20000, 8000)]),
                              val_ratio=0.0, seed=7, chunk_bytes=CHUNK)
    assert out["train_bytes"] == 20000
    assert os.path.getsize(out["train_path"]) == 20000


def test_each_source_contributes_exactly_its_drawn_bytes(tmp_path, monkeypatch):
    _corpus(tmp_path, monkeypatch, {"a": 40000, "b": 20000})
    out = mix_planner.compose("uni", _plan([("a", 0.6, 40000, 12000),
                                            ("b", 0.4, 20000, 8000)]),
                              val_ratio=0.0, seed=7, chunk_bytes=CHUNK)
    with open(out["train_path"], "rb") as f:
        blob = f.read()
    assert blob.count(b"a") == 12000
    assert blob.count(b"b") == 8000


def test_sources_interleave_rather_than_forming_blocks(tmp_path, monkeypatch):
    """A composed corpus that is just source-A-then-source-B would train the
    model on one distribution and then the other. Chunks must be shuffled."""
    _corpus(tmp_path, monkeypatch, {"a": 40000, "b": 20000})
    out = mix_planner.compose("uni", _plan([("a", 0.6, 40000, 12000),
                                            ("b", 0.4, 20000, 8000)]),
                              val_ratio=0.0, seed=7, chunk_bytes=CHUNK)
    with open(out["train_path"], "rb") as f:
        blob = f.read()
    switches = sum(1 for i in range(1, len(blob)) if blob[i] != blob[i - 1])
    assert switches > 1, "sources were concatenated as blocks, not interleaved"


def test_train_and_val_never_share_a_byte(tmp_path, monkeypatch):
    """Val is drawn from a disjoint tail region, so a source drawn for many
    epochs still cannot leak its val bytes into train."""
    monkeypatch.setattr(mix_planner.paths, "CORPUS_ROOT", str(tmp_path))
    path = os.path.join(tmp_path, "a_train.bin")
    with open(path, "wb") as f:
        f.write(b"T" * 36000 + b"V" * 4000)   # head and tail are tellable apart
    monkeypatch.setattr(mix_planner.corpus_reader, "resolve_paths", lambda stem: (path, None))

    out = mix_planner.compose("uni", _plan([("a", 1.0, 40000, 30000)]),
                              val_ratio=0.1, seed=3, chunk_bytes=CHUNK)
    with open(out["train_path"], "rb") as f:
        assert set(f.read()) == {ord("T")}
    with open(out["val_path"], "rb") as f:
        assert set(f.read()) == {ord("V")}


def test_a_source_drawn_past_its_size_wraps_instead_of_running_out(tmp_path, monkeypatch):
    _corpus(tmp_path, monkeypatch, {"a": 4000})
    out = mix_planner.compose("uni", _plan([("a", 1.0, 4000, 10000)]),
                              val_ratio=0.0, seed=1, chunk_bytes=CHUNK)
    assert out["train_bytes"] == 10000


def test_same_seed_reproduces_the_same_corpus(tmp_path, monkeypatch):
    _corpus(tmp_path, monkeypatch, {"a": 40000, "b": 20000})
    rows = [("a", 0.6, 40000, 12000), ("b", 0.4, 20000, 8000)]
    first  = mix_planner.compose("uni", _plan(rows), val_ratio=0.0, seed=11, chunk_bytes=CHUNK)
    second = mix_planner.compose("uni", _plan(rows), val_ratio=0.0, seed=11, chunk_bytes=CHUNK)
    assert first["sha256_train"] == second["sha256_train"]


def test_a_different_seed_changes_the_ordering(tmp_path, monkeypatch):
    _corpus(tmp_path, monkeypatch, {"a": 40000, "b": 20000})
    rows = [("a", 0.6, 40000, 12000), ("b", 0.4, 20000, 8000)]
    first  = mix_planner.compose("uni", _plan(rows), val_ratio=0.0, seed=11, chunk_bytes=CHUNK)
    second = mix_planner.compose("uni", _plan(rows), val_ratio=0.0, seed=12, chunk_bytes=CHUNK)
    assert first["sha256_train"] != second["sha256_train"]


def test_an_empty_plan_is_refused(tmp_path, monkeypatch):
    _corpus(tmp_path, monkeypatch, {"a": 4000})
    with pytest.raises(ValueError, match="no bytes"):
        mix_planner.compose("uni", _plan([("a", 1.0, 4000, 0)]),
                            val_ratio=0.0, seed=1, chunk_bytes=CHUNK)


def test_a_missing_source_is_refused_by_name(tmp_path, monkeypatch):
    _corpus(tmp_path, monkeypatch, {"a": 4000})
    monkeypatch.setattr(mix_planner.corpus_reader, "resolve_paths", lambda stem: (None, None))
    with pytest.raises(ValueError, match="ghost"):
        mix_planner.compose("uni", _plan([("ghost", 1.0, 4000, 1000)]),
                            val_ratio=0.0, seed=1, chunk_bytes=CHUNK)


def test_an_out_of_range_val_ratio_is_refused(tmp_path, monkeypatch):
    _corpus(tmp_path, monkeypatch, {"a": 4000})
    with pytest.raises(ValueError, match="val_ratio"):
        mix_planner.compose("uni", _plan([("a", 1.0, 4000, 1000)]),
                            val_ratio=1.0, seed=1, chunk_bytes=CHUNK)


def test_the_manifest_reports_per_source_byte_counts(tmp_path, monkeypatch):
    _corpus(tmp_path, monkeypatch, {"a": 40000, "b": 20000})
    out = mix_planner.compose("uni", _plan([("a", 0.6, 40000, 12000),
                                            ("b", 0.4, 20000, 8000)]),
                              val_ratio=0.0, seed=7, chunk_bytes=CHUNK)
    by_stem = {r["stem"]: r for r in out["sources"]}
    assert by_stem["a"]["train_bytes"] == 12000
    assert by_stem["b"]["train_bytes"] == 8000
    assert sum(r["train_bytes"] for r in out["sources"]) == out["train_bytes"]


def test_composing_onto_a_source_is_refused(tmp_path, monkeypatch):
    """Reading a source while overwriting it would shred the source mid-run."""
    _corpus(tmp_path, monkeypatch, {"a": 40000, "b": 20000})
    with pytest.raises(ValueError, match="both the output and one of the sources"):
        mix_planner.compose("a", _plan([("a", 0.6, 40000, 12000),
                                        ("b", 0.4, 20000, 8000)]),
                            val_ratio=0.0, seed=7, chunk_bytes=CHUNK)
