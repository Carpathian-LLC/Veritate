# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - build_base_corpus assembles the one from-scratch pretraining corpus. Pinned here are
#   the parts that decide what a model reads and are wrong silently: --mix resolution
#   (json file, weighted spec, unweighted spec, empty), document splitting on the
#   separator including the small-row batching HF sources need, the filter and dedup
#   chain, and the assembly's budget and train/val split. No network: catalog and filters
#   are stubbed, sources are local files.
# tests/corpus/test_build_base_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import json

import build_base_corpus as bbc

# ------------------------------------------------------------------------------------
# Constants

EOT = bbc.EOT

# ------------------------------------------------------------------------------------
# Functions


def _write_docs(path, docs, sep=EOT):
    path.write_bytes(sep.join(d.encode() for d in docs))
    return str(path)


def test_an_empty_mix_takes_the_profiles_stems(monkeypatch):
    """No --mix means the planner decides both the sources and their weights."""
    monkeypatch.setattr(bbc.mix_planner, "profile_stems", lambda p: ["a", "b"])
    assert bbc._selection("", "general") == (["a", "b"], None)


def test_a_json_mix_file_pins_the_weights(tmp_path):
    p = tmp_path / "mix.json"
    p.write_text(json.dumps({"fineweb_edu": 0.7, "skills": 0.3}), encoding="utf-8")
    stems, weights = bbc._selection(str(p), "general")
    assert stems == ["fineweb_edu", "skills"]
    assert weights == {"fineweb_edu": 0.7, "skills": 0.3}


def test_a_weighted_spec_pins_the_weights():
    stems, weights = bbc._selection("a:0.6,b:0.4", "general")
    assert stems == ["a", "b"] and weights == {"a": 0.6, "b": 0.4}


def test_a_spec_without_weights_selects_the_sources_only():
    """`a+b` picks what goes in and leaves the weighting to the planner."""
    stems, weights = bbc._selection("a+b", "general")
    assert stems == ["a", "b"] and weights is None


def test_documents_split_on_the_separator(tmp_path):
    path = _write_docs(tmp_path / "c.bin", ["one", "two", "three"])
    assert list(bbc.iter_documents(path, EOT, 0)) == [b"one", b"two", b"three"]


def test_small_rows_are_batched_up_to_the_minimum(tmp_path):
    """An HF row is one short line; emitting each as a document would make dedup and the
    quality filter operate on fragments. Rows accumulate to min_doc_bytes first."""
    path = _write_docs(tmp_path / "c.bin", ["x" * 10] * 6)
    docs = list(bbc.iter_documents(path, EOT, 25))
    assert len(docs) < 6
    assert all(len(d) >= 25 for d in docs[:-1])


def test_the_filter_chain_rejects_counts_and_dedups(tmp_path, monkeypatch):
    """clean -> quality -> pii -> dedup, with every drop counted by reason so the manifest
    reports what the corpus lost."""
    # long enough to produce a minhash sketch; the deduper ignores fragments by design
    keep = "the lighthouse keeper walks the stairs each evening and lights the lamp " * 3
    path = _write_docs(tmp_path / "c.bin", [keep, "dirty", "junky", keep])
    monkeypatch.setattr(bbc.corpus_filters, "clean_document",
                        lambda t: None if t == "dirty" else t)
    monkeypatch.setattr(bbc.corpus_filters, "quality_reject_reason",
                        lambda t: "too_short" if t == "junky" else None)
    monkeypatch.setattr(bbc.corpus_filters, "strip_pii", lambda t: t)
    source = bbc._local_source("s", path, budget_bytes=10_000)
    out = list(bbc.filtered_documents(source, bbc.corpus_filters.DocumentDeduper()))
    assert out == [keep.encode()]
    stat = source["stat"]
    assert stat["docs_kept"] == 1
    assert stat["dedup_drops"] == 1
    assert stat["rejects"] == {bbc.CLEAN_REJECT: 1, "too_short": 1}


def test_assemble_stops_each_source_at_its_budget(tmp_path, monkeypatch):
    """The budget is what makes the mix a mix; a source that overruns it silently reweights
    the corpus."""
    monkeypatch.setattr(bbc.corpus_filters, "clean_document", lambda t: t)
    monkeypatch.setattr(bbc.corpus_filters, "quality_reject_reason", lambda t: None)
    monkeypatch.setattr(bbc.corpus_filters, "strip_pii", lambda t: t)
    big = _write_docs(tmp_path / "big.bin", [f"doc{i:03d}" for i in range(100)])
    plan = [bbc._local_source("big", big, budget_bytes=60)]
    written = bbc.assemble(plan, str(tmp_path / "t.bin"), str(tmp_path / "v.bin"), 0.0, seed=1)
    assert 60 <= plan[0]["stat"]["bytes_written"] < 60 + len("doc000") + len(bbc.RECORD_SUFFIX)
    assert written == plan[0]["stat"]["bytes_written"]


def test_the_tail_of_the_run_becomes_the_val_bin(tmp_path, monkeypatch):
    """val_ratio splits by position, not by source, so validation is the last slice of the
    same assembled stream."""
    monkeypatch.setattr(bbc.corpus_filters, "clean_document", lambda t: t)
    monkeypatch.setattr(bbc.corpus_filters, "quality_reject_reason", lambda t: None)
    monkeypatch.setattr(bbc.corpus_filters, "strip_pii", lambda t: t)
    src = _write_docs(tmp_path / "s.bin", [f"doc{i:03d}" for i in range(60)])
    train, val = tmp_path / "t.bin", tmp_path / "v.bin"
    plan = [bbc._local_source("s", src, budget_bytes=400)]
    bbc.assemble(plan, str(train), str(val), 0.25, seed=1)
    assert train.stat().st_size > 0 and val.stat().st_size > 0
    assert val.stat().st_size < train.stat().st_size
