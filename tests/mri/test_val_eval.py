# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers tools/val_eval.py: the general-capability probe. Model loading is stubbed;
#   what is pinned here is corpus resolution, the seeded-draw contract that makes two
#   checkpoints comparable, and the baseline arithmetic.
# tests/mri/test_val_eval.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "veritate_mri"))
sys.path.insert(0, REPO)

from tools import val_eval

# ------------------------------------------------------------------------------------
# Tests


def test_a_stem_resolves_to_the_val_bin(tmp_path, monkeypatch):
    """'mixed_chat' means mixed_chat_val.bin in the corpus root, not a literal filename."""
    monkeypatch.setattr(val_eval, "CORPUS_ROOT", str(tmp_path))
    (tmp_path / "mixed_chat_val.bin").write_bytes(b"x" * 16)
    assert val_eval.resolve_val_bin("mixed_chat") == str(tmp_path / "mixed_chat_val.bin")


def test_a_missing_corpus_raises_instead_of_scoring_nothing(tmp_path, monkeypatch):
    """A typo'd corpus name must fail loudly; a silent skip would report a null as a result."""
    monkeypatch.setattr(val_eval, "CORPUS_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        val_eval.resolve_val_bin("no_such_corpus")


def test_the_baseline_is_scored_on_the_same_corpus_as_every_row(tmp_path, monkeypatch):
    """Comparing a tuned checkpoint to a baseline scored on different bytes is meaningless."""
    monkeypatch.setattr(val_eval, "CORPUS_ROOT", str(tmp_path))
    (tmp_path / "mixed_chat_val.bin").write_bytes(b"x" * 16)
    seen = []

    def fake_score(model_name, step, val_path, iters, batch, device="cpu"):
        seen.append((model_name, step, val_path))
        return 1.0 if model_name == "base" else 1.1

    monkeypatch.setattr(val_eval, "score", fake_score)
    val_eval.run("tuned", [10], baseline="base:0")
    assert len({p for _m, _s, p in seen}) == 1


def test_delta_is_reported_against_the_baseline_not_the_previous_step(tmp_path, monkeypatch):
    """A drift where each step sits inside tolerance is exactly how a model walks away
    from its base unnoticed, so every row is measured against the fixed reference."""
    monkeypatch.setattr(val_eval, "CORPUS_ROOT", str(tmp_path))
    (tmp_path / "mixed_chat_val.bin").write_bytes(b"x" * 16)
    vals = {("base", 0): 1.0, ("tuned", 10): 1.05, ("tuned", 20): 1.10}
    monkeypatch.setattr(val_eval, "score",
                        lambda m, s, *a, **k: vals[(m, s)])
    rep = val_eval.run("tuned", [10, 20], baseline="base:0")
    assert [r["delta_pct"] for r in rep["rows"]] == [5.0, 10.0]


def test_without_a_baseline_rows_carry_no_delta(tmp_path, monkeypatch):
    """An absolute val loss is not a comparison; inventing a delta would imply one."""
    monkeypatch.setattr(val_eval, "CORPUS_ROOT", str(tmp_path))
    (tmp_path / "mixed_chat_val.bin").write_bytes(b"x" * 16)
    monkeypatch.setattr(val_eval, "score", lambda m, s, *a, **k: 1.23)
    rep = val_eval.run("tuned", [10])
    assert rep["baseline"] is None
    assert "delta_pct" not in rep["rows"][0]


def test_the_report_is_written_as_json_when_asked(tmp_path, monkeypatch):
    """An unattended sweep is only useful if its numbers outlive the process."""
    monkeypatch.setattr(val_eval, "CORPUS_ROOT", str(tmp_path))
    (tmp_path / "mixed_chat_val.bin").write_bytes(b"x" * 16)
    monkeypatch.setattr(val_eval, "score", lambda m, s, *a, **k: 0.5)
    out = tmp_path / "rep.json"
    val_eval.run("tuned", [10], out_path=str(out))
    assert json.loads(out.read_text())["rows"][0]["val"] == 0.5


def test_a_checkpoint_that_scored_nothing_reports_none_rather_than_zero(tmp_path, monkeypatch):
    """evaluate() returns None when every iteration was skipped; a 0.0 there would read
    as a perfect model."""
    monkeypatch.setattr(val_eval, "CORPUS_ROOT", str(tmp_path))
    (tmp_path / "mixed_chat_val.bin").write_bytes(b"x" * 16)
    monkeypatch.setattr(val_eval, "score", lambda m, s, *a, **k: None)
    rep = val_eval.run("tuned", [10], baseline=None)
    assert rep["rows"][0]["val"] is None


def test_repo_root_is_the_directory_that_holds_models():
    """tools/ -> veritate_mri/ -> repo root: stopping a level short points config lookups
    at veritate_mri/models/, which does not exist."""
    assert os.path.basename(val_eval.REPO) != "veritate_mri"
    assert val_eval.REPO == REPO
