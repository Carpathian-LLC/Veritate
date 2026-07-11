# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - model capability tiers: backward-compatible read (legacy flat + versioned
#   schema), corpus->tier resolution, additivity in mark, and schema-on-write.
# tests/mri/test_capabilities.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from readers import capabilities as caps

# ------------------------------------------------------------------------------------
# Functions

def _seed(monkeypatch, tmp_path, block):
    cfg = tmp_path / "config.json"
    data = {"description": "toy"}
    if block is not None:
        data["capabilities"] = block
    cfg.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(caps, "_config_path", lambda name: str(cfg))
    return str(cfg)


def test_read_legacy_flat_block(monkeypatch, tmp_path):
    """read() accepts the legacy flat capabilities block with no migration."""
    _seed(monkeypatch, tmp_path, {"autocomplete": {"status": "trained"},
                                  "chat": {"status": "trained"},
                                  "agent": {"status": "untrained"}})
    assert caps.read("m")["chat"]["status"] == "trained"


def test_read_missing_block_falls_back_autocomplete(monkeypatch, tmp_path):
    """A config with no capabilities key reads autocomplete=trained, rest untrained."""
    _seed(monkeypatch, tmp_path, None)
    b = caps.read("m")
    assert b["autocomplete"]["status"] == "trained"
    assert b["chat"]["status"] == "untrained"


def test_read_new_schema_block(monkeypatch, tmp_path):
    """read() accepts the versioned {schema,pipeline_tag,tasks} block."""
    _seed(monkeypatch, tmp_path, {"schema": 1, "pipeline_tag": "text-generation",
                                  "tasks": {"chat": {"status": "trained"}}})
    assert caps.read("m")["chat"]["status"] == "trained"


def test_modes_for_corpus_single_chat():
    """A chat catalog stem resolves to the chat tier."""
    assert caps.modes_for_corpus("chat_50mb") == ("chat",)


def test_modes_for_corpus_mix_unions_tiers():
    """A chat+agent mix spec unions both tiers in TIERS order."""
    assert caps.modes_for_corpus("chat_50mb:1,agent_15mb:1") == ("chat", "agent")


def test_modes_for_corpus_custom_stem_empty():
    """A stem absent from the catalog resolves to no tiers."""
    assert caps.modes_for_corpus("fineweb_edu:0.5,chat_v1:0.5") == ()


def test_mark_writes_versioned_schema(monkeypatch, tmp_path):
    """mark() persists the versioned HF-aligned block on disk."""
    cfg = _seed(monkeypatch, tmp_path, None)
    caps.mark("m", "chat", "trained", trainer="native/trainer", step=10)
    disk = json.loads(open(cfg, encoding="utf-8").read())["capabilities"]
    assert disk["schema"] == 1 and disk["pipeline_tag"] == "text-generation" and "tasks" in disk


def test_mark_additivity_lifts_lower_tier(monkeypatch, tmp_path):
    """Marking chat trained lifts an untrained autocomplete to trained (implied)."""
    _seed(monkeypatch, tmp_path, {"autocomplete": {"status": "untrained"},
                                  "chat": {"status": "untrained"},
                                  "agent": {"status": "untrained"}})
    caps.mark("m", "chat", "trained", trainer="native/trainer", step=10)
    entry = caps.read("m")["autocomplete"]
    assert entry["status"] == "trained" and entry.get("implied") is True


def test_mark_failed_does_not_regress_trained(monkeypatch, tmp_path):
    """A failed mark is dropped when the tier is already trained."""
    _seed(monkeypatch, tmp_path, {"autocomplete": {"status": "trained"},
                                  "chat": {"status": "trained"},
                                  "agent": {"status": "untrained"}})
    caps.mark("m", "chat", "failed")
    assert caps.read("m")["chat"]["status"] == "trained"
