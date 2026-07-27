# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - a mix that blends ChatML-framed and legacy <|user|>-framed chat corpora trains
#   the wrong turn framing: hybrid_routes.STOP_MARKERS is ChatML-only, so the
#   legacy marker leaks into the served answer. The planner must surface the
#   legacy-framed member rather than blending it silently.
# - hermetic: catalog, settings, and stem->path resolution are stubbed; the corpus
#   bytes are written into tmp_path.
# tests/training/test_mix_planner_framing.py
# ------------------------------------------------------------------------------------
# Imports:

import json

import pytest
from routes import hybrid_routes
from training import mix_planner

# ------------------------------------------------------------------------------------
# Constants

CHATML_HEAD = b"<|im_start|>user\nhi<|im_end|>\n"
LEGACY_HEAD = b"<|user|>\nhi\n<|assistant|>\n"
FILLER      = b"x" * 4096

CHATML_STEM = "chatml_probe"
LEGACY_STEM = "legacy_probe"
STEMS       = [CHATML_STEM, LEGACY_STEM]

TARGET      = 1_000_000
MAX_EPOCHS  = 4.0
CATALOG     = {s: {"stem": s, "label": s, "topic": "chat",
                   "size_train": len(CHATML_HEAD) + len(FILLER),
                   "recommended_min_params": None, "recommended_max_params": None}
               for s in STEMS}

FRAMING_GAP_REASON = ("live product bug: mix_planner.plan() has no chat-framing check, so a "
                      "legacy <|user|>-framed corpus blends into a ChatML mix with no warning "
                      "and the model learns a turn framing STOP_MARKERS cannot cut")

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def planner(monkeypatch, tmp_path):
    """Planner wired to two on-disk chat corpora: one ChatML-framed, one legacy-framed."""
    paths = {}
    for stem, head in ((CHATML_STEM, CHATML_HEAD), (LEGACY_STEM, LEGACY_HEAD)):
        p = tmp_path / f"{stem}_train.bin"
        p.write_bytes(head + FILLER)
        paths[stem] = str(p)
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"profiles": {}}), encoding="utf-8")
    monkeypatch.setattr(mix_planner.settings_mod, "get", lambda: {
        mix_planner.SETTING_MAX_EPOCHS:    MAX_EPOCHS,
        mix_planner.SETTING_PROFILE:       "",
        mix_planner.SETTING_PROFILES_PATH: str(profiles),
    })
    monkeypatch.setattr(mix_planner, "_catalog_by_stem", lambda: dict(CATALOG))
    monkeypatch.setattr(mix_planner.corpus_reader, "resolve_paths",
                        lambda stem: (paths.get(stem), None))
    return mix_planner


def _plan(planner):
    return planner.plan(STEMS, TARGET)


def test_stop_markers_cannot_cut_the_legacy_marker():
    """The served stop markers do not include the legacy <|user|> turn marker."""
    assert "<|user|>" not in hybrid_routes.STOP_MARKERS


def test_mix_of_both_framings_plans_both_stems(planner):
    """A mix of a ChatML and a legacy chat corpus currently plans both members."""
    assert sorted(r["stem"] for r in _plan(planner)["sources"]) == sorted(STEMS)


@pytest.mark.xfail(strict=True, reason=FRAMING_GAP_REASON)
def test_mix_warns_when_a_member_is_legacy_framed(planner):
    """A mix containing a legacy-framed chat corpus warns and names that corpus."""
    assert [w for w in _plan(planner)["warnings"] if LEGACY_STEM in w] != []
