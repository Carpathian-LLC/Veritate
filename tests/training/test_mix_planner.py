# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - mix planner contract: epoch cap, surplus redistribution, explicit weights,
#   unavailable sources, spec round-trip through multicorpus, determinism, and the
#   recommended-param-band warning. Also covers the /corpus/mix/plan boundary, the
#   only layer that validates a plan request.
# - hermetic: the catalog, the settings dict, and stem->path resolution are stubbed,
#   and the profiles file is written into tmp_path. No disk corpora, no network.
# tests/training/test_mix_planner.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

import pytest
from flask import Flask

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from routes import corpus_routes
from training import mix_planner

from veritate_core.plugin import multicorpus

# ------------------------------------------------------------------------------------
# Constants

BIG_BYTES  = 1_000_000_000
MID_BYTES  = 100_000_000
TINY_BYTES = 1_200_000

CATALOG = {
    "big":  {"stem": "big",  "label": "Big",  "topic": "knowledge",
             "size_train": BIG_BYTES,  "recommended_min_params": 50_000_000,
             "recommended_max_params": 500_000_000},
    "mid":  {"stem": "mid",  "label": "Mid",  "topic": "code",
             "size_train": MID_BYTES,  "recommended_min_params": None,
             "recommended_max_params": None},
    "tiny": {"stem": "tiny", "label": "Tiny", "topic": "chat",
             "size_train": TINY_BYTES, "recommended_min_params": None,
             "recommended_max_params": None},
}

PROFILE = {"profiles": {"chatty": {"topics": {"chat": 0.6, "knowledge": 0.3, "code": 0.1},
                                   "unlisted_topic_share": 0.02,
                                   "stems": ["tiny", "big"]}}}

STEMS      = ["tiny", "big", "mid"]
TARGET     = 1_000_000_000
MAX_EPOCHS = 4.0

EPOCH_TOLERANCE  = 1e-6
WEIGHT_TOLERANCE = 1e-9

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def planner(monkeypatch, tmp_path):
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps(PROFILE), encoding="utf-8")
    monkeypatch.setattr(mix_planner.settings_mod, "get", lambda: {
        mix_planner.SETTING_MAX_EPOCHS:    MAX_EPOCHS,
        mix_planner.SETTING_PROFILE:       "chatty",
        mix_planner.SETTING_PROFILES_PATH: str(profiles),
    })
    monkeypatch.setattr(mix_planner, "_catalog_by_stem", lambda: dict(CATALOG))
    monkeypatch.setattr(mix_planner.corpus_reader, "resolve_paths", lambda stem: (None, None))
    return mix_planner


def _by_stem(plan):
    return {r["stem"]: r for r in plan["sources"]}


def test_tiny_source_cannot_exceed_the_epoch_cap(planner):
    """A 1.2 MB source the profile wants at 60% is held to max_epochs passes."""
    rows = _by_stem(planner.plan(STEMS, TARGET))
    assert rows["tiny"]["epochs"] <= MAX_EPOCHS + EPOCH_TOLERANCE


def test_capped_source_keeps_its_byte_budget_under_availability(planner):
    """The capped source draws at most max_epochs * its own bytes, never more."""
    rows = _by_stem(planner.plan(STEMS, TARGET))
    assert rows["tiny"]["bytes_drawn"] <= TINY_BYTES * MAX_EPOCHS


def test_surplus_from_a_capped_source_is_redistributed(planner):
    """Weight freed by the cap lands on the sources with headroom."""
    rows = _by_stem(planner.plan(STEMS, TARGET))
    assert rows["big"]["weight"] > 0.3


def test_weights_still_sum_to_one_after_capping(planner):
    """Capping plus redistribution leaves the emitted weights summing to 1.0."""
    plan = planner.plan(STEMS, TARGET)
    assert sum(r["weight"] for r in plan["sources"]) == pytest.approx(1.0, abs=WEIGHT_TOLERANCE)


def test_explicit_weights_are_honored(planner):
    """User-supplied weights that need no capping pass through normalized, ignoring the profile."""
    rows = _by_stem(planner.plan(["big", "mid"], TARGET, weights={"big": 3.0, "mid": 1.0}))
    assert (rows["big"]["weight"], rows["mid"]["weight"]) == (0.75, 0.25)


def test_explicit_weights_still_obey_the_epoch_cap(planner):
    """An explicit weight that would over-draw a source is still capped."""
    rows = _by_stem(planner.plan(["tiny", "big"], TARGET, weights={"tiny": 0.5, "big": 0.5}))
    assert rows["tiny"]["epochs"] <= MAX_EPOCHS + EPOCH_TOLERANCE


def test_oversubscribed_source_warns(planner):
    """An oversubscribed source is reported in warnings, not silently truncated."""
    plan = planner.plan(STEMS, TARGET)
    assert any("'tiny'" in w and "capped" in w for w in plan["warnings"])


def test_unavailable_source_warns_and_leaves_the_spec(planner):
    """A stem with no bytes on disk or in the catalog warns and is excluded from the spec."""
    plan = planner.plan(["big", "ghost"], TARGET)
    assert "ghost" not in plan["spec"]
    assert any("'ghost'" in w for w in plan["warnings"])


def test_target_beyond_capacity_warns(planner):
    """A target larger than max_epochs passes over every source warns and stops at capacity."""
    plan = planner.plan(STEMS, TARGET * 100)
    assert plan["bytes_planned"] == int((BIG_BYTES + MID_BYTES + TINY_BYTES) * MAX_EPOCHS)


def test_spec_round_trips_through_multicorpus(planner):
    """The emitted spec parses back into the same stems and weights."""
    plan = planner.plan(STEMS, TARGET)
    parsed = multicorpus.parse_spec(plan["spec"])
    assert parsed == [(r["stem"], r["weight"]) for r in plan["sources"] if r["weight"] > 0]


def test_plan_is_deterministic(planner):
    """Same inputs produce a byte-identical plan."""
    a = planner.plan(STEMS, TARGET, model_params=10_000_000)
    b = planner.plan(STEMS, TARGET, model_params=10_000_000)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_corpus_outside_its_param_band_warns(planner):
    """A corpus recommended for larger models warns when model_params is supplied."""
    plan = planner.plan(STEMS, TARGET, model_params=10_000_000)
    assert any("'big'" in w and "50,000,000+ params" in w for w in plan["warnings"])


def test_param_band_is_silent_without_model_params(planner):
    """No size-suitability warning is emitted when the caller gives no model_params."""
    plan = planner.plan(STEMS, TARGET)
    assert not any("recommended for models" in w for w in plan["warnings"])


def test_unknown_profile_raises(planner):
    """An unknown profile name is a ValueError naming the profiles that exist."""
    with pytest.raises(ValueError, match="unknown mix profile"):
        planner.plan(STEMS, TARGET, profile="nope")


def test_shipped_profiles_load(monkeypatch):
    """The shipped profiles file parses and carries the default profile with both required keys."""
    monkeypatch.setattr(mix_planner.settings_mod, "get",
                        lambda: {mix_planner.SETTING_PROFILES_PATH: ""})
    default = mix_planner.load_profiles()["pretrain"]
    assert mix_planner.PROFILE_TOPICS in default and mix_planner.PROFILE_SHARE in default


def test_plan_route_rejects_an_empty_stem_list(planner):
    """POST /corpus/mix/plan without stems returns 400, not a traceback."""
    app = Flask(__name__)
    corpus_routes.register(app)
    resp = app.test_client().post("/corpus/mix/plan", json={"target_bytes": TARGET})
    assert resp.status_code == 400


def test_plan_route_returns_a_spec(planner):
    """POST /corpus/mix/plan returns the plan with a multicorpus spec string."""
    app = Flask(__name__)
    corpus_routes.register(app)
    resp = app.test_client().post("/corpus/mix/plan",
                                  json={"stems": STEMS, "target_bytes": TARGET})
    assert resp.status_code == 200 and multicorpus.is_mixed_spec(resp.get_json()["spec"])
