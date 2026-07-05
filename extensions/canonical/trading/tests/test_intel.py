# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Deterministic tests for the Market Intel LLM brief layer: strict-JSON parse with
#   malformed fallback, the mechanical no-news pump-risk override, and the
#   (symbol, headline-hash) cache. No network: model + news fetch monkeypatched.
# extensions/canonical/trading/tests/test_intel.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import intel
import scanner as sc

# ------------------------------------------------------------------------------------
# Fixtures

GOOD_REPLY = ('noise before {"why_moving":"ETF approval headline.","news_driven":true,'
              '"meme":false,"pump_risk":"low","caution":"Already public."} noise after')
SPIKE = {"price_z": sc.PRICE_Z_MIN + 1, "vol_z": sc.VOL_Z_MIN + 1}


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    """Isolate the brief cache and model state per test."""
    monkeypatch.setattr(intel, "_BRIEFS", {})
    monkeypatch.setattr(intel, "_MODEL_STATE", "untried")

# ------------------------------------------------------------------------------------
# Parse

def test_parse_brief_extracts_strict_json():
    """A valid JSON object inside model prose parses into the five brief fields."""
    b = intel.parse_brief(GOOD_REPLY)
    assert b == {"why_moving": "ETF approval headline.", "news_driven": True,
                 "meme": False, "pump_risk": "low", "caution": "Already public."}


def test_parse_brief_malformed_returns_none():
    """No braces or broken JSON parses to None, never raises."""
    assert intel.parse_brief("i cannot answer that") is None
    assert intel.parse_brief('{"why_moving": unquoted}') is None


def test_parse_brief_clamps_unknown_risk():
    """An off-menu pump_risk value ('medium', 'severe') maps into the allowed set."""
    assert intel.parse_brief('{"pump_risk":"medium"}')["pump_risk"] == "med"
    assert intel.parse_brief('{"pump_risk":"severe"}')["pump_risk"] == "low"

# ------------------------------------------------------------------------------------
# Brief + mechanical override

def test_brief_no_news_spike_forced_high(monkeypatch):
    """No headlines + spiking metrics stamps pump_risk high even when the model says low."""
    monkeypatch.setattr(intel, "headlines", lambda q: [])
    monkeypatch.setattr(intel, "_complete", lambda p, m: '{"why_moving":"Calm.","news_driven":false,"meme":false,"pump_risk":"low","caution":"Fine."}')
    b = intel.brief("AAA", SPIKE)
    assert b["pump_risk"] == "high" and b["caution"] == intel.CAUTION_PUMP


def test_brief_news_driven_spike_not_forced(monkeypatch):
    """A spike the model ties to real headlines keeps the model's own risk call."""
    monkeypatch.setattr(intel, "headlines", lambda q: ["Exchange lists AAA"])
    monkeypatch.setattr(intel, "_complete", lambda p, m: GOOD_REPLY)
    assert intel.brief("AAA", SPIKE)["pump_risk"] == "low"


def test_brief_model_offline_fallback(monkeypatch):
    """Model unreachable yields the honest metrics-only fallback, still risk-stamped."""
    monkeypatch.setattr(intel, "headlines", lambda q: [])

    def _offline(p, m):
        intel._MODEL_STATE = "offline"
        return ""
    monkeypatch.setattr(intel, "_complete", _offline)
    b = intel.brief("AAA", SPIKE)
    assert b["why_moving"] == intel.WHY_OFFLINE and b["pump_risk"] == "high"

# ------------------------------------------------------------------------------------
# Cache

def test_brief_cached_by_headline_hash(monkeypatch):
    """Unchanged headlines never re-hit the model; changed headlines do."""
    calls = []
    monkeypatch.setattr(intel, "headlines", lambda q: ["same headline"])
    monkeypatch.setattr(intel, "_complete", lambda p, m: calls.append(1) or GOOD_REPLY)
    intel.brief("AAA")
    intel.brief("AAA")
    assert len(calls) == 1
    monkeypatch.setattr(intel, "headlines", lambda q: ["new headline"])
    intel.brief("AAA")
    assert len(calls) == 2


def test_brief_fallback_not_cached(monkeypatch):
    """An offline fallback is not cached, so the next call retries the model."""
    monkeypatch.setattr(intel, "headlines", lambda q: [])
    monkeypatch.setattr(intel, "_complete", lambda p, m: "")
    intel.brief("AAA")
    assert intel.cached_brief("AAA") is None


def test_cached_brief_absent_is_none():
    """cached_brief never fetches; unknown symbol is None."""
    assert intel.cached_brief("ZZZ") is None
