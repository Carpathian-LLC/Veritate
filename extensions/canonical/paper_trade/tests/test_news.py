# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the Paper Trading sentiment layer
#   (extensions/canonical/paper_trade/server/{scraper,sentiment}.py): RSS pubDate parsing,
#   sentiment-JSON extraction, time-decay aggregation, and the score path with the model
#   call mocked. No network, no model.
# extensions/canonical/paper_trade/tests/test_news.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.normpath(os.path.join(HERE, "..", "server"))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import scraper
import sentiment as sent

# ------------------------------------------------------------------------------------
# scraper

def test_ts_parses_rfc822():
    """RSS RFC-822 pubDate parses to a positive unix timestamp."""
    assert scraper._ts("Wed, 02 Oct 2024 13:00:00 GMT") > 1_700_000_000


def test_ts_fallback_on_garbage():
    """An unparseable pubDate falls back to ~now, never raises."""
    assert scraper._ts("not a date") > 1_700_000_000

# ------------------------------------------------------------------------------------
# sentiment parsing

def test_parse_clean_json():
    """A clean sentiment JSON parses and clamps into range."""
    o = sent._parse('{"asset":"eth","sentiment":0.8,"confidence":0.9}')
    assert o == {"asset": "ETH", "sentiment": 0.8, "confidence": 0.9}


def test_parse_json_embedded_in_prose():
    """The {...} object is extracted even when the model adds prose around it."""
    o = sent._parse('Sure! {"asset":"BTC","sentiment":-0.5,"confidence":0.6} hope that helps')
    assert o["asset"] == "BTC" and o["sentiment"] == -0.5


def test_parse_clamps_out_of_range():
    """Out-of-range sentiment is clamped to [-1,1]."""
    assert sent._parse('{"asset":"X","sentiment":5,"confidence":2}')["sentiment"] == 1.0


def test_parse_invalid_returns_none():
    """Text with no JSON object yields None (dropped downstream)."""
    assert sent._parse("no json here") is None

# ------------------------------------------------------------------------------------
# aggregation

def test_aggregate_time_decay_favors_fresh():
    """A fresh confident item outweighs a stale one of equal magnitude, opposite sign."""
    now = 1_000_000.0
    scored = [{"asset": "BTC", "sentiment": 1.0, "confidence": 1.0, "ts": now},
              {"asset": "BTC", "sentiment": -1.0, "confidence": 1.0, "ts": now - 6 * 3600}]
    sig = sent.aggregate(scored, half_life_s=6 * 3600, now=now)
    assert sig["BTC"]["n"] == 2 and sig["BTC"]["score"] > 0


def test_score_items_uses_model(monkeypatch):
    """score_items scores via the (mocked) completion and attaches the parsed fields."""
    monkeypatch.setattr(sent, "_complete",
                        lambda prompt, provider, model, url: '{"asset":"SOL","sentiment":0.4,"confidence":0.7}')
    out = sent.score_items([{"source": "x", "title": "Solana up", "url": "", "ts": 1}])
    assert len(out) == 1 and out[0]["asset"] == "SOL" and out[0]["title"] == "Solana up"
