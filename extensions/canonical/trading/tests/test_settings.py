# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Deterministic tests for the shared scraper's settings + channel registry
#   (extensions/canonical/trading/server/scraper.py): channel normalization/CRUD,
#   settings persistence, reddit hot.json parsing (fixture, no network), and
#   per-channel health marking on blocked fetches.
# extensions/canonical/trading/tests/test_settings.py
# ------------------------------------------------------------------------------------
# Imports:

import io
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import scraper

# ------------------------------------------------------------------------------------
# Fixtures

REDDIT_FIXTURE = {
    "data": {"children": [
        {"data": {"title": "Sticky rules post", "permalink": "/r/x/1", "created_utc": 100, "stickied": True}},
        {"data": {"title": "BTC breaks resistance", "permalink": "/r/x/2", "created_utc": 1700000000}},
        {"data": {"title": "  ", "permalink": "/r/x/3", "created_utc": 1700000001}},
        {"data": {"title": "ETH staking news", "permalink": "/r/x/4", "created_utc": 1700000002}},
    ]}
}


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(scraper, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(scraper, "_HEALTH", {})
    return tmp_path

# ------------------------------------------------------------------------------------
# Functions

def test_load_settings_defaults(tmp_settings):
    """load_settings with no file returns the defaults including the default channels."""
    s = scraper.load_settings()
    assert s["model"] == scraper.DEFAULT_SETTINGS["model"]
    assert [c["id"] for c in s["channels"]] == [c["id"] for c in scraper.DEFAULT_CHANNELS]


def test_save_settings_persists_roundtrip(tmp_settings):
    """save_settings writes settings.json and load_settings reads the same values back."""
    scraper.save_settings({"model": "m1", "news_interval": 60, "intel_interval": 120,
                           "news_fee_bps": 5.5})
    s = scraper.load_settings()
    assert (s["model"], s["news_interval"], s["intel_interval"], s["news_fee_bps"]) == ("m1", 60, 120, 5.5)


def test_save_settings_clamps_interval(tmp_settings):
    """save_settings clamps a too-small interval to the 15s floor."""
    s = scraper.save_settings({"news_interval": 1})
    assert s["news_interval"] == 15


def test_channel_add_and_delete(tmp_settings):
    """Posting a new channel list replaces the registry; deleting shrinks it."""
    chs = [dict(c) for c in scraper.DEFAULT_CHANNELS]
    chs.append({"type": "reddit", "value": "CryptoCurrency", "enabled": True})
    saved = scraper.save_settings({"channels": chs})
    assert any(c["type"] == "reddit" and c["value"] == "CryptoCurrency" for c in saved["channels"])
    fewer = [c for c in saved["channels"] if c["type"] != "reddit"]
    saved2 = scraper.save_settings({"channels": fewer})
    assert not any(c["type"] == "reddit" for c in saved2["channels"])


def test_normalize_channels_drops_invalid(tmp_settings):
    """normalize_channels drops unknown types and empty values, keeps valid rows."""
    out = scraper.normalize_channels([
        {"type": "rss", "value": "https://a.example/feed"},
        {"type": "carrier_pigeon", "value": "coo"},
        {"type": "gnews", "value": ""},
        {"type": "reddit", "value": "wallstreetbets", "enabled": False},
    ])
    assert [(c["type"], c["enabled"]) for c in out] == [("rss", True), ("reddit", False)]


def test_normalize_channels_unique_ids(tmp_settings):
    """normalize_channels assigns unique ids to duplicate values."""
    out = scraper.normalize_channels([
        {"type": "gnews", "value": "solana"},
        {"type": "gnews", "value": "solana"},
    ])
    assert len({c["id"] for c in out}) == 2


def test_normalize_channels_empty_falls_back(tmp_settings):
    """An all-invalid channel list falls back to the defaults, never an empty registry."""
    out = scraper.normalize_channels([{"type": "bad", "value": "x"}])
    assert [c["id"] for c in out] == [c["id"] for c in scraper.DEFAULT_CHANNELS]


def test_parse_reddit_fixture(tmp_settings):
    """parse_reddit maps hot.json children to items, skipping stickied and empty titles."""
    items = scraper.parse_reddit(REDDIT_FIXTURE, "r_cryptocurrency")
    assert [i["title"] for i in items] == ["BTC breaks resistance", "ETH staking news"]
    assert items[0]["url"] == "https://old.reddit.com/r/x/2"
    assert items[0]["ts"] == 1700000000
    assert all(i["source"] == "r_cryptocurrency" for i in items)


def test_fetch_reddit_blocked_health(tmp_settings, monkeypatch):
    """A 429 from reddit marks the channel blocked and returns no items."""
    def _boom(url):
        raise urllib.error.HTTPError(url, 429, "too many", {}, io.BytesIO(b""))
    monkeypatch.setattr(scraper, "_get", _boom)
    assert scraper.fetch_reddit("r_cc", "CryptoCurrency") == []
    monkeypatch.setattr(scraper, "SETTINGS_PATH", scraper.SETTINGS_PATH)
    health = {h["id"]: h for h in scraper.channel_health()}
    assert scraper._HEALTH["r_cc"]["status"] == "blocked"
    assert all(h["status"] == "unknown" for h in health.values())


def test_fetch_reddit_ok_health(tmp_settings, monkeypatch):
    """A parseable hot.json marks the channel ok with the item count."""
    import json as _json
    monkeypatch.setattr(scraper, "_get", lambda url: _json.dumps(REDDIT_FIXTURE).encode())
    items = scraper.fetch_reddit("r_cc", "CryptoCurrency")
    assert len(items) == 2
    assert scraper._HEALTH["r_cc"] == {"status": "ok", "last_fetch": scraper._HEALTH["r_cc"]["last_fetch"], "items": 2}


def test_channel_health_annotates_channels(tmp_settings):
    """channel_health returns every configured channel with unknown status pre-fetch."""
    rows = scraper.channel_health()
    assert {r["id"] for r in rows} == {c["id"] for c in scraper.DEFAULT_CHANNELS}
    assert all(r["status"] == "unknown" and r["last_fetch"] is None for r in rows)


def test_fear_greed_disabled_channel(tmp_settings, monkeypatch):
    """fear_greed returns None when the index channel is disabled (no fetch attempted)."""
    chs = [dict(c) for c in scraper.DEFAULT_CHANNELS]
    for c in chs:
        if c["type"] == "index":
            c["enabled"] = False
    scraper.save_settings({"channels": chs})
    monkeypatch.setattr(scraper, "_get", lambda url: (_ for _ in ()).throw(AssertionError("fetched")))
    assert scraper.fear_greed() is None


def test_scrape_uses_enabled_channels_only(tmp_settings, monkeypatch):
    """scrape pulls headlines only from enabled non-index channels."""
    scraper.save_settings({"channels": [
        {"id": "a", "type": "rss", "value": "https://a/feed", "enabled": True},
        {"id": "b", "type": "rss", "value": "https://b/feed", "enabled": False},
    ]})
    seen = []
    def _fake(c):
        seen.append(c["id"])
        return [{"source": c["id"], "title": f"t-{c['id']}", "url": "", "ts": 1}]
    monkeypatch.setattr(scraper, "_fetch_channel", _fake)
    items = scraper.scrape(limit=10)
    assert seen == ["a"]
    assert [i["title"] for i in items] == ["t-a"]
