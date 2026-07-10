# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Route-registration smoke for the Trading extension: register(app) mounts every
#   /ext/trading/* route group on a bare Flask app, plus the system view/stop routes.
#   No network, no threads (resume is a no-op with state paths pointed at tmp).
# extensions/canonical/trading/tests/test_register.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib.util
import os
import sys

import flask
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import news_trader as nt
import scanner as sc
import scraper

# ------------------------------------------------------------------------------------
# Fixtures

@pytest.fixture
def app(tmp_path, monkeypatch):
    """register(app) on a bare Flask app with every state path at tmp (no resume side effects)."""
    monkeypatch.setattr(nt, "LEDGER_PATH", str(tmp_path / "account.json"))
    monkeypatch.setattr(nt, "LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(sc, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(scraper, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    path = os.path.join(os.path.dirname(__file__), "..", "server", "register.py")
    spec = importlib.util.spec_from_file_location("trading_register_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = flask.Flask(__name__)
    mod.register(a)
    return a

# ------------------------------------------------------------------------------------
# Functions

def test_register_mounts_market_routes(app):
    """register(app) adds the /ext/trading/market/* model-serving routes."""
    rules = {r.rule for r in app.url_map.iter_rules()}
    want = {"/ext/trading/market/veritate_models", "/ext/trading/market/veritate_hindcast",
            "/ext/trading/market/veritate_benchmark", "/ext/trading/market/veritate_data_report",
            "/ext/trading/market/veritate_live", "/ext/trading/market/instruments",
            "/ext/trading/market/paper_signal", "/ext/trading/market/paper_backtest",
            "/ext/trading/market/paper_decide"}
    assert want <= rules


def test_register_mounts_paper_routes(app):
    """register(app) adds the /ext/trading/paper/* trader routes."""
    rules = {r.rule for r in app.url_map.iter_rules()}
    want = {"/ext/trading/paper/sentiment", "/ext/trading/paper/feed",
            "/ext/trading/paper/trader/start", "/ext/trading/paper/trader/stop",
            "/ext/trading/paper/trader/status", "/ext/trading/paper/trader/update",
            "/ext/trading/paper/account", "/ext/trading/paper/exp/start",
            "/ext/trading/paper/exp/stop", "/ext/trading/paper/exp/status",
            "/ext/trading/paper/exp/update", "/ext/trading/paper/exp/accounts",
            "/ext/trading/paper/xsmom/start", "/ext/trading/paper/xsmom/stop",
            "/ext/trading/paper/xsmom/status", "/ext/trading/paper/xsmom/accounts",
            "/ext/trading/paper/eqmom/start", "/ext/trading/paper/eqmom/stop",
            "/ext/trading/paper/eqmom/status", "/ext/trading/paper/eqmom/accounts",
            "/ext/trading/paper/ml7/start", "/ext/trading/paper/ml7/stop",
            "/ext/trading/paper/ml7/status", "/ext/trading/paper/ml7/accounts"}
    assert want <= rules


def test_register_mounts_intel_settings_system_routes(app):
    """register(app) adds intel, settings, system, and models-page routes."""
    rules = {r.rule for r in app.url_map.iter_rules()}
    want = {"/ext/trading/intel/scan", "/ext/trading/intel/trending", "/ext/trading/intel/pumps",
            "/ext/trading/intel/brief", "/ext/trading/intel/status", "/ext/trading/intel/candles",
            "/ext/trading/intel/watch/start", "/ext/trading/intel/watch/stop",
            "/ext/trading/settings", "/ext/trading/system", "/ext/trading/system/stop",
            "/ext/trading/system/stop_all", "/ext/trading/models"}
    assert want <= rules


def test_system_lists_every_runnable(app):
    """GET /ext/trading/system lists news, xsmom arms, eqmom, ml7, and intel watch, all stopped."""
    j = app.test_client().get("/ext/trading/system").get_json()
    ids = {r["id"] for r in j["runnables"]}
    assert {"news_main", "xsmom_base", "xsmom_gated", "eqmom_main", "ml7_main",
            "intel_watch"} <= ids
    assert j["running"] == 0


def test_system_stop_all_idles_clean(app):
    """POST /ext/trading/system/stop_all succeeds with nothing running."""
    j = app.test_client().post("/ext/trading/system/stop_all", json={}).get_json()
    assert j["ok"] is True


def test_system_stop_unknown_id_rejected(app):
    """POST /ext/trading/system/stop with an unknown id returns ok=false."""
    j = app.test_client().post("/ext/trading/system/stop", json={"id": "nope"}).get_json()
    assert j["ok"] is False


def test_intel_candles_route_serves_and_404s(app, monkeypatch):
    """GET /ext/trading/intel/candles returns scanner.candles rows; 404 when None."""
    monkeypatch.setattr(sc, "candles", lambda sym, bar, limit: [
        {"t": 1, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 2.0}])
    j = app.test_client().get("/ext/trading/intel/candles?symbol=btc&bar=1H").get_json()
    assert j["ok"] is True and j["symbol"] == "BTC" and len(j["candles"]) == 1
    monkeypatch.setattr(sc, "candles", lambda sym, bar, limit: None)
    assert app.test_client().get("/ext/trading/intel/candles?symbol=zzz").status_code == 404


def test_settings_get_returns_defaults(app):
    """GET /ext/trading/settings returns defaults with health-annotated channels."""
    j = app.test_client().get("/ext/trading/settings").get_json()
    assert j["ok"] is True
    assert j["model"] == scraper.DEFAULT_SETTINGS["model"]
    assert {c["id"] for c in j["channels"]} == {c["id"] for c in scraper.DEFAULT_CHANNELS}
    assert all(c["status"] == "unknown" for c in j["channels"])


def test_settings_post_persists_channels(app, tmp_path):
    """POST /ext/trading/settings saves a channel edit and GET reflects it."""
    c = app.test_client()
    body = {"model": "qwen2.5:14b-instruct", "news_interval": 900,
            "channels": [{"type": "reddit", "value": "CryptoCurrency", "enabled": True}]}
    j = c.post("/ext/trading/settings", json=body).get_json()
    assert j["ok"] is True
    j2 = c.get("/ext/trading/settings").get_json()
    assert j2["model"] == "qwen2.5:14b-instruct"
    assert j2["news_interval"] == 900
    assert [ch["type"] for ch in j2["channels"]] == ["reddit"]
