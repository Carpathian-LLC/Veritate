# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract test for the autonomous loop's tick
#   (extensions/canonical/market/server/autotrader.py): the live feed and model forecast
#   are mocked (no network, no checkpoint), policy + execution run for real against a fake
#   broker, verifying the forecast->decision->order wiring end to end.
# extensions/canonical/market/tests/test_autotrader.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.normpath(os.path.join(HERE, "..", "server"))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import autotrader as at
from test_execution import FakeBroker

# ------------------------------------------------------------------------------------
# tick wiring

def test_tick_forecast_to_order(monkeypatch):
    """A bullish forecast flows through policy.decide and submits a buy on the fake broker."""
    idx = pd.date_range("2026-01-01", periods=70, freq="1min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)
    monkeypatch.setattr(at.lv, "fetch", lambda symbol, base="1m", limit=400: (df, 0))
    monkeypatch.setattr(at.vz, "predict_next",
                        lambda model, seq, closed, stride: {"p_up": 0.7, "confidence": 0.4,
                                                            "expected_move": 0.02, "vol": 0.01})
    monkeypatch.setattr(at.vz, "trailing_premium", lambda closed: 0.001)
    broker = FakeBroker(equity=10000.0, qty=0.0)
    overrides = {"mode": "directional", "fee": 0.0005, "conf_gate": 0.0, "move_gate": 1.0, "sizing": "confidence"}
    r = at.tick("BTCUSDT", (None, 256, 9), broker, overrides)
    assert r["ok"] and r["exec"]["action"] == "buy"
    assert len(broker.orders) == 1 and broker.orders[0]["side"] == "buy"


def test_tick_no_data_returns_error(monkeypatch):
    """A short/empty feed yields a clean error record, no order."""
    monkeypatch.setattr(at.lv, "fetch", lambda symbol, base="1m", limit=400: (None, None))
    broker = FakeBroker()
    r = at.tick("BTCUSDT", (None, 256, 9), broker, {"mode": "directional"})
    assert r["ok"] is False and broker.orders == []
