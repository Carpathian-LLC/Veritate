# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the autonomous news trader's pure ledger logic
#   (extensions/canonical/paper_trade/server/news_trader.py): sentiment -> target exposure,
#   paper rebalance (buy to target, exit untargeted, fee, dust skip), mark-to-market.
#   No network, no model, no file I/O.
# extensions/canonical/paper_trade/tests/test_news_trader.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.normpath(os.path.join(HERE, "..", "server"))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import news_trader as nt

# ------------------------------------------------------------------------------------
# Fixtures

def _fresh():
    return {"cash": 10000.0, "positions": {}, "start_cash": 10000.0, "history": []}

# ------------------------------------------------------------------------------------
# targets

def test_targets_gate_universe_and_block():
    """Above-gate tradable assets size long; non-tradable (MARKET) and below-gate are dropped."""
    sig = {"BTC": {"score": 0.8, "n": 3}, "MARKET": {"score": 0.9, "n": 2},
           "ETH": {"score": 0.1, "n": 1}, "SOL": {"score": -0.5, "n": 1}}
    t = nt.targets(sig, gate=0.25, max_size=1.0)
    assert t["BTCUSDT"] == 0.8
    assert "MARKET" + nt.QUOTE not in t       # not a tradable coin
    assert "ETHUSDT" not in t and "SOLUSDT" not in t   # below gate / negative


def test_targets_caps_at_max_size():
    """Target exposure is capped at max_size."""
    t = nt.targets({"BTC": {"score": 0.9, "n": 1}}, gate=0.25, max_size=0.5)
    assert t["BTCUSDT"] == 0.5


def test_targets_normalizes_total_exposure_to_one():
    """Two strong longs are scaled so total exposure is 100%, never leveraged (no neg cash)."""
    t = nt.targets({"BTC": {"score": 0.6, "n": 1}, "ETH": {"score": 0.6, "n": 1}}, gate=0.25, max_size=1.0)
    assert abs(sum(t.values()) - 1.0) < 1e-9
    assert t["BTCUSDT"] == 0.5 and t["ETHUSDT"] == 0.5

# ------------------------------------------------------------------------------------
# rebalance + equity

def test_rebalance_buys_to_target_and_charges_fee():
    """From flat, a 0.5 target (past the band) buys 0.5*equity/price; equity drops only by fee."""
    led = _fresh()
    acts = nt.rebalance(led, {"BTCUSDT": 0.5}, {"BTCUSDT": 100.0}, 0.002, 0.1)
    assert led["positions"]["BTCUSDT"] == 50.0
    assert len(acts) == 1 and acts[0]["side"] == "buy"
    assert abs(nt.equity(led, {"BTCUSDT": 100.0}) - 9990.0) < 1e-6     # 10000 - 10 fee


def test_rebalance_exits_untargeted_position():
    """A held symbol no longer targeted is fully sold to cash (exposure exceeds the band)."""
    led = _fresh(); led["positions"]["BTCUSDT"] = 50.0; led["cash"] = 4990.0
    acts = nt.rebalance(led, {}, {"BTCUSDT": 100.0}, 0.002, 0.1)
    assert led["positions"]["BTCUSDT"] == 0.0
    assert any(a["side"] == "sell" for a in acts)


def test_rebalance_deadband_skips_small_drift():
    """No trade when target vs current exposure is within the band (no fee churn)."""
    led = _fresh(); led["positions"]["BTCUSDT"] = 50.0; led["cash"] = 4990.0   # ~50% exposure
    acts = nt.rebalance(led, {"BTCUSDT": 0.55}, {"BTCUSDT": 100.0}, 0.002, 0.12)  # +5% < band
    assert acts == []

# ------------------------------------------------------------------------------------
# market hours (stocks only trade when the exchange is open)

def test_market_open_crypto_always():
    """Crypto is 24/7, so market_open('crypto') is always True."""
    assert nt.market_open("crypto") is True


def test_is_market_hours_gates_us_session():
    """US equity gate: open midday on a weekday, closed after hours and on weekends."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    assert nt._is_market_hours(datetime(2026, 6, 17, 10, 0, tzinfo=et)) is True    # Wed 10:00 ET
    assert nt._is_market_hours(datetime(2026, 6, 17, 20, 0, tzinfo=et)) is False   # Wed 20:00 ET
    assert nt._is_market_hours(datetime(2026, 6, 20, 11, 0, tzinfo=et)) is False   # Saturday
