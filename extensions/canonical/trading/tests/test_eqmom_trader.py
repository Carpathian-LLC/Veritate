# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Deterministic tests for the equity momentum paper trader: 12-1 window, decile
#   targets, monthly cadence, resume flag. No network: fetchers monkeypatched.
# extensions/canonical/trading/tests/test_eqmom_trader.py
# ------------------------------------------------------------------------------------
# Imports:

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import eqmom_trader as eq
import news_trader as nt

# ------------------------------------------------------------------------------------
# Functions

def test_momentum_skips_latest_month():
    """12-1 momentum measures t-252 to t-21 and ignores the most recent month entirely."""
    closes = [100.0] * 300
    closes[-253] = 100.0
    closes[-22] = 150.0
    closes[-1] = 1.0
    assert math.isclose(eq.momentum_12_1(closes), 0.5)


def test_momentum_none_on_short_history():
    """Names with under a year of history are excluded, not misranked."""
    assert eq.momentum_12_1([100.0] * 100) is None


def test_targets_top_decile_equal_weight():
    """targets() holds the strongest decile at equal weight summing to 1."""
    moms = {f"T{i:03d}": i / 100.0 for i in range(200)}
    tgt = eq.targets(moms)
    assert len(tgt) == 20 and "T199" in tgt and "T000" not in tgt
    assert math.isclose(sum(tgt.values()), 1.0)


def test_tick_rebalances_once_per_month(tmp_path, monkeypatch):
    """A second tick in the same month marks the book but places no trades."""
    monkeypatch.setattr(eq, "universe", lambda: [f"T{i:03d}" for i in range(120)])
    monkeypatch.setattr(eq, "daily_adjcloses", lambda t: [100.0 + int(t[1:]) * (i / 300.0) for i in range(300)])
    monkeypatch.setattr(eq, "FETCH_PAUSE", 0.0)
    monkeypatch.setattr(nt, "stock_price", lambda s: 100.0)
    monkeypatch.setattr(nt, "market_open", lambda m: True)
    path = str(tmp_path / "led.json")
    first = eq.tick("main", path)
    second = eq.tick("main", path)
    assert len(first["acts"]) > 0 and second["acts"] == []


def test_tick_holds_outside_market_hours(tmp_path, monkeypatch):
    """No rebalance fires outside the US regular session; the month stays unstamped."""
    monkeypatch.setattr(eq, "universe", lambda: ["A", "B"])
    monkeypatch.setattr(nt, "stock_price", lambda s: 100.0)
    monkeypatch.setattr(nt, "market_open", lambda m: False)
    path = str(tmp_path / "led.json")
    eq.tick("main", path)
    assert "month" not in nt.load_ledger(path)


def test_stop_clears_resume_flag(tmp_path, monkeypatch):
    """stop_thread clears the ledger auto flag so resume() will not restart the arm."""
    monkeypatch.setattr(nt, "ledger_for", lambda label: str(tmp_path / f"{label}.json"))
    led = nt.load_ledger(nt.ledger_for(eq.label_for("main")))
    led["auto"] = True
    nt.save_ledger(led, nt.ledger_for(eq.label_for("main")))
    eq.stop_thread("main")
    assert nt.load_ledger(nt.ledger_for(eq.label_for("main"))).get("auto") is False
