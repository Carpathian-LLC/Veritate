# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Deterministic tests for the XS momentum paper trader: ranking, vol gate, signed
#   rebalance math, weekly cadence. No network: fetchers monkeypatched.
# extensions/canonical/trading/tests/test_xsmom_trader.py
# ------------------------------------------------------------------------------------
# Imports:

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import news_trader as nt
import xsmom_trader as xt

# ------------------------------------------------------------------------------------
# Functions

def _rets(n=20):
    return {f"C{i:02d}": (i - n / 2) / 100.0 for i in range(n)}


def test_targets_longs_top_and_shorts_bottom():
    """targets() puts positive weight on the best trailing returns, negative on the worst."""
    tgt = xt.targets(_rets(), True)
    assert tgt[xt.sym_for("C19")] > 0 and tgt[xt.sym_for("C00")] < 0


def test_targets_gross_is_one():
    """Sum of |weights| equals GROSS with equal weight per name."""
    tgt = xt.targets(_rets(), True)
    assert math.isclose(sum(abs(w) for w in tgt.values()), xt.GROSS)


def test_targets_empty_when_flat():
    """A gated-flat week produces no targets."""
    assert xt.targets(_rets(), False) == {}


def test_targets_empty_when_universe_thin():
    """Fewer fetchable names than two legs -> no trade rather than a lopsided book."""
    assert xt.targets({"A": 0.1, "B": -0.1}, True) == {}


def test_vol_gate_flat_in_high_vol_regime():
    """A calm year followed by a violent month puts current vol above the median -> flat."""
    closes = [100.0 * (1 + 0.001 * (i % 2)) for i in range(400)]
    closes += [closes[-1] * (1.1 if i % 2 else 0.9) for i in range(30)]
    assert xt.vol_gate_invested(closes) is False


def test_vol_gate_invested_in_calm_regime():
    """A violent year followed by a calm month puts current vol below the median -> invested."""
    closes = [100.0 * (1.05 if i % 2 else 0.95) ** (i % 3) for i in range(400)]
    closes += [closes[-1] * (1 + 0.0001 * (i % 2)) for i in range(40)]
    assert xt.vol_gate_invested(closes) is True


def test_rebalance_short_adds_cash_and_charges_fee():
    """Opening a short raises cash by the notional minus the per-side fee."""
    led = {"cash": 10000.0, "positions": {}, "start_cash": 10000.0, "history": []}
    xt.rebalance_to(led, {"AAAUSDT": -0.5}, {"AAAUSDT": 100.0}, 0.001)
    assert math.isclose(led["cash"], 10000.0 + 5000.0 - 5.0)


def test_rebalance_exits_untargeted_position():
    """A held symbol absent from targets is closed out."""
    led = {"cash": 5000.0, "positions": {"BBBUSDT": 10.0}, "start_cash": 10000.0, "history": []}
    xt.rebalance_to(led, {}, {"BBBUSDT": 100.0}, 0.0)
    assert "BBBUSDT" not in led["positions"]


def test_tick_rebalances_once_per_week(tmp_path, monkeypatch):
    """The second tick inside the same ISO week marks the book but places no trades."""
    monkeypatch.setattr(xt, "daily_closes", lambda stem, n: [100.0 + i + (sum(map(ord, stem)) % 50) for i in range(n)])
    monkeypatch.setattr(xt, "mark_price", lambda stem: 100.0)
    path = str(tmp_path / "led.json")
    first = xt.tick("base", path)
    second = xt.tick("base", path)
    assert len(first["acts"]) > 0 and second["acts"] == []


def test_tick_history_entry_shape(tmp_path, monkeypatch):
    """Each tick appends a history row carrying equity and the BTC benchmark price."""
    monkeypatch.setattr(xt, "daily_closes", lambda stem, n: [100.0 + i for i in range(n)])
    monkeypatch.setattr(xt, "mark_price", lambda stem: 250.0)
    path = str(tmp_path / "led.json")
    xt.tick("base", path)
    h = nt.load_ledger(path)["history"][-1]
    assert h["bench_px"] == 250.0 and h["bench_asset"] == "BTC" and "equity" in h
