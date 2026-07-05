# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Deterministic tests for the ml7 paper trader: tranche targets, single-slot
#   rebalance accounting, fee-on-delta, daily stamp, model-missing guard, resume
#   stamp, cache round-trip. No network: signal and prices monkeypatched.
# extensions/canonical/trading/tests/test_ml7_trader.py
# ------------------------------------------------------------------------------------
# Imports:

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import ml7_trader as mt
import news_trader as nt

# ------------------------------------------------------------------------------------
# Constants

W = mt.GROSS / mt.TRANCHES / 2.0 / 8          # per-name weight at k=8
PX = 100.0

# ------------------------------------------------------------------------------------
# Fixtures / helpers

@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "HIST_DIR", str(tmp_path / "hist"))
    monkeypatch.setattr(mt, "FUND_CACHE_DIR", str(tmp_path / "hist" / "funding"))
    monkeypatch.setattr(mt, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(mt, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setattr(mt, "_MODEL", {"models": None, "manifest": None})
    return tmp_path


def _scores(n=40):
    return {f"C{i:02d}USDT": i / n for i in range(n)}


def _led(cash=10000.0):
    return {"cash": cash, "positions": {}, "start_cash": cash, "history": [], "tranches": {}}

# ------------------------------------------------------------------------------------
# Functions

def test_tranche_targets_quintiles():
    """40 scored names -> 8 longs at +w and 8 shorts at -w, w = (1/7)/2/8."""
    tgt = mt.tranche_targets(_scores())
    longs = [s for s, w in tgt.items() if w > 0]
    shorts = [s for s, w in tgt.items() if w < 0]
    assert len(longs) == 8 and len(shorts) == 8
    assert all(math.isclose(abs(w), W) for w in tgt.values())
    assert "C39USDT" in longs and "C00USDT" in shorts


def test_tranche_targets_thin_universe():
    """Fewer scored names than MIN_NAMES -> no targets (hold, retry next tick)."""
    assert mt.tranche_targets(_scores(mt.MIN_NAMES - 1)) == {}


def test_book_gross_is_one_across_tranches():
    """Seven full tranches sum to GROSS 1.0 of gross exposure."""
    gross_one = sum(abs(w) for w in mt.tranche_targets(_scores()).values())
    assert math.isclose(mt.TRANCHES * gross_one, mt.GROSS)


def test_rebalance_touches_only_its_slot():
    """Rebalancing slot 0 leaves other tranches' positions and books untouched."""
    led = _led()
    led["tranches"] = {"0": {"AAAUSDT": 1.0}, "1": {"BBBUSDT": 2.0}}
    led["positions"] = {"AAAUSDT": 1.0, "BBBUSDT": 2.0}
    acts = mt.rebalance_tranche(led, 0, {"CCCUSDT": W}, {s: PX for s in
                                ["AAAUSDT", "BBBUSDT", "CCCUSDT"]}, 0.0)
    assert led["positions"]["BBBUSDT"] == 2.0
    assert led["tranches"]["1"] == {"BBBUSDT": 2.0}
    assert {a["sym"] for a in acts} <= {"AAAUSDT", "CCCUSDT"}


def test_rebalance_fee_on_traded_delta():
    """A dollar-neutral tranche entry costs exactly fee * traded notional in cash."""
    led = _led()
    tgt = {"AAAUSDT": 0.05, "BBBUSDT": -0.05}
    mt.rebalance_tranche(led, 3, tgt, {"AAAUSDT": PX, "BBBUSDT": PX}, mt.FEE)
    assert math.isclose(led["cash"], 10000.0 - mt.FEE * 1000.0)


def test_rebalance_exits_untargeted_name():
    """A held tranche name absent from the new targets is closed out of the book."""
    led = _led()
    led["tranches"] = {"2": {"AAAUSDT": 5.0}}
    led["positions"] = {"AAAUSDT": 5.0}
    mt.rebalance_tranche(led, 2, {}, {"AAAUSDT": PX}, 0.0)
    assert "AAAUSDT" not in led["positions"] and led["tranches"]["2"] == {}


def test_tick_trades_once_per_day(tmp_path, monkeypatch, paths):
    """The second tick on the same closed UTC day marks the book but places no trades."""
    monkeypatch.setattr(mt, "_signal", lambda day: _scores())
    monkeypatch.setattr(mt.xt, "mark_price", lambda stem: PX)
    path = str(tmp_path / "led.json")
    first = mt.tick("main", path)
    second = mt.tick("main", path)
    assert len(first["acts"]) == 16 and second["acts"] == []


def test_tick_tranche_gross_is_seventh_of_equity(tmp_path, monkeypatch, paths):
    """A first tick deploys one tranche: gross notional ~ equity / 7."""
    monkeypatch.setattr(mt, "_signal", lambda day: _scores())
    monkeypatch.setattr(mt.xt, "mark_price", lambda stem: PX)
    path = str(tmp_path / "led.json")
    r = mt.tick("main", path)
    gross = sum(abs(a["qty"]) * a["price"] for a in r["acts"])
    assert math.isclose(gross, 10000.0 / mt.TRANCHES, rel_tol=1e-6)


def test_tick_history_row_shape(tmp_path, monkeypatch, paths):
    """Each tick appends a history row with equity and the BTC benchmark price."""
    monkeypatch.setattr(mt, "_signal", lambda day: None)
    monkeypatch.setattr(mt.xt, "mark_price", lambda stem: 250.0)
    path = str(tmp_path / "led.json")
    mt.tick("main", path)
    h = nt.load_ledger(path)["history"][-1]
    assert h["bench_px"] == 250.0 and h["bench_asset"] == "BTC" and "equity" in h


def test_tick_without_model_holds(tmp_path, monkeypatch, paths):
    """No trained model on disk -> tick marks the ledger but never trades."""
    monkeypatch.setattr(mt.xt, "mark_price", lambda stem: PX)
    path = str(tmp_path / "led.json")
    r = mt.tick("main", path)
    assert r["acts"] == [] and len(nt.load_ledger(path)["history"]) == 1


def test_model_state_missing_is_graceful(paths):
    """model_state() reports the missing frozen model instead of raising."""
    assert mt.model_state().startswith("missing")


def test_resume_honors_auto_stamp(tmp_path, monkeypatch):
    """resume() restarts only arms whose ledger carries auto=true."""
    monkeypatch.setattr(nt, "LEDGER_DIR", str(tmp_path))
    started = []
    monkeypatch.setattr(mt, "start_thread", lambda a: started.append(a) or True)
    assert mt.resume() == {}
    led = _led()
    led["auto"] = True
    nt.save_ledger(led, nt.ledger_for(mt.label_for("main")))
    assert mt.resume() == {"main": True} and started == ["main"]


def test_hist_cache_roundtrip_exact(paths):
    """The daily history cache round-trips float64 values exactly through CSV."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    df = pd.DataFrame({c: rng.normal(100, 5, 20) for c in
                       ["open", "high", "low", "close", "volume"]}, index=idx)
    df["nmin"] = 1440
    mt._save_hist("AAAUSDT", df)
    back = mt._load_hist("AAAUSDT")
    assert np.array_equal(back.to_numpy(), df.to_numpy())
    assert list(back.index) == list(idx)
