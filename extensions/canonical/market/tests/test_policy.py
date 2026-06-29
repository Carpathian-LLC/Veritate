# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the trading-policy layer
#   (extensions/canonical/market/server/policy.py): the backtest curve output the paper
#   trading page renders, the vol-harvest premium gate, the implied-premium override, and
#   the per-trade row builder. Pure numpy, deterministic, no model or network.
# extensions/canonical/market/tests/test_policy.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.normpath(os.path.join(HERE, "..", "server"))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import policy as pol

# ------------------------------------------------------------------------------------
# Fixtures

def _series(n=200, ret=0.01, p_up=0.7, exp_move=0.01, conf=0.4):
    """Constant-signal arrays: price walks with ret, the rest held flat for determinism."""
    ret_next = np.full(n, ret)
    price = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], ret_next[:-1]])))
    return {"t": list(range(n)), "price": price.tolist(),
            "p_up": np.full(n, p_up), "conf": np.full(n, conf),
            "exp_move": np.full(n, exp_move), "vol": np.full(n, 0.01), "ret_next": ret_next}

# ------------------------------------------------------------------------------------
# backtest output

def test_backtest_returns_aligned_curve():
    """backtest returns equity/gate/lean/size/pnl_bps arrays aligned to the input length."""
    s = _series()
    r = pol.backtest(s["price"], s["p_up"], s["conf"], s["exp_move"], s["vol"], s["ret_next"])
    n = len(s["ret_next"])
    assert len(r["equity"]) == n and len(r["gate"]) == n and len(r["pnl_bps"]) == n


def test_directional_profits_on_correct_lean():
    """Directional mode nets positive when the lean and the realized move agree every bar."""
    s = _series(p_up=0.7, ret=0.01)
    r = pol.backtest(s["price"], s["p_up"], s["conf"], s["exp_move"], s["vol"], s["ret_next"],
                     mode="directional", fee=0.0005)
    assert r["equity"][-1] > 0


def test_vol_harvest_move_gate_blocks_trades():
    """A move gate far above the premium leaves the vol-harvest policy with zero trades."""
    s = _series(exp_move=0.05)
    r = pol.backtest(s["price"], s["p_up"], s["conf"], s["exp_move"], s["vol"], s["ret_next"],
                     mode="vol_harvest", move_gate=100.0)
    assert r["n_trades"] == 0


def test_implied_premium_override_changes_gating():
    """Passing a high implied premium array gates out trades a trailing premium would take."""
    s = _series(exp_move=0.05, ret=0.02)
    base = pol.backtest(s["price"], s["p_up"], s["conf"], s["exp_move"], s["vol"], s["ret_next"],
                        mode="vol_harvest", move_gate=1.0)
    implied = pol.backtest(s["price"], s["p_up"], s["conf"], s["exp_move"], s["vol"], s["ret_next"],
                           mode="vol_harvest", move_gate=1.0, premium=np.full(len(s["ret_next"]), 0.10))
    assert implied["n_trades"] < base["n_trades"]


def test_trades_rows_only_gated_bars():
    """trades() emits one row per gated bar with the expected keys, aligned to the series."""
    s = _series()
    r = pol.backtest(s["price"], s["p_up"], s["conf"], s["exp_move"], s["vol"], s["ret_next"],
                     mode="directional", fee=0.0005)
    rows = pol.trades(s, r)
    assert len(rows) == sum(r["gate"])
    assert set(rows[0]) == {"t", "price", "side", "lean", "size", "pnl_bps"}
