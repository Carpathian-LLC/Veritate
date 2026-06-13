# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the market data layer (veritate_mri/market/data.py): the
#   load-bearing correctness claims the byte-model serving path depends on, namely
#   timestamp-unit normalization and no-lookahead resample boundaries.
# tests/mri/test_market.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MARKET_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "veritate_mri", "market"))
if MARKET_DIR not in sys.path:
    sys.path.insert(0, MARKET_DIR)

import data as md

# ------------------------------------------------------------------------------------
# Fixtures

def _synth(n=4000, seed=0):
    """Synthetic OHLCV with volatility clustering, UTC minute index."""
    rng = np.random.default_rng(seed)
    vol = 0.001 * (1.0 + 0.8 * np.sin(np.arange(n) / 200.0) ** 2)
    r = rng.normal(0, 1, n) * vol
    c = 100.0 * np.exp(np.cumsum(r))
    o = c * np.exp(-r * 0.5)
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0005, n)))
    v = rng.uniform(1e3, 5e3, n) * (1 + 5 * vol)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)

# ------------------------------------------------------------------------------------
# data layer

def test_normalize_time_detects_units():
    t = np.array([1502942400000, 1780271940000000], dtype=np.int64)   # ms (2017), us (2026)
    dt = md.normalize_time(t)
    assert dt[0].year == 2017 and dt[1].year == 2026


def test_resample_boundary_no_lookahead():
    df = _synth(600)
    h = md.resample(df, "15m")
    b0 = h.index[0]
    inside = df[(df.index >= b0) & (df.index < b0 + pd.Timedelta("15min"))]
    assert h["close"].iloc[0] == inside["close"].iloc[-1]          # close = last inside bar
    assert h["high"].iloc[0] == inside["high"].max()
