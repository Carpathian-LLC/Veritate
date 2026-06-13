# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the experimental market platform (veritate_mri/market/).
#   Guards the load-bearing correctness claims: timestamp-unit normalization, resample
#   boundaries, NO-LOOKAHEAD features, forward-label alignment, purged-split
#   non-overlap, and a tiny end-to-end train -> predict -> cone -> replay.
# tests/mri/test_market.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MARKET_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "veritate_mri", "market"))
if MARKET_DIR not in sys.path:
    sys.path.insert(0, MARKET_DIR)

import backtest as mbt
import data as md
import dataset as mds
import features as mf
import models as mmd

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

# ------------------------------------------------------------------------------------
# features

def test_features_no_lookahead():
    df = _synth(3000)
    full = mf.compute(df)
    k = 2000
    trunc = mf.compute(df.iloc[:k])
    m = k - mf.warmup_bars() - 5
    a, b = full.iloc[:m].to_numpy(), trunc.iloc[:m].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    assert both.any()
    assert np.allclose(a[both], b[both], atol=1e-9, rtol=1e-6)


def test_features_count_and_finite():
    f = mf.compute(_synth(2000)).dropna()
    assert list(f.columns) == mf.FEATURES and len(mf.FEATURES) == 32
    assert np.isfinite(f.to_numpy()).all()

# ------------------------------------------------------------------------------------
# labels + split

def test_forward_labels_align():
    df = _synth(500)
    fr, fv = mds.make_labels(df, horizon=10)
    lc = np.log(df["close"].to_numpy())
    assert np.isclose(fr[100], lc[110] - lc[100])                  # forward return is strictly future
    assert np.isnan(fr[-1]) and fv[-1] != fv[-1]                   # last rows have no future


def test_purged_split_no_overlap(monkeypatch):
    # build a dataset from two synthetic instruments via the real builder, mocked loader
    frames = {"AAA": _synth(2500, 1), "BBB": _synth(2500, 2)}
    monkeypatch.setattr(md, "list_instruments", lambda s="crypto": list(frames))
    monkeypatch.setattr(md, "load", lambda sym, base="1m", source="crypto", cols=None: frames[sym])
    d = mds.build(["AAA", "BBB"], horizon=15, base="1m")
    tr, va = mds.purged_split(d, val_frac=0.2)
    assert tr.sum() > 0 and va.sum() > 0
    assert d["t"][tr].max() < d["t"][va].min()                    # purge: train strictly before val

# ------------------------------------------------------------------------------------
# end-to-end model

def test_train_predict_cone_replay(monkeypatch, tmp_path):
    frames = {f"S{i}": _synth(6000, i) for i in range(3)}
    monkeypatch.setattr(md, "list_instruments", lambda s="crypto": list(frames))
    monkeypatch.setattr(md, "load", lambda sym, base="1m", source="crypto", cols=None: frames[sym])
    monkeypatch.setattr(mmd, "MODEL_DIR", str(tmp_path))
    summary = mmd.train_and_save(symbols=list(frames), horizons=[5], base="1m",
                                 max_bars=6000, train_stride=4)
    assert "1m_h5" in summary and summary["1m_h5"]["vol_r2"] == summary["1m_h5"]["vol_r2"]

    mm = mmd.MarketModel.load("1m", 5)
    pred = mm.predict_latest(frames["S0"])
    assert set(pred) >= {"vol_fwd", "p_up", "confidence", "regime"}
    assert 0.0 <= pred["p_up"] <= 1.0 and pred["vol_fwd"] > 0
    assert pred["regime"] in ("calm", "normal", "turbulent")

    cone = mm.cone(pred["vol_fwd"], pred["p_up"])
    band = cone[0.8]
    w_first = band[0][1] - band[0][0]
    w_last = band[-1][1] - band[-1][0]
    assert w_last > w_first > 0                                    # cone widens with horizon

    res = mbt.replay(mm, frames["S1"])
    assert res is not None
    assert {"summary", "vol_series", "reliability", "equity"} <= set(res)
    assert res["summary"]["n"] > 0
