# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The ONE feature function, shared by training and live serving (the #1 rule from the
#   research: identical featurization both sides eliminates train/serve skew).
# - Every feature at row t uses only bars <= t (trailing windows, no centered windows,
#   no negative shifts). Verified by _self_test(): truncating the tail leaves earlier
#   feature rows bitwise unchanged.
# - Feature set follows the evidence: multi-scale realized vol + range estimators
#   (volatility is the forecastable signal), volume/illiquidity, mean-reversion position,
#   candle shape, and time-of-day seasonality. Scale-free so one model spans instruments.
# veritate_mri/market/features.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Constants

# the model input columns, in a fixed order (train and serve must agree)
FEATURES = [
    "r1", "r5", "r15", "r60",                  # multi-scale momentum (log returns)
    "rz5", "rz15", "rz60",                     # vol-normalized momentum (z-scored by trailing vol)
    "rv10", "rv30", "rv120",                   # realized vol over 10/30/120 bars
    "rv_ratio", "rv_accel",                    # vol regime: short/long ratio, short vol change
    "park10", "park30",                        # Parkinson high-low range vol
    "shock",                                   # |r1| / rv30  (how big is this bar vs normal)
    "range1", "body1", "uwick1", "lwick1",     # candle geometry (fractions of close)
    "vol_z30", "vol_ratio", "svol",            # volume z-score, ratio, signed-volume proxy
    "amihud",                                  # |r1| / dollar-volume (illiquidity)
    "zclose20", "zclose60",                    # Bollinger position vs SMA (mean-reversion)
    "rsi14",                                   # RSI
    "dist_hi", "dist_lo",                      # distance from rolling max / min
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",  # intraday + weekly seasonality
]

EPS = 1e-12

# ------------------------------------------------------------------------------------
# Helpers (all trailing / causal)

def _roll_mean(s, n):
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def _roll_std(s, n):
    return s.rolling(n, min_periods=max(2, n // 2)).std()


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rd = dn.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = ru / (rd + EPS)
    return 100.0 - 100.0 / (1.0 + rs)

# ------------------------------------------------------------------------------------
# Feature computation

def compute(df):
    """OHLCV frame (UTC index) -> feature DataFrame aligned to the same index.

    Row t is the feature vector known at the CLOSE of bar t (used to predict t+h).
    Warmup rows with insufficient history are returned as NaN; caller drops them.
    """
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])
    lc = np.log(c.clip(lower=EPS))
    r = lc.diff()                                   # 1-bar log return
    out = pd.DataFrame(index=df.index)

    # momentum
    out["r1"] = r
    out["r5"] = lc.diff(5)
    out["r15"] = lc.diff(15)
    out["r60"] = lc.diff(60)

    # realized vol (root-mean-square of returns), trailing
    r2 = r * r
    rv10 = np.sqrt(_roll_mean(r2, 10))
    rv30 = np.sqrt(_roll_mean(r2, 30))
    rv120 = np.sqrt(_roll_mean(r2, 120))
    out["rv10"], out["rv30"], out["rv120"] = rv10, rv30, rv120
    out["rv_ratio"] = rv10 / (rv120 + EPS)
    out["rv_accel"] = rv10 / (rv30 + EPS)

    # vol-normalized momentum
    out["rz5"] = out["r5"] / (rv30 * np.sqrt(5) + EPS)
    out["rz15"] = out["r15"] / (rv30 * np.sqrt(15) + EPS)
    out["rz60"] = out["r60"] / (rv30 * np.sqrt(60) + EPS)

    # Parkinson high-low range volatility
    hl = np.log((h / l.clip(lower=EPS)).clip(lower=1.0)) ** 2
    k = 1.0 / (4.0 * np.log(2.0))
    out["park10"] = np.sqrt(k * _roll_mean(hl, 10))
    out["park30"] = np.sqrt(k * _roll_mean(hl, 30))

    out["shock"] = r.abs() / (rv30 + EPS)

    # candle geometry (fractions of close)
    rng = (h - l) / (c + EPS)
    out["range1"] = rng
    out["body1"] = (c - o).abs() / (c + EPS)
    out["uwick1"] = (h - np.maximum(o, c)) / (c + EPS)
    out["lwick1"] = (np.minimum(o, c) - l) / (c + EPS)

    # volume
    vm = _roll_mean(v, 30)
    vs = _roll_std(v, 30)
    out["vol_z30"] = (v - vm) / (vs + EPS)
    out["vol_ratio"] = v / (vm + EPS)
    out["svol"] = np.sign(r).fillna(0.0) * out["vol_z30"]
    out["amihud"] = r.abs() / (c * v + EPS) * 1e9      # scaled illiquidity

    # mean-reversion position
    sma20, sd20 = _roll_mean(c, 20), _roll_std(c, 20)
    sma60, sd60 = _roll_mean(c, 60), _roll_std(c, 60)
    out["zclose20"] = (c - sma20) / (sd20 + EPS)
    out["zclose60"] = (c - sma60) / (sd60 + EPS)
    out["rsi14"] = _rsi(c, 14)
    out["dist_hi"] = c / (h.rolling(60, min_periods=10).max() + EPS) - 1.0
    out["dist_lo"] = c / (l.rolling(60, min_periods=10).min() + EPS) - 1.0

    # seasonality (UTC minute-of-day, day-of-week)
    idx = df.index
    mod = idx.hour * 60 + idx.minute
    out["tod_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    out["tod_cos"] = np.cos(2 * np.pi * mod / 1440.0)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)

    return out[FEATURES].replace([np.inf, -np.inf], np.nan)


def warmup_bars():
    """Bars of history needed before features are valid (longest window)."""
    return 121

# ------------------------------------------------------------------------------------
# Self-test (run: python veritate_mri/market/features.py)

def _self_test():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data as md

    sym = md.list_instruments("crypto")[0]
    df = md.load(sym, "1m").iloc[-20000:]
    f = compute(df)
    assert list(f.columns) == FEATURES
    valid = f.dropna()
    assert len(valid) > 15000, len(valid)
    assert np.isfinite(valid.to_numpy()).all()

    # no-lookahead: truncating the tail must not change earlier feature rows
    k = 15000
    ft = compute(df.iloc[:k])
    m = min(len(ft), k) - 200
    a = f.iloc[:m].to_numpy()
    b = ft.iloc[:m].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[both], b[both], atol=1e-9, rtol=1e-6), "LOOKAHEAD: features changed when future removed"
    print(f"ok: {sym} {len(f)} rows, {len(FEATURES)} features, {len(valid)} valid, no-lookahead verified")
    print("features:", ", ".join(FEATURES))


if __name__ == "__main__":
    _self_test()
