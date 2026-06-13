# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Canonical data layer for the market platform. Loads raw 1m OHLCV CSVs from
#   external_data/ and exposes clean, timezone-aware, multi-horizon bars.
# - Binance switched its kline timestamp unit mid-history (ms in 2017 -> us in 2025+),
#   so a single file mixes units. normalize_time() detects unit per-row, otherwise
#   resampling onto minute boundaries silently corrupts. Tested by _self_test().
# - Resampling is right-edge bar aggregation (open=first, high=max, low=min,
#   close=last, volume=sum) with no lookahead.
# veritate_mri/market/data.py
# ------------------------------------------------------------------------------------
# Imports:

import glob
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
EXTERNAL_DIR = os.path.join(ROOT, "external_data")

OHLCV = ["open", "high", "low", "close", "volume"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

# pandas resample rules per named horizon
HORIZONS = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}

_US = 1_000_000_000_000_000   # >= 1e15 -> microseconds
_MS = 1_000_000_000_000       # >= 1e12 -> milliseconds
_S = 1_000_000_000            # >= 1e9  -> seconds

# ------------------------------------------------------------------------------------
# Functions

def index_ns(index):
    """Int64 nanoseconds-since-epoch for a DatetimeIndex, regardless of its native
    resolution (pandas indexes may be us/ms/ns; downstream epoch math assumes ns)."""
    return index.values.astype("datetime64[ns]").view("int64")


def normalize_time(t):
    """Mixed ms/us/s epoch integers -> UTC DatetimeIndex (unit detected per-row)."""
    t = np.asarray(t, dtype=np.int64)
    ns = np.where(t >= _US, t * 1_000,
         np.where(t >= _MS, t * 1_000_000,
         np.where(t >= _S, t * 1_000_000_000, t)))
    return pd.to_datetime(ns, utc=True)


def load_1m(path, cols=OHLCV):
    """Load one raw OHLCV CSV -> clean frame indexed by UTC time, dedup+sorted.
    Schema-flexible: crypto uses a numeric epoch 'time' column, stocks a 'date' string."""
    head = list(pd.read_csv(path, nrows=0).columns)
    tcol = "time" if "time" in head else ("date" if "date" in head else head[0])
    use = [tcol] + [c for c in (cols or OHLCV) if c in head and c != tcol]
    df = pd.read_csv(path, usecols=use)
    tv = df[tcol].to_numpy()
    df.index = normalize_time(tv) if np.issubdtype(tv.dtype, np.number) else pd.to_datetime(df[tcol], utc=True)
    df = df.drop(columns=[tcol])
    df = df[np.isfinite(df["close"]) & (df["close"] > 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def resample(df, horizon):
    """Aggregate a 1m OHLCV frame to a coarser horizon ('5m','1h',...). No lookahead."""
    rule = HORIZONS.get(horizon, horizon)
    agg = {k: AGG[k] for k in df.columns if k in AGG}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out[np.isfinite(out["close"]) & (out["close"] > 0)]


def list_instruments(source="crypto"):
    """Symbol names available as raw 1m CSVs under external_data/<source>/."""
    files = sorted(glob.glob(os.path.join(EXTERNAL_DIR, source, "*.csv")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def path_for(symbol, source="crypto"):
    return os.path.join(EXTERNAL_DIR, source, f"{symbol}.csv")


def load(symbol, horizon="1m", source="crypto", cols=OHLCV):
    """Load one instrument at a given horizon (the platform's main entry point)."""
    df = load_1m(path_for(symbol, source), cols=cols)
    return df if horizon in ("1m", "1min") else resample(df, horizon)


def load_tail(symbol, n_bars, base="1m", source="crypto"):
    """Read only the last ~n_bars of a (possibly huge) 1m CSV and resample to `base`.
    Reads from the end of the file so a web request never loads the whole history."""
    path = path_for(symbol, source)
    mins = HORIZONS_MIN.get(base, 1)
    need_1m = int((n_bars + 160) * mins)
    sz = os.path.getsize(path)
    if sz < 50_000_000:                            # small file (e.g. daily stocks): read fully
        df = load_1m(path)
        out = df if base in ("1m", "1min") else resample(df, base)
        return out.iloc[-(n_bars + 150):]
    approx = min(sz, need_1m * 72 + 8192)
    with open(path, "rb") as f:
        f.seek(max(0, sz - approx))
        chunk = f.read().decode("ascii", "replace")
    rows = []
    for ln in chunk.splitlines()[1:]:          # drop the partial first line
        c = ln.split(",")
        if len(c) < 6:
            continue
        try:
            rows.append((int(float(c[0])), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])))
        except ValueError:
            continue
    if not rows:
        return None
    a = np.array(rows, dtype=np.float64)
    df = pd.DataFrame(a[:, 1:6], columns=OHLCV)
    df.index = normalize_time(a[:, 0].astype(np.int64))
    df = df[np.isfinite(df["close"]) & (df["close"] > 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = df if base in ("1m", "1min") else resample(df, base)
    return out.iloc[-(n_bars + 150):]


HORIZONS_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


# ------------------------------------------------------------------------------------
# Self-test (run: python veritate_mri/market/data.py)

def _self_test():
    t = np.array([1502942400000, 1502942460000, 1780271940000000], dtype=np.int64)
    dt = normalize_time(t)
    assert str(dt[0].date()) == "2017-08-17", dt[0]
    assert dt[2].year == 2026, dt[2]
    syms = list_instruments("crypto")
    assert syms, "no crypto instruments found"
    df = load(syms[0], "1m")
    assert list(df.columns) == OHLCV
    assert df.index.is_monotonic_increasing
    h = resample(df, "1h")
    assert len(h) < len(df)
    # resampled close at a boundary equals the last 1m close inside that bar
    b0 = h.index[0]
    b1 = b0 + pd.Timedelta("1h")
    inside = df[(df.index >= b0) & (df.index < b1)]
    assert h["close"].iloc[0] == inside["close"].iloc[-1]
    assert np.isclose(h["volume"].iloc[0], inside["volume"].sum(), rtol=1e-6)
    print(f"ok: {syms[0]} 1m={len(df)} bars {df.index[0]}..{df.index[-1]}  1h={len(h)} bars")
    print(f"instruments: {len(syms)}")


if __name__ == "__main__":
    _self_test()
