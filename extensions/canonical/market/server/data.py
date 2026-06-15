# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Canonical data layer for the market platform. Loads raw 1m OHLCV CSVs from the
#   extension data dir and exposes clean, timezone-aware, multi-horizon bars.
# - Binance switched its kline timestamp unit mid-history (ms in 2017 -> us in 2025+),
#   so a single file mixes units. normalize_time() detects unit per-row, otherwise
#   resampling onto minute boundaries silently corrupts. Tested by _self_test().
# - Resampling is right-edge bar aggregation (open=first, high=max, low=min,
#   close=last, volume=sum) with no lookahead.
# extensions/canonical/market/server/data.py
# ------------------------------------------------------------------------------------
# Imports:

import glob
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
# Per-extension data cache (gitignored). Under installed/ so it survives canonical-
# vs-installed and is never copied on install.
DATA_DIR = os.path.join(ROOT, "extensions", "installed", "market", "data")
# Downloadable add-on datasets live under here (disposable cache). Active serving
# sources stay at data/<source>; everything else falls back here, resolved by
# source_dir().
EXTENSION_DIR = os.path.join(DATA_DIR, "extension_data")
FUNDING_DIR = os.path.join(DATA_DIR, "funding")
SENTIMENT_PATH = os.path.join(DATA_DIR, "sentiment", "fng.csv")

# sources whose bars carry crypto context channels (perp funding + crypto fear-greed)
CRYPTO_SOURCES = {"crypto", "crypto_extra", "crypto_of", "crypto_1s"}

OHLCV = ["open", "high", "low", "close", "volume"]
# taker-buy volume + trade count: present in Binance-fetched CSVs, absent in legacy files.
OHLCV_EXT = OHLCV + ["trades", "taker_buy"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
       "trades": "sum", "taker_buy": "sum"}

# pandas resample rules per named horizon
HORIZONS = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}

# minutes per named horizon (single owner; dataset.py imports this)
HORIZONS_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

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


def load_1m(path, cols=OHLCV_EXT):
    """Load one raw OHLCV CSV -> clean frame indexed by UTC time, dedup+sorted.
    Schema-flexible: crypto uses a numeric epoch 'time' column, stocks a 'date' string.
    Reads the optional trades/taker_buy columns when the file carries them."""
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


def _fetch():
    """The sibling fetch module, importable whether data loads as a package or standalone."""
    try:
        import fetch
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, HERE)
        import fetch
    return fetch


def source_dir(source):
    """Directory holding `source`'s CSVs. Active sources live at data/<source>;
    downloadable extensions (stocks, forex, the broader crypto sets, ...) live under
    data/extension_data/<source>. Prefer the root location, fall back to the
    extension cache, default to root for a fresh source."""
    root = os.path.join(DATA_DIR, source)
    if os.path.isdir(root):
        return root
    ext = os.path.join(EXTENSION_DIR, source)
    return ext if os.path.isdir(ext) else root


def list_instruments(source="crypto"):
    """Symbols for `source`: local raw CSVs, plus the fetchable crypto majors so a fresh
    install (empty data dir) still has instruments to pick (they backfill on first run).
    Crypto is ordered high-volume majors first (in their canonical order), then the rest
    alphabetically, so the dropdown leads with what most users want."""
    files = sorted(glob.glob(os.path.join(source_dir(source), "*.csv")))
    local = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if source == "crypto":
        majors = _fetch().fetchable_symbols()
        rest = sorted(set(local) - set(majors))
        seen = set()
        return [s for s in majors + rest if not (s in seen or seen.add(s))]
    return sorted(set(local))


def path_for(symbol, source="crypto"):
    return os.path.join(source_dir(source), f"{symbol}.csv")


def _context_series(path, valcol):
    """A (time,valcol) context CSV -> ascending value series indexed by UTC time."""
    df = pd.read_csv(path, usecols=["time", valcol])
    df.index = normalize_time(df["time"].to_numpy())
    return df[valcol].sort_index()


def join_context(df, symbol):
    """Forward-fill perp funding (per-symbol) + fear-greed sentiment (global) onto df's
    bar index as 'funding' / 'fng' columns. No lookahead: each bar carries the last value
    at or before its timestamp. Missing source -> column absent (codec emits the absent byte)."""
    fpath = os.path.join(FUNDING_DIR, f"{symbol}.csv")
    if os.path.isfile(fpath):
        df = df.assign(funding=_context_series(fpath, "funding").reindex(df.index, method="ffill").to_numpy())
    if os.path.isfile(SENTIMENT_PATH):
        df = df.assign(fng=_context_series(SENTIMENT_PATH, "value").reindex(df.index, method="ffill").to_numpy())
    return df


def load(symbol, horizon="1m", source="crypto", cols=OHLCV):
    """Load one instrument at a given horizon (the platform's main entry point)."""
    df = load_1m(path_for(symbol, source), cols=cols)
    df = df if horizon in ("1m", "1min") else resample(df, horizon)
    return join_context(df, symbol) if source in CRYPTO_SOURCES else df


def load_tail(symbol, n_bars, base="1m", source="crypto"):
    """Read only the last ~n_bars of a (possibly huge) 1m CSV and resample to `base`.
    Reads from the end of the file so a web request never loads the whole history."""
    path = path_for(symbol, source)
    mins = HORIZONS_MIN.get(base, 1)
    need_1m = int((n_bars + 160) * mins)
    if not os.path.isfile(path):                    # fresh install / new symbol -> backfill on demand
        if not _fetch().ensure(symbol, source, need_1m, path):
            return None                             # could not fetch (offline, or non-crypto)
    sz = os.path.getsize(path)
    if sz < 50_000_000:                            # small file (e.g. daily stocks): read fully
        df = load_1m(path)
        out = df if base in ("1m", "1min") else resample(df, base)
        out = out.iloc[-(n_bars + 150):]
        return join_context(out, symbol) if source in CRYPTO_SOURCES else out
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
            vals = [int(float(c[0])), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
            if len(c) >= 8:                    # crypto schema: ...,volume,trades,taker_buy
                vals += [float(c[6]), float(c[7])]
            rows.append(tuple(vals))
        except ValueError:
            continue
    if not rows:
        return None
    ncol = min(len(r) for r in rows)
    a = np.array([r[:ncol] for r in rows], dtype=np.float64)
    df = pd.DataFrame(a[:, 1:ncol], columns=OHLCV_EXT if ncol >= 8 else OHLCV)
    df.index = normalize_time(a[:, 0].astype(np.int64))
    df = df[np.isfinite(df["close"]) & (df["close"] > 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = df if base in ("1m", "1min") else resample(df, base)
    out = out.iloc[-(n_bars + 150):]
    return join_context(out, symbol) if source in CRYPTO_SOURCES else out


# ------------------------------------------------------------------------------------
# Self-test (run: python extensions/canonical/market/server/data.py)

def _self_test():
    t = np.array([1502942400000, 1502942460000, 1780271940000000], dtype=np.int64)
    dt = normalize_time(t)
    assert str(dt[0].date()) == "2017-08-17", dt[0]
    assert dt[2].year == 2026, dt[2]
    syms = list_instruments("crypto")
    assert syms, "no crypto instruments found"
    on_disk = [s for s in syms if os.path.isfile(path_for(s, "crypto"))]
    assert on_disk, "no crypto instruments on disk"
    df = load(on_disk[0], "1m")
    assert list(df.columns)[:len(OHLCV)] == OHLCV     # OHLCV prefix; crypto appends funding/fng context
    assert df.index.is_monotonic_increasing
    h = resample(df, "1h")
    assert len(h) < len(df)
    # resampled close at a boundary equals the last 1m close inside that bar
    b0 = h.index[0]
    b1 = b0 + pd.Timedelta("1h")
    inside = df[(df.index >= b0) & (df.index < b1)]
    assert h["close"].iloc[0] == inside["close"].iloc[-1]
    assert np.isclose(h["volume"].iloc[0], inside["volume"].sum(), rtol=1e-6)
    print(f"ok: {on_disk[0]} 1m={len(df)} bars {df.index[0]}..{df.index[-1]}  1h={len(h)} bars")
    print(f"instruments: {len(syms)}")


if __name__ == "__main__":
    _self_test()
