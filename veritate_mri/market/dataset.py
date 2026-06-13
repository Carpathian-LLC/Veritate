# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Builds a pooled, anonymous, multi-instrument supervised dataset from raw bars:
#   features (features.compute) + forward labels, with a PURGED time split so no
#   training label's window overlaps the validation period (the leakage guard from
#   the validation research).
# - Three labels per row, all strictly forward (known only after t):
#     y_ret : forward log return over the next `horizon` bars   (direction/quantiles)
#     y_vol : forward realized vol = sqrt(sum r^2) next horizon  (the forecastable target)
#     y_dir : -1/0/+1 with a flat band scaled by trailing vol    (triple-barrier-lite)
# - Pooling across instruments with no ticker label => one instrument-agnostic model.
# veritate_mri/market/dataset.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as md
import features as ff

# ------------------------------------------------------------------------------------
# Constants

BASE_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

# ------------------------------------------------------------------------------------
# Labels

def make_labels(df, horizon):
    """Forward return and forward realized vol over the next `horizon` bars."""
    c = df["close"].to_numpy(dtype=np.float64)
    lc = np.log(np.clip(c, 1e-12, None))
    n = len(c)
    idx = np.arange(n)

    fr = np.full(n, np.nan)
    ok = idx + horizon < n
    fr[ok] = lc[idx[ok] + horizon] - lc[idx[ok]]

    r = np.diff(lc, prepend=lc[0])
    csum = np.concatenate([[0.0], np.cumsum(r * r)])     # csum[k] = sum r^2 over [0,k)
    fv = np.full(n, np.nan)
    hi = idx + 1 + horizon
    okv = hi <= n
    fv[okv] = np.sqrt(np.maximum(csum[hi[okv]] - csum[idx[okv] + 1], 0.0))
    return fr, fv

# ------------------------------------------------------------------------------------
# Dataset assembly

def build(symbols, horizon=15, base="1m", flat_z=0.33, max_bars=None, train_stride=1, source="crypto"):
    """Pool features+labels across instruments. Returns a dict of stacked arrays."""
    step_ns = np.int64(BASE_MINUTES[base] * 60 * 1_000_000_000)
    avail = set(md.list_instruments(source))
    cols = ["X", "y_ret", "y_vol", "y_dir", "t", "sym"]
    acc = {k: [] for k in cols}
    used = []
    for si, s in enumerate(symbols):
        if s not in avail:
            continue
        try:
            df = md.load(s, base, source=source)
        except Exception:
            continue
        if max_bars:
            df = df.iloc[-max_bars:]
        if len(df) < ff.warmup_bars() + horizon + 200:
            continue
        f = ff.compute(df)
        fr, fv = make_labels(df, horizon)
        rv = f["rv30"].to_numpy(dtype=np.float64)
        band = flat_z * rv * np.sqrt(horizon)
        ydir = np.where(fr > band, 1, np.where(fr < -band, -1, 0)).astype(np.int8)

        Xv = f.to_numpy(dtype=np.float32)
        m = np.isfinite(Xv).all(axis=1) & np.isfinite(fr) & np.isfinite(fv)
        tns = md.index_ns(f.index)
        rows = np.flatnonzero(m)
        acc["X"].append(Xv[rows])
        acc["y_ret"].append(fr[rows].astype(np.float32))
        acc["y_vol"].append(fv[rows].astype(np.float32))
        acc["y_dir"].append(ydir[rows])
        acc["t"].append(tns[rows])
        acc["sym"].append(np.full(len(rows), si, dtype=np.int16))
        used.append(s)

    out = {k: (np.concatenate(acc[k]) if acc[k] else np.array([])) for k in cols}
    out["features"] = ff.FEATURES
    out["symbols"] = used
    out["horizon"] = horizon
    out["base"] = base
    out["step_ns"] = step_ns
    return out


def purged_split(ds, val_frac=0.2):
    """Time-ordered split with purge+embargo: train labels fully resolve before the
    val features start, and an embargo of `horizon` bars separates them."""
    t = ds["t"]
    h_ns = ds["step_ns"] * (ds["horizon"] + 1)
    cutoff = np.quantile(t, 1.0 - val_frac)
    train = (t + h_ns) <= cutoff                 # label window ends before cutoff (purge)
    val = t >= (cutoff + h_ns)                    # features start after embargo
    return train, val
