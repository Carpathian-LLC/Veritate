# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - builds a byte-level price-series corpus from raw OHLCV CSVs in external_data/.
# - one corpus per asset class (stocks, crypto). instruments are anonymous: no
#   ticker label, so the model learns one instrument-agnostic tape dynamic.
# - per-instrument time split: the oldest (1-val_ratio) of each instrument's bars
#   are train, the newest val_ratio are val (forecast the held-out future).
# - encoding + feature math live in series_codec.py (shared with the predict page).
# - the bar timestamp (crypto epoch-ms 'time', stock 'date' index) is threaded as
#   epoch-ns into compute_features so the session channel can be derived. sources
#   without a timestamp encode the session fallback byte.
# - streams to disk per instrument so a GB-scale (1m) corpus never sits in memory.
# - run:
#     python veritate_mri/tools/build_series_corpus.py --source stocks
#     python veritate_mri/tools/build_series_corpus.py --source crypto
# veritate_mri/tools/build_series_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    from series_codec import (
        BAR_STRIDE, FEAT_WINDOW, SEQ_SEP, compute_features, encode_sequence,
    )
except ModuleNotFoundError:
    from tools.series_codec import (
        BAR_STRIDE, FEAT_WINDOW, SEQ_SEP, compute_features, encode_sequence,
    )

_MRI = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _MRI not in sys.path:
    sys.path.insert(0, _MRI)
from market import data as md

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
EXTERNAL_DIR = os.path.join(ROOT, "external_data")
CORPUS_DIR = os.path.join(ROOT, "trainers", "corpus")

DEFAULT_VAL_RATIO = 0.1
MIN_BARS = 250
STOCK_COLS = ["open", "high", "low", "close", "adjclose", "volume"]
CRYPTO_COLS = ["open", "high", "low", "close", "volume"]
STOCK_TIME_COL = "date"
CRYPTO_TIME_COL = "time"

# ------------------------------------------------------------------------------------
# Functions

def _clean(o, h, l, c, v, adj, ts_ns, tb=None, ntr=None, fund=None, fng=None):
    mask = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(v) & np.isfinite(adj)
    mask &= (c > 0) & (adj > 0)
    keep = lambda x: None if x is None else x[mask]
    return (o[mask], h[mask], l[mask], c[mask], v[mask], adj[mask], ts_ns[mask],
            keep(tb), keep(ntr), keep(fund), keep(fng))


def load_stock_csv(path, horizon="1m", context=False):
    df = pd.read_csv(path, usecols=STOCK_COLS + [STOCK_TIME_COL])
    g = lambda k: df[k].to_numpy(dtype=np.float64)
    ts_ns = pd.to_datetime(df[STOCK_TIME_COL], utc=True).values.astype("datetime64[ns]").view("int64")
    return _clean(g("open"), g("high"), g("low"), g("close"), g("volume"), g("adjclose"), ts_ns)


def load_crypto_csv(path, horizon="1m", context=False):
    sym = os.path.splitext(os.path.basename(path))[0]
    df = md.load_1m(path, cols=md.OHLCV_EXT)
    if horizon not in ("1m", "1min"):
        df = md.resample(df, horizon)
    if context:
        df = md.join_context(df, sym)
    g = lambda k: df[k].to_numpy(dtype=np.float64) if k in df.columns else None
    c = g("close")
    ts_ns = md.index_ns(df.index)
    return _clean(g("open"), g("high"), g("low"), c, g("volume"), c.copy(), ts_ns,
                  g("taker_buy"), g("trades"), g("funding"), g("fng"))


LOADERS = {
    "stocks": load_stock_csv, "stocks_1m": load_stock_csv,
    "indices": load_stock_csv, "futures": load_stock_csv,
    "crypto": load_crypto_csv, "crypto_1s": load_crypto_csv,
    "crypto_extra": load_crypto_csv, "forex": load_crypto_csv,
    "crypto_of": load_crypto_csv,
}


def build(source, val_ratio, no_order_flow=False, horizon="1m"):
    loader = LOADERS[source]
    files = sorted(glob.glob(os.path.join(md.source_dir(source), "*.csv")))
    os.makedirs(CORPUS_DIR, exist_ok=True)
    stem = source if horizon in ("1m", "1min") else f"{source}_{horizon}"
    if no_order_flow:
        stem = f"{stem}_noflow"
    train_path = os.path.join(CORPUS_DIR, f"{stem}_train.bin")
    val_path = os.path.join(CORPUS_DIR, f"{stem}_val.bin")
    sep = SEQ_SEP.encode("ascii")
    ctx = source in md.CRYPTO_SOURCES
    n_inst = n_bars = n_skip = 0
    with open(train_path, "wb") as tf, open(val_path, "wb") as vf:
        for path in files:
            try:
                o, h, l, c, v, adj, ts_ns, tb, ntr, fund, fng = loader(path, horizon, ctx)
            except Exception:
                n_skip += 1
                continue
            if no_order_flow:                          # A/B before-arm: drop the order-flow channels
                tb = ntr = None
            if len(c) < MIN_BARS + FEAT_WINDOW + 2:
                n_skip += 1
                continue
            rz, gr, vr, yr, ss, bp, tr, fr, st = compute_features(o, h, l, c, v, adj, ts_ns, tb, ntr, fund, fng)
            if len(rz) < MIN_BARS:
                n_skip += 1
                continue
            seq = encode_sequence(rz, gr, vr, yr, ss, bp, tr, fr, st).encode("ascii")
            cut = int(len(rz) * (1.0 - val_ratio)) * BAR_STRIDE
            tf.write(seq[:cut]); tf.write(sep)
            vf.write(seq[cut:]); vf.write(sep)
            n_inst += 1
            n_bars += len(rz)
    return {
        "instruments": n_inst, "skipped": n_skip, "bars": n_bars,
        "train_path": train_path, "val_path": val_path,
        "train_mb": os.path.getsize(train_path) / 1e6, "val_mb": os.path.getsize(val_path) / 1e6,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a byte-level price-series corpus.")
    ap.add_argument("--source", required=True, choices=list(LOADERS))
    ap.add_argument("--horizon", default="1m", choices=list(md.HORIZONS),
                    help="bar horizon: resample 1m source bars before encoding (crypto only); writes <source>_<horizon>_*.bin")
    ap.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--no-order-flow", action="store_true",
                    help="encode buy-pressure/trade-activity as the absent-fallback (A/B before-arm); writes <source>_noflow_*.bin")
    args = ap.parse_args(argv)
    s = build(args.source, args.val_ratio, args.no_order_flow, args.horizon)
    print(f"{args.source} @ {args.horizon}: {s['instruments']} instruments, {s['bars']} bars, {s['skipped']} skipped")
    print(f"  train: {s['train_path']}  ({s['train_mb']:.2f} MB)")
    print(f"  val:   {s['val_path']}  ({s['val_mb']:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
