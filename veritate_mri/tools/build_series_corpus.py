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

# ------------------------------------------------------------------------------------
# Functions

def _clean(o, h, l, c, v, adj):
    mask = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(v) & np.isfinite(adj)
    mask &= (c > 0) & (adj > 0)
    return o[mask], h[mask], l[mask], c[mask], v[mask], adj[mask]


def load_stock_csv(path):
    df = pd.read_csv(path, usecols=STOCK_COLS)
    g = lambda k: df[k].to_numpy(dtype=np.float64)
    return _clean(g("open"), g("high"), g("low"), g("close"), g("volume"), g("adjclose"))


def load_crypto_csv(path):
    df = pd.read_csv(path, usecols=CRYPTO_COLS)
    g = lambda k: df[k].to_numpy(dtype=np.float64)
    c = g("close")
    return _clean(g("open"), g("high"), g("low"), c, g("volume"), c.copy())


LOADERS = {"stocks": load_stock_csv, "crypto": load_crypto_csv}


def build(source, val_ratio):
    loader = LOADERS[source]
    files = sorted(glob.glob(os.path.join(EXTERNAL_DIR, source, "*.csv")))
    os.makedirs(CORPUS_DIR, exist_ok=True)
    train_path = os.path.join(CORPUS_DIR, f"{source}_train.bin")
    val_path = os.path.join(CORPUS_DIR, f"{source}_val.bin")
    sep = SEQ_SEP.encode("ascii")
    n_inst = n_bars = n_skip = 0
    with open(train_path, "wb") as tf, open(val_path, "wb") as vf:
        for path in files:
            try:
                o, h, l, c, v, adj = loader(path)
            except Exception:
                n_skip += 1
                continue
            if len(c) < MIN_BARS + FEAT_WINDOW + 2:
                n_skip += 1
                continue
            rz, gr, vr = compute_features(o, h, l, c, v, adj)
            if len(rz) < MIN_BARS:
                n_skip += 1
                continue
            seq = encode_sequence(rz, gr, vr).encode("ascii")
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
    ap.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    args = ap.parse_args(argv)
    s = build(args.source, args.val_ratio)
    print(f"{args.source}: {s['instruments']} instruments, {s['bars']} bars, {s['skipped']} skipped")
    print(f"  train: {s['train_path']}  ({s['train_mb']:.2f} MB)")
    print(f"  val:   {s['val_path']}  ({s['val_mb']:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
