# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Live bar fetcher for the real-time forecast feed. Pulls recent CLOSED klines from
#   the Binance.US public REST endpoint (no API key, weight 1). api.binance.com is
#   geo-blocked (HTTP 451) in the US; api.binance.us is the legal US source.
# - The last kline Binance returns is the still-forming current bar; predict() drops it
#   and forecasts from the last CLOSED bar (the closed-bar rule — no acting on a partial
#   candle). Same features.compute as training, so no train/serve skew.
# veritate_mri/market/live.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import ssl
import urllib.request

import certifi
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Constants

REST = "https://api.binance.us/api/v3/klines"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
TIMEOUT = 15

# ------------------------------------------------------------------------------------
# Functions

def fetch(symbol, base="1m", limit=400):
    """Recent klines for `symbol` at timeframe `base`. Returns (df, last_open_ms).
    df rows are bars in ascending time; the LAST row is the in-progress bar."""
    interval = INTERVAL.get(base, "1m")
    url = f"{REST}?symbol={symbol.upper()}&interval={interval}&limit={int(limit)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=SSL_CTX) as r:
        raw = json.loads(r.read())
    if not raw:
        return None, None
    a = np.array([[float(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
                  for k in raw], dtype=np.float64)
    df = pd.DataFrame(a[:, 1:6], columns=["open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(a[:, 0].astype(np.int64), unit="ms", utc=True)
    df = df[np.isfinite(df["close"]) & (df["close"] > 0)]
    return df, int(raw[-1][0])


def predict(mm, symbol):
    """Live forecast from the last CLOSED bar. Returns df_closed, forming_bar, pred."""
    df, _ = fetch(symbol, base=mm.base, limit=400)
    if df is None or len(df) < 150:
        return None, None, None
    forming = df.iloc[-1]
    closed = df.iloc[:-1]                       # drop the still-forming current bar
    pred = mm.predict_latest(closed)
    forming_bar = {"t": int(df.index[-1].value // 1_000_000_000),
                   "o": float(forming["open"]), "h": float(forming["high"]),
                   "l": float(forming["low"]), "c": float(forming["close"])}
    return closed, forming_bar, pred
