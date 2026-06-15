# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - On-demand crypto backfill for the market platform. When external_data/crypto/
#   <symbol>.csv is missing, fetch the needed 1m bars from Binance and cache them in the
#   schema data.py reads, so a fresh install's market page works with zero manual data.
# - Hosts tried in order: api.binance.com (global) then api.binance.us (US; api.binance.com
#   is HTTP 451 geo-blocked in the US). First host that answers for the symbol is used.
# - Always fetches 1m (the native resolution); coarser resolutions resample in data.py.
#   Falls back to a hosted CSV catalog (market_data_catalog.json) when the API is unreachable.
# - Writes atomically (tmp + os.replace) so concurrent fetches of the same symbol never
#   leave a half-written file.
# veritate_mri/market/fetch.py
# ------------------------------------------------------------------------------------
# Imports:

import csv
import json
import os
import ssl
import urllib.request

import certifi

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "market_data_catalog.json")
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
HOSTS = ("https://api.binance.com/api/v3/klines", "https://api.binance.us/api/v3/klines")
PAGE = 1000                # klines per request (Binance hard cap)
MAX_1M = 15000             # safety cap on one backfill (~10.4 days of 1m, covers the 1-week max window)
TIMEOUT = 20
# Default fetchable majors so a fresh install (empty external_data) still lists instruments.
MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
          "AVAXUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", "XLMUSDT"]

# ------------------------------------------------------------------------------------
# Functions

def fetchable_symbols():
    return list(MAJORS)


def _get(host, symbol, end_ms):
    """One Binance 1m kline page for `symbol`, ending at end_ms (None = most recent)."""
    q = f"{host}?symbol={symbol.upper()}&interval=1m&limit={PAGE}"
    if end_ms is not None:
        q += f"&endTime={int(end_ms)}"
    req = urllib.request.Request(q, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=SSL_CTX) as r:
        return json.loads(r.read())


def _klines_1m(symbol, need, cap=MAX_1M):
    """Page Binance 1m klines backward until `need` traded bars (or history runs out). Returns
    rows [(open_ms, o, h, l, c, v, trades, taker_buy), ...] ascending, [] if no host answered.
    Zero-volume bars are dropped: api.binance.us emits synthetic no-trade fill bars (open==high==
    low==close, volume 0) for ~50-75% of recent 1m bars on majors, which carry no signal."""
    need = min(int(need), cap)
    rows = {}
    end = None
    host = None
    while len(rows) < need:
        batch = None
        if host is None:                       # pick the first host that answers for this symbol
            for h in HOSTS:
                try:
                    batch = _get(h, symbol, end)
                    host = h
                    break
                except Exception:
                    continue
            if host is None:
                break
        else:
            try:
                batch = _get(host, symbol, end)
            except Exception:
                break
        if not batch:
            break
        for k in batch:                        # kline[8]=trade count, kline[9]=taker-buy base volume
            vol = float(k[5])
            if vol == 0.0:                     # drop synthetic no-trade fill bars
                continue
            rows[int(k[0])] = (int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]),
                               vol, int(k[8]), float(k[9]))
        end = int(batch[0][0]) - 1             # step before the earliest bar we got
        if len(batch) < PAGE:
            break
    return [rows[t] for t in sorted(rows)]


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume", "trades", "taker_buy"])
        w.writerows(rows)
    os.replace(tmp, path)


def _hosted(symbol, source, path):
    """Download a hosted CSV for (source, symbol) from market_data_catalog.json, if listed."""
    if not os.path.isfile(CATALOG):
        return False
    try:
        with open(CATALOG) as f:
            cat = json.load(f)
    except Exception:
        return False
    url = (cat.get(source) or {}).get(symbol.upper())
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=SSL_CTX) as r:
            data = r.read()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def ensure(symbol, source, need_1m, path, cap=MAX_1M):
    """Make `path` exist for (source, symbol): fetch ~need_1m 1m bars from Binance, else a
    hosted CSV. Returns True if the file now exists. Crypto only (no public source for stocks).
    `cap` bounds one fetch; the on-demand market path keeps MAX_1M, the bulk CLI lifts it."""
    if os.path.isfile(path):
        return True
    if source != "crypto":
        return False
    rows = _klines_1m(symbol, need_1m, cap)
    if rows:
        _write_csv(path, rows)
        return True
    return _hosted(symbol, source, path)


def main(argv=None):
    """Bulk-fetch crypto 1m bars (with the trades + taker_buy fields) into
    external_data/<source>/<SYMBOL>.csv, for building an order-flow training corpus.
    Existing files are left untouched. Run: python -m market.fetch BTCUSDT ETHUSDT --bars 400000"""
    import argparse
    ap = argparse.ArgumentParser(description="Bulk-fetch crypto 1m bars with taker fields.")
    ap.add_argument("symbols", nargs="*", help="symbols to fetch (default: the majors)")
    ap.add_argument("--source", default="crypto_extra", help="external_data subdir to write into")
    ap.add_argument("--bars", type=int, default=400000, help="1m bars per symbol to pull")
    args = ap.parse_args(argv)
    root = os.path.normpath(os.path.join(HERE, "..", "..", "external_data", args.source))
    syms = args.symbols or MAJORS
    for s in syms:
        path = os.path.join(root, f"{s.upper()}.csv")
        ok = ensure(s, "crypto", args.bars, path, cap=args.bars)
        n = sum(1 for _ in open(path)) - 1 if ok and os.path.isfile(path) else 0
        print(f"{s.upper()}: {'ok' if ok else 'FAILED'} ({n} bars) -> {path}")


if __name__ == "__main__":
    main()
