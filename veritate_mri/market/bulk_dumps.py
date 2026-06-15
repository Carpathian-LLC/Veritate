# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Bulk historical crypto downloader for building a training corpus that carries the
#   order-flow channels. Pulls Binance's public monthly kline dumps from
#   data.binance.vision (no API key, no rate limit, full history) and writes them in the
#   8-column schema data.py / build_series_corpus read: time,open,high,low,close,volume,
#   trades,taker_buy. The REST fetcher (fetch.py) is for the live page's small windows;
#   this is the GB-scale path for training. Writes into its own external_data/crypto_of/
#   dir (NOT crypto_extra, which holds OHLCV-only pairs) so the order-flow corpus is clean.
# - Monthly dump CSVs are the 12-column Binance kline format (open_time, OHLC, volume,
#   close_time, quote_volume, count, taker_buy_base, taker_buy_quote, ignore); some 2025+
#   files carry a header row, detected and skipped. open_time is passed through raw
#   (ms/us) and normalized downstream by market.data.normalize_time.
# - run: python -m market.bulk_dumps BTCUSDT ETHUSDT --start 2019-01 --source crypto_of
# veritate_mri/market/bulk_dumps.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import datetime as dt
import io
import os
import ssl
import time
import urllib.error
import urllib.request
import zipfile

import certifi

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
EXTERNAL_DIR = os.path.join(ROOT, "external_data")
BASE = "https://data.binance.vision/data/spot/monthly/klines"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
TIMEOUT = 120
OUT_HEADER = "time,open,high,low,close,volume,trades,taker_buy\n"
# 12-col kline dump -> our 8 columns: open_time, OHLC, volume, count(trades), taker_buy_base.
SRC_COLS = (0, 1, 2, 3, 4, 5, 8, 9)
DEFAULT_START = "2017-08"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
                   "AVAXUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", "XLMUSDT",
                   "TRXUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
                   "SUIUSDT", "AAVEUSDT", "ALGOUSDT", "ICPUSDT", "HBARUSDT", "NEARUSDT", "DOTUSDT",
                   "SANDUSDT", "GALAUSDT", "FTMUSDT", "AXSUSDT", "EGLDUSDT", "THETAUSDT", "GRTUSDT",
                   "CRVUSDT", "MKRUSDT", "RUNEUSDT", "ENJUSDT", "CHZUSDT"]

# ------------------------------------------------------------------------------------
# Functions

def _months(start_ym, end_ym):
    y, m = (int(x) for x in start_ym.split("-"))
    ey, em = (int(x) for x in end_ym.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m = 1; y += 1


def _fetch_month(symbol, ym, retries=5):
    """One monthly zip -> list of mapped 8-col rows (strings). [] when the month is absent
    (404). Transient network errors retry with backoff; a persistent one raises."""
    url = f"{BASE}/{symbol}/1m/{symbol}-1m-{ym}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    blob = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=SSL_CTX) as r:
                blob = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == retries - 1:
                raise
        time.sleep(2 * (attempt + 1))
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode("ascii", "replace").splitlines()
    rows = []
    for ln in raw:
        c = ln.split(",")
        if len(c) < 10 or not c[0][:1].isdigit():        # skip header / short lines
            continue
        rows.append(",".join(c[i] for i in SRC_COLS))
    return rows


def fetch_symbol(symbol, start, end, source):
    """Download every monthly dump for `symbol` in [start, end] and write one deduped,
    time-sorted 8-col CSV. Skips work if the output already exists. Returns bar count."""
    symbol = symbol.upper()
    out = os.path.join(EXTERNAL_DIR, source, f"{symbol}.csv")
    if os.path.isfile(out):
        return -1
    by_time = {}
    for ym in _months(start, end):
        for row in _fetch_month(symbol, ym):
            by_time[int(row.split(",", 1)[0])] = row
    if not by_time:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.tmp"
    with open(tmp, "w") as f:
        f.write(OUT_HEADER)
        for t in sorted(by_time):
            f.write(by_time[t]); f.write("\n")
    os.replace(tmp, out)
    return len(by_time)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bulk-download Binance monthly 1m klines (with order-flow fields).")
    ap.add_argument("symbols", nargs="*", help="symbols (default: a 40-pair high-volume set)")
    ap.add_argument("--start", default=DEFAULT_START, help="first month YYYY-MM")
    ap.add_argument("--end", default=None, help="last month YYYY-MM (default: last complete month)")
    ap.add_argument("--source", default="crypto_of", help="external_data subdir to write into")
    args = ap.parse_args(argv)
    end = args.end or (dt.date.today().replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")
    syms = [s.upper() for s in (args.symbols or DEFAULT_SYMBOLS)]
    print(f"bulk dumps: {len(syms)} symbols, {args.start}..{end} -> external_data/{args.source}/", flush=True)
    total = 0; failed = []
    for i, s in enumerate(syms, 1):
        try:
            n = fetch_symbol(s, args.start, end, args.source)
        except Exception as e:                          # one bad symbol never aborts the run
            failed.append(s)
            print(f"[{i}/{len(syms)}] {s}: FAILED ({type(e).__name__}: {e})", flush=True)
            continue
        if n == -1:
            print(f"[{i}/{len(syms)}] {s}: exists, skipped", flush=True)
        else:
            total += max(0, n)
            print(f"[{i}/{len(syms)}] {s}: {n} bars", flush=True)
    print(f"done: {total} bars across {len(syms)} symbols; failed: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
