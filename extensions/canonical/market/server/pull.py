# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - scratch puller for the price-prediction experiment. raw OHLCV only, no corpus build.
# - stocks: yahoo v8 chart json (no key). crypto: cryptodatadownload daily csv (no key).
# - re-runnable + idempotent: skips an instrument whose csv already exists.
# - usage: python3 pull.py [stocks|crypto|all] [--limit N]
# extensions/canonical/market/server/pull.py
# ------------------------------------------------------------------------------------
# Imports:

import csv
import json
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone

import certifi

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")), "extensions", "installed", "market", "data")
OUT_STOCKS = os.path.join(HERE, "stocks")
OUT_CRYPTO = os.path.join(HERE, "crypto")

SP500_LIST_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=-2208988800&period2=9999999999&interval=1d&events=div%2Csplit"
CRYPTO_CSV_URL = "https://www.cryptodatadownload.com/cdd/Binance_{pair}_d.csv"

CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT",
    "DOTUSDT", "MATICUSDT", "LTCUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT",
    "XLMUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "NEARUSDT", "ALGOUSDT",
    "VETUSDT", "ICPUSDT", "HBARUSDT", "EOSUSDT", "AAVEUSDT", "MKRUSDT", "EGLDUSDT",
    "SANDUSDT", "MANAUSDT", "AXSUSDT", "THETAUSDT", "FTMUSDT", "XTZUSDT", "GRTUSDT",
    "CHZUSDT", "ENJUSDT", "ZECUSDT", "DASHUSDT", "COMPUSDT", "SNXUSDT", "CRVUSDT",
    "1INCHUSDT", "SUSHIUSDT", "YFIUSDT", "UNIUSDT", "RUNEUSDT", "KSMUSDT", "CAKEUSDT",
    "ZILUSDT", "BATUSDT", "IOTAUSDT", "NEOUSDT", "QTUMUSDT", "WAVESUSDT", "KAVAUSDT",
    "RVNUSDT", "ONEUSDT", "ANKRUSDT", "STORJUSDT",
]

SSL_CTX = ssl.create_default_context(cafile=certifi.where())
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
TIMEOUT = 30
DELAY = 0.4
RETRIES = 4
STOCK_HEADER = ["date", "open", "high", "low", "close", "adjclose", "volume"]

# ------------------------------------------------------------------------------------
# Functions

def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=SSL_CTX) as resp:
                return resp.read()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last


def get_sp500_symbols():
    raw = http_get(SP500_LIST_URL).decode("utf-8")
    rows = list(csv.DictReader(raw.splitlines()))
    return [r["Symbol"].strip().replace(".", "-") for r in rows if r.get("Symbol")]


def pull_stock(sym):
    path = os.path.join(OUT_STOCKS, sym + ".csv")
    if os.path.exists(path):
        return "skip", 0
    doc = json.loads(http_get(YAHOO_CHART_URL.format(sym=sym)))
    res = doc["chart"]["result"]
    if not res:
        return "empty", 0
    res = res[0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(ts))
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(STOCK_HEADER)
        for i, t in enumerate(ts):
            c = q["close"][i]
            if c is None:
                continue
            d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
            w.writerow([d, q["open"][i], q["high"][i], q["low"][i], c, adj[i], q["volume"][i]])
            n += 1
    return "ok", n


def pull_crypto(pair):
    path = os.path.join(OUT_CRYPTO, pair + ".csv")
    if os.path.exists(path):
        return "skip", 0
    try:
        raw = http_get(CRYPTO_CSV_URL.format(pair=pair))
    except Exception:
        return "miss", 0
    lines = raw.decode("utf-8", "replace").splitlines()
    body = [ln for ln in lines if "," in ln and not ln.startswith("http")]
    if len(body) < 2:
        return "miss", 0
    with open(path, "w") as fh:
        fh.write("\n".join(body) + "\n")
    return "ok", len(body) - 1


def run(target, limit):
    stats = {"stocks": {}, "crypto": {}}
    if target in ("stocks", "all"):
        syms = get_sp500_symbols()
        if limit:
            syms = syms[:limit]
        for i, s in enumerate(syms):
            try:
                status, n = pull_stock(s)
            except Exception as exc:
                status, n = "err:" + type(exc).__name__, 0
            stats["stocks"][s] = {"status": status, "rows": n}
            print(f"[stock {i+1}/{len(syms)}] {s} {status} {n}", flush=True)
            time.sleep(DELAY)
    if target in ("crypto", "all"):
        pairs = CRYPTO_PAIRS[:limit] if limit else CRYPTO_PAIRS
        for i, p in enumerate(pairs):
            status, n = pull_crypto(p)
            stats["crypto"][p] = {"status": status, "rows": n}
            print(f"[crypto {i+1}/{len(pairs)}] {p} {status} {n}", flush=True)
            time.sleep(DELAY)
    with open(os.path.join(HERE, "_manifest.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    ok_s = sum(1 for v in stats["stocks"].values() if v["status"] in ("ok", "skip"))
    ok_c = sum(1 for v in stats["crypto"].values() if v["status"] in ("ok", "skip"))
    print(f"DONE stocks_ok={ok_s} crypto_ok={ok_c}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tgt = args[0] if args else "all"
    lim = 0
    if "--limit" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1])
    run(tgt, lim)
