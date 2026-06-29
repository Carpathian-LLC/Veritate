# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - scratch puller for daily index levels and continuous-futures bars from the
#   yahoo v8 chart json (free, no key). same endpoint pull.py uses for stocks.
# - indices: cash index levels (^GSPC, ^NDX, ^DJI, ...). futures: yahoo continuous
#   front-month contracts (ES=F, CL=F, GC=F, ...), the only free continuous series.
#   CME tick/minute data is paid; this is the honest free substitute.
# - both write the canonical stock schema date,open,high,low,close,adjclose,volume.
#   indices/futures carry no split-adjustment, so adjclose mirrors close.
# - one csv per symbol, full available history, time-ascending. re-runnable: skips
#   a symbol whose csv already exists.
# - usage: python3 pull_yahoo.py indices
#          python3 pull_yahoo.py futures
# extensions/canonical/market/server/pull_yahoo.py
# ------------------------------------------------------------------------------------
# Imports:

import csv
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import certifi

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")), "extensions", "installed", "market", "data")

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=-2208988800&period2=9999999999&interval=1d"

INDICES = [
    "^GSPC", "^NDX", "^IXIC", "^DJI", "^RUT", "^VIX", "^FTSE", "^GDAXI",
    "^FCHI", "^STOXX50E", "^N225", "^HSI", "^GSPTSE", "^AXJO", "^BSESN",
    "^KS11", "^TWII", "^BVSP", "^MXX", "^TNX",
]

FUTURES = [
    "ES=F", "NQ=F", "YM=F", "RTY=F", "CL=F", "GC=F", "SI=F", "HG=F",
    "NG=F", "ZB=F", "ZN=F", "ZC=F", "ZS=F", "ZW=F", "KC=F", "CT=F",
    "SB=F", "CC=F", "PL=F", "PA=F",
]

TARGETS = {"indices": (INDICES, "indices"), "futures": (FUTURES, "futures")}

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


def pull_symbol(sym, out_dir):
    path = os.path.join(out_dir, sym.replace("^", "") + ".csv")
    if os.path.exists(path):
        return "skip", 0
    url = CHART_URL.format(sym=urllib.parse.quote(sym))
    doc = json.loads(http_get(url))
    res = doc["chart"]["result"]
    if not res:
        return "empty", 0
    res = res[0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(STOCK_HEADER)
        for i, t in enumerate(ts):
            c = q["close"][i]
            if c is None:
                continue
            d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
            w.writerow([d, q["open"][i], q["high"][i], q["low"][i], c, c, q["volume"][i]])
            n += 1
    return "ok", n


def run(target):
    syms, sub = TARGETS[target]
    out_dir = os.path.join(HERE, sub)
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for i, s in enumerate(syms):
        try:
            status, n = pull_symbol(s, out_dir)
        except Exception as exc:
            status, n = "err:" + type(exc).__name__, 0
        stats[s] = {"status": status, "rows": n}
        print(f"[{i+1}/{len(syms)}] {s} {status} {n}", flush=True)
        time.sleep(DELAY)
    with open(os.path.join(HERE, f"_manifest_{target}.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    total = sum(v["rows"] for v in stats.values())
    ok = sum(1 for v in stats.values() if v["status"] in ("ok", "skip"))
    print(f"DONE {target} symbols_ok={ok} total_rows={total}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tgt = args[0] if args else "indices"
    run(tgt)
