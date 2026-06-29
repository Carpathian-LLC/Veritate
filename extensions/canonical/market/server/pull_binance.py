# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - scratch puller for granular crypto klines from the Binance Vision archive.
#   api.binance.com is geo-blocked (HTTP 451) in the US; api.binance.us is used
#   only to rank pairs by volume, the public Vision S3 archive serves the bars.
# - writes one canonical csv per symbol: time,open,high,low,close,volume (quote/USDT
#   volume). rows are appended in chronological month order, already time-ascending.
# - re-runnable: skips a symbol whose csv already exists.
# - --top mode: ranks USDT pairs by binance.us 24h volume (caps at ~200 pairs).
# - --full mode: enumerates EVERY USDT pair from the Vision S3 archive directly
#   (paginated prefix listing), writes the ones NOT already present in crypto/ to
#   crypto_extra/ so the two dirs together cover the whole archive with no dupes.
# - usage: python3 pull_binance.py --interval 1m --top 60
#          python3 pull_binance.py --interval 1m --full
# extensions/canonical/market/server/pull_binance.py
# ------------------------------------------------------------------------------------
# Imports:

import io
import json
import os
import re
import ssl
import sys
import time
import urllib.request
import zipfile

import certifi

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")), "extensions", "installed", "market", "data")
OUT = os.path.join(HERE, "crypto")
OUT_EXTRA = os.path.join(HERE, "crypto_extra")

TICKER_URL = "https://api.binance.us/api/v3/ticker/24hr"
LIST_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
VISION_BASE = "https://data.binance.vision/"
KLINE_PREFIX = "data/spot/monthly/klines/{sym}/{interval}/"
SYMBOL_PREFIX = "data/spot/monthly/klines/"

STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "UST", "USTC", "EUR", "GBP", "AEUR"}
QUOTE = "USDT"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
TIMEOUT = 45
RETRIES = 4
DEFAULT_INTERVAL = "1m"
DEFAULT_TOP = 60
CSV_HEADER = "time,open,high,low,close,volume\n"

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


def top_symbols(n):
    rows = json.loads(http_get(TICKER_URL))
    pairs = []
    for x in rows:
        s = x["symbol"]
        if not s.endswith("USDT") or s[:-4] in STABLE_BASES:
            continue
        pairs.append((float(x["quoteVolume"]), s))
    pairs.sort(reverse=True)
    return [s for _, s in pairs[:n]]


def all_usdt_symbols():
    syms, marker = [], ""
    while True:
        url = LIST_BASE + "?delimiter=/&prefix=" + SYMBOL_PREFIX
        if marker:
            url += "&marker=" + marker
        body = http_get(url).decode()
        found = re.findall(r"<Prefix>" + re.escape(SYMBOL_PREFIX) + r"([^/]+)/</Prefix>", body)
        syms += found
        if "<IsTruncated>true</IsTruncated>" not in body:
            break
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", body)
        marker = m.group(1) if m else SYMBOL_PREFIX + found[-1] + "/"
    keep = [s for s in syms if s.endswith(QUOTE) and s[:-len(QUOTE)] not in STABLE_BASES]
    return sorted(set(keep))


def list_months(sym, interval):
    url = LIST_BASE + "?delimiter=/&prefix=" + KLINE_PREFIX.format(sym=sym, interval=interval)
    return sorted(re.findall(r"<Key>([^<]+\.zip)</Key>", http_get(url).decode()))


def pull_symbol(sym, interval, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, sym + ".csv")
    if os.path.exists(path) or os.path.exists(os.path.join(OUT, sym + ".csv")):
        return "skip", 0
    keys = list_months(sym, interval)
    if not keys:
        return "miss", 0
    n = 0
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        fh.write(CSV_HEADER)
        for k in keys:
            try:
                zf = zipfile.ZipFile(io.BytesIO(http_get(VISION_BASE + k)))
            except Exception:
                continue
            for name in zf.namelist():
                for row in zf.read(name).decode().splitlines():
                    c = row.split(",")
                    if len(c) < 8:
                        continue
                    try:
                        float(c[1])
                    except ValueError:
                        continue
                    fh.write(f"{c[0]},{c[1]},{c[2]},{c[3]},{c[4]},{c[7]}\n")
                    n += 1
    os.replace(tmp, path)
    return "ok", n


def run(interval, top, full):
    if full:
        syms = all_usdt_symbols()
        out_dir = OUT_EXTRA if interval == DEFAULT_INTERVAL else os.path.join(HERE, "crypto_" + interval)
        manifest = "_manifest_binance_full.json"
    else:
        syms = top_symbols(top)
        out_dir = OUT if interval == DEFAULT_INTERVAL else os.path.join(HERE, "crypto_" + interval)
        manifest = "_manifest_binance.json"
    stats = {}
    for i, s in enumerate(syms):
        try:
            status, n = pull_symbol(s, interval, out_dir)
        except Exception as exc:
            status, n = "err:" + type(exc).__name__, 0
        stats[s] = {"status": status, "rows": n}
        print(f"[{i+1}/{len(syms)}] {s} {status} {n}", flush=True)
    with open(os.path.join(HERE, manifest), "w") as fh:
        json.dump({"interval": interval, "symbols": stats}, fh, indent=2)
    total = sum(v["rows"] for v in stats.values())
    ok = sum(1 for v in stats.values() if v["status"] in ("ok", "skip"))
    print(f"DONE interval={interval} symbols_ok={ok} total_rows={total}", flush=True)


if __name__ == "__main__":
    interval = DEFAULT_INTERVAL
    top = DEFAULT_TOP
    full = "--full" in sys.argv
    if "--interval" in sys.argv:
        interval = sys.argv[sys.argv.index("--interval") + 1]
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    run(interval, top, full)
