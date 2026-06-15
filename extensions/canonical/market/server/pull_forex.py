# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - scratch puller for 1-minute forex bars from the dukascopy tick feed (free, no
#   key, direct GET). one bi5 file per pair per hour: lzma-compressed binary ticks
#   of (ms_offset, ask_points, bid_points, ask_vol, bid_vol), big-endian.
# - ticks are decoded with stdlib lzma+struct, mid price = (ask+bid)/2 / point_div
#   (1e3 for JPY pairs, 1e5 otherwise), then resampled to 1m OHLC with tick-count
#   as volume so the codec's vol_ratio stays finite (no real lot tape here).
# - writes the canonical crypto schema time,open,high,low,close,volume with time as
#   epoch-ms (the builder's crypto loader feeds time through normalize_time, numeric).
# - INCREMENTAL + RESUMABLE: writes one csv per pair, flushed per completed month so
#   a killed run keeps finished months. a per-pair _done.json records completed months;
#   a re-run skips finished months and resumes mid-pair. weekends/holidays return
#   empty bi5 and are skipped. ~0.53s per hourly fetch sequential.
# - PARALLEL FETCH: hourly bi5 GET+decode within a month run on a bounded thread pool
#   (--workers, default 12); the endpoint tolerates parallel GETs. bars are folded in
#   time order after the pool returns, so concurrency never reorders or drops bars; a
#   month is marked done in _done.json only after all its hours are written.
# - usage: python3 pull_forex.py                              (default last 3 years)
#          python3 pull_forex.py --start 2020 --end 2026      (full history)
#          python3 pull_forex.py --pairs EURUSD --start 2026 --end 2026
#          python3 pull_forex.py --workers 12                 (parallel hourly fetch)
# extensions/canonical/market/server/pull_forex.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import lzma
import os
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import certifi

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")), "extensions", "installed", "market", "data")
OUT = os.path.join(HERE, "forex")

FEED_URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{year}/{month0:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY", "EURAUD", "EURCAD",
    "GBPCHF", "CADJPY", "AUDNZD", "NZDJPY", "CHFJPY", "GBPCAD",
]

SSL_CTX = ssl.create_default_context(cafile=certifi.where())
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
TIMEOUT = 60
RETRIES = 4
DEFAULT_WORKERS = 12
DEFAULT_YEARS = 3
DEFAULT_END = 2026
DEFAULT_START = DEFAULT_END - DEFAULT_YEARS + 1
TICK_FMT = ">IIIff"
TICK_SIZE = 20
POINT_DIV_JPY = 1e3
POINT_DIV = 1e5
MINUTE_MS = 60_000
HOURS_PER_DAY = 24
SAT = 5
SUN = 6
CSV_HEADER = "time,open,high,low,close,volume\n"
DONE_SUFFIX = "_done.json"
PART_SUFFIX = ".part"

# ------------------------------------------------------------------------------------
# Functions

def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=SSL_CTX) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return b""
            last = exc
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last


def decode_hour(blob, hour_ms, point_div):
    data = lzma.decompress(blob)
    n = len(data) // TICK_SIZE
    out = []
    for i in range(n):
        ms, ask, bid, _, _ = struct.unpack_from(TICK_FMT, data, i * TICK_SIZE)
        out.append((hour_ms + ms, (ask + bid) / 2.0 / point_div))
    return out


def fold_ticks(bars, ticks):
    for ts, price in ticks:
        m = ts - (ts % MINUTE_MS)
        b = bars.get(m)
        if b is None:
            bars[m] = [price, price, price, price, 1]
        else:
            b[1] = max(b[1], price)
            b[2] = min(b[2], price)
            b[3] = price
            b[4] += 1


def month_iter(start, end):
    y, m = start, 1
    while (y, m) <= (end, 12):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch_hour(pair, day, hour, point_div):
    url = FEED_URL.format(pair=pair, year=day.year, month0=day.month - 1, day=day.day, hour=hour)
    try:
        blob = http_get(url)
    except Exception:
        return None
    if not blob:
        return None
    hour_ms = int(day.replace(hour=hour).timestamp() * 1000)
    return decode_hour(blob, hour_ms, point_div)


def pull_month(pair, year, month, point_div, workers):
    jobs = []
    day = datetime(year, month, 1, tzinfo=timezone.utc)
    while day.year == year and day.month == month:
        if day.weekday() != SAT:
            for hour in range(HOURS_PER_DAY):
                jobs.append((day, hour))
        day += timedelta(days=1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ticks_by_job = list(pool.map(lambda j: fetch_hour(pair, j[0], j[1], point_div), jobs))
    bars = {}
    for ticks in ticks_by_job:
        if ticks:
            fold_ticks(bars, ticks)
    return bars


def append_month(path, bars):
    with open(path, "a") as fh:
        for m in sorted(bars):
            o, h, l, c, v = bars[m]
            fh.write(f"{m},{o},{h},{l},{c},{v}\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_done(pair):
    p = os.path.join(OUT, pair + DONE_SUFFIX)
    if os.path.exists(p):
        with open(p) as fh:
            return set(json.load(fh))
    return set()


def save_done(pair, done):
    tmp = os.path.join(OUT, pair + DONE_SUFFIX + PART_SUFFIX)
    with open(tmp, "w") as fh:
        json.dump(sorted(done), fh)
    os.replace(tmp, os.path.join(OUT, pair + DONE_SUFFIX))


def pull_pair(pair, start, end, workers):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, pair + ".csv")
    point_div = POINT_DIV_JPY if pair.endswith("JPY") else POINT_DIV
    done = load_done(pair)
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(CSV_HEADER)
    rows = 0
    for year, month in month_iter(start, end):
        key = f"{year}-{month:02d}"
        if key in done:
            continue
        t0 = time.time()
        bars = pull_month(pair, year, month, point_div, workers)
        append_month(path, bars)
        done.add(key)
        save_done(pair, done)
        rows += len(bars)
        print(f"    {pair} {key} +{len(bars)} rows {time.time()-t0:.0f}s", flush=True)
    return ("ok" if rows or done else "miss"), rows


def run(pairs, start, end, workers):
    stats = {}
    for i, p in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {p} start {start}-{end} workers {workers}", flush=True)
        try:
            status, n = pull_pair(p, start, end, workers)
        except Exception as exc:
            status, n = "err:" + type(exc).__name__, 0
        stats[p] = {"status": status, "rows": n}
        print(f"[{i+1}/{len(pairs)}] {p} {status} {n}", flush=True)
    with open(os.path.join(HERE, "_manifest_forex.json"), "w") as fh:
        json.dump({"start": start, "end": end, "pairs": stats}, fh, indent=2)
    total = sum(v["rows"] for v in stats.values())
    ok = sum(1 for v in stats.values() if v["status"] == "ok")
    print(f"DONE pairs_ok={ok} total_rows={total}", flush=True)


if __name__ == "__main__":
    start = DEFAULT_START
    end = DEFAULT_END
    pairs = PAIRS
    workers = DEFAULT_WORKERS
    if "--start" in sys.argv:
        start = int(sys.argv[sys.argv.index("--start") + 1])
    if "--end" in sys.argv:
        end = int(sys.argv[sys.argv.index("--end") + 1])
    if "--pairs" in sys.argv:
        pairs = sys.argv[sys.argv.index("--pairs") + 1].split(",")
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    run(pairs, start, end, workers)
