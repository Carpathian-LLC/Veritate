# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Free, keyless market surveillance data layer: OKX spot tickers (all USDT pairs, one
#   call) + CoinGecko trending searches + meme-token board. Persists rolling per-scan
#   snapshots so price/volume z-scores use trailing baselines across scans, not just 24h
#   fields. Flags PUMP when price z and volume z spike together (the validated detection
#   recipe from the 2026-06-26 pump probe; buying flagged pumps MEAN-REVERTS, so a flag
#   is a warning, never a signal). Every flagged event appends one JSON line to
#   events.jsonl: the forward event dataset prior research lacked.
# - Background watch thread rescans every SCAN_SEC and stamps auto-resume in state.json;
#   register.py calls resume() at boot. CoinGecko free tier is rate-limited (~10-30
#   req/min), so its two calls are disk-cached and refreshed at most once per CG_TTL_S.
# extensions/canonical/trading/server/scanner.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import ssl
import threading
import time
import urllib.request

import certifi

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
DATA_DIR = os.path.join(ROOT, "extensions", "installed", "trading", "data", "intel")
SNAP_DIR = os.path.join(DATA_DIR, "snapshots")
EVENTS_PATH = os.path.join(DATA_DIR, "events.jsonl")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
CG_CACHE_PATH = os.path.join(DATA_DIR, "cg_cache.json")

OKX_TICKERS_URL = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles?instId={}-USDT&bar={}&limit={}"
CG_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"
CG_MEME_URL = ("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
               "&category=meme-token&order=market_cap_desc&per_page=100"
               "&price_change_percentage=1h,24h,7d")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
TIMEOUT = 15
CTX = ssl.create_default_context(cafile=certifi.where())

QUOTE = "-USDT"
MIN_VOL_USD = 50000.0     # 24h quote volume floor; thinner books are uninvestable noise
BASELINE_N = 14           # trailing snapshots per z-score baseline
BASELINE_MIN = 5          # snapshots required before z-scores are trusted (else 0)
SNAP_KEEP = 60            # rolling snapshot files kept on disk
RET1H_S = 3600            # age of the snapshot used for ret_1h
PRICE_Z_MIN = 2.5         # pump flag: price return z at/above this...
VOL_Z_MIN = 3.0           # ...AND volume z at/above this (vol-z>3 probe recipe)
COOLDOWN_S = 6 * 3600     # one event per symbol per window, not 50 per pump
W_PRICE = 1.0             # anomaly score weights
W_VOL = 1.0
W_RANGE = 0.5
ROWS_MAX = 120            # scored rows returned per scan
MEME_MAX = 30             # meme board rows returned
EVENTS_MAX = 5000         # events.jsonl line cap (oldest rotated out)
CG_TTL_S = 300            # CoinGecko refresh at most once per scan cadence
SCAN_TTL_S = 240          # /scan serves the cached result inside this window
SCAN_SEC = 300            # watch thread cadence
CANDLE_BARS = ("1H", "4H", "1Dutc")   # chart bars; 1Dutc avoids OKX's UTC+8 daily alignment
CANDLE_MAX = 300          # OKX per-call row cap; every fetch grabs the full window
CANDLE_TTL_S = 60         # per (sym, bar) cache window

_LOCK = threading.Lock()
_CACHE = None             # last scan result
_RUN = None               # watch thread {thread, stop, interval, model}
_CANDLES = {}             # (sym, bar) -> {ts, rows}

# ------------------------------------------------------------------------------------
# Functions

def _get_json(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def okx_rows():
    """All OKX spot USDT pairs above the volume floor: {sym: {px, open24h, high, low, vol_usd}}.
    None when the feed is unreachable (caller keeps the last scan)."""
    d = _get_json(OKX_TICKERS_URL)
    if not d or d.get("code") != "0":
        return None
    out = {}
    for r in d["data"]:
        inst = r["instId"]
        if not inst.endswith(QUOTE):
            continue
        try:
            px, vol = float(r["last"]), float(r["volCcy24h"])
        except (ValueError, KeyError):
            continue
        if px <= 0 or vol < MIN_VOL_USD:
            continue
        out[inst[:-len(QUOTE)]] = {"px": px, "open24h": float(r["open24h"]),
                                   "high": float(r["high24h"]), "low": float(r["low24h"]),
                                   "vol_usd": vol}
    return out


def candles(sym, bar, limit=CANDLE_MAX):
    """OHLCV rows oldest-first for one USDT pair: [{t, o, h, l, c, v}], v = quote USD volume.
    TTL-cached per (sym, bar); a failed refetch serves the stale cache. None on bad input
    or unreachable feed with no cache."""
    sym = "".join(ch for ch in str(sym).upper() if ch.isalnum())
    if not sym or bar not in CANDLE_BARS:
        return None
    limit = max(2, min(CANDLE_MAX, int(limit)))
    with _LOCK:
        c = _CANDLES.get((sym, bar))
    if c and time.time() - c["ts"] < CANDLE_TTL_S:
        return c["rows"][-limit:]
    d = _get_json(OKX_CANDLES_URL.format(sym, bar, CANDLE_MAX))
    if not d or d.get("code") != "0" or not d.get("data"):
        return c["rows"][-limit:] if c else None
    rows = [{"t": int(r[0]) // 1000, "o": float(r[1]), "h": float(r[2]), "l": float(r[3]),
             "c": float(r[4]), "v": float(r[7])} for r in reversed(d["data"])]
    with _LOCK:
        _CANDLES[(sym, bar)] = {"ts": time.time(), "rows": rows}
    return rows[-limit:]


def cg_boards(refresh=False):
    """CoinGecko trending searches + meme-token board, disk-cached CG_TTL_S (free tier is
    rate-limited; at most one refresh per scan). Stale cache beats no data."""
    cached = _read_json(CG_CACHE_PATH, None)
    if cached and not refresh and time.time() - cached.get("ts", 0) < CG_TTL_S:
        return cached
    trending, meme = [], []
    d = _get_json(CG_TRENDING_URL)
    for it in (d or {}).get("coins", []):
        c = it.get("item", {})
        trending.append({"sym": str(c.get("symbol", "")).upper(), "name": c.get("name", ""),
                         "rank": c.get("market_cap_rank")})
    d = _get_json(CG_MEME_URL)
    for c in d or []:
        meme.append({"sym": str(c.get("symbol", "")).upper(), "name": c.get("name", ""),
                     "px": c.get("current_price"),
                     "ret_1h": c.get("price_change_percentage_1h_in_currency"),
                     "ret_24h": c.get("price_change_percentage_24h_in_currency"),
                     "ret_7d": c.get("price_change_percentage_7d_in_currency"),
                     "vol_usd": c.get("total_volume"), "mcap": c.get("market_cap")})
    if not trending and not meme and cached:
        return cached
    out = {"ts": int(time.time()), "trending": trending, "meme": meme}
    _write_json(CG_CACHE_PATH, out)
    return out


def _snaps():
    """Trailing snapshots oldest-first: [{ts, syms: {sym: [px, vol_usd, range]}}]."""
    if not os.path.isdir(SNAP_DIR):
        return []
    names = sorted(n for n in os.listdir(SNAP_DIR) if n.endswith(".json"))
    return [s for s in (_read_json(os.path.join(SNAP_DIR, n), None) for n in names) if s]


def _save_snap(snap):
    _write_json(os.path.join(SNAP_DIR, f"{snap['ts']}.json"), snap)
    names = sorted(n for n in os.listdir(SNAP_DIR) if n.endswith(".json"))
    for n in names[:-SNAP_KEEP]:
        os.remove(os.path.join(SNAP_DIR, n))


def zscore(x, hist):
    """(x - mean) / std over the trailing baseline; 0 when the baseline is short or flat."""
    if len(hist) < BASELINE_MIN:
        return 0.0
    m = sum(hist) / len(hist)
    var = sum((v - m) ** 2 for v in hist) / len(hist)
    if var <= 0:
        return 0.0
    return (x - m) / var ** 0.5


def anomaly_score(price_z, vol_z, range_x):
    """Composite: directionless price shock + positive volume shock + range expansion."""
    return round(W_PRICE * abs(price_z) + W_VOL * max(vol_z, 0.0) + W_RANGE * max(range_x - 1.0, 0.0), 3)


def pump_flag(price_z, vol_z):
    """The validated detection recipe: price z AND volume z spiking together."""
    return price_z >= PRICE_Z_MIN and vol_z >= VOL_Z_MIN


def append_event(ev):
    """One JSON line per flagged event: the forward pump dataset. Rotates at EVENTS_MAX."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(EVENTS_PATH, "a") as f:
        f.write(json.dumps(ev) + "\n")
    with open(EVENTS_PATH) as f:
        lines = f.readlines()
    if len(lines) > EVENTS_MAX:
        with open(EVENTS_PATH, "w") as f:
            f.writelines(lines[-EVENTS_MAX:])


def events(limit=50):
    """Recent flagged events, newest first."""
    try:
        with open(EVENTS_PATH) as f:
            lines = f.readlines()
    except OSError:
        return []
    out = []
    for ln in reversed(lines[-max(limit, 0):] if limit else lines):
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def event_count():
    try:
        with open(EVENTS_PATH) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _ret_1h(sym, snaps, now, px):
    old = [s for s in snaps if now - s["ts"] >= RET1H_S]
    if not old:
        return None
    p0 = old[-1]["syms"].get(sym)
    return round(px / p0[0] - 1.0, 6) if p0 and p0[0] > 0 else None


def _score_rows(tick, snaps, now, state):
    """Per-symbol metrics vs trailing baselines + pump flags. Flags append events and
    stamp a per-symbol cooldown so one pump is one event."""
    tail = snaps[-BASELINE_N:]
    cooldown = state.setdefault("cooldown", {})
    rows, new_events = [], []
    for sym, t in tick.items():
        px_h, vol_h, rng_h = [], [], []
        for s in tail:
            v = s["syms"].get(sym)
            if v:
                px_h.append(v[0])
                vol_h.append(v[1])
                rng_h.append(v[2])
        rets_h = [px_h[i] / px_h[i - 1] - 1.0 for i in range(1, len(px_h)) if px_h[i - 1] > 0]
        r_now = t["px"] / px_h[-1] - 1.0 if px_h and px_h[-1] > 0 else 0.0
        rng = (t["high"] - t["low"]) / t["px"]
        price_z = round(zscore(r_now, rets_h), 2)
        vol_z = round(zscore(t["vol_usd"], vol_h), 2)
        rng_m = sum(rng_h) / len(rng_h) if rng_h else 0.0
        range_x = round(rng / rng_m, 2) if rng_m > 0 else 1.0
        row = {"sym": sym, "px": t["px"], "vol_usd": round(t["vol_usd"]),
               "ret_1h": _ret_1h(sym, snaps, now, t["px"]),
               "ret_24h": round(t["px"] / t["open24h"] - 1.0, 6) if t["open24h"] > 0 else None,
               "price_z": price_z, "vol_z": vol_z, "range_x": range_x,
               "score": anomaly_score(price_z, vol_z, range_x), "pump": False}
        if pump_flag(price_z, vol_z) and now - cooldown.get(sym, 0) >= COOLDOWN_S:
            cooldown[sym] = now
            row["pump"] = True
            ev = {k: row[k] for k in ("sym", "px", "ret_1h", "ret_24h", "price_z",
                                      "vol_z", "range_x", "score", "vol_usd")}
            ev["ts"] = now
            append_event(ev)
            new_events.append(ev)
        rows.append(row)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows, new_events


def scan(refresh=False):
    """One surveillance pass: OKX tick + CG boards -> scored rows vs trailing snapshot
    baselines -> pump events -> snapshot persisted. Cached SCAN_TTL_S unless refresh."""
    global _CACHE
    with _LOCK:
        now = int(time.time())
        if _CACHE and not refresh and now - _CACHE["ts"] < SCAN_TTL_S:
            return _CACHE
        tick = okx_rows()
        if tick is None:
            return _CACHE or {"ts": now, "rows": [], "trending": [], "meme": [],
                              "events": [], "error": "market feed unreachable"}
        cg = cg_boards(refresh)
        snaps = _snaps()
        state = _read_json(STATE_PATH, {})
        rows, new_events = _score_rows(tick, snaps, now, state)
        meme_syms = {m["sym"] for m in cg["meme"]}
        trend_syms = {t["sym"] for t in cg["trending"]}
        names = {b["sym"]: b["name"] for b in cg["meme"] + cg["trending"] if b.get("name")}
        for r in rows:
            r["meme"] = r["sym"] in meme_syms
            r["trending"] = r["sym"] in trend_syms
            if r["sym"] in names:
                r["name"] = names[r["sym"]]
        by_sym = {r["sym"]: r for r in rows}
        trending = [{**t, "okx": by_sym.get(t["sym"])} for t in cg["trending"]]
        meme = sorted(cg["meme"], key=lambda m: abs(m.get("ret_24h") or 0.0), reverse=True)
        _save_snap({"ts": now, "syms": {s: [t["px"], t["vol_usd"],
                                            (t["high"] - t["low"]) / t["px"]]
                                        for s, t in tick.items()}})
        fresh = _read_json(STATE_PATH, {})           # keep a watch stamp written mid-scan
        fresh["cooldown"] = state.get("cooldown", {})
        _write_json(STATE_PATH, fresh)
        _CACHE = {"ts": now, "rows": rows[:ROWS_MAX], "trending": trending,
                  "meme": meme[:MEME_MAX], "events": new_events}
        return _CACHE


def row_for(sym):
    """Last-scan metrics row for one symbol, or None before any scan / for unknown syms."""
    if not _CACHE:
        return None
    return next((r for r in _CACHE["rows"] if r["sym"] == sym), None)


def status():
    return {"last_scan": _CACHE["ts"] if _CACHE else None, "events": event_count(),
            "watching": bool(_RUN and _RUN["thread"].is_alive()),
            "interval": (_RUN or {}).get("interval", SCAN_SEC),
            "started": (_RUN or {}).get("started"), "model": (_RUN or {}).get("model")}

# ------------------------------------------------------------------------------------
# Watch thread (scan every SCAN_SEC; auto-resume stamped in state.json)

def start_thread(interval=SCAN_SEC, model=None):
    global _RUN
    if _RUN and _RUN["thread"].is_alive():
        return False
    state = _read_json(STATE_PATH, {})
    state["watch"] = {"on": True, "interval": interval, "model": model}
    _write_json(STATE_PATH, state)
    ev = threading.Event()
    _RUN = {"thread": None, "stop": ev, "interval": interval, "model": model,
            "started": int(time.time())}

    def _run():
        while not ev.is_set():
            try:
                s = scan(refresh=True)
                import intel
                intel.auto_brief(s, model=model)
            except Exception:
                pass
            ev.wait(interval)
    t = threading.Thread(target=_run, name="market-intel-watch", daemon=True)
    _RUN["thread"] = t
    t.start()
    return True


def stop_thread():
    global _RUN
    if _RUN:
        _RUN["stop"].set()
        _RUN = None
    state = _read_json(STATE_PATH, {})
    state["watch"] = {"on": False}
    _write_json(STATE_PATH, state)
    return True


def resume():
    """Restart the watch if state says it was on (server reboot must not gap the dataset)."""
    w = _read_json(STATE_PATH, {}).get("watch") or {}
    if w.get("on"):
        return start_thread(int(w.get("interval") or SCAN_SEC), w.get("model"))
    return False
