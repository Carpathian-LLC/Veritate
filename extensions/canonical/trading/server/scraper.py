# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Shared news + sentiment-context scraper for the Trading extension. Headlines come
#   from a user-editable CHANNEL registry persisted in settings.json: rss (feed URL),
#   reddit (subreddit name, old.reddit.com hot.json), gnews (Google News search query),
#   index (the alternative.me fear-greed feed). Both the news-sentiment loop and the
#   intel briefs read through this module, so one channel list drives all scraping.
# - Per-channel health (ok / blocked / error + last fetch) is tracked in memory and
#   surfaced via channel_health() for the settings page. One unreachable channel is
#   skipped, never fatal. This module also owns settings.json (load/save/normalize).
# - RSS parsed with the stdlib (no feedparser dep). Stdlib + certifi only, so a missing
#   optional dep never breaks dashboard startup.
# extensions/canonical/trading/server/scraper.py
# ------------------------------------------------------------------------------------
# Imports:

import email.utils
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import certifi

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
SETTINGS_PATH = os.path.join(ROOT, "extensions", "installed", "trading", "data", "settings.json")

GNEWS_URL = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
REDDIT_URL = "https://old.reddit.com/r/{}/hot.json?limit=25"
FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) veritate-trading/1.0"
TIMEOUT = 12
CTX = ssl.create_default_context(cafile=certifi.where())
MAX_PER_FEED = 25
BLOCKED_CODES = (403, 429)

CHANNEL_TYPES = ("rss", "reddit", "gnews", "index")

DEFAULT_CHANNELS = [
    {"id": "cointelegraph", "type": "rss", "value": "https://cointelegraph.com/rss", "enabled": True},
    {"id": "decrypt", "type": "rss", "value": "https://decrypt.co/feed", "enabled": True},
    {"id": "cryptoslate", "type": "rss", "value": "https://cryptoslate.com/feed/", "enabled": True},
    {"id": "bitcoinmagazine", "type": "rss", "value": "https://bitcoinmagazine.com/feed", "enabled": True},
    {"id": "marketwatch", "type": "rss", "value": "https://feeds.marketwatch.com/marketwatch/topstories/", "enabled": True},
    {"id": "cnbc", "type": "rss", "value": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "enabled": True},
    {"id": "yahoofinance", "type": "rss", "value": "https://finance.yahoo.com/news/rssindex", "enabled": True},
    {"id": "fear_greed", "type": "index", "value": FNG_URL, "enabled": True},
]

DEFAULT_SETTINGS = {
    "model": "qwen2.5:7b-instruct",
    "news_interval": 300,
    "intel_interval": 300,
    "news_fee_bps": 20.0,
    "channels": DEFAULT_CHANNELS,
}

# Token -> search names/aliases. Picking a coin auto-pulls news about THAT coin (Google News
# search RSS) and ranks token-matching headlines first, so the model reads what's relevant.
TOKEN_NAMES = {
    "BTC": ["Bitcoin", "BTC"], "ETH": ["Ethereum", "ETH"], "SOL": ["Solana", "SOL"],
    "DOGE": ["Dogecoin", "DOGE"], "XRP": ["XRP", "Ripple"], "LINK": ["Chainlink", "LINK"],
    "AVAX": ["Avalanche", "AVAX"], "ADA": ["Cardano", "ADA"], "LTC": ["Litecoin", "LTC"],
    "BCH": ["Bitcoin Cash", "BCH"], "DOT": ["Polkadot", "DOT"], "ATOM": ["Cosmos", "ATOM"],
    "NEAR": ["NEAR Protocol", "NEAR"], "UNI": ["Uniswap", "UNI"], "XLM": ["Stellar", "XLM"],
    "AAVE": ["Aave", "AAVE"], "SHIB": ["Shiba Inu", "SHIB"], "PEPE": ["Pepe coin", "PEPE"],
    "ARB": ["Arbitrum", "ARB"], "OP": ["Optimism crypto", "OP token"], "APT": ["Aptos", "APT"],
    "SUI": ["Sui crypto", "SUI"], "HYPE": ["Hyperliquid", "HYPE"],
}
STOCK_NAMES = {
    "AAPL": ["Apple"], "MSFT": ["Microsoft"], "NVDA": ["Nvidia"], "TSLA": ["Tesla"],
    "AMZN": ["Amazon"], "GOOGL": ["Google", "Alphabet"], "META": ["Meta", "Facebook"],
    "AMD": ["AMD"], "NFLX": ["Netflix"], "INTC": ["Intel"], "BA": ["Boeing"], "DIS": ["Disney"],
    "PYPL": ["PayPal"], "COIN": ["Coinbase"], "MSTR": ["MicroStrategy", "Strategy stock"],
    "GME": ["GameStop"], "AMC": ["AMC Entertainment"], "PLTR": ["Palantir"], "SOFI": ["SoFi"],
    "NIO": ["NIO"], "RIVN": ["Rivian"], "LCID": ["Lucid Motors"], "F": ["Ford Motor"],
    "JPM": ["JPMorgan"], "BAC": ["Bank of America"], "WMT": ["Walmart"], "KO": ["Coca-Cola"],
    "PFE": ["Pfizer"], "MRNA": ["Moderna"], "SPY": ["S&P 500"], "QQQ": ["Nasdaq 100"],
}

_HEALTH = {}              # channel id -> {status, last_fetch, items}

# ------------------------------------------------------------------------------------
# Settings (single owner of settings.json)

def load_settings():
    """Persisted settings merged over defaults, so new keys appear without migration."""
    try:
        with open(SETTINGS_PATH) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    out = {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}
    out["channels"] = normalize_channels(out.get("channels"))
    return out


def save_settings(changes):
    """Merge validated changes into settings.json atomically; returns the saved settings."""
    cur = load_settings()
    if isinstance(changes.get("model"), str) and changes["model"]:
        cur["model"] = changes["model"]
    for k in ("news_interval", "intel_interval"):
        if changes.get(k) is not None:
            cur[k] = max(15, int(changes[k]))
    if changes.get("news_fee_bps") is not None:
        cur["news_fee_bps"] = max(0.0, float(changes["news_fee_bps"]))
    if "channels" in changes:
        cur["channels"] = normalize_channels(changes["channels"])
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f)
    os.replace(tmp, SETTINGS_PATH)
    return cur


def _slug(value):
    keep = "".join(c if (c.isalnum() or c in "._-") else "_" for c in value.lower())
    return keep.strip("_")[:40] or "channel"


def normalize_channels(raw):
    """Validate a channel list: known type, non-empty value, unique filesystem-safe id.
    Invalid entries are dropped; an empty/absent list falls back to the defaults."""
    if not isinstance(raw, list):
        return [dict(c) for c in DEFAULT_CHANNELS]
    out, seen = [], set()
    for c in raw:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        value = str(c.get("value") or "").strip()
        if ctype not in CHANNEL_TYPES or not value:
            continue
        cid = str(c.get("id") or "").strip() or _slug(value)
        base, n = cid, 1
        while cid in seen:
            cid = f"{base}_{n}"
            n += 1
        seen.add(cid)
        out.append({"id": cid, "type": ctype, "value": value, "enabled": bool(c.get("enabled", True))})
    return out if out else [dict(c) for c in DEFAULT_CHANNELS]


def channels(enabled_only=True):
    chs = load_settings()["channels"]
    return [c for c in chs if c["enabled"]] if enabled_only else chs


def channel_health():
    """Channel list annotated with the in-memory fetch health (unknown before first fetch)."""
    out = []
    for c in channels(enabled_only=False):
        h = _HEALTH.get(c["id"], {})
        out.append({**c, "status": h.get("status", "unknown"),
                    "last_fetch": h.get("last_fetch"), "items": h.get("items", 0)})
    return out

# ------------------------------------------------------------------------------------
# Fetchers

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
        return r.read()


def _mark(cid, status, n=0):
    _HEALTH[cid] = {"status": status, "last_fetch": int(time.time()), "items": n}


def _ts(pubdate):
    """RFC-822 RSS pubDate -> unix seconds, or now() if unparseable."""
    try:
        return int(email.utils.parsedate_to_datetime(pubdate).timestamp())
    except (TypeError, ValueError):
        return int(time.time())


def fetch_rss(source, url):
    """One RSS feed -> [{source, title, url, ts}]. Unreachable/malformed -> []."""
    try:
        root = ET.fromstring(_get(url))
    except Exception:
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        out.append({"source": source, "title": title,
                    "url": (item.findtext("link") or "").strip(),
                    "ts": _ts(item.findtext("pubDate"))})
        if len(out) >= MAX_PER_FEED:
            break
    return out


def parse_reddit(payload, source):
    """A subreddit hot.json payload -> [{source, title, url, ts}]. Stickied posts skipped."""
    out = []
    for child in ((payload or {}).get("data") or {}).get("children") or []:
        d = child.get("data") or {}
        title = (d.get("title") or "").strip()
        if not title or d.get("stickied"):
            continue
        out.append({"source": source, "title": title,
                    "url": "https://old.reddit.com" + (d.get("permalink") or ""),
                    "ts": int(d.get("created_utc") or time.time())})
        if len(out) >= MAX_PER_FEED:
            break
    return out


def fetch_reddit(source, subreddit):
    """Subreddit hot posts as headline items. 403/429 marks the channel blocked."""
    try:
        payload = json.loads(_get(REDDIT_URL.format(urllib.parse.quote(subreddit))))
    except urllib.error.HTTPError as e:
        _mark(source, "blocked" if e.code in BLOCKED_CODES else "error")
        return []
    except Exception:
        _mark(source, "error")
        return []
    items = parse_reddit(payload, source)
    _mark(source, "ok", len(items))
    return items


def _fetch_channel(c):
    if c["type"] == "reddit":
        return fetch_reddit(c["id"], c["value"])
    url = GNEWS_URL.format(urllib.parse.quote(c["value"])) if c["type"] == "gnews" else c["value"]
    items = fetch_rss(c["id"], url)
    _mark(c["id"], "ok" if items else "error", len(items))
    return items


def fear_greed():
    """Current crypto fear-greed index 0..100, or None when unreachable or disabled."""
    ch = next((c for c in channels() if c["type"] == "index"), None)
    if ch is None:
        return None
    try:
        d = json.loads(_get(ch["value"]))["data"][0]
        _mark(ch["id"], "ok", 1)
        return {"value": int(d["value"]), "label": d.get("value_classification", "")}
    except Exception:
        _mark(ch["id"], "error")
        return None


def gnews(names, qualifier="cryptocurrency"):
    """Google News search RSS for a name's aliases -> focused headlines (qualifier disambiguates,
    e.g. 'cryptocurrency' or 'stock')."""
    q = " OR ".join(f'"{n}"' for n in names) + " " + qualifier
    return fetch_rss("googlenews", GNEWS_URL.format(urllib.parse.quote(q)))


def scrape(limit=40, focus=None, market="crypto"):
    """Recent headlines from every enabled channel, deduped by title. When `focus` is a
    ticker (SOL, NVDA), a Google News query for that name is pulled first and matching
    headlines rank to the top, so choosing an asset auto-focuses the news. `market` picks
    the focus name table and query qualifier; the channel list itself is market-agnostic
    (the sentiment layer tags assets, and only universe names ever trade)."""
    if market == "stocks":
        names, qual = STOCK_NAMES.get((focus or "").upper()), "stock"
    else:
        names, qual = TOKEN_NAMES.get((focus or "").upper()), "cryptocurrency"
    items, seen = [], set()
    feeds = [c for c in channels() if c["type"] != "index"]
    for feed in ([None] if names else []) + feeds:
        got = gnews(names, qual) if feed is None else _fetch_channel(feed)
        for it in got:
            key = it["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(it)
    if names:
        low = [n.lower() for n in names]
        items.sort(key=lambda x: (not any(w in x["title"].lower() for w in low), -x["ts"]))
    else:
        items.sort(key=lambda x: x["ts"], reverse=True)
    return items[:limit]
