# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Free crypto news + sentiment-context scraper for the paper-trading sentiment loop.
#   Pulls recent headlines from public RSS feeds (no key) and the alternative.me
#   fear-greed index. Returns plain dicts; no model, no platform internals. The sentiment
#   layer (sentiment.py) scores these via the /teacher/complete endpoint.
# - RSS parsed with the stdlib (no feedparser dep). One unreachable feed is skipped, never
#   fatal. Stdlib + certifi only, so a missing optional dep never breaks dashboard startup.
# extensions/canonical/paper_trade/server/scraper.py
# ------------------------------------------------------------------------------------
# Imports:

import email.utils
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import certifi

# ------------------------------------------------------------------------------------
# Constants

NEWS_FEEDS = [
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("decrypt", "https://decrypt.co/feed"),
    ("cryptoslate", "https://cryptoslate.com/feed/"),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/feed"),
]
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
GNEWS_URL = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
TIMEOUT = 12
CTX = ssl.create_default_context(cafile=certifi.where())
MAX_PER_FEED = 25

# ------------------------------------------------------------------------------------
# Functions

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
        return r.read()


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


def fear_greed():
    """Current crypto fear-greed index 0..100 (alternative.me), or None if unreachable."""
    try:
        import json
        d = json.loads(_get(FNG_URL))["data"][0]
        return {"value": int(d["value"]), "label": d.get("value_classification", "")}
    except Exception:
        return None


def gnews(names):
    """Google News search RSS for a token's names -> token-specific headlines."""
    q = " OR ".join(f'"{n}"' for n in names) + " cryptocurrency"
    return fetch_rss("googlenews", GNEWS_URL.format(urllib.parse.quote(q)))


def scrape(limit=40, focus=None):
    """Recent crypto headlines, deduped by title. When `focus` is a ticker (e.g. SOL), pull a
    token-specific Google News query first and rank headlines mentioning that coin to the top, so
    choosing a coin automatically focuses the news the model reads. Otherwise: all feeds, newest first."""
    names = TOKEN_NAMES.get((focus or "").upper())
    items, seen = [], set()
    sources = ([("googlenews", None)] if names else []) + NEWS_FEEDS
    for source, url in sources:
        feed = gnews(names) if url is None else fetch_rss(source, url)
        for it in feed:
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
