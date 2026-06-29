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
# Stocks: general financial feeds (broad market/macro) + ticker -> company names for focused pulls.
STOCK_FEEDS = [
    ("marketwatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("cnbc", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("yahoofinance", "https://finance.yahoo.com/news/rssindex"),
]
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


def gnews(names, qualifier="cryptocurrency"):
    """Google News search RSS for a name's aliases -> focused headlines (qualifier disambiguates,
    e.g. 'cryptocurrency' or 'stock')."""
    q = " OR ".join(f'"{n}"' for n in names) + " " + qualifier
    return fetch_rss("googlenews", GNEWS_URL.format(urllib.parse.quote(q)))


def scrape(limit=40, focus=None, market="crypto"):
    """Recent headlines for a market, deduped by title. `market` picks the feed set (crypto RSS vs
    general financial RSS). When `focus` is a ticker (SOL, NVDA), a Google News query for that
    name is pulled first and matching headlines rank to the top, so choosing an asset auto-focuses
    the news. With no focus it scans the broad feeds (general crypto, or general market/macro)."""
    if market == "stocks":
        names = STOCK_NAMES.get((focus or "").upper())
        feeds, qual = STOCK_FEEDS, "stock"
    else:
        names = TOKEN_NAMES.get((focus or "").upper())
        feeds, qual = NEWS_FEEDS, "cryptocurrency"
    items, seen = [], set()
    sources = ([("googlenews", None)] if names else []) + feeds
    for source, url in sources:
        feed = gnews(names, qual) if url is None else fetch_rss(source, url)
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
