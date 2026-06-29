# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Sentiment layer: scores scraped headlines (scraper.py) with a user-added model via the
#   platform endpoint POST /teacher/complete (HTTP, no platform-internal imports), then
#   aggregates per-asset into a time-decayed sentiment signal the trader consumes.
# - Recommended scorer model: a general instruct model (e.g. ollama qwen2.5:7b-instruct).
#   Defaults to the configured teacher when provider/model are omitted.
# extensions/canonical/paper_trade/server/sentiment.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import ssl
import time
import urllib.request

import certifi

# ------------------------------------------------------------------------------------
# Constants

COMPLETE_URL = "http://127.0.0.1:8001/teacher/complete"
CTX = ssl.create_default_context(cafile=certifi.where())
TIMEOUT = 30
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
SENT_SYSTEM = (
    "You are a financial markets sentiment scorer for crypto and stocks. Judge the EXPECTED SHORT-TERM "
    "PRICE IMPACT of the headline BEYOND what the market already knows. Rate routine, repeated, or "
    "already-priced-in news near 0; give large magnitudes only to genuinely new, surprising, market-moving "
    'information. Output ONLY compact JSON: {"asset":"<TICKER or MARKET>","sentiment":<-1.0..1.0>,"confidence":<0..1>}. '
    "asset is the ticker the news is about (e.g. BTC, ETH, SOL, AAPL, NVDA, TSLA) or MARKET if broad / not "
    "asset-specific; confidence reflects how novel and actionable the news is. No prose."
)
HALF_LIFE_S = 6 * 3600          # sentiment weight halves every 6h
SCORE_MAX_TOKENS = 80

# ------------------------------------------------------------------------------------
# Functions

def _complete(prompt, provider, model, url):
    """Score via the platform /teacher/complete endpoint; if the dashboard is not running
    (standalone CLI use), fall back to the teacher client directly so the loop still runs."""
    body = {"prompt": prompt, "system": SENT_SYSTEM, "max_tokens": SCORE_MAX_TOKENS, "temperature": 0.0}
    if provider:
        body["provider"] = provider
    if model:
        body["model"] = model
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
            d = json.loads(r.read())
        if d.get("ok"):
            return d.get("text", "")
    except Exception:
        pass
    return _complete_direct(prompt, provider, model)


def _complete_direct(prompt, provider, model):
    """Standalone fallback: call the teacher client in-process (first-party CLI use only)."""
    try:
        import os
        import sys
        root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "veritate_mri"))
        if root not in sys.path:
            sys.path.insert(0, root)
        import teacher
        return teacher.complete(provider or DEFAULT_PROVIDER, model or DEFAULT_MODEL,
                                [{"role": "user", "content": prompt}],
                                system=SENT_SYSTEM, max_tokens=SCORE_MAX_TOKENS, temperature=0.0)
    except Exception:
        return ""


def _parse(text):
    """Pull the {...} sentiment object out of the model text; None if absent/invalid."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        o = json.loads(text[i:j + 1])
        return {"asset": str(o.get("asset", "MARKET")).upper(),
                "sentiment": max(-1.0, min(1.0, float(o.get("sentiment", 0.0)))),
                "confidence": max(0.0, min(1.0, float(o.get("confidence", 0.0))))}
    except (ValueError, TypeError):
        return None


def score_items(items, provider=None, model=None, url=COMPLETE_URL, cache=None):
    """Score each headline; attach asset/sentiment/confidence. Unscored items are dropped.
    If `cache` (a dict keyed by headline title) is given, scores are memoized, so a re-scan only
    pays model latency for genuinely NEW headlines. This keeps a slower, stronger scorer (e.g.
    qwen2.5:72b, better at judging whether news is already priced in) affordable in steady state."""
    out = []
    for it in items:
        title = it["title"]
        score = cache.get(title) if cache is not None else None
        if score is None:
            score = _parse(_complete(title, provider, model, url))
            if cache is not None and score is not None:
                cache[title] = score
        if score is not None:
            out.append({**it, **score})
    return out


def aggregate(scored, half_life_s=HALF_LIFE_S, now=None):
    """Per-asset time-decayed sentiment signal: {asset: {score, n}} with score in [-1,1].
    Weight = confidence * 0.5**(age/half_life), so fresh, confident calls dominate."""
    now = now if now is not None else time.time()
    acc = {}
    for s in scored:
        w = s["confidence"] * (0.5 ** (max(0.0, now - s["ts"]) / half_life_s))
        a = acc.setdefault(s["asset"], [0.0, 0.0, 0])
        a[0] += w * s["sentiment"]
        a[1] += w
        a[2] += 1
    return {asset: {"score": (v[0] / v[1] if v[1] > 0 else 0.0), "n": v[2]}
            for asset, v in acc.items()}
