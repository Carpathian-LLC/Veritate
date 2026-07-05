# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - LLM brief layer: pulls recent headlines for a coin through the shared scraper
#   (scraper.gnews, the extension's one scraping surface) and asks a local model via the
#   platform endpoint POST /teacher/complete for strict JSON: why it is moving,
#   news-driven or not, meme or not, pump risk, one-line caution. Briefs are cached by
#   (symbol, headline-hash), so only genuinely new material hits the model.
# - Belt and braces: a high-anomaly move with NO news is stamped pump_risk high
#   mechanically, whatever the model says. A detected pump is exit liquidity, never a
#   buy signal; the caution copy states that.
# extensions/canonical/trading/server/intel.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import ssl
import time
import urllib.request

import certifi

import scanner
import scraper

# ------------------------------------------------------------------------------------
# Constants

COMPLETE_URL = "http://127.0.0.1:8001/teacher/complete"
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
MODEL_TIMEOUT = 60
CTX = ssl.create_default_context(cafile=certifi.where())
HEADLINES_N = 5
MAX_TOKENS = 200
TOP_BRIEF_N = 5
RISKS = ("low", "med", "high")
SYSTEM = (
    "You are a crypto market surveillance analyst. Given a coin, its live move metrics, and "
    "recent headlines (possibly none), explain the move for a dashboard. Output ONLY compact "
    'JSON: {"why_moving":"<one plain sentence>","news_driven":<true|false>,"meme":<true|false>,'
    '"pump_risk":"low"|"med"|"high","caution":"<one plain sentence>"}. news_driven is true only '
    "if a headline plausibly explains the move. A sharp price+volume spike with no news smells "
    "like a pump: say so. Never recommend buying. No prose outside the JSON."
)
CAUTION_PUMP = ("Pump pattern: price and volume spiking with no news. Detected pumps "
                "historically mean-revert; buyers here are the exit liquidity.")
CAUTION_DEFAULT = "Public spikes are usually already priced in by the time they are visible."
WHY_OFFLINE = "model offline"
WHY_FALLBACK = "No model read; metrics only."

_BRIEFS = {}              # sym -> {hhash, brief, ts, model}
_MODEL_STATE = "untried"  # untried | ok | offline

# ------------------------------------------------------------------------------------
# Functions

def headlines(query):
    """Recent headlines for a coin name via the shared scraper -> [titles]."""
    return [it["title"] for it in scraper.gnews([query], "crypto")[:HEADLINES_N]]


def _complete(prompt, model):
    global _MODEL_STATE
    body = {"prompt": prompt, "system": SYSTEM, "max_tokens": MAX_TOKENS, "temperature": 0.0,
            "provider": DEFAULT_PROVIDER, "model": model or DEFAULT_MODEL}
    try:
        req = urllib.request.Request(COMPLETE_URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=MODEL_TIMEOUT, context=CTX) as r:
            d = json.loads(r.read())
        if d.get("ok"):
            _MODEL_STATE = "ok"
            return d.get("text", "")
    except Exception:
        pass
    _MODEL_STATE = "offline"
    return ""


def parse_brief(text):
    """Pull the strict-JSON brief out of the model text; None if absent or malformed."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        o = json.loads(text[i:j + 1])
        risk = str(o.get("pump_risk", "low")).lower()[:4].strip()
        return {"why_moving": str(o.get("why_moving", "")).strip() or WHY_FALLBACK,
                "news_driven": bool(o.get("news_driven")),
                "meme": bool(o.get("meme")),
                "pump_risk": risk if risk in RISKS else ("med" if risk.startswith("med") else "low"),
                "caution": str(o.get("caution", "")).strip() or CAUTION_DEFAULT}
    except (ValueError, TypeError):
        return None


def _spike(metrics):
    return ((metrics.get("price_z") or 0) >= scanner.PRICE_Z_MIN
            and (metrics.get("vol_z") or 0) >= scanner.VOL_Z_MIN)


def _enforce(brief, metrics, heads):
    """No-news + spiking metrics -> pump_risk high mechanically, whatever the model said."""
    if _spike(metrics) and (not heads or not brief["news_driven"]):
        brief["pump_risk"] = "high"
        brief["caution"] = CAUTION_PUMP
    return brief


def _prompt(sym, name, metrics, heads):
    m = ", ".join(f"{k}={metrics.get(k)}" for k in ("ret_1h", "ret_24h", "price_z", "vol_z",
                                                    "range_x", "vol_usd") if metrics.get(k) is not None)
    h = "\n".join(f"- {t}" for t in heads) if heads else "(none found)"
    return f"Coin: {name or sym} ({sym})\nMetrics: {m or 'n/a'}\nRecent headlines:\n{h}"


def brief(sym, metrics=None, name=None, model=None):
    """Brief one symbol. Cached by (symbol, headline-hash): only new material hits the
    model. Model unreachable or malformed reply -> honest metrics-only fallback, not cached,
    so the next call retries."""
    metrics = metrics or {}
    heads = headlines(name or sym)
    hhash = hashlib.sha1("|".join(heads).encode()).hexdigest()[:12]
    c = _BRIEFS.get(sym)
    if c and c["hhash"] == hhash:
        return c["brief"]
    b = parse_brief(_complete(_prompt(sym, name, metrics, heads), model))
    if b is None:
        why = WHY_OFFLINE if _MODEL_STATE == "offline" else WHY_FALLBACK
        return _enforce({"why_moving": why, "news_driven": False, "meme": bool(metrics.get("meme")),
                         "pump_risk": "low", "caution": CAUTION_DEFAULT}, metrics, heads)
    b = _enforce(b, metrics, heads)
    _BRIEFS[sym] = {"hhash": hhash, "brief": b, "ts": int(time.time()), "model": model or DEFAULT_MODEL}
    return b


def cached_brief(sym):
    """Cached brief for a symbol without touching the model or the news; None if absent."""
    c = _BRIEFS.get(sym)
    return c["brief"] if c else None


def auto_brief(scan_result, model=None):
    """Watch-tick pass: brief new pump events + the top anomalies/trending rows. Cache-gated,
    so steady state costs no model calls when headlines have not changed."""
    rows = scan_result.get("rows", [])
    targets = list(scan_result.get("events", []))
    targets += [r for r in rows if r.get("trending")][:TOP_BRIEF_N]
    targets += rows[:TOP_BRIEF_N]
    seen = set()
    for t in targets:
        sym = t["sym"]
        if sym in seen:
            continue
        seen.add(sym)
        row = scanner.row_for(sym) or t
        brief(sym, row, name=row.get("name"), model=model)


def model_state():
    return _MODEL_STATE
