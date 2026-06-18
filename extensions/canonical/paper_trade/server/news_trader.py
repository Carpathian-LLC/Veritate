# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Autonomous news-sentiment paper trader (the "script"). Each tick: scrape free crypto
#   news (scraper) -> score with a user-added model via /teacher/complete (sentiment) ->
#   per-asset time-decayed sentiment -> long-only target exposure -> rebalance a SIMULATED
#   paper ledger marked to live prices -> persist + log. Runs forward on its own; this is
#   the only honest way to validate LLM-news trading (no trustworthy historical backtest).
# - FAKE money only: a JSON ledger, no broker, no keys. Robinhood-shaped (long-or-cash,
#   spread as a fee). Sentiment drives DIRECTION (the chart model is a coin flip there);
#   an optional chart-model gate (--use_chart) vetoes tiny-move bars via /market/paper_decide.
# - Run: python extensions/canonical/market/server/../paper_trade/server/news_trader.py --model qwen2.5:7b-instruct
# extensions/canonical/paper_trade/server/news_trader.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import ssl
import threading
import time
import urllib.request

import certifi

import scraper
import sentiment as sent

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
LEDGER_PATH = os.path.join(ROOT, "extensions", "installed", "paper_trade", "data", "account.json")
TICKER_URL = "https://api.binance.us/api/v3/ticker/price?symbol={}"
DECIDE_URL = "http://127.0.0.1:8001/market/paper_decide"
CTX = ssl.create_default_context(cafile=certifi.where())
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
TIMEOUT = 12
QUOTE = "USDT"
START_CASH = 10000.0
SENT_GATE = 0.25          # act only on sentiment above this
FEE = 0.0020              # round-trip spread, fraction
MAX_SIZE = 1.0
REBAL_BAND = 0.12         # event-driven: only trade when target exposure shifts > this (of equity)
INTERVAL = 300            # SCAN news every 5 min; trading is event-driven, not on this clock
# Tradable universe: liquid coins on the price feed. The LLM tags many entities (MARKET,
# COINBASE, ALIBABA, ...); only these are traded/charted, so noise never becomes a position.
TRADABLE = {"BTC", "ETH", "SOL", "DOGE", "XRP", "LINK", "AVAX", "ADA", "LTC", "BCH", "DOT",
            "ATOM", "NEAR", "UNI", "XLM", "AAVE", "SHIB", "PEPE", "ARB", "OP", "APT", "SUI", "HYPE"}

# ------------------------------------------------------------------------------------
# Prices + ledger

def price(symbol):
    try:
        req = urllib.request.Request(TICKER_URL.format(symbol), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
            return float(json.loads(r.read())["price"])
    except Exception:
        return None


def load_ledger():
    if os.path.isfile(LEDGER_PATH):
        try:
            with open(LEDGER_PATH) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {"cash": START_CASH, "positions": {}, "start_cash": START_CASH, "history": []}


def save_ledger(led):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    tmp = LEDGER_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(led, f)
    os.replace(tmp, LEDGER_PATH)


def equity(led, prices):
    eq = led["cash"]
    for sym, qty in led["positions"].items():
        p = prices.get(sym)
        if p:
            eq += qty * p
    return eq

# ------------------------------------------------------------------------------------
# Signal -> targets -> rebalance

def targets(signal, gate, max_size, focus=None):
    """Per-symbol target long exposure (0..1) from per-asset sentiment. Long-only: positive
    sentiment above the gate -> sized long; otherwise flat. MARKET/unknown assets skipped.
    When `focus` is a ticker, only that coin is eligible (single-token mode)."""
    out = {}
    foc = (focus or "").upper() or None
    for asset, v in signal.items():
        if asset not in TRADABLE:
            continue
        if foc and asset != foc:
            continue
        s = v["score"]
        if s >= gate:
            out[asset + QUOTE] = min(max_size, max(0.0, s))
    total = sum(out.values())
    if total > 1.0:                       # long-only spot: never deploy over 100% of equity (no leverage)
        out = {k: w / total for k, w in out.items()}
    return out


def chart_ok(symbol, source, model):
    """Optional chart-model gate: skip when the byte model sees no tradable move."""
    try:
        q = f"model={model}&symbol={symbol}&source={source}&mode=vol_harvest&move_gate=1.0"
        req = urllib.request.Request(DECIDE_URL + "?" + q, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
            d = json.loads(r.read())
        return bool(d.get("ok") and d.get("decision", {}).get("act"))
    except Exception:
        return True            # gate is advisory; never block on its failure


def rebalance(led, tgt, prices, fee, band):
    """Event-driven, fee-aware: only move a position when its target exposure diverges from
    the current by more than `band` (fraction of equity). Small sentiment drift -> no trade
    (no fee churn); a real news shift -> rebalance. Untargeted holds (sentiment fell off) exit
    once the position exceeds the band. This is the cost-aware filter that makes ML signals
    tradable; it trades MORE when news breaks and nothing when quiet."""
    eq = equity(led, prices)
    acts = []
    for sym in set(tgt) | set(led["positions"]):
        p = prices.get(sym)
        if not p:
            continue
        frac = tgt.get(sym, 0.0)
        cur = led["positions"].get(sym, 0.0)
        cur_frac = (cur * p / eq) if eq > 0 else 0.0
        if abs(frac - cur_frac) < band:
            continue
        want_qty = frac * eq / p
        dqty = want_qty - cur
        led["cash"] -= dqty * p + abs(dqty * p) * fee
        led["positions"][sym] = want_qty
        acts.append({"sym": sym, "side": "buy" if dqty > 0 else "sell",
                     "qty": round(dqty, 8), "price": p})
    return acts

# ------------------------------------------------------------------------------------
# One tick + loop

_SCORE_CACHE = {}        # title -> parsed score; only NEW headlines hit the model each scan

def tick(model, provider, gate, max_size, fee, use_chart, source, band, focus=None):
    items = scraper.scrape(limit=20, focus=focus)
    scored = sent.score_items(items, provider=provider, model=model, cache=_SCORE_CACHE)
    if len(_SCORE_CACHE) > 2000:
        _SCORE_CACHE.clear()
    signal = sent.aggregate(scored)
    led = load_ledger()
    tgt = targets(signal, gate, max_size, focus)
    if use_chart:
        tgt = {s: f for s, f in tgt.items() if chart_ok(s, source, model)}
    track = [a for a in signal if a in TRADABLE]
    foc = (focus or "").upper()
    if foc and foc in TRADABLE and foc not in track:
        track.append(foc)
    syms = set(tgt) | set(led["positions"]) | {a + QUOTE for a in track} | {"BTC" + QUOTE}
    prices = {s: price(s) for s in syms}
    acts = rebalance(led, tgt, prices, fee, band)
    eq = equity(led, prices)
    led["history"].append({"t": int(time.time()), "equity": round(eq, 2),
                           "btc": prices.get("BTC" + QUOTE),
                           "signal": {a: round(signal[a]["score"], 3) for a in track if a in signal},
                           "prices": {a: prices.get(a + QUOTE) for a in track if prices.get(a + QUOTE)},
                           "acts": acts})
    led["history"] = led["history"][-1000:]
    save_ledger(led)
    return {"equity": eq, "pnl": eq - led["start_cash"], "signal": signal, "acts": acts, "n_scored": len(scored)}


def loop(model, provider, gate, max_size, fee, use_chart, source, interval, band, focus=None):
    print(f"news_trader [PAPER] model={model or '(configured)'} focus={focus or 'all'} gate={gate} "
          f"fee={fee*1e4:.0f}bps band={band} scan every {interval}s (trades event-driven) -> {LEDGER_PATH}", flush=True)
    while True:
        try:
            r = tick(model, provider, gate, max_size, fee, use_chart, source, band, focus)
            pos = ", ".join(f"{a}={v['score']:+.2f}" for a, v in sorted(r["signal"].items())[:6])
            print(f"{time.strftime('%H:%M:%S')} eq=${r['equity']:,.2f} pnl=${r['pnl']:+,.2f} "
                  f"scored={r['n_scored']} acts={len(r['acts'])} | {pos}", flush=True)
        except Exception as e:
            print(f"{time.strftime('%H:%M:%S')} tick error: {type(e).__name__}: {e}", flush=True)
        time.sleep(interval)

# ------------------------------------------------------------------------------------
# Managed thread (UI start/stop from the dashboard)

_RUN = {"thread": None, "stop": None, "cfg": {}}

def start_thread(model, provider, gate, max_size, fee, use_chart, source, interval, band, focus=None):
    if _RUN["thread"] and _RUN["thread"].is_alive():
        return False
    ev = threading.Event()
    _RUN["stop"] = ev
    _RUN["cfg"] = {"model": model, "gate": gate, "interval": interval, "band": band, "focus": focus or "all",
                   "source": source, "fee_bps": round(fee * 1e4, 1), "started": int(time.time())}

    def _run():
        while not ev.is_set():
            try:
                tick(model, provider, gate, max_size, fee, use_chart, source, band, focus)
            except Exception:
                pass
            ev.wait(interval)
    t = threading.Thread(target=_run, name="news-trader", daemon=True)
    _RUN["thread"] = t
    t.start()
    return True


def stop_thread():
    if _RUN["stop"]:
        _RUN["stop"].set()
    _RUN["thread"] = None
    return True


def status():
    alive = bool(_RUN["thread"] and _RUN["thread"].is_alive())
    return {"running": alive, **(_RUN.get("cfg", {}) if alive else {})}

# ------------------------------------------------------------------------------------
# CLI

def main():
    p = argparse.ArgumentParser(description="Autonomous news-sentiment paper trader (fake money).")
    p.add_argument("--model", default="qwen2.5:7b-instruct", help="scorer model (Ollama)")
    p.add_argument("--provider", default=None)
    p.add_argument("--gate", type=float, default=SENT_GATE)
    p.add_argument("--max_size", type=float, default=MAX_SIZE)
    p.add_argument("--fee_bps", type=float, default=FEE * 1e4)
    p.add_argument("--source", default="crypto_of")
    p.add_argument("--use_chart", action="store_true", help="gate entries with the byte model (/market/paper_decide)")
    p.add_argument("--interval", type=int, default=INTERVAL, help="seconds between news scans")
    p.add_argument("--band", type=float, default=REBAL_BAND, help="event-driven rebalance deadband (of equity)")
    p.add_argument("--focus", default=None, help="trade only this ticker (e.g. SOL); also focuses the news pull")
    p.add_argument("--once", action="store_true", help="run a single tick and exit")
    a = p.parse_args()
    if a.once:
        r = tick(a.model, a.provider, a.gate, a.max_size, a.fee_bps / 1e4, a.use_chart, a.source, a.band, a.focus)
        print(json.dumps({"equity": r["equity"], "pnl": r["pnl"], "acts": r["acts"],
                          "signal": {k: round(v["score"], 3) for k, v in r["signal"].items()}}, indent=2))
        return
    loop(a.model, a.provider, a.gate, a.max_size, a.fee_bps / 1e4, a.use_chart, a.source, a.interval, a.band, a.focus)


if __name__ == "__main__":
    main()
