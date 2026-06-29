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
LEDGER_DIR = os.path.dirname(LEDGER_PATH)
TICKER_URL = "https://api.binance.us/api/v3/ticker/price?symbol={}"
STOCK_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range=1d"
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
# Tradable universes by market. The LLM tags many entities; only names in the universe are
# traded, so noise never becomes a position. A symbol with no live price is silently skipped,
# so a generous list is safe (dead tickers just never trade).
# Barbell, grounded in the sentiment-trading research: liquid crypto majors (where news sentiment
# actually links to price) + the high-retail-attention meme tier (strong signal, size small). Crypto
# breadth is mostly illusory (majors correlate >0.85 in stress), so this stays focused, not sprawling.
TRADABLE = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC", "BCH", "TRX",
            "DOT", "UNI", "ATOM", "NEAR", "APT", "SUI", "XLM", "AAVE",
            "SHIB", "PEPE", "WIF", "BONK", "FLOKI"}
# US equities: highest retail-attention / most news-covered names (where breadth genuinely helps),
# plus SPY/QQQ as macro proxies / broad-stock benchmark.
STOCK_UNIVERSE = {"TSLA", "NVDA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "AMD", "PLTR", "GME",
                  "AMC", "HOOD", "SMCI", "SOUN", "COIN", "MSTR", "SOFI", "NIO", "RIVN", "INTC",
                  "NFLX", "BABA", "SHOP", "F", "BA", "DIS", "PFE", "SPY", "QQQ"}
ET_ZONE = "America/New_York"
MKT_OPEN_MIN = 9 * 60 + 30        # US equities regular session open, 09:30 ET
MKT_CLOSE_MIN = 16 * 60           # close, 16:00 ET

# ------------------------------------------------------------------------------------
# Prices (market-aware) + ledger

def price(symbol, market="crypto"):
    """Live price for a symbol. crypto -> Binance.US spot; stocks -> Yahoo Finance quote. None
    on any failure (caller skips symbols with no price), so an unknown ticker never crashes a tick."""
    if market == "stocks":
        return stock_price(symbol)
    try:
        req = urllib.request.Request(TICKER_URL.format(symbol), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
            return float(json.loads(r.read())["price"])
    except Exception:
        return None


def stock_price(ticker):
    try:
        req = urllib.request.Request(STOCK_URL.format(ticker), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
            d = json.loads(r.read())
        return float(d["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:
        return None


def universe(market="crypto"):
    return STOCK_UNIVERSE if market == "stocks" else TRADABLE


def sym_for(asset, market="crypto"):
    """Trade symbol for an asset: crypto appends the quote (BTC -> BTCUSDT); stocks use the bare ticker."""
    return asset if market == "stocks" else asset + QUOTE


def _is_market_hours(now):
    if now.weekday() >= 5:
        return False
    m = now.hour * 60 + now.minute
    return MKT_OPEN_MIN <= m < MKT_CLOSE_MIN


def market_open(market):
    """True when the market is tradable now. Crypto trades 24/7; stocks only in the US regular
    session, because the free Yahoo quote freezes after hours, so trading then is pure fee loss."""
    if market != "stocks":
        return True
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return _is_market_hours(datetime.now(ZoneInfo(ET_ZONE)))
    except Exception:
        return True


def ledger_for(label):
    """Path to a named run's ledger. 'main' (or empty) -> the default account.json; any other
    label (e.g. an A/B arm 'btc'/'doge') gets its own account_<label>.json, so multiple runs
    never share a ledger."""
    return LEDGER_PATH if (not label or label == "main") else os.path.join(LEDGER_DIR, f"account_{label}.json")


def load_ledger(path=LEDGER_PATH):
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {"cash": START_CASH, "positions": {}, "start_cash": START_CASH, "history": []}


def save_ledger(led, path=LEDGER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(led, f)
    os.replace(tmp, path)


def equity(led, prices):
    eq = led["cash"]
    for sym, qty in led["positions"].items():
        p = prices.get(sym)
        if p:
            eq += qty * p
    return eq

# ------------------------------------------------------------------------------------
# Signal -> targets -> rebalance

def targets(signal, gate, max_size, focus=None, market="crypto", mode="follow"):
    """Per-symbol target long exposure (0..1) from per-asset sentiment. Long-only. `mode`:
    'follow' (momentum) sizes long on POSITIVE sentiment; 'fade' (contrarian) sizes long on
    NEGATIVE sentiment (the research says news sentiment mean-reverts, so fading the spike is the
    other hypothesis worth testing). Only names in the market's universe; focus restricts to one."""
    out = {}
    foc = (focus or "").upper() or None
    uni = universe(market)
    for asset, v in signal.items():
        if asset not in uni:
            continue
        if foc and asset != foc:
            continue
        sig = v["score"] if mode != "fade" else -v["score"]
        if sig >= gate:
            out[sym_for(asset, market)] = min(max_size, max(0.0, sig))
    total = sum(out.values())
    if total > 1.0:                       # long-only: never deploy over 100% of equity (no leverage)
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

def tick(model, provider, gate, max_size, fee, use_chart, source, band, focus=None, ledger_path=LEDGER_PATH,
         market="crypto", mode="follow", risk_off=True):
    items = scraper.scrape(limit=20, focus=focus, market=market)
    scored = sent.score_items(items, provider=provider, model=model, cache=_SCORE_CACHE)
    if len(_SCORE_CACHE) > 2000:
        _SCORE_CACHE.clear()
    signal = sent.aggregate(scored)
    led = load_ledger(ledger_path)
    tgt = targets(signal, gate, max_size, focus, market, mode)
    # Research-backed macro risk-off gate (the one documented drawdown-reducer): when broad MARKET
    # sentiment is strongly negative, cut new long exposure hard (follow mode only; fade wants the dip).
    if risk_off and mode != "fade":
        ms = signal.get("MARKET", {}).get("score", 0.0)
        if ms <= -0.35:
            tgt = {s: f * 0.25 for s, f in tgt.items()}
    if use_chart and market == "crypto":
        tgt = {s: f for s, f in tgt.items() if chart_ok(s, source, model)}
    uni = universe(market)
    track = [a for a in signal if a in uni]
    foc = (focus or "").upper()
    if foc and foc in uni and foc not in track:
        track.append(foc)
    bench_asset = foc if (foc and foc in uni) else ("SPY" if market == "stocks" else "BTC")
    syms = set(tgt) | set(led["positions"]) | {sym_for(a, market) for a in track} | {sym_for(bench_asset, market)}
    prices = {s: price(s, market) for s in syms}
    acts = rebalance(led, tgt, prices, fee, band) if market_open(market) else []
    eq = equity(led, prices)
    led["history"].append({"t": int(time.time()), "equity": round(eq, 2), "market": market,
                           "bench_px": prices.get(sym_for(bench_asset, market)), "bench_asset": bench_asset,
                           "signal": {a: round(signal[a]["score"], 3) for a in track if a in signal},
                           "prices": {a: prices.get(sym_for(a, market)) for a in track if prices.get(sym_for(a, market))},
                           "acts": acts})
    led["history"] = led["history"][-1000:]
    save_ledger(led, ledger_path)
    return {"equity": eq, "pnl": eq - led["start_cash"], "signal": signal, "acts": acts, "n_scored": len(scored)}


def loop(model, provider, gate, max_size, fee, use_chart, source, interval, band, focus=None, ledger_path=LEDGER_PATH, market="crypto"):
    print(f"news_trader [PAPER] model={model or '(configured)'} market={market} focus={focus or 'all'} gate={gate} "
          f"fee={fee*1e4:.0f}bps band={band} scan every {interval}s (trades event-driven) -> {ledger_path}", flush=True)
    while True:
        try:
            r = tick(model, provider, gate, max_size, fee, use_chart, source, band, focus, ledger_path, market)
            pos = ", ".join(f"{a}={v['score']:+.2f}" for a, v in sorted(r["signal"].items())[:6])
            print(f"{time.strftime('%H:%M:%S')} eq=${r['equity']:,.2f} pnl=${r['pnl']:+,.2f} "
                  f"scored={r['n_scored']} acts={len(r['acts'])} | {pos}", flush=True)
        except Exception as e:
            print(f"{time.strftime('%H:%M:%S')} tick error: {type(e).__name__}: {e}", flush=True)
        time.sleep(interval)

# ------------------------------------------------------------------------------------
# Managed thread (UI start/stop from the dashboard)

_RUNS = {}          # label -> {"thread", "stop", "cfg"}; supports concurrent runs on separate ledgers

def start_thread(model, provider, gate, max_size, fee, use_chart, source, interval, band, focus=None,
                 label="main", market="crypto", mode="follow", risk_off=True):
    r = _RUNS.get(label)
    if r and r["thread"] and r["thread"].is_alive():
        return False
    ledger_path = ledger_for(label)
    ev = threading.Event()
    # cfg is the LIVE config: each tick reads from it, so update_run() changes take effect with no restart.
    cfg = {"label": label, "model": model, "provider": provider, "gate": gate, "max_size": max_size,
           "fee": fee, "use_chart": use_chart, "source": source, "interval": interval, "band": band,
           "focus": focus, "market": market, "mode": mode, "risk_off": risk_off,
           "fee_bps": round(fee * 1e4, 1), "started": int(time.time())}
    _RUNS[label] = {"thread": None, "stop": ev, "cfg": cfg}

    def _run():
        while not ev.is_set():
            c = _RUNS.get(label, {}).get("cfg", cfg)
            try:
                tick(c["model"], c["provider"], c["gate"], c["max_size"], c["fee"], c["use_chart"],
                     c["source"], c["band"], c["focus"], ledger_path, c.get("market", "crypto"),
                     c.get("mode", "follow"), c.get("risk_off", True))
            except Exception:
                pass
            ev.wait(max(15, int(c.get("interval", interval))))
    t = threading.Thread(target=_run, name=f"news-trader-{label}", daemon=True)
    _RUNS[label]["thread"] = t
    t.start()
    return True


def update_run(label="main", **changes):
    """Mutate a RUNNING run's live config in place (model, gate, band, interval, focus, market, ...).
    The loop picks up the change on its next tick. No restart, ledger untouched. False if not running."""
    r = _RUNS.get(label)
    if not (r and r["thread"] and r["thread"].is_alive()):
        return False
    cfg = r["cfg"]
    for k in ("model", "provider", "gate", "band", "interval", "focus", "use_chart", "source",
              "max_size", "fee", "market", "mode", "risk_off"):
        if k in changes and (changes[k] is not None or k == "focus"):   # focus may be cleared to None (AUTO)
            cfg[k] = changes[k]
    cfg["fee_bps"] = round(cfg["fee"] * 1e4, 1)
    return True


def stop_thread(label="main"):
    if label is None:                       # stop every run
        for lbl in list(_RUNS):
            stop_thread(lbl)
        return True
    r = _RUNS.get(label)
    if r and r["stop"]:
        r["stop"].set()
    _RUNS.pop(label, None)
    return True


def status(label="main"):
    r = _RUNS.get(label)
    alive = bool(r and r["thread"] and r["thread"].is_alive())
    return {"running": alive, **(r["cfg"] if (alive and r) else {})}


def status_all():
    return {lbl: status(lbl) for lbl in list(_RUNS)}

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
    p.add_argument("--market", default="crypto", choices=["crypto", "stocks"], help="crypto (Binance.US) or stocks (Yahoo)")
    p.add_argument("--label", default="main", help="named run -> its own ledger account_<label>.json")
    p.add_argument("--once", action="store_true", help="run a single tick and exit")
    a = p.parse_args()
    ledger_path = ledger_for(a.label)
    if a.once:
        r = tick(a.model, a.provider, a.gate, a.max_size, a.fee_bps / 1e4, a.use_chart, a.source, a.band, a.focus, ledger_path, a.market)
        print(json.dumps({"equity": r["equity"], "pnl": r["pnl"], "acts": r["acts"],
                          "signal": {k: round(v["score"], 3) for k, v in r["signal"].items()}}, indent=2))
        return
    loop(a.model, a.provider, a.gate, a.max_size, a.fee_bps / 1e4, a.use_chart, a.source, a.interval, a.band, a.focus, ledger_path, a.market)


if __name__ == "__main__":
    main()
