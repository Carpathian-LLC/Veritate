# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Monthly US equity momentum forward paper trader (fake money, long-only). Canonical
#   12-1 cross-sectional momentum passed the backtest bar (+5.6pp/yr over a survivorship-
#   matched null; trading_model_plan.md section 13) but free historical data cannot remove
#   survivorship, so the universe was FROZEN 2026-07-03 (data/equity_universe.json) and the
#   claim is validated forward from here.
# - First trading day of each month (US regular session): rank 12-1 momentum (return from
#   t-252 to t-21 trading days, Yahoo daily adjclose), hold the top decile equal weight,
#   3bp/side. Hourly ticks mark to live quotes; SPY is the benchmark on every history row.
# - FAKE money: JSON ledger, no broker, no keys.
#   Run standalone: python extensions/canonical/trading/server/eqmom_trader.py
# extensions/canonical/trading/server/eqmom_trader.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone

import news_trader as nt

# ------------------------------------------------------------------------------------
# Constants

UNIVERSE_PATH = os.path.join(nt.LEDGER_DIR, "equity_universe.json")
HIST_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range=2y"
ARMS = ("main",)
BENCH = "SPY"
LOOKBACK_D = 252          # momentum window start, trading days back
SKIP_D = 21               # skip the most recent month (12-1 spec)
TOP_FRAC = 0.10           # hold the top decile
FEE = 0.0003              # 3bp per side
MIN_TRADE_USD = 1.0
FETCH_PAUSE = 0.05        # politeness between Yahoo history calls
TICK_SEC = 3600
HISTORY_CAP = 5000

# ------------------------------------------------------------------------------------
# Functions

def universe():
    try:
        with open(UNIVERSE_PATH) as f:
            return json.load(f)["tickers"]
    except (OSError, ValueError, KeyError):
        return []


def daily_adjcloses(ticker):
    """Two years of daily adjusted closes, oldest first. None on any failure (name skipped)."""
    try:
        req = urllib.request.Request(HIST_URL.format(ticker), headers={"User-Agent": nt.UA})
        with urllib.request.urlopen(req, timeout=nt.TIMEOUT, context=nt.CTX) as r:
            d = json.loads(r.read())
        res = d["chart"]["result"][0]
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        return [x for x in adj if x]
    except Exception:
        return None


def momentum_12_1(closes):
    """12-1 momentum: return from t-252 to t-21 trading days. None when history is short."""
    if not closes or len(closes) < LOOKBACK_D + 1 or closes[-LOOKBACK_D - 1] <= 0:
        return None
    return closes[-SKIP_D - 1] / closes[-LOOKBACK_D - 1] - 1.0


def targets(moms):
    """Equal-weight long-only top decile of the ranked momentum dict."""
    n = max(1, int(len(moms) * TOP_FRAC))
    top = sorted(moms, key=moms.get, reverse=True)[:n]
    w = 1.0 / n
    return {t: w for t in top}


def _month_now():
    d = datetime.now(timezone.utc)
    return f"{d.year}-{d.month:02d}"


def tick(arm, ledger_path):
    """One cycle: on a new month during the US session, re-rank and rebalance; every cycle,
    mark held names + SPY and append an _exp_view-compatible history row."""
    led = nt.load_ledger(ledger_path)
    month = _month_now()
    acts = []
    if led.get("month") != month and nt.market_open("stocks"):
        moms = {}
        for t in universe():
            m = momentum_12_1(daily_adjcloses(t))
            if m is not None:
                moms[t] = m
            time.sleep(FETCH_PAUSE)
        if len(moms) >= 100:
            tgt = targets(moms)
            syms = set(tgt) | set(led["positions"])
            prices = {s: nt.stock_price(s) for s in syms}
            acts = _rebalance_to(led, tgt, prices)
            led["month"] = month
    last_px = led.setdefault("last_px", {})
    for sym in led["positions"]:
        p = nt.stock_price(sym)
        if p:
            last_px[sym] = p
    marks = {s: last_px.get(s) for s in led["positions"]}
    eq = nt.equity(led, marks)
    led["history"].append({"t": int(time.time()), "equity": round(eq, 2), "market": "stocks",
                           "bench_px": nt.stock_price(BENCH), "bench_asset": BENCH,
                           "prices": {s: p for s, p in marks.items() if p}, "acts": acts})
    led["history"] = led["history"][-HISTORY_CAP:]
    nt.save_ledger(led, ledger_path)
    return {"equity": eq, "pnl": eq - led["start_cash"], "acts": acts}


def _rebalance_to(led, tgt, prices):
    eq = nt.equity(led, prices)
    acts = []
    for sym in set(tgt) | set(led["positions"]):
        p = prices.get(sym)
        if not p:
            continue
        want = tgt.get(sym, 0.0) * eq / p
        dqty = want - led["positions"].get(sym, 0.0)
        if abs(dqty * p) < MIN_TRADE_USD:
            continue
        led["cash"] -= dqty * p + abs(dqty * p) * FEE
        if want:
            led["positions"][sym] = want
        else:
            led["positions"].pop(sym, None)
        acts.append({"sym": sym, "side": "buy" if dqty > 0 else "sell",
                     "qty": round(dqty, 6), "price": p})
    return acts

# ------------------------------------------------------------------------------------
# Managed threads + resume + CLI

_RUNS = {}


def label_for(arm):
    return "eqmom_" + arm


def start_thread(arm):
    if arm not in ARMS:
        return False
    label = label_for(arm)
    r = _RUNS.get(label)
    if r and r["thread"] and r["thread"].is_alive():
        return False
    path = nt.ledger_for(label)
    led = nt.load_ledger(path)
    led["auto"] = True
    nt.save_ledger(led, path)
    ev = threading.Event()
    cfg = {"label": label, "arm": arm, "fee_bps": round(FEE * 1e4, 1), "interval": TICK_SEC,
           "started": int(time.time())}
    _RUNS[label] = {"thread": None, "stop": ev, "cfg": cfg}

    def _run():
        while not ev.is_set():
            try:
                tick(arm, path)
            except Exception:
                pass
            ev.wait(TICK_SEC)
    t = threading.Thread(target=_run, name=f"eqmom-{arm}", daemon=True)
    _RUNS[label]["thread"] = t
    t.start()
    return True


def stop_thread(arm=None):
    for label in ([label_for(arm)] if arm else list(_RUNS)):
        r = _RUNS.get(label)
        if r and r["stop"]:
            r["stop"].set()
        _RUNS.pop(label, None)
        path = nt.ledger_for(label)
        led = nt.load_ledger(path)
        led["auto"] = False
        nt.save_ledger(led, path)
    return True


def resume():
    """Restart arms flagged running in their ledger (server reboot must not gap the record)."""
    return {a: start_thread(a) for a in ARMS if nt.load_ledger(nt.ledger_for(label_for(a))).get("auto")}


def status_all():
    out = {}
    for arm in ARMS:
        r = _RUNS.get(label_for(arm))
        alive = bool(r and r["thread"] and r["thread"].is_alive())
        out[arm] = {"running": alive, **(r["cfg"] if (alive and r) else {})}
    return out


def main():
    p = argparse.ArgumentParser(description="Monthly 12-1 equity momentum paper trader (fake money).")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    if a.once:
        r = tick("main", nt.ledger_for(label_for("main")))
        print(json.dumps({"equity": r["equity"], "pnl": r["pnl"], "acts": r["acts"][:12]}, indent=2))
        return
    start_thread("main")
    print(f"eqmom_trader [PAPER] 12-1 top-decile monthly, fee={FEE * 1e4:.0f}bp/side -> "
          f"{nt.ledger_for(label_for('main'))}", flush=True)
    while True:
        time.sleep(TICK_SEC)


if __name__ == "__main__":
    main()
