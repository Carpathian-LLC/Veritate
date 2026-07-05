# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Weekly cross-sectional momentum forward paper trader (fake money, long/short). The one
#   strategy net-positive across every out-of-sample year at real fees in the 2026-07
#   research campaign (developer_documentation/market/trading_model_plan.md sections 7+9).
#   Two arms: "base" (always invested) and "gated" (flat when BTC 30d vol >= its trailing
#   365d median). The gate cleared the backtest bar (test Sharpe 0.93) but was
#   test-adjudicated, so both arms run forward and the live record decides.
# - Each hourly tick marks the ledger to live OKX prices; on a new ISO week it re-ranks
#   trailing 7d returns across the majors universe and rebalances to long top-8 / short
#   bottom-8 at 3.5bp/side. No funding credit, no cash yield: both omissions conservative
#   (backtest funding was a +1.85%/yr tailwind to this book).
# - FAKE money: JSON ledger, no broker, no keys. Prices are OKX public (US-reachable).
#   Run standalone: python extensions/canonical/trading/server/xsmom_trader.py --arm both
# extensions/canonical/trading/server/xsmom_trader.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import math
import ssl
import statistics
import threading
import time
import urllib.request
from datetime import datetime, timezone

import certifi

import news_trader as nt

# ------------------------------------------------------------------------------------
# Constants

OKX_CANDLES = "https://www.okx.com/api/v5/market/candles?instId={}-USDT&bar=1D&limit={}"
OKX_HIST = "https://www.okx.com/api/v5/market/history-candles?instId={}-USDT&bar=1D&after={}&limit=100"
OKX_TICKER = "https://www.okx.com/api/v5/market/ticker?instId={}-USDT"
CTX = ssl.create_default_context(cafile=certifi.where())
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
TIMEOUT = 12
UNIVERSE = ["AAVE", "ADA", "ALGO", "APT", "ARB", "ATOM", "AVAX", "AXS", "BCH", "BNB",
            "BTC", "CHZ", "CRV", "DOGE", "DOT", "EGLD", "ENJ", "ETC", "ETH", "FIL",
            "FTM", "GALA", "GRT", "HBAR", "ICP", "INJ", "LINK", "LTC", "MKR", "NEAR",
            "OP", "RUNE", "SAND", "SOL", "SUI", "THETA", "TRX", "UNI", "XLM", "XRP"]
ARMS = ("base", "gated")
LOOKBACK_D = 7            # trailing return window for the ranking, days
N_SIDE = 8                # names per leg (quintile-ish of the fetchable universe)
GROSS = 1.0               # total gross exposure; half long, half short
FEE = 0.00035             # 3.5bp per side, the primary fee point of the backtests
VOL_WIN = 30              # BTC realized-vol window, days
VOL_MED_WIN = 365         # trailing window for the vol median, days
VOL_GATE_MIN = 90         # observations required before the gate may flatten
MIN_TRADE_USD = 1.0       # dust filter
TICK_SEC = 3600           # mark hourly; trading is weekly
HISTORY_CAP = 5000

# ------------------------------------------------------------------------------------
# Functions

def _get(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=CTX) as r:
            d = json.loads(r.read())
        return d["data"] if d.get("code") == "0" else None
    except Exception:
        return None


def daily_closes(stem, n):
    """Oldest-first closed daily closes for a pair, paging back through history-candles
    until n bars. None when the pair is unfetchable (delisted names just never trade)."""
    rows = _get(OKX_CANDLES.format(stem, min(300, n + 1)))
    if not rows:
        return None
    rows = [r for r in rows if r[8] == "1"]
    while 0 < len(rows) < n:
        older = _get(OKX_HIST.format(stem, rows[-1][0], ))
        older = [r for r in older if r[8] == "1"] if older else []
        if not older:
            break
        rows += older
    rows = rows[:n]
    rows.reverse()
    return [float(r[4]) for r in rows]


def mark_price(stem):
    rows = _get(OKX_TICKER.format(stem))
    return float(rows[0]["last"]) if rows else None


def sym_for(stem):
    return stem + nt.QUOTE


def vol_gate_invested(closes):
    """True when BTC 30d annualized vol sits below its trailing 365d median (calm-regime
    gate, thresholds fixed from the backtest). Under VOL_GATE_MIN observations -> invested."""
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    vols = [statistics.stdev(rets[i - VOL_WIN:i]) * math.sqrt(365.0)
            for i in range(VOL_WIN, len(rets) + 1)]
    vols = vols[-(VOL_MED_WIN + 1):]
    if len(vols) < VOL_GATE_MIN:
        return True
    return vols[-1] < statistics.median(vols[:-1])


def targets(rets, invested):
    """Signed target weights: +GROSS/2/N_SIDE on the N_SIDE best trailing returns, the same
    negative on the N_SIDE worst. Empty dict when the vol gate says flat."""
    if not invested or len(rets) < 2 * N_SIDE:
        return {}
    ranked = sorted(rets, key=rets.get)
    w = GROSS / 2.0 / N_SIDE
    out = {sym_for(s): -w for s in ranked[:N_SIDE]}
    out.update({sym_for(s): w for s in ranked[-N_SIDE:]})
    return out


def rebalance_to(led, tgt, prices, fee):
    """Full weekly rebalance to signed target weights. Fee per side on traded notional.
    Short opens add cash (proceeds held; no margin interest modeled)."""
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
        led["cash"] -= dqty * p + abs(dqty * p) * fee
        if want:
            led["positions"][sym] = want
        else:
            led["positions"].pop(sym, None)
        acts.append({"sym": sym, "side": "buy" if dqty > 0 else "sell",
                     "qty": round(dqty, 8), "price": p})
    return acts


def _iso_week_now():
    y, w, _ = datetime.now(timezone.utc).isocalendar()
    return f"{y}-W{w:02d}"


def tick(arm, ledger_path):
    """One cycle: on a new ISO week, rank + rebalance (gate checked at rebalance time only,
    matching the backtest cadence); every cycle, mark equity to live prices and append history."""
    led = nt.load_ledger(ledger_path)
    week = _iso_week_now()
    acts = []
    invested = led.get("invested", True)
    if led.get("week") != week:
        if arm == "gated":
            btc = daily_closes("BTC", VOL_MED_WIN + VOL_WIN + 5)
            invested = vol_gate_invested(btc) if btc else invested
        else:
            invested = True
        rets = {}
        for stem in UNIVERSE:
            c = daily_closes(stem, LOOKBACK_D + 2)
            if c and len(c) > LOOKBACK_D and c[-LOOKBACK_D - 1] > 0:
                rets[stem] = c[-1] / c[-LOOKBACK_D - 1] - 1.0
        if invested and len(rets) < 2 * N_SIDE:
            acts = []                      # feed failure: hold, retry next tick, week not stamped
        else:
            tgt = targets(rets, invested)
            syms = set(tgt) | set(led["positions"])
            prices = {s: mark_price(s[:-len(nt.QUOTE)]) for s in syms}
            acts = rebalance_to(led, tgt, prices, FEE)
            led["week"] = week
            led["invested"] = invested
    last_px = led.setdefault("last_px", {})
    for sym in led["positions"]:
        p = mark_price(sym[:-len(nt.QUOTE)])
        if p:
            last_px[sym] = p
    marks = {s: last_px.get(s) for s in led["positions"]}
    eq = nt.equity(led, marks)
    btc_px = mark_price("BTC")
    led["history"].append({"t": int(time.time()), "equity": round(eq, 2), "market": "crypto",
                           "bench_px": btc_px, "bench_asset": "BTC", "invested": led.get("invested", True),
                           "prices": {s[:-len(nt.QUOTE)]: p for s, p in marks.items() if p},
                           "acts": acts})
    led["history"] = led["history"][-HISTORY_CAP:]
    nt.save_ledger(led, ledger_path)
    return {"equity": eq, "pnl": eq - led["start_cash"], "acts": acts, "invested": led.get("invested", True)}

# ------------------------------------------------------------------------------------
# Managed threads (dashboard start/stop) + CLI

_RUNS = {}


def label_for(arm):
    return "xsmom_" + arm


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
    t = threading.Thread(target=_run, name=f"xsmom-{arm}", daemon=True)
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
    p = argparse.ArgumentParser(description="Weekly XS momentum L/S paper trader (fake money).")
    p.add_argument("--arm", default="both", choices=["base", "gated", "both"])
    p.add_argument("--once", action="store_true", help="one tick per requested arm, then exit")
    a = p.parse_args()
    arms = list(ARMS) if a.arm == "both" else [a.arm]
    if a.once:
        for arm in arms:
            r = tick(arm, nt.ledger_for(label_for(arm)))
            print(json.dumps({"arm": arm, **{k: r[k] for k in ("equity", "pnl", "invested")},
                              "acts": r["acts"]}, indent=2))
        return
    for arm in arms:
        start_thread(arm)
    print(f"xsmom_trader [PAPER] arms={arms} fee={FEE * 1e4:.1f}bp/side tick={TICK_SEC}s "
          f"rebalance=weekly -> {[nt.ledger_for(label_for(x)) for x in arms]}", flush=True)
    while True:
        time.sleep(TICK_SEC)
        s = {a_: v.get("running") for a_, v in status_all().items() if a_ in arms}
        eqs = {a_: nt.load_ledger(nt.ledger_for(label_for(a_))).get("history", [])[-1:] for a_ in arms}
        line = " ".join(f"{a_}: ${h[0]['equity']:,.2f}" for a_, h in eqs.items() if h)
        print(f"{time.strftime('%H:%M:%S')} alive={s} {line}", flush=True)


if __name__ == "__main__":
    main()
