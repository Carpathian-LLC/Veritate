# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Autonomous paper-trading loop: each tick pulls the latest CLOSED bar, runs the byte
#   model forecast (veritate), turns it into a trade decision (policy), and drives the
#   broker position toward it (execution). Standalone CLI, like recorder.py.
# - SAFETY: paper by default (Alpaca paper endpoint, simulated money). Live requires
#   --live AND real keys. dry_run logs orders without sending. A run bails after
#   MAX_ERRORS consecutive failures. This is the test rig; it does not create an edge.
# - Run: python extensions/canonical/market/server/autotrader.py --model <name> --symbol BTCUSDT
# extensions/canonical/market/server/autotrader.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import time

import execution as ex
import live as lv
import policy as pol
import veritate as vz

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_INTERVAL = 60          # seconds between ticks
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_SOURCE = "crypto_of"
BASE = "1m"
LIMIT = 400
MAX_ERRORS = 5
MIN_BARS = 60

# ------------------------------------------------------------------------------------
# One tick

def tick(symbol, bundle, broker, overrides):
    """One decision+execution cycle on the latest closed bar. Pure orchestration of
    already-tested pieces; returns a record for logging."""
    model, seq, stride = bundle
    df, _ = lv.fetch(symbol, base=BASE, limit=LIMIT)
    if df is None or len(df) < MIN_BARS:
        return {"ok": False, "error": "no live data"}
    closed = df.iloc[:-1]                                  # closed-bar rule: never act on the forming candle
    pred = vz.predict_next(model, seq, closed, stride)
    if pred is None:
        return {"ok": False, "error": "forecast failed"}
    premium = vz.trailing_premium(closed)
    decision = pol.decide(pred["p_up"], pred["confidence"], pred["expected_move"],
                          pred["vol"], premium=premium, **overrides)
    price = float(closed["close"].iloc[-1])
    result = ex.rebalance(broker, symbol, decision, price)
    return {"ok": True, "price": price, "p_up": pred["p_up"],
            "exp_move_bps": pred["expected_move"] * 1e4, "premium_bps": premium * 1e4,
            "decision": decision, "exec": result}

# ------------------------------------------------------------------------------------
# Loop

def loop(symbol, source, model_name, broker, overrides, interval):
    model, seq, step, stride = vz.load_model(model_name)
    if model is None:
        print(f"no checkpoint for model {model_name!r}", flush=True)
        return
    bundle = (model, seq, stride)
    mode = "DRY-RUN" if broker.dry_run else ("PAPER" if broker.paper else "LIVE")
    print(f"autotrader [{mode}] {model_name} step {step} stride {stride} -> {symbol} every {interval}s", flush=True)
    errors = 0
    while True:
        try:
            r = tick(symbol, bundle, broker, overrides)
            if r.get("ok"):
                errors = 0
                d = r["decision"]
                act = (d.get("side") or "flat").upper() if d.get("act") else "FLAT"
                print(f"{_stamp()} {symbol} {r['price']:.4f} p_up={r['p_up']:.3f} "
                      f"move={r['exp_move_bps']:.1f}bps prem={r['premium_bps']:.1f}bps "
                      f"-> {act} | {r['exec']['action']} {r['exec'].get('qty','')}", flush=True)
            else:
                errors += 1
                print(f"{_stamp()} tick error ({errors}/{MAX_ERRORS}): {r.get('error')}", flush=True)
                if errors >= MAX_ERRORS:
                    print("too many errors, stopping.", flush=True)
                    return
        except Exception as e:
            errors += 1
            print(f"{_stamp()} exception ({errors}/{MAX_ERRORS}): {type(e).__name__}: {e}", flush=True)
            if errors >= MAX_ERRORS:
                return
        time.sleep(interval)


def _stamp():
    return time.strftime("%H:%M:%S")

# ------------------------------------------------------------------------------------
# CLI

def main():
    p = argparse.ArgumentParser(description="Autonomous paper trader (Alpaca paper by default).")
    p.add_argument("--model", required=True)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    p.add_argument("--mode", default="vol_harvest", choices=["vol_harvest", "directional"])
    p.add_argument("--fee_bps", type=float, default=5.0)
    p.add_argument("--conf_gate", type=float, default=0.0)
    p.add_argument("--move_gate", type=float, default=1.3)
    p.add_argument("--sizing", default="confidence", choices=["confidence", "fixed", "vol_target"])
    p.add_argument("--live", action="store_true", help="REAL money (requires real keys). Default is paper.")
    p.add_argument("--dry_run", action="store_true", help="log orders without sending")
    a = p.parse_args()
    overrides = {"mode": a.mode, "fee": a.fee_bps / 1e4, "conf_gate": a.conf_gate,
                 "move_gate": a.move_gate, "sizing": a.sizing}
    broker = ex.Broker(paper=not a.live, dry_run=a.dry_run)
    loop(a.symbol, a.source, a.model, broker, overrides, a.interval)


if __name__ == "__main__":
    main()
