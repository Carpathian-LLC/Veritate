# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Entry point for the Paper Trading extension's server routes. register(app) adds the
#   /ext/paper_trade/* API: a live sentiment feed (scrape free crypto news -> score with a
#   user-added model via /teacher/complete -> per-asset time-decayed signal). The trade
#   forecasts/backtest still come from the Market LLM extension's /market/* API; this server
#   owns only the sentiment side. Server modules import by bare name (server/ is on sys.path
#   at register time); route bodies import lazily so a missing dep never breaks startup.
# extensions/canonical/paper_trade/server/register.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import time

from flask import request

# ------------------------------------------------------------------------------------
# Constants

N_DEFAULT = 10
N_MAX = 30
AB_ARMS = (("BTC", "btc"), ("DOGE", "doge"))   # head-to-head: each arm trades one coin, own ledger
AB_BAND = 0.25                                  # wider deadband: trade only on big shifts, not a clock
EXP_MAX_ARMS = 10                               # A/B/C/D... cap on parallel experiment arms

# ------------------------------------------------------------------------------------
# Functions

def _safe(fn):
    try:
        return fn()
    except Exception as e:
        from runtime import logs as logmod
        msg = f"{type(e).__name__}: {e}"
        logmod.error("paper_trade", msg)
        return ({"ok": False, "error": msg}, 500)


def _holdings(led, eq, last_prices):
    """Per-asset holdings marked to the last tick price: qty, price, $ value, % of account.
    Positions are keyed by symbol (BTCUSDT); history prices are keyed by asset (BTC)."""
    out = []
    for sym, qty in led.get("positions", {}).items():
        if not qty:
            continue
        asset = sym[:-4] if sym.endswith("USDT") else sym
        px = (last_prices or {}).get(asset)
        val = round(qty * px, 2) if px else None
        out.append({"sym": sym, "asset": asset, "qty": round(qty, 8), "price": px, "value": val,
                    "weight": (round(val / eq, 4) if (val and eq) else None)})
    return sorted(out, key=lambda h: (h["value"] or 0), reverse=True)


def _ab_view(led, coin):
    """Account view for one A/B arm, benchmarked against HOLDING that arm's coin (not BTC). The
    coin's price each tick is in history `prices[coin]`; BTC also has a dedicated `btc` field."""
    hist = led.get("history", [])
    eq = hist[-1]["equity"] if hist else led.get("start_cash", 0.0)
    last = hist[-1] if hist else {}
    start_cash = led.get("start_cash", 0.0)
    tail = hist[-300:]

    def _coin_px(h):
        return (h.get("prices", {}) or {}).get(coin) or (h.get("btc") if coin == "BTC" else None)
    p0 = next((p for p in (_coin_px(h) for h in tail) if p), None)
    bench, last_p = [], None
    for h in tail:
        last_p = _coin_px(h) or last_p
        v = round(start_cash * last_p / p0, 2) if (p0 and last_p) else None
        bench.append({"t": h["t"], "equity": v})
    last_prices = (last.get("prices") or {})
    return {"coin": coin, "start_cash": start_cash, "equity": eq, "pnl": round(eq - start_cash, 2),
            "cash": led.get("cash", 0.0), "cash_weight": round(led.get("cash", 0.0) / eq, 4) if eq else None,
            "positions": {k: round(v, 6) for k, v in led.get("positions", {}).items() if v},
            "holdings": _holdings(led, eq, last_prices),
            "curve": [{"t": h["t"], "equity": h["equity"]} for h in tail], "bench": bench,
            "recent": [{"t": h["t"], "equity": h["equity"], "acts": h.get("acts", [])} for h in reversed(hist[-10:])],
            "series": [{"t": h["t"], "signal": h.get("signal", {}), "prices": h.get("prices", {})}
                       for h in hist[-240:]]}


def _exp_view(led):
    """Generic per-arm account view (any market). Benchmarked against HOLDING the arm's benchmark
    asset, whose price is stored each tick as `bench_px` (BTC for broad crypto, SPY for broad
    stocks, or the focused ticker). Falls back to the legacy `btc` field for old ledgers."""
    hist = led.get("history", [])
    eq = hist[-1]["equity"] if hist else led.get("start_cash", 0.0)
    last = hist[-1] if hist else {}
    start_cash = led.get("start_cash", 0.0)
    tail = hist[-300:]

    def _bpx(h):
        return h.get("bench_px") or h.get("btc")
    p0 = next((p for p in (_bpx(h) for h in tail) if p), None)
    bench, last_p = [], None
    for h in tail:
        last_p = _bpx(h) or last_p
        v = round(start_cash * last_p / p0, 2) if (p0 and last_p) else None
        bench.append({"t": h["t"], "equity": v})
    last_prices = (last.get("prices") or {})
    trades = sum(len(h.get("acts", [])) for h in hist)
    return {"start_cash": start_cash, "equity": eq, "pnl": round(eq - start_cash, 2), "trades": trades,
            "ticks": len(hist),
            "cash": led.get("cash", 0.0), "cash_weight": round(led.get("cash", 0.0) / eq, 4) if eq else None,
            "bench_asset": last.get("bench_asset"), "market": last.get("market", "crypto"),
            "holdings": _holdings(led, eq, last_prices), "last_signal": last.get("signal", {}),
            "curve": [{"t": h["t"], "equity": h["equity"]} for h in tail], "bench": bench,
            "recent": [{"t": h["t"], "equity": h["equity"], "acts": h.get("acts", [])} for h in reversed(hist[-10:])],
            "series": [{"t": h["t"], "signal": h.get("signal", {}), "prices": h.get("prices", {})}
                       for h in hist[-240:]]}


def _exp_path():
    import news_trader as nt
    return os.path.join(nt.LEDGER_DIR, "experiment.json")


def _exp_load():
    try:
        with open(_exp_path()) as f:
            return json.load(f)
    except Exception:
        return {"arms": []}


def _exp_save(spec):
    import news_trader as nt
    os.makedirs(nt.LEDGER_DIR, exist_ok=True)
    tmp = _exp_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(spec, f)
    os.replace(tmp, _exp_path())


# Aggressiveness -> (entry gate, trade band, max position size). Conservative = choosy + small +
# low turnover; Aggressive = acts on weak signal, full size, more turnover.
AGGR = {"cons": (0.20, 0.15, 0.5), "bal": (0.10, 0.10, 1.0), "aggr": (0.05, 0.06, 1.0)}


def _arm_spec(a, seen):
    """Normalize one requested arm {market, focus, aggr, mode} -> a spec with a stable, unique,
    filesystem-safe label (encodes the config so the same subject at different settings is a
    distinct arm) so its ledger persists across restarts."""
    market = "stocks" if a.get("market") == "stocks" else "crypto"
    focus = a.get("focus") or None
    if focus and str(focus).upper() in ("AUTO", "ALL", "BROAD", ""):
        focus = None
    focus = focus.upper() if focus else None
    aggr = a.get("aggr") if a.get("aggr") in AGGR else "bal"
    mode = "fade" if a.get("mode") == "fade" else "follow"
    base = (focus.lower() if focus else ("broad_" + market)) + ("_" + aggr if aggr != "bal" else "") + ("_fade" if mode == "fade" else "")
    label = base
    n = 1
    while label in seen:
        label = f"{base}_{n}"
        n += 1
    seen.add(label)
    title = focus if focus else ("All stocks" if market == "stocks" else "All crypto")
    return {"label": label, "market": market, "focus": focus, "aggr": aggr, "mode": mode, "title": title}


def register(app):
    @app.route("/ext/paper_trade/sentiment")
    def pt_sentiment():
        def _do():
            import scraper
            import sentiment as sent
            n = min(N_MAX, max(3, int(request.args.get("n", N_DEFAULT))))
            model = request.args.get("model") or None
            provider = request.args.get("provider") or None
            focus = request.args.get("token") or request.args.get("focus") or None
            if focus and focus.upper() == "AUTO":
                focus = None
            items = scraper.scrape(limit=n, focus=focus)
            scored = sent.score_items(items, provider=provider, model=model)
            return {"ok": True, "fear_greed": scraper.fear_greed(),
                    "signal": sent.aggregate(scored), "scored": scored, "n": len(scored)}
        return _safe(_do)

    @app.route("/ext/paper_trade/feed")
    def pt_feed():
        def _do():
            import scraper
            n = min(N_MAX, max(3, int(request.args.get("n", N_DEFAULT))))
            return {"ok": True, "fear_greed": scraper.fear_greed(), "items": scraper.scrape(limit=n)}
        return _safe(_do)

    @app.route("/ext/paper_trade/trader/start", methods=["POST"])
    def pt_trader_start():
        def _do():
            import news_trader as nt
            import sentiment as sent
            b = request.get_json(silent=True) or {}
            focus = b.get("focus") or None
            if focus and str(focus).upper() == "AUTO":
                focus = None
            started = nt.start_thread(
                b.get("model") or sent.DEFAULT_MODEL, b.get("provider") or None,
                float(b.get("gate", nt.SENT_GATE)), nt.MAX_SIZE,
                float(b.get("fee_bps", nt.FEE * 1e4)) / 1e4, bool(b.get("use_chart")),
                b.get("source", "crypto_of"), int(b.get("interval", nt.INTERVAL)),
                float(b.get("band", nt.REBAL_BAND)), focus)
            return {"ok": True, "started": started, **nt.status()}
        return _safe(_do)

    @app.route("/ext/paper_trade/trader/stop", methods=["POST"])
    def pt_trader_stop():
        def _do():
            import news_trader as nt
            nt.stop_thread()
            return {"ok": True, **nt.status()}
        return _safe(_do)

    @app.route("/ext/paper_trade/trader/status")
    def pt_trader_status():
        def _do():
            import news_trader as nt
            return {"ok": True, **nt.status()}
        return _safe(_do)

    @app.route("/ext/paper_trade/trader/update", methods=["POST"])
    def pt_trader_update():
        def _do():
            import news_trader as nt
            b = request.get_json(silent=True) or {}
            ch = {}
            if b.get("model"):
                ch["model"] = b["model"]
            if b.get("gate") is not None:
                ch["gate"] = float(b["gate"])
            if b.get("band") is not None:
                ch["band"] = float(b["band"])
            if b.get("interval") is not None:
                ch["interval"] = int(b["interval"])
            if "focus" in b:
                f = b.get("focus")
                ch["focus"] = None if (f is None or str(f).upper() == "AUTO") else f
            ok = nt.update_run("main", **ch)
            return {"ok": ok, "applied": ok, **nt.status("main")}
        return _safe(_do)

    @app.route("/ext/paper_trade/account")
    def pt_account():
        def _do():
            import news_trader as nt
            led = nt.load_ledger()
            hist = led.get("history", [])
            eq = hist[-1]["equity"] if hist else led.get("start_cash", 0.0)
            last = hist[-1] if hist else {}
            start_cash = led.get("start_cash", 0.0)
            tail = hist[-300:]

            def _bpx(h):
                return h.get("bench_px") or h.get("btc")
            b0 = next((_bpx(h) for h in tail if _bpx(h)), None)   # buy-hold base price
            bench, last_btc = [], None
            for h in tail:
                last_btc = _bpx(h) or last_btc
                v = round(start_cash * last_btc / b0, 2) if (b0 and last_btc) else None
                bench.append({"t": h["t"], "equity": v})
            return {"ok": True, "running": bool(hist), "start_cash": start_cash,
                    "focus": nt.status().get("focus"),
                    "cash": led.get("cash", 0.0), "equity": eq,
                    "cash_weight": round(led.get("cash", 0.0) / eq, 4) if eq else None,
                    "positions": {k: round(v, 6) for k, v in led.get("positions", {}).items() if v},
                    "holdings": _holdings(led, eq, (last.get("prices") or {})),
                    "signal": last.get("signal", {}), "last_t": last.get("t"),
                    "curve": [{"t": h["t"], "equity": h["equity"]} for h in tail],
                    "bench": bench,
                    "series": [{"t": h["t"], "equity": h["equity"], "signal": h.get("signal", {}),
                                "prices": h.get("prices", {})} for h in hist[-240:]],
                    "recent": list(reversed(hist[-12:]))}
        return _safe(_do)

    @app.route("/ext/paper_trade/ab/start", methods=["POST"])
    def pt_ab_start():
        def _do():
            import news_trader as nt
            import sentiment as sent
            b = request.get_json(silent=True) or {}
            model = b.get("model") or sent.DEFAULT_MODEL
            gate = float(b.get("gate", nt.SENT_GATE))
            interval = int(b.get("interval", nt.INTERVAL))
            band = float(b.get("band", AB_BAND))
            started = {}
            for coin, label in AB_ARMS:
                started[label] = nt.start_thread(model, b.get("provider") or None, gate, nt.MAX_SIZE,
                                                 nt.FEE, False, b.get("source", "crypto_of"),
                                                 interval, band, coin, label)
            return {"ok": True, "started": started, "band": band, "model": model, **nt.status_all()}
        return _safe(_do)

    @app.route("/ext/paper_trade/ab/stop", methods=["POST"])
    def pt_ab_stop():
        def _do():
            import news_trader as nt
            for _coin, label in AB_ARMS:
                nt.stop_thread(label)
            return {"ok": True}
        return _safe(_do)

    @app.route("/ext/paper_trade/ab/status")
    def pt_ab_status():
        def _do():
            import news_trader as nt
            return {"ok": True, **{label: nt.status(label) for _c, label in AB_ARMS}}
        return _safe(_do)

    @app.route("/ext/paper_trade/ab/update", methods=["POST"])
    def pt_ab_update():
        def _do():
            import news_trader as nt
            b = request.get_json(silent=True) or {}
            ch = {}
            if b.get("model"):
                ch["model"] = b["model"]
            if b.get("gate") is not None:
                ch["gate"] = float(b["gate"])
            if b.get("band") is not None:
                ch["band"] = float(b["band"])
            if b.get("interval") is not None:
                ch["interval"] = int(b["interval"])
            updated = {label: nt.update_run(label, **ch) for _c, label in AB_ARMS}
            return {"ok": True, "updated": updated}
        return _safe(_do)

    @app.route("/ext/paper_trade/ab/accounts")
    def pt_ab_accounts():
        def _do():
            import news_trader as nt
            out = {"ok": True}
            for coin, label in AB_ARMS:
                led = nt.load_ledger(nt.ledger_for(label))
                out[label] = {"running": nt.status(label).get("running", False), **_ab_view(led, coin)}
            return out
        return _safe(_do)

    # ---- generic N-arm experiment (A/B/C/D...): each arm = its own market+focus+ledger, one shared model ----
    @app.route("/ext/paper_trade/exp/start", methods=["POST"])
    def pt_exp_start():
        def _do():
            import news_trader as nt
            import sentiment as sent
            b = request.get_json(silent=True) or {}
            saved = _exp_load()
            model = b.get("model") or saved.get("model") or sent.DEFAULT_MODEL
            interval = int(b.get("interval", saved.get("interval", nt.INTERVAL)))
            risk_off = bool(b.get("risk_off", saved.get("risk_off", True)))
            reset = b.get("reset", True)
            if b.get("arms"):                          # fresh experiment from the UI
                seen = set()
                arms = [_arm_spec(a, seen) for a in b["arms"][:EXP_MAX_ARMS]]
            else:                                      # resume the saved experiment (cron restart path)
                arms = saved.get("arms", [])
            for a in saved.get("arms", []):            # stop any prior arms first
                nt.stop_thread(a["label"])
            for a in arms:
                nt.stop_thread(a["label"])
            time.sleep(0.4)                            # let stopped threads exit before we reset/restart
            started = {}
            for a in arms:
                g, bd, ms = AGGR.get(a.get("aggr", "bal"), AGGR["bal"])
                if reset:
                    nt.save_ledger({"cash": nt.START_CASH, "positions": {}, "start_cash": nt.START_CASH,
                                    "history": []}, nt.ledger_for(a["label"]))
                started[a["label"]] = nt.start_thread(model, None, g, ms, nt.FEE, False, "crypto_of",
                                                      interval, bd, a["focus"], a["label"], a["market"],
                                                      a.get("mode", "follow"), risk_off)
            spec = {"arms": arms, "model": model, "interval": interval, "risk_off": risk_off}
            _exp_save(spec)
            return {"ok": True, "started": started, **spec}
        return _safe(_do)

    @app.route("/ext/paper_trade/exp/stop", methods=["POST"])
    def pt_exp_stop():
        def _do():
            import news_trader as nt
            for a in _exp_load().get("arms", []):
                nt.stop_thread(a["label"])
            return {"ok": True}
        return _safe(_do)

    @app.route("/ext/paper_trade/exp/status")
    def pt_exp_status():
        def _do():
            import news_trader as nt
            spec = _exp_load()
            running = {a["label"]: nt.status(a["label"]).get("running", False) for a in spec.get("arms", [])}
            return {"ok": True, "running_any": any(running.values()), "running": running, **spec}
        return _safe(_do)

    @app.route("/ext/paper_trade/exp/update", methods=["POST"])
    def pt_exp_update():
        def _do():
            import news_trader as nt
            b = request.get_json(silent=True) or {}
            ch = {}
            if b.get("model"):
                ch["model"] = b["model"]
            if b.get("interval") is not None:
                ch["interval"] = int(b["interval"])
            if b.get("risk_off") is not None:
                ch["risk_off"] = bool(b["risk_off"])
            spec = _exp_load()
            updated = {a["label"]: nt.update_run(a["label"], **ch) for a in spec.get("arms", [])}
            spec.update(ch)
            _exp_save(spec)
            return {"ok": True, "updated": updated}
        return _safe(_do)

    @app.route("/ext/paper_trade/exp/accounts")
    def pt_exp_accounts():
        def _do():
            import news_trader as nt
            spec = _exp_load()
            arms = []
            for a in spec.get("arms", []):
                led = nt.load_ledger(nt.ledger_for(a["label"]))
                arms.append({"label": a["label"], "title": a["title"], "market": a["market"],
                             "focus": a.get("focus"), "aggr": a.get("aggr", "bal"), "mode": a.get("mode", "follow"),
                             "running": nt.status(a["label"]).get("running", False), **_exp_view(led)})
            return {"ok": True, "model": spec.get("model"), "gate": spec.get("gate"),
                    "band": spec.get("band"), "interval": spec.get("interval"), "arms": arms}
        return _safe(_do)
