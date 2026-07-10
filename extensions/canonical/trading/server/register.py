# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Entry point for the Trading extension. register(app) mounts everything under
#   /ext/trading in three groups: /ext/trading/market/* (byte-model serving, signals,
#   backtests, instruments), /ext/trading/paper/* (news / experiment / xsmom / eqmom /
#   ml7 paper traders, all simulated capital), /ext/trading/intel/* (market scanner,
#   pump radar, model briefs). Plus /ext/trading/settings (model default, scan intervals,
#   fee assumption, scraper channel registry), /ext/trading/system (one view of every
#   runnable + stop controls), and the /ext/trading/models analytics page.
# - Nothing auto-starts at boot except resume() of runs a user explicitly started
#   (ledger/state auto-stamps). The recorder and autotrader stay CLI-only. Server
#   modules import by bare name (server/ is on sys.path at register time); route
#   bodies import lazily so a missing dep never breaks dashboard startup.
# extensions/canonical/trading/server/register.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import time

from flask import request, send_from_directory

# ------------------------------------------------------------------------------------
# Constants

LOG_SOURCE = "trading"
N_DEFAULT = 10
N_MAX = 30
EXP_MAX_ARMS = 10                               # A/B/C/D... cap on parallel experiment arms
PUMPS_DEFAULT = 50
PUMPS_MAX = 500
BRIEF_ROWS = 20                       # rows per scan response annotated with cached briefs
PAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page")
MODELS_PAGE = "models.html"

# Aggressiveness -> (entry gate, trade band, max position size). Conservative = choosy + small +
# low turnover; Aggressive = acts on weak signal, full size, more turnover.
AGGR = {"cons": (0.20, 0.15, 0.5), "bal": (0.10, 0.10, 1.0), "aggr": (0.05, 0.06, 1.0)}

# ------------------------------------------------------------------------------------
# Functions

def _safe(fn):
    """Run a route handler; log any exception and return a JSON error body + 500 so the
    frontend gets parseable bytes instead of Flask's HTML error page."""
    try:
        return fn()
    except Exception as e:
        from runtime import logs as logmod
        msg = f"{type(e).__name__}: {e}"
        logmod.error(LOG_SOURCE, msg)
        return ({"ok": False, "error": msg}, 500)


def _policy_args(a):
    """Trading-policy overrides from query args; fee in bps, gates as fractions."""
    return {"mode": a.get("mode", "vol_harvest"), "sizing": a.get("sizing", "confidence"),
            "fee": float(a.get("fee_bps", 5)) / 1e4, "conf_gate": float(a.get("conf_gate", 0)),
            "move_gate": float(a.get("move_gate", 1.0)), "max_size": float(a.get("max_size", 1.0))}


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


def _with_briefs(rows):
    import intel
    for r in rows[:BRIEF_ROWS]:
        b = intel.cached_brief(r["sym"])
        if b:
            r["brief"] = b
    return rows


def _system_view():
    """Every runnable with its live state: the one call the overview strip polls."""
    import eqmom_trader as eq
    import ml7_trader as mt
    import news_trader as nt
    import scanner
    import xsmom_trader as xt
    out = []
    st = nt.status("main")
    out.append({"id": "news_main", "group": "news", "name": "News sentiment trader",
                "running": st.get("running", False), "started": st.get("started")})
    for a in _exp_load().get("arms", []):
        s = nt.status(a["label"])
        out.append({"id": "exp_" + a["label"], "group": "news",
                    "name": f"Experiment arm: {a.get('title', a['label'])}",
                    "running": s.get("running", False), "started": s.get("started")})
    xs_names = {"base": "Crypto momentum (baseline)", "gated": "Crypto momentum (vol-gated)"}
    for arm, s in xt.status_all().items():
        out.append({"id": "xsmom_" + arm, "group": "xsmom", "name": xs_names.get(arm, arm),
                    "running": s.get("running", False), "started": s.get("started")})
    for arm, s in eq.status_all().items():
        out.append({"id": "eqmom_" + arm, "group": "eqmom", "name": "Equity momentum (12-1)",
                    "running": s.get("running", False), "started": s.get("started")})
    for arm, s in mt.status_all().items():
        out.append({"id": "ml7_" + arm, "group": "ml7", "name": "ML weekly book (7-day ranks)",
                    "running": s.get("running", False), "started": s.get("started")})
    sc = scanner.status()
    out.append({"id": "intel_watch", "group": "intel", "name": "Intel watch (pump radar)",
                "running": bool(sc.get("watching")), "started": sc.get("started")})
    return out


def _system_stop(run_id):
    """Stop one runnable by its system id; True when the id maps to a known runnable."""
    import eqmom_trader as eq
    import ml7_trader as mt
    import news_trader as nt
    import scanner
    import xsmom_trader as xt
    if run_id == "news_main":
        return nt.stop_thread("main")
    if run_id.startswith("exp_"):
        return nt.stop_thread(run_id[len("exp_"):])
    if run_id.startswith("xsmom_"):
        return xt.stop_thread(run_id[len("xsmom_"):])
    if run_id.startswith("eqmom_"):
        return eq.stop_thread(run_id[len("eqmom_"):])
    if run_id.startswith("ml7_"):
        return mt.stop_thread(run_id[len("ml7_"):])
    if run_id == "intel_watch":
        return scanner.stop_thread()
    return False

# ------------------------------------------------------------------------------------
# Routes

def register(app):
    app.add_url_rule("/ext/trading/models", "ext_page_trading_models",
                     lambda: send_from_directory(PAGE_DIR, MODELS_PAGE))

    # ---- market: byte-model serving, signals, backtests ----
    @app.route("/ext/trading/market/veritate_models")
    def trading_veritate_models():
        def _do():
            import veritate as vz
            return {"ok": True, "models": vz.list_models()}
        return _safe(_do)

    @app.route("/ext/trading/market/veritate_hindcast")
    def trading_veritate_hindcast():
        def _do():
            import data as md
            import veritate as vz
            model_name = request.args.get("model")
            source = request.args.get("source", "crypto")
            symbol = request.args.get("symbol", "BTCUSDT")
            base = request.args.get("base", "1m")
            n = min(20000, max(300, int(request.args.get("n", 1500))))
            if not model_name or model_name not in vz.list_models():
                return ({"ok": False, "error": "pick a trained Veritate model."}, 400)
            df = md.load_tail(symbol, n_bars=n, base=base, source=source)
            if df is None:
                return ({"ok": False, "error": f"no data for {symbol}."}, 404)
            model, seq, step, stride = vz.load_model(model_name)
            g = vz.hindcast(model, seq, df, base=base, stride=stride)
            if g is None:
                return ({"ok": False, "error": "not enough bars to run the model."}, 404)
            g.update({"ok": True, "symbol": symbol, "model": model_name, "step": step})
            return g
        return _safe(_do)

    @app.route("/ext/trading/market/veritate_benchmark")
    def trading_veritate_benchmark():
        def _do():
            import data as md
            import veritate as vz
            model_name = request.args.get("model")
            source = request.args.get("source", "crypto")
            symbol = request.args.get("symbol", "BTCUSDT")
            base = request.args.get("base", "1m")
            n = min(20000, max(300, int(request.args.get("n", 1500))))
            if not model_name or model_name not in vz.list_models():
                return ({"ok": False, "error": "pick a trained Veritate model."}, 400)
            df = md.load_tail(symbol, n_bars=n, base=base, source=source)
            if df is None:
                return ({"ok": False, "error": f"no data for {symbol}."}, 404)
            model, seq, step, stride = vz.load_model(model_name)
            g = vz.benchmark(model, seq, df, base=base, stride=stride)
            if g is None:
                return ({"ok": False, "error": "not enough bars to benchmark."}, 404)
            g.update({"ok": True, "symbol": symbol, "model": model_name, "step": step})
            return g
        return _safe(_do)

    @app.route("/ext/trading/market/veritate_data_report")
    def trading_veritate_data_report():
        def _do():
            import veritate as vz
            source = request.args.get("source", "crypto")
            r = vz.data_report(source)
            r["ok"] = True
            r["source"] = source
            return r
        return _safe(_do)

    @app.route("/ext/trading/market/veritate_live")
    def trading_veritate_live():
        def _do():
            import data as md
            import live as lv
            import veritate as vz
            model_name = request.args.get("model")
            symbol = request.args.get("symbol", "BTCUSDT")
            source = request.args.get("source", "crypto")
            if not model_name or model_name not in vz.list_models():
                return ({"ok": False, "error": "pick a trained Veritate model."}, 400)
            model, seq, step, stride = vz.load_model(model_name)
            if source == "crypto":
                try:
                    df, _ = lv.fetch(symbol, base="1m", limit=400)
                except Exception as e:
                    return ({"ok": False, "error": f"live feed error ({symbol} may not be on Binance.US): {e}"}, 502)
                if df is None or len(df) < 60:
                    return ({"ok": False, "error": "no live data."}, 404)
                closed = df.iloc[:-1]
            else:
                closed = md.load_tail(symbol, n_bars=400, base="1d", source=source)
                if closed is None:
                    return ({"ok": False, "error": "no data."}, 404)
            pred = vz.predict_next(model, seq, closed, stride)
            if pred is None:
                return ({"ok": False, "error": "prediction failed."}, 500)
            last_close = float(closed["close"].iloc[-1])
            last_t = int(closed.index[-1].value // 1_000_000_000)
            pred.update({"ok": True, "symbol": symbol, "model": model_name, "last_close": last_close,
                         "last_t": last_t, "expected_move_bps": pred["expected_move"] * 1e4})
            return pred
        return _safe(_do)

    @app.route("/ext/trading/market/instruments")
    def trading_instruments():
        def _do():
            import data as md
            source = request.args.get("source", "crypto")
            return {"ok": True, "source": source, "instruments": md.list_instruments(source)}
        return _safe(_do)

    @app.route("/ext/trading/market/paper_signal")
    def trading_paper_signal():
        def _do():
            import data as md
            import veritate as vz
            model_name = request.args.get("model")
            source = request.args.get("source", "crypto")
            symbol = request.args.get("symbol", "BTCUSDT")
            base = request.args.get("base", "1h")
            n = min(20000, max(300, int(request.args.get("n", 1500))))
            if not model_name or model_name not in vz.list_models():
                return ({"ok": False, "error": "pick a trained Veritate model."}, 400)
            df = md.load_tail(symbol, n_bars=n, base=base, source=source)
            if df is None:
                return ({"ok": False, "error": f"no data for {symbol}."}, 404)
            model, seq, step, stride = vz.load_model(model_name)
            sig = vz.signal_series(model, seq, df, base=base, stride=stride)
            if sig is None:
                return ({"ok": False, "error": "not enough bars to run the model."}, 404)
            sig.update({"ok": True, "symbol": symbol, "model": model_name, "step": step})
            return sig
        return _safe(_do)

    @app.route("/ext/trading/market/paper_backtest")
    def trading_paper_backtest():
        def _do():
            import data as md
            import policy as pol
            import veritate as vz
            model_name = request.args.get("model")
            source = request.args.get("source", "crypto")
            symbol = request.args.get("symbol", "BTCUSDT")
            base = request.args.get("base", "1h")
            n = min(20000, max(300, int(request.args.get("n", 1500))))
            if not model_name or model_name not in vz.list_models():
                return ({"ok": False, "error": "pick a trained Veritate model."}, 400)
            df = md.load_tail(symbol, n_bars=n, base=base, source=source)
            if df is None:
                return ({"ok": False, "error": f"no data for {symbol}."}, 404)
            model, seq, step, stride = vz.load_model(model_name)
            sig = vz.signal_series(model, seq, df, base=base, stride=stride)
            if sig is None:
                return ({"ok": False, "error": "not enough bars to run the model."}, 404)
            res = pol.backtest(sig["price"], sig["p_up"], sig["conf"], sig["exp_move"],
                               sig["vol"], sig["ret_next"], **_policy_args(request.args))
            res["trades"] = pol.trades(sig, res)
            res.update({"ok": True, "symbol": symbol, "model": model_name, "step": step,
                        "base": base, "n": sig["n"], "t": sig["t"], "price": sig["price"]})
            return res
        return _safe(_do)

    @app.route("/ext/trading/market/paper_decide")
    def trading_paper_decide():
        def _do():
            import data as md
            import live as lv
            import policy as pol
            import veritate as vz
            model_name = request.args.get("model")
            symbol = request.args.get("symbol", "BTCUSDT")
            source = request.args.get("source", "crypto")
            if not model_name or model_name not in vz.list_models():
                return ({"ok": False, "error": "pick a trained Veritate model."}, 400)
            model, seq, step, stride = vz.load_model(model_name)
            if source == "crypto":
                try:
                    df, _ = lv.fetch(symbol, base="1m", limit=400)
                except Exception as e:
                    return ({"ok": False, "error": f"live feed error ({symbol} may not be on Binance.US): {e}"}, 502)
                if df is None or len(df) < 60:
                    return ({"ok": False, "error": "no live data."}, 404)
                closed = df.iloc[:-1]
            else:
                closed = md.load_tail(symbol, n_bars=400, base="1d", source=source)
                if closed is None:
                    return ({"ok": False, "error": "no data."}, 404)
            pred = vz.predict_next(model, seq, closed, stride)
            if pred is None:
                return ({"ok": False, "error": "prediction failed."}, 500)
            premium = vz.trailing_premium(closed)
            dec = pol.decide(pred["p_up"], pred["confidence"], pred["expected_move"],
                             pred["vol"], premium=premium, **_policy_args(request.args))
            return {"ok": True, "symbol": symbol, "model": model_name, "step": step,
                    "last_close": float(closed["close"].iloc[-1]),
                    "last_t": int(closed.index[-1].value // 1_000_000_000),
                    "p_up": pred["p_up"], "confidence": pred["confidence"],
                    "expected_move": pred["expected_move"], "expected_move_bps": pred["expected_move"] * 1e4,
                    "premium": premium, "premium_bps": premium * 1e4, "decision": dec}
        return _safe(_do)

    # ---- paper: news sentiment feed + traders (simulated capital) ----
    @app.route("/ext/trading/paper/sentiment")
    def trading_sentiment():
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

    @app.route("/ext/trading/paper/feed")
    def trading_feed():
        def _do():
            import scraper
            n = min(N_MAX, max(3, int(request.args.get("n", N_DEFAULT))))
            return {"ok": True, "fear_greed": scraper.fear_greed(), "items": scraper.scrape(limit=n)}
        return _safe(_do)

    @app.route("/ext/trading/paper/trader/start", methods=["POST"])
    def trading_trader_start():
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

    @app.route("/ext/trading/paper/trader/stop", methods=["POST"])
    def trading_trader_stop():
        def _do():
            import news_trader as nt
            nt.stop_thread()
            return {"ok": True, **nt.status()}
        return _safe(_do)

    @app.route("/ext/trading/paper/trader/status")
    def trading_trader_status():
        def _do():
            import news_trader as nt
            return {"ok": True, **nt.status()}
        return _safe(_do)

    @app.route("/ext/trading/paper/trader/update", methods=["POST"])
    def trading_trader_update():
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

    @app.route("/ext/trading/paper/account")
    def trading_account():
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
            cum_notional = led.get("cum_notional", 0.0)
            fee_side = led.get("fee_side", nt.FEE)
            equity_alt = eq + (fee_side - nt.ALT_FEE) * cum_notional   # same tape, documented 10bp/side
            return {"ok": True, "running": bool(hist), "start_cash": start_cash,
                    "focus": nt.status().get("focus"),
                    "cum_notional": round(cum_notional, 2), "fee_bps_live": round(fee_side * 1e4, 1),
                    "alt_fee_bps": round(nt.ALT_FEE * 1e4, 1), "equity_alt": round(equity_alt, 2),
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

    # ---- generic N-arm experiment (A/B/C/D...): each arm = its own market+focus+ledger ----
    @app.route("/ext/trading/paper/exp/start", methods=["POST"])
    def trading_exp_start():
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

    @app.route("/ext/trading/paper/exp/stop", methods=["POST"])
    def trading_exp_stop():
        def _do():
            import news_trader as nt
            for a in _exp_load().get("arms", []):
                nt.stop_thread(a["label"])
            return {"ok": True}
        return _safe(_do)

    @app.route("/ext/trading/paper/exp/status")
    def trading_exp_status():
        def _do():
            import news_trader as nt
            spec = _exp_load()
            running = {a["label"]: nt.status(a["label"]).get("running", False) for a in spec.get("arms", [])}
            return {"ok": True, "running_any": any(running.values()), "running": running, **spec}
        return _safe(_do)

    @app.route("/ext/trading/paper/exp/update", methods=["POST"])
    def trading_exp_update():
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

    @app.route("/ext/trading/paper/exp/accounts")
    def trading_exp_accounts():
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

    @app.route("/ext/trading/paper/xsmom/start", methods=["POST"])
    def trading_xsmom_start():
        def _do():
            import xsmom_trader as xt
            b = request.get_json(silent=True) or {}
            arms = b.get("arms") or list(xt.ARMS)
            return {"ok": True, "started": {a: xt.start_thread(a) for a in arms if a in xt.ARMS}}
        return _safe(_do)

    @app.route("/ext/trading/paper/xsmom/stop", methods=["POST"])
    def trading_xsmom_stop():
        def _do():
            import xsmom_trader as xt
            xt.stop_thread()
            return {"ok": True}
        return _safe(_do)

    @app.route("/ext/trading/paper/xsmom/status")
    def trading_xsmom_status():
        def _do():
            import xsmom_trader as xt
            return {"ok": True, "arms": xt.status_all()}
        return _safe(_do)

    @app.route("/ext/trading/paper/xsmom/accounts")
    def trading_xsmom_accounts():
        def _do():
            import news_trader as nt
            import xsmom_trader as xt
            arms = []
            for arm in xt.ARMS:
                led = nt.load_ledger(nt.ledger_for(xt.label_for(arm)))
                arms.append({"label": xt.label_for(arm), "title": f"XS momentum ({arm})", "arm": arm,
                             "invested": led.get("invested", True),
                             "running": xt.status_all()[arm]["running"], **_exp_view(led)})
            return {"ok": True, "fee_bps": 3.5, "rebalance": "weekly", "arms": arms}
        return _safe(_do)

    @app.route("/ext/trading/paper/eqmom/start", methods=["POST"])
    def trading_eqmom_start():
        def _do():
            import eqmom_trader as eq
            return {"ok": True, "started": {a: eq.start_thread(a) for a in eq.ARMS}}
        return _safe(_do)

    @app.route("/ext/trading/paper/eqmom/stop", methods=["POST"])
    def trading_eqmom_stop():
        def _do():
            import eqmom_trader as eq
            eq.stop_thread()
            return {"ok": True}
        return _safe(_do)

    @app.route("/ext/trading/paper/eqmom/status")
    def trading_eqmom_status():
        def _do():
            import eqmom_trader as eq
            return {"ok": True, "arms": eq.status_all()}
        return _safe(_do)

    @app.route("/ext/trading/paper/eqmom/accounts")
    def trading_eqmom_accounts():
        def _do():
            import eqmom_trader as eq
            import news_trader as nt
            arms = []
            for arm in eq.ARMS:
                led = nt.load_ledger(nt.ledger_for(eq.label_for(arm)))
                arms.append({"label": eq.label_for(arm), "title": "Equity momentum 12-1", "arm": arm,
                             "invested": True,
                             "running": eq.status_all()[arm]["running"], **_exp_view(led)})
            return {"ok": True, "fee_bps": 3.0, "rebalance": "monthly", "arms": arms}
        return _safe(_do)

    @app.route("/ext/trading/paper/ml7/start", methods=["POST"])
    def trading_ml7_start():
        def _do():
            import ml7_trader as mt
            return {"ok": True, "started": {a: mt.start_thread(a) for a in mt.ARMS},
                    "model": mt.model_state()}
        return _safe(_do)

    @app.route("/ext/trading/paper/ml7/stop", methods=["POST"])
    def trading_ml7_stop():
        def _do():
            import ml7_trader as mt
            mt.stop_thread()
            return {"ok": True}
        return _safe(_do)

    @app.route("/ext/trading/paper/ml7/status")
    def trading_ml7_status():
        def _do():
            import ml7_trader as mt
            return {"ok": True, "arms": mt.status_all(), "model": mt.model_state()}
        return _safe(_do)

    @app.route("/ext/trading/paper/ml7/accounts")
    def trading_ml7_accounts():
        def _do():
            import ml7_trader as mt
            import news_trader as nt
            arms = []
            for arm in mt.ARMS:
                led = nt.load_ledger(nt.ledger_for(mt.label_for(arm)))
                arms.append({"label": mt.label_for(arm), "title": "ML weekly book (7-day ranks)",
                             "arm": arm, "invested": True, "day": led.get("day"),
                             "running": mt.status_all()[arm]["running"], **_exp_view(led)})
            return {"ok": True, "fee_bps": 3.5, "rebalance": "daily tranche",
                    "model": mt.model_state(), "arms": arms}
        return _safe(_do)

    # ---- intel: scanner, pump radar, briefs ----
    @app.route("/ext/trading/intel/scan")
    def trading_intel_scan():
        def _do():
            import scanner
            s = scanner.scan(refresh=request.args.get("refresh") == "1")
            _with_briefs(s.get("rows", []))
            return {"ok": "error" not in s, **s}
        return _safe(_do)

    @app.route("/ext/trading/intel/trending")
    def trading_intel_trending():
        def _do():
            import scanner
            s = scanner.scan()
            return {"ok": True, "ts": s["ts"], "trending": s.get("trending", []),
                    "meme": s.get("meme", []),
                    "movers": _with_briefs(list(s.get("rows", [])))}
        return _safe(_do)

    @app.route("/ext/trading/intel/pumps")
    def trading_intel_pumps():
        def _do():
            import intel
            import scanner
            n = min(PUMPS_MAX, max(1, int(request.args.get("n", PUMPS_DEFAULT))))
            evs = scanner.events(n)
            for ev in evs:
                b = intel.cached_brief(ev["sym"])
                if b:
                    ev["brief"] = b
            return {"ok": True, "events": evs, "count": scanner.event_count()}
        return _safe(_do)

    @app.route("/ext/trading/intel/brief")
    def trading_intel_brief():
        def _do():
            import intel
            import scanner
            sym = (request.args.get("symbol") or "").upper()
            if not sym:
                return {"ok": False, "error": "symbol required"}
            row = scanner.row_for(sym) or {}
            b = intel.brief(sym, row, name=row.get("name"),
                            model=request.args.get("model") or None)
            return {"ok": True, "symbol": sym, "brief": b, "model_state": intel.model_state()}
        return _safe(_do)

    @app.route("/ext/trading/intel/candles")
    def trading_intel_candles():
        def _do():
            import scanner
            sym = (request.args.get("symbol") or "").upper()
            bar = request.args.get("bar", "1H")
            rows = scanner.candles(sym, bar, int(request.args.get("limit", scanner.CANDLE_MAX)))
            if rows is None:
                return ({"ok": False, "error": f"no OKX candles for {sym} ({bar})"}, 404)
            return {"ok": True, "symbol": sym, "bar": bar, "candles": rows}
        return _safe(_do)

    @app.route("/ext/trading/intel/status")
    def trading_intel_status():
        def _do():
            import intel
            import scanner
            return {"ok": True, **scanner.status(), "model_state": intel.model_state()}
        return _safe(_do)

    @app.route("/ext/trading/intel/watch/start", methods=["POST"])
    def trading_intel_watch_start():
        def _do():
            import scanner
            b = request.get_json(silent=True) or {}
            started = scanner.start_thread(int(b.get("interval") or scanner.SCAN_SEC),
                                           b.get("model") or None)
            return {"ok": True, "started": started, **scanner.status()}
        return _safe(_do)

    @app.route("/ext/trading/intel/watch/stop", methods=["POST"])
    def trading_intel_watch_stop():
        def _do():
            import scanner
            scanner.stop_thread()
            return {"ok": True, **scanner.status()}
        return _safe(_do)

    # ---- settings: model default, intervals, fee assumption, scraper channels ----
    @app.route("/ext/trading/settings", methods=["GET", "POST"])
    def trading_settings():
        def _do():
            import scraper
            if request.method == "POST":
                saved = scraper.save_settings(request.get_json(silent=True) or {})
                return {"ok": True, **saved, "channels": scraper.channel_health()}
            s = scraper.load_settings()
            return {"ok": True, **s, "channels": scraper.channel_health()}
        return _safe(_do)

    # ---- system: one view of every runnable + stop controls ----
    @app.route("/ext/trading/system")
    def trading_system():
        def _do():
            runs = _system_view()
            return {"ok": True, "runnables": runs,
                    "running": sum(1 for r in runs if r["running"])}
        return _safe(_do)

    @app.route("/ext/trading/system/stop", methods=["POST"])
    def trading_system_stop():
        def _do():
            b = request.get_json(silent=True) or {}
            run_id = str(b.get("id") or "")
            ok = _system_stop(run_id)
            return {"ok": bool(ok), "id": run_id, "runnables": _system_view()}
        return _safe(_do)

    @app.route("/ext/trading/system/stop_all", methods=["POST"])
    def trading_system_stop_all():
        def _do():
            import eqmom_trader as eq
            import ml7_trader as mt
            import news_trader as nt
            import scanner
            import xsmom_trader as xt
            nt.stop_thread(None)
            xt.stop_thread()
            eq.stop_thread()
            mt.stop_thread()
            scanner.stop_thread()
            return {"ok": True, "runnables": _system_view()}
        return _safe(_do)

    # resume ONLY what a user explicitly started (ledger/state auto-stamps persist intent)
    try:
        import eqmom_trader as eq
        import ml7_trader as mt
        import news_trader as nt
        import scanner
        import xsmom_trader as xt
        xt.resume()
        eq.resume()
        mt.resume()
        nt.resume()
        scanner.resume()
    except Exception as e:
        from runtime import logs as logmod
        logmod.error(LOG_SOURCE, f"resume: {type(e).__name__}: {e}")
