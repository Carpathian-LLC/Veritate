# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Entry point for the market extension. register(app) adds the /market/* API routes;
#   the registry mounts the page from manifest.page. Serves Veritate byte-LLM forecasts
#   (hindcast, benchmark, live), the data report, and instrument lists.
# - Fully isolated from the canonical trainer/chat/RAG pipelines: it only reads its
#   own data-dir CSVs and Veritate model checkpoints. Server modules import by bare
#   name (server/ is on sys.path at register time). Lazy imports so a missing dep
#   never breaks dashboard startup.
# extensions/canonical/market/register.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request

# ------------------------------------------------------------------------------------
# Functions

def _safe(source, fn, *a, **kw):
    """Run a route handler; log any exception and return a JSON error body + 500 so the
    frontend gets parseable bytes instead of Flask's HTML error page."""
    try:
        return fn(*a, **kw)
    except Exception as e:
        from runtime import logs as logmod
        msg = f"{type(e).__name__}: {e}"
        logmod.error(source, msg)
        return ({"ok": False, "error": msg}, 500)


def _policy_args(a):
    """Trading-policy overrides from query args; fee in bps, gates as fractions."""
    return {"mode": a.get("mode", "vol_harvest"), "sizing": a.get("sizing", "confidence"),
            "fee": float(a.get("fee_bps", 5)) / 1e4, "conf_gate": float(a.get("conf_gate", 0)),
            "move_gate": float(a.get("move_gate", 1.0)), "max_size": float(a.get("max_size", 1.0))}

# ------------------------------------------------------------------------------------
# Routes

def register(app):
    @app.route("/market/veritate_models")
    def market_veritate_models():
        def _do():
            import veritate as vz
            return {"ok": True, "models": vz.list_models()}
        return _safe("market", _do)

    @app.route("/market/veritate_hindcast")
    def market_veritate_hindcast():
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
        return _safe("market", _do)

    @app.route("/market/veritate_benchmark")
    def market_veritate_benchmark():
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
        return _safe("market", _do)

    @app.route("/market/veritate_data_report")
    def market_veritate_data_report():
        def _do():
            import veritate as vz
            source = request.args.get("source", "crypto")
            r = vz.data_report(source)
            r["ok"] = True
            r["source"] = source
            return r
        return _safe("market", _do)

    @app.route("/market/veritate_live")
    def market_veritate_live():
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
        return _safe("market", _do)

    @app.route("/market/instruments")
    def market_instruments():
        def _do():
            import data as md
            source = request.args.get("source", "crypto")
            return {"ok": True, "source": source, "instruments": md.list_instruments(source)}
        return _safe("market", _do)

    @app.route("/market/paper_signal")
    def market_paper_signal():
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
        return _safe("market", _do)

    @app.route("/market/paper_backtest")
    def market_paper_backtest():
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
        return _safe("market", _do)

    @app.route("/market/paper_decide")
    def market_paper_decide():
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
        return _safe("market", _do)
