# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Backend for the experimental /market decision-support dashboard. Serves Veritate
#   byte-LLM forecasts (hindcast, benchmark, live), the data report, and instrument lists.
# - Fully isolated from the canonical trainer/chat/RAG pipelines: it only reads
#   external_data CSVs and Veritate model checkpoints. Lazy imports so a missing dep
#   never breaks dashboard startup.
# veritate_mri/routes/market_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request

from ._common import safe_route as _safe

# ------------------------------------------------------------------------------------
# Routes

def register(app):
    @app.route("/market/veritate_models")
    def market_veritate_models():
        def _do():
            from market import veritate as vz
            return {"ok": True, "models": vz.list_models()}
        return _safe("market", _do)

    @app.route("/market/veritate_hindcast")
    def market_veritate_hindcast():
        def _do():
            from market import data as md
            from market import veritate as vz
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
            from market import data as md
            from market import veritate as vz
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
            from market import veritate as vz
            source = request.args.get("source", "crypto")
            r = vz.data_report(source)
            r["ok"] = True
            r["source"] = source
            return r
        return _safe("market", _do)

    @app.route("/market/veritate_live")
    def market_veritate_live():
        def _do():
            from market import data as md
            from market import live as lv
            from market import veritate as vz
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
            from market import data as md
            source = request.args.get("source", "crypto")
            return {"ok": True, "source": source, "instruments": md.list_instruments(source)}
        return _safe("market", _do)

    @app.route("/market/extensions/catalog")
    def market_extensions_catalog():
        def _do():
            from market import extensions as ex
            return ex.catalog()
        return _safe("market", _do)

    @app.route("/market/extensions/download", methods=["POST"])
    def market_extensions_download():
        def _do():
            from market import extensions as ex
            return ex.download((request.get_json(silent=True) or {}).get("source"))
        return _safe("market", _do)

    @app.route("/market/extensions/delete", methods=["POST"])
    def market_extensions_delete():
        def _do():
            from market import extensions as ex
            return ex.delete((request.get_json(silent=True) or {}).get("source"))
        return _safe("market", _do)
