# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Backend for the experimental /market decision-support dashboard. Serves GBDT
#   forecasts (volatility + calibrated direction), a forward probability cone, regime,
#   EV/Kelly sizing, and a per-instrument honest backtest replay.
# - Fully isolated from the canonical trainer/chat/RAG pipelines: it only reads
#   external_data CSVs and models/market/*.joblib. Lazy imports so a missing sklearn
#   never breaks dashboard startup.
# veritate_mri/routes/market_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import math
import os

from flask import request

from readers import paths

from ._common import safe_route as _safe

# ------------------------------------------------------------------------------------
# Constants

EXTERNAL_DIR = os.path.join(paths.REPO_ROOT, "external_data")
BASE_SEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
SCAN_MAJORS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "DOTUSDT", "ATOMUSDT", "NEARUSDT",
    "UNIUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "INJUSDT", "SUIUSDT", "XLMUSDT", "HBARUSDT", "RUNEUSDT", "AAVEUSDT", "ICPUSDT",
]
CONE_LEVELS = (0.5, 0.8, 0.95)
DEFAULT_COST_BPS = 10.0
CANDLES_OUT = 120

# ------------------------------------------------------------------------------------
# Helpers

def _market():
    """Lazy import of the isolated market package (models, data, backtest)."""
    from market import backtest as bt
    from market import data as md
    from market import models as mk
    return mk, md, bt


def _instruments(source):
    d = os.path.join(EXTERNAL_DIR, source)
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".csv")) if os.path.isdir(d) else []


def _kelly(p_up, exp_move, cost):
    """Symmetric-payoff fractional Kelly + expected value (in return units)."""
    edge = (p_up - 0.5) * 2.0
    side = 1 if edge >= 0 else -1
    p = p_up if side > 0 else (1.0 - p_up)
    full_kelly = max(0.0, 2.0 * p - 1.0)
    ev = abs(edge) * exp_move - cost
    return {"side": side, "edge": edge, "kelly_quarter": 0.25 * full_kelly,
            "ev": ev, "ev_bps": ev * 1e4}

# ------------------------------------------------------------------------------------
# Routes

def register(app):
    @app.route("/market/status")
    def market_status():
        def _do():
            mk, md, bt = _market()
            avail = mk.MarketModel.available()
            summ = {}
            sp = os.path.join(mk.MODEL_DIR, "summary.json")
            if os.path.isfile(sp):
                import json
                summ = json.load(open(sp))
            crypto = _instruments("crypto")
            return {"ok": True, "models": avail, "summary": summ,
                    "instruments": {"crypto": len(crypto), "stocks": len(_instruments("stocks"))},
                    "cone_levels": list(CONE_LEVELS)}
        return _safe("market", _do)

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
            n = min(20000, max(300, int(request.args.get("n", 1500))))
            if not model_name or model_name not in vz.list_models():
                return ({"ok": False, "error": "pick a trained Veritate model."}, 400)
            df = md.load_tail(symbol, n_bars=n, base="1m", source=source)
            if df is None:
                return ({"ok": False, "error": f"no data for {symbol}."}, 404)
            model, seq, step = vz.load_model(model_name)
            g = vz.hindcast(model, seq, df)
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
            n = min(20000, max(300, int(request.args.get("n", 1500))))
            if not model_name or model_name not in vz.list_models():
                return ({"ok": False, "error": "pick a trained Veritate model."}, 400)
            df = md.load_tail(symbol, n_bars=n, base="1m", source=source)
            if df is None:
                return ({"ok": False, "error": f"no data for {symbol}."}, 404)
            model, seq, step = vz.load_model(model_name)
            g = vz.benchmark(model, seq, df)
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
            model, seq, step = vz.load_model(model_name)
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
            pred = vz.predict_next(model, seq, closed)
            if pred is None:
                return ({"ok": False, "error": "prediction failed."}, 500)
            last_close = float(closed["close"].iloc[-1])
            last_t = int(closed.index[-1].value // 1_000_000_000)
            pred.update({"ok": True, "symbol": symbol, "model": model_name, "last_close": last_close,
                         "last_t": last_t, "expected_move_bps": pred["expected_move"] * 1e4})
            return pred
        return _safe("market", _do)

    @app.route("/market/scan")
    def market_scan():
        def _do():
            mk, md, bt = _market()
            base = request.args.get("base", "1m")
            horizon = int(request.args.get("horizon", 15))
            source = request.args.get("source", "crypto")
            if f"{base}_h{horizon}" not in mk.MarketModel.available():
                return ({"ok": False, "error": f"no model {base}_h{horizon} trained yet."}, 404)
            mm = mk.MarketModel.load(base, horizon)
            avail = set(_instruments(source))
            syms = [s for s in SCAN_MAJORS if s in avail][:24]
            cost = (DEFAULT_COST_BPS * 1e-4) * 2.0
            rows = []
            for s in syms:
                try:
                    df = md.load_tail(s, n_bars=200, base=base, source=source)
                    if df is None:
                        continue
                    p = mm.predict_latest(df)
                    if not p:
                        continue
                    k = _kelly(p["p_up"], p["vol_fwd"], cost)
                    rows.append({"symbol": s, "p_up": p["p_up"], "confidence": p["confidence"],
                                 "vol_fwd_bps": p["vol_fwd"] * 1e4, "regime": p["regime"],
                                 "ev_bps": k["ev_bps"], "side": k["side"],
                                 "last_close": float(df["close"].iloc[-1])})
                except Exception:
                    continue
            rows.sort(key=lambda r: r["confidence"], reverse=True)
            return {"ok": True, "base": base, "horizon": horizon, "n": len(rows), "rows": rows}
        return _safe("market", _do)

    @app.route("/market/corpus")
    def market_corpus():
        def _do():
            from market import corpus_manifest as cm
            from runtime import settings as settings_mod
            m = cm.collect()
            m["ok"] = True
            m["s3_url"] = (settings_mod.get().get("market_corpus_s3_url") or "").strip()
            return m
        return _safe("market", _do)

    @app.route("/market/instruments")
    def market_instruments():
        def _do():
            source = request.args.get("source", "crypto")
            return {"ok": True, "source": source, "instruments": _instruments(source)}
        return _safe("market", _do)

    @app.route("/market/forecast")
    def market_forecast():
        def _do():
            mk, md, bt = _market()
            source = request.args.get("source", "crypto")
            symbol = request.args.get("symbol", "BTCUSDT")
            base = request.args.get("base", "1m")
            horizon = int(request.args.get("horizon", 15))
            if f"{base}_h{horizon}" not in mk.MarketModel.available():
                return ({"ok": False, "error": f"no model {base}_h{horizon} trained yet."}, 404)
            mm = mk.MarketModel.load(base, horizon)
            df = md.load_tail(symbol, n_bars=max(CANDLES_OUT, 240), base=base, source=source)
            if df is None or len(df) < mm.features.__len__():
                return ({"ok": False, "error": f"not enough data for {symbol}."}, 404)
            pred = mm.predict_latest(df)
            if pred is None:
                return ({"ok": False, "error": "feature warmup failed."}, 500)

            last_close = float(df["close"].iloc[-1])
            last_t = int(df.index[-1].value // 1_000_000_000)
            sec = BASE_SEC.get(base, 60)
            # candles for the chart
            tail = df.iloc[-CANDLES_OUT:]
            candles = [{"t": int(ix.value // 1_000_000_000), "o": float(o), "h": float(h),
                        "l": float(l), "c": float(c)}
                       for ix, o, h, l, c in zip(tail.index, tail["open"], tail["high"],
                                                 tail["low"], tail["close"])]
            # cone -> price space (anchor at last close, project H steps)
            cone = mm.cone(pred["vol_fwd"], pred["p_up"], levels=CONE_LEVELS)
            cone_px = {}
            for lv, band in cone.items():
                pts = [{"t": last_t + (k + 1) * sec,
                        "lo": last_close * math.exp(lo), "hi": last_close * math.exp(hi),
                        "mid": last_close * math.exp((lo + hi) / 2.0)}
                       for k, (lo, hi) in enumerate(band)]
                cone_px[str(lv)] = pts
            cost = (DEFAULT_COST_BPS * 1e-4) * 2.0
            kelly = _kelly(pred["p_up"], pred["vol_fwd"], cost)
            return {"ok": True, "symbol": symbol, "base": base, "horizon": horizon,
                    "last_close": last_close, "last_t": last_t, "step_sec": sec,
                    "vol_fwd": pred["vol_fwd"], "vol_fwd_bps": pred["vol_fwd"] * 1e4,
                    "p_up": pred["p_up"], "confidence": pred["confidence"],
                    "regime": pred["regime"], "candles": candles, "cone": cone_px,
                    "decision": kelly, "metrics": mm.metrics}
        return _safe("market", _do)

    @app.route("/market/live")
    def market_live():
        def _do():
            mk, md, bt = _market()
            from market import live as lv
            symbol = request.args.get("symbol", "BTCUSDT")
            base = request.args.get("base", "1m")
            horizon = int(request.args.get("horizon", 15))
            if f"{base}_h{horizon}" not in mk.MarketModel.available():
                return ({"ok": False, "error": f"no model {base}_h{horizon} trained yet."}, 404)
            mm = mk.MarketModel.load(base, horizon)
            try:
                closed, forming, pred = lv.predict(mm, symbol)
            except Exception as e:
                return ({"ok": False, "error": f"live feed error ({symbol} may not be on Binance.US): {e}"}, 502)
            if pred is None:
                return ({"ok": False, "error": f"no live data for {symbol}."}, 404)

            last_close = float(closed["close"].iloc[-1])
            last_t = int(closed.index[-1].value // 1_000_000_000)
            sec = BASE_SEC.get(base, 60)
            tail = closed.iloc[-CANDLES_OUT:]
            candles = [{"t": int(ix.value // 1_000_000_000), "o": float(o), "h": float(h),
                        "l": float(l), "c": float(c)}
                       for ix, o, h, l, c in zip(tail.index, tail["open"], tail["high"],
                                                 tail["low"], tail["close"])]
            cone = mm.cone(pred["vol_fwd"], pred["p_up"], levels=CONE_LEVELS)
            cone_px = {str(lv): [{"t": last_t + (k + 1) * sec,
                                  "lo": last_close * math.exp(lo), "hi": last_close * math.exp(hi),
                                  "mid": last_close * math.exp((lo + hi) / 2.0)}
                                 for k, (lo, hi) in enumerate(band)]
                       for lv, band in cone.items()}
            cost = (DEFAULT_COST_BPS * 1e-4) * 2.0
            return {"ok": True, "live": True, "symbol": symbol, "base": base, "horizon": horizon,
                    "last_close": last_close, "last_t": last_t, "step_sec": sec, "forming": forming,
                    "vol_fwd": pred["vol_fwd"], "vol_fwd_bps": pred["vol_fwd"] * 1e4,
                    "p_up": pred["p_up"], "confidence": pred["confidence"], "regime": pred["regime"],
                    "candles": candles, "cone": cone_px, "decision": _kelly(pred["p_up"], pred["vol_fwd"], cost),
                    "metrics": mm.metrics}
        return _safe("market", _do)

    @app.route("/market/hindcast")
    def market_hindcast():
        def _do():
            mk, md, bt = _market()
            source = request.args.get("source", "crypto")
            symbol = request.args.get("symbol", "BTCUSDT")
            base = request.args.get("base", "1m")
            horizon = int(request.args.get("horizon", 15))
            n = min(20000, max(300, int(request.args.get("n", 1500))))
            if f"{base}_h{horizon}" not in mk.MarketModel.available():
                return ({"ok": False, "error": f"no model {base}_h{horizon} trained yet."}, 404)
            mm = mk.MarketModel.load(base, horizon)
            df = md.load_tail(symbol, n_bars=n, base=base, source=source)
            if df is None:
                return ({"ok": False, "error": f"no data for {symbol}."}, 404)
            res = bt.hindcast(mm, df)
            if res is None:
                return ({"ok": False, "error": "not enough bars to hindcast."}, 404)
            res.update({"ok": True, "symbol": symbol, "base": base, "horizon": horizon})
            return res
        return _safe("market", _do)

    @app.route("/market/backtest")
    def market_backtest():
        def _do():
            mk, md, bt = _market()
            source = request.args.get("source", "crypto")
            symbol = request.args.get("symbol", "BTCUSDT")
            base = request.args.get("base", "1m")
            horizon = int(request.args.get("horizon", 15))
            n = min(40000, max(500, int(request.args.get("n", 6000))))
            if f"{base}_h{horizon}" not in mk.MarketModel.available():
                return ({"ok": False, "error": f"no model {base}_h{horizon} trained yet."}, 404)
            mm = mk.MarketModel.load(base, horizon)
            df = md.load_tail(symbol, n_bars=n, base=base, source=source)
            if df is None:
                return ({"ok": False, "error": f"no data for {symbol}."}, 404)
            res = bt.replay(mm, df, cost_bps=DEFAULT_COST_BPS)
            if res is None:
                return ({"ok": False, "error": "not enough bars to replay."}, 404)
            res.update({"ok": True, "symbol": symbol, "base": base, "horizon": horizon})
            return res
        return _safe("market", _do)
