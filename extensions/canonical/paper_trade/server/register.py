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

from flask import request

# ------------------------------------------------------------------------------------
# Constants

N_DEFAULT = 10
N_MAX = 30

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
            b0 = next((h.get("btc") for h in tail if h.get("btc")), None)   # buy-hold base price
            bench, last_btc = [], None
            for h in tail:
                last_btc = h.get("btc") or last_btc
                v = round(start_cash * last_btc / b0, 2) if (b0 and last_btc) else None
                bench.append({"t": h["t"], "equity": v})
            return {"ok": True, "running": bool(hist), "start_cash": start_cash,
                    "focus": nt.status().get("focus"),
                    "cash": led.get("cash", 0.0), "equity": eq,
                    "positions": {k: round(v, 6) for k, v in led.get("positions", {}).items() if v},
                    "signal": last.get("signal", {}), "last_t": last.get("t"),
                    "curve": [{"t": h["t"], "equity": h["equity"]} for h in tail],
                    "bench": bench,
                    "series": [{"t": h["t"], "equity": h["equity"], "signal": h.get("signal", {}),
                                "prices": h.get("prices", {})} for h in hist[-240:]],
                    "recent": list(reversed(hist[-12:]))}
        return _safe(_do)
