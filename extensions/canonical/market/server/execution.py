# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Execution adapter for the autonomous paper trader. Thin REST client over the Alpaca
#   PAPER endpoint (simulated money) plus the pure decision-to-order translation the loop
#   uses. SPOT, long-only venue: policy decisions map to a target long exposure (0..1);
#   short and straddle are NOT expressible here (need margin / perps / options), so they
#   map to flat. The honest constraint, surfaced in code.
# - Keys from env (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY) or passed in. Defaults to the
#   paper base URL; live requires paper=False AND keys. Network goes through one seam,
#   Broker._http (mocked in tests). dry_run logs instead of sending. Lazy stdlib-only deps.
# extensions/canonical/market/server/execution.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import ssl
import urllib.request

# ------------------------------------------------------------------------------------
# Constants

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
ORDERS = "/v2/orders"
ACCOUNT = "/v2/account"
POSITIONS = "/v2/positions"
KEY_ENV = "ALPACA_API_KEY_ID"
SECRET_ENV = "ALPACA_API_SECRET_KEY"
QUOTE = "USD"                          # Alpaca crypto quote currency
DEFAULT_TIF = "gtc"                    # valid crypto: gtc, ioc, fok, gtd
MIN_NOTIONAL = 1.0                     # skip rebalances smaller than this (USD), avoids dust orders
TIMEOUT = 15

# ------------------------------------------------------------------------------------
# Pure translation (decision -> order). No network; the testable core.

def to_alpaca_symbol(symbol):
    """Model/Binance symbol -> Alpaca crypto pair: BTCUSDT -> BTC/USD, ETHUSD -> ETH/USD."""
    s = symbol.upper()
    for suffix in ("USDT", "USDC", "USD"):
        if s.endswith(suffix):
            return f"{s[:-len(suffix)]}/{QUOTE}"
    return f"{s}/{QUOTE}"


def target_qty_for(decision, equity, price):
    """Target long quantity from a policy.decide() result on a spot long-only venue.
    Only a directional 'long' is executable here; flat / short / straddle -> 0 (skipped)."""
    if not decision.get("act") or decision.get("side") != "long" or price <= 0:
        return 0.0
    fraction = max(0.0, min(1.0, float(decision.get("size", 0.0))))
    return fraction * equity / price


def plan_order(current_qty, target_qty, min_qty):
    """Order to move current -> target, or None when the gap is below min_qty (dust)."""
    diff = target_qty - current_qty
    if abs(diff) < min_qty:
        return None
    return {"side": "buy" if diff > 0 else "sell", "qty": round(abs(diff), 8)}

# ------------------------------------------------------------------------------------
# Broker (Alpaca REST; paper by default)

class Broker:
    def __init__(self, key=None, secret=None, paper=True, dry_run=False):
        self.key = key or os.environ.get(KEY_ENV)
        self.secret = secret or os.environ.get(SECRET_ENV)
        self.base = PAPER_URL if paper else LIVE_URL
        self.paper = paper
        self.dry_run = dry_run

    def _http(self, method, path, body=None):
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={
            "Apca-Api-Key-Id": self.key or "", "Apca-Api-Secret-Key": self.secret or "",
            "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as r:
            return json.loads(r.read())

    def _require_keys(self):
        if not self.key or not self.secret:
            raise RuntimeError(f"Alpaca keys missing: set {KEY_ENV} and {SECRET_ENV} (free paper keys).")

    def account(self):
        self._require_keys()
        return self._http("GET", ACCOUNT)

    def equity(self):
        return float(self.account()["equity"])

    def position_qty(self, alpaca_symbol):
        self._require_keys()
        for p in self._http("GET", POSITIONS):
            if p.get("symbol") in (alpaca_symbol, alpaca_symbol.replace("/", "")):
                return float(p.get("qty", 0.0))
        return 0.0

    def submit_order(self, alpaca_symbol, qty, side, tif=DEFAULT_TIF):
        order = {"symbol": alpaca_symbol, "qty": str(qty), "side": side,
                 "type": "market", "time_in_force": tif}
        if self.dry_run:
            return {"dry_run": True, **order}
        self._require_keys()
        return self._http("POST", ORDERS, order)


def rebalance(broker, symbol, decision, price):
    """Drive the spot position toward the decision's target exposure with one market order.
    Returns the action taken (hold / buy / sell) and the broker response."""
    asym = to_alpaca_symbol(symbol)
    equity = broker.equity()
    current = broker.position_qty(asym)
    target = target_qty_for(decision, equity, price)
    plan = plan_order(current, target, MIN_NOTIONAL / price if price > 0 else MIN_NOTIONAL)
    if plan is None:
        return {"action": "hold", "symbol": asym, "current": current, "target": target}
    order = broker.submit_order(asym, plan["qty"], plan["side"])
    return {"action": plan["side"], "symbol": asym, "qty": plan["qty"],
            "current": current, "target": target, "order": order}
