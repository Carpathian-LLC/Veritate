# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the execution adapter
#   (extensions/canonical/market/server/execution.py): symbol mapping, decision-to-order
#   translation, rebalance planning, and the order payload. Broker network is never hit:
#   pure functions are exercised directly, the Broker's one network seam (_http) is mocked.
# extensions/canonical/market/tests/test_execution.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.normpath(os.path.join(HERE, "..", "server"))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import execution as ex

# ------------------------------------------------------------------------------------
# Fakes

class FakeBroker:
    """Records submitted orders; fixed equity + position. No network."""
    def __init__(self, equity=10000.0, qty=0.0):
        self._equity = equity; self._qty = qty; self.orders = []
    def equity(self):
        return self._equity
    def position_qty(self, asym):
        return self._qty
    def submit_order(self, asym, qty, side, tif=ex.DEFAULT_TIF):
        o = {"symbol": asym, "qty": qty, "side": side}; self.orders.append(o); return o

# ------------------------------------------------------------------------------------
# pure translation

def test_to_alpaca_symbol_maps_quote():
    """Binance-style symbols become Alpaca pairs against USD."""
    assert ex.to_alpaca_symbol("BTCUSDT") == "BTC/USD"
    assert ex.to_alpaca_symbol("ETHUSD") == "ETH/USD"


def test_target_qty_long_scales_by_size():
    """A long decision targets size-fraction of equity converted to quantity."""
    q = ex.target_qty_for({"act": True, "side": "long", "size": 0.5}, 10000.0, 100.0)
    assert q == 50.0


def test_target_qty_non_long_is_flat():
    """Flat, short, and straddle are not executable on spot long-only -> target 0."""
    assert ex.target_qty_for({"act": False}, 10000.0, 100.0) == 0.0
    assert ex.target_qty_for({"act": True, "side": "short", "size": 1.0}, 10000.0, 100.0) == 0.0
    assert ex.target_qty_for({"act": True, "side": "straddle", "size": 1.0}, 10000.0, 100.0) == 0.0


def test_plan_order_buy_sell_and_dust():
    """plan_order buys up to target, sells down to it, and skips sub-min gaps."""
    assert ex.plan_order(0.0, 1.0, 0.01) == {"side": "buy", "qty": 1.0}
    assert ex.plan_order(1.0, 0.4, 0.01) == {"side": "sell", "qty": 0.6}
    assert ex.plan_order(1.0, 1.0, 0.01) is None

# ------------------------------------------------------------------------------------
# rebalance

def test_rebalance_buys_to_reach_target():
    """From flat, a long decision submits a buy for the full target quantity."""
    b = FakeBroker(equity=10000.0, qty=0.0)
    r = ex.rebalance(b, "BTCUSDT", {"act": True, "side": "long", "size": 1.0}, price=100.0)
    assert r["action"] == "buy" and len(b.orders) == 1 and b.orders[0]["side"] == "buy"


def test_rebalance_holds_when_already_at_target():
    """No order when the current position already matches the target exposure."""
    b = FakeBroker(equity=10000.0, qty=100.0)        # 100 units = full equity at price 100
    r = ex.rebalance(b, "BTCUSDT", {"act": True, "side": "long", "size": 1.0}, price=100.0)
    assert r["action"] == "hold" and b.orders == []

# ------------------------------------------------------------------------------------
# Broker (network seam mocked)

def test_submit_order_dry_run_skips_network():
    """dry_run returns a simulated ack and never calls the HTTP seam."""
    b = ex.Broker(key="k", secret="s", dry_run=True)
    b._http = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network called in dry_run"))
    out = b.submit_order("BTC/USD", 0.01, "buy")
    assert out.get("dry_run") is True


def test_submit_order_builds_market_order_payload():
    """A live submit posts a well-formed market order to the orders endpoint."""
    b = ex.Broker(key="k", secret="s", dry_run=False)
    seen = {}
    b._http = lambda method, path, body=None: seen.update(method=method, path=path, body=body) or {"id": "1"}
    b.submit_order("BTC/USD", 0.01, "buy")
    assert seen["method"] == "POST" and seen["path"] == ex.ORDERS
    assert seen["body"]["symbol"] == "BTC/USD" and seen["body"]["side"] == "buy"
    assert seen["body"]["type"] == "market" and seen["body"]["time_in_force"] == ex.DEFAULT_TIF
