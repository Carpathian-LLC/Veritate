# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Trading-policy layer. Converts the byte model's per-bar forecast into trade
#   decisions and backtests them with fees. The model's validated edge is MAGNITUDE/
#   volatility (expected-|z| vs realized-|z| corr ~0.25-0.30, rising with horizon),
#   while direction is ~coin-flip; so the policy is built around vol-harvesting, with a
#   directional mode kept for comparison.
# - Model-agnostic: operates on a signal series (price, p_up, confidence, exp_move,
#   vol, ret_next), NOT the model. veritate.py produces the signal; this scores policies
#   over it. decide() is the single-bar live decision the MCP trader calls; backtest()
#   is the vectorized historical scorer.
# - Two modes. "vol_harvest": direction-agnostic straddle proxy - buy volatility when the
#   model forecasts a move bigger than the prevailing premium; payoff = |realized move| -
#   premium - fee. Monetizes the magnitude edge. "directional": trade the lean only when
#   the expected move clears the fee and confidence is high, sized by confidence/inverse
#   vol; kept to confirm direction stays unprofitable.
# - Fees are round-trip fractions (0.0020 = 20 bps taker, 0.0005 = aggressive maker).
# extensions/canonical/market/server/policy.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np

# ------------------------------------------------------------------------------------
# Config

DEFAULTS = {
    "mode": "vol_harvest",      # "vol_harvest" | "directional"
    "fee": 0.0005,              # round-trip cost as a fraction
    "conf_gate": 0.0,           # min confidence (0..1) to act
    "move_gate": 1.0,           # vol_harvest: require exp_move >= move_gate * premium
                                # directional: require exp_move >= move_gate * fee
    "premium_window": 96,       # trailing bars defining the vol "premium" (fair straddle cost)
    "premium": None,            # vol_harvest: per-bar implied premium array (e.g. DVOL); None = trailing |ret|
    "sizing": "confidence",     # "fixed" | "confidence" | "vol_target"
    "max_size": 1.0,
    "vol_target": 0.01,         # vol_target sizing: target per-trade risk (return units)
    "stop": None,               # directional: cap per-trade loss at this fraction (None = bar close)
}


def _cfg(overrides):
    c = dict(DEFAULTS)
    if overrides:
        c.update(overrides)
    return c


def _size(conf, vol, c):
    s = c["sizing"]
    if s == "fixed":
        size = np.ones_like(conf)
    elif s == "vol_target":
        size = c["vol_target"] / np.clip(vol, 1e-9, None)
    else:                                   # confidence: scale 0 at 0.5 -> 1 at full conf
        size = np.clip(conf, 0.0, 1.0)
    return np.clip(size, 0.0, c["max_size"])

# ------------------------------------------------------------------------------------
# Backtest (vectorized over a signal series)

def backtest(price, p_up, conf, exp_move, vol, ret_next, **overrides):
    """All inputs are 1-D arrays aligned per bar; ret_next[i] is the realized log return
    from bar i to i+1 (the trade outcome of acting at bar i). Returns metrics + per-bar
    pnl. Trades only gated bars; pnl is net of round-trip fee on the traded size."""
    c = _cfg(overrides)
    price = np.asarray(price, float); p_up = np.asarray(p_up, float)
    conf = np.asarray(conf, float); exp_move = np.asarray(exp_move, float)
    vol = np.asarray(vol, float); ret_next = np.asarray(ret_next, float)
    size = _size(conf, vol, c)
    fee = c["fee"]
    lean = np.sign(p_up - 0.5)

    if c["mode"] == "directional":
        gate = (conf >= c["conf_gate"]) & (exp_move >= c["move_gate"] * fee)
        raw = lean * ret_next
        if c["stop"] is not None:
            raw = np.maximum(raw, -abs(c["stop"]))      # floor the loss at the stop
        pnl = np.where(gate, raw * size - fee * size, 0.0)
    else:                                                # vol_harvest
        prem = (np.asarray(c["premium"], float) if c["premium"] is not None
                else _trailing_mean_abs(ret_next, c["premium_window"]))
        gate = (exp_move >= c["move_gate"] * prem) & (conf >= c["conf_gate"])
        payoff = np.abs(ret_next) - prem                 # long straddle vs the prevailing premium
        pnl = np.where(gate, payoff * size - fee * size, 0.0)

    out = _metrics(pnl, gate, size, ret_next, c)
    out["equity"] = np.cumsum(pnl * 1e4).round(4).tolist()   # cumulative net pnl, bps
    out["gate"] = gate.astype(int).tolist()
    out["lean"] = lean.astype(int).tolist()
    out["size"] = np.round(size, 4).tolist()
    out["pnl_bps"] = np.round(pnl * 1e4, 4).tolist()
    return out


def trades(sig, res, limit=200):
    """Per-trade rows from a backtest `res` and its aligned signal series `sig`: the
    most-recent `limit` gated bars. side is 'straddle' for vol_harvest, else long/short."""
    gate = res["gate"]; lean = res["lean"]; size = res["size"]; pnl = res["pnl_bps"]
    t = sig["t"]; price = sig["price"]; vol_harvest = res["mode"] == "vol_harvest"
    rows = [{"t": t[i], "price": round(price[i], 6),
             "side": "straddle" if vol_harvest else ("long" if lean[i] >= 0 else "short"),
             "lean": "up" if lean[i] >= 0 else "down", "size": size[i], "pnl_bps": pnl[i]}
            for i, g in enumerate(gate) if g]
    return rows[-limit:]


def _trailing_mean_abs(ret, w):
    a = np.abs(ret)
    out = np.full(len(a), np.nan)
    csum = np.concatenate([[0.0], np.cumsum(a)])
    for i in range(len(a)):
        lo = max(0, i - w + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def _metrics(pnl, gate, size, ret_next, c):
    traded = gate & (size > 0)
    r = pnl[traded]
    n = int(traded.sum())
    if n == 0:
        return {"mode": c["mode"], "fee_bps": c["fee"] * 1e4, "n_trades": 0,
                "net": 0.0, "mean_bps": None, "win_rate": None, "sharpe": None,
                "max_dd": 0.0, "exposure": 0.0, "equity": []}
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    sd = float(r.std())
    return {
        "mode": c["mode"], "fee_bps": round(c["fee"] * 1e4, 1),
        "conf_gate": c["conf_gate"], "move_gate": c["move_gate"], "sizing": c["sizing"],
        "n_trades": n,
        "net": float(r.sum()),
        "mean_bps": float(r.mean() * 1e4),
        "win_rate": float((r > 0).mean()),
        "sharpe": (float(r.mean()) / sd) if sd > 0 else None,
        "max_dd": float((peak - eq).max() * 1e4),         # bps, matching mean_bps/equity
        "exposure": float(traded.mean()),
    }

# ------------------------------------------------------------------------------------
# Live decision (single bar; what the MCP trader calls)

def decide(p_up, conf, exp_move, vol, premium=None, **overrides):
    """One-bar trade decision from the current forecast. premium is the trailing vol
    premium (required for vol_harvest). Returns {act, side, size, reason}."""
    c = _cfg(overrides)
    size = float(_size(np.array([conf]), np.array([vol]), c)[0])
    if c["mode"] == "directional":
        if conf < c["conf_gate"] or exp_move < c["move_gate"] * c["fee"]:
            return {"act": False, "reason": "below conf/move gate"}
        return {"act": True, "side": "long" if p_up >= 0.5 else "short",
                "size": size, "reason": "directional lean, move clears fee"}
    prem = premium if premium is not None else c["fee"]
    if conf < c["conf_gate"] or exp_move < c["move_gate"] * prem:
        return {"act": False, "reason": "forecast move below premium gate"}
    return {"act": True, "side": "straddle", "size": size,
            "lean": "long" if p_up >= 0.5 else "short",
            "reason": "forecast move exceeds premium; buy volatility"}
