# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Live OKX context recorder. The signals that move short-term crypto direction
#   (order-book imbalance, open interest, live funding) have NO downloadable history:
#   they exist only going forward. This appends a periodic snapshot per pair so a
#   future codec channel / v2 model can train on them once enough has accumulated.
# - One row per tick per pair: mid, spread, top-N book imbalance, open interest,
#   funding. Appended to data/live/<SYM>.csv (the same SYM stem the corpus builder
#   uses, so a later join_context-style merge lines up by symbol).
# - Public market data only (no auth). The bundled Python here lacks CA certs, so
#   an unverified TLS context is used for these read-only public endpoints.
# - run: python extensions/canonical/market/server/recorder.py   (loops until killed)
# extensions/canonical/market/server/recorder.py
# ------------------------------------------------------------------------------------
# Imports:

import csv
import json
import os
import ssl
import time
import urllib.request

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "extensions", "installed", "market", "data", "live")

OKX = "https://www.okx.com/api/v5"
PAIRS = ["DOGEUSDT", "ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT",
         "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "BNBUSDT"]
INTERVAL_SEC = 60
BOOK_DEPTH = 20
REQ_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
FIELDS = ["time", "mid", "spread", "book_imbalance", "open_interest", "funding"]
_CTX = ssl._create_unverified_context()

# ------------------------------------------------------------------------------------
# Functions

def _inst(sym):
    return f"{sym[:-4]}-USDT-SWAP" if sym.endswith("USDT") else sym


def _get(path):
    req = urllib.request.Request(OKX + path, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=REQ_TIMEOUT, context=_CTX) as r:
        return json.load(r).get("data") or []


def _snapshot(inst):
    book = _get(f"/market/books?instId={inst}&sz={BOOK_DEPTH}")
    oi = _get(f"/public/open-interest?instType=SWAP&instId={inst}")
    fund = _get(f"/public/funding-rate?instId={inst}")
    if not book or not book[0].get("bids") or not book[0].get("asks"):
        return None
    bids = book[0]["bids"]; asks = book[0]["asks"]
    bid = float(bids[0][0]); ask = float(asks[0][0])
    bid_sz = sum(float(b[1]) for b in bids); ask_sz = sum(float(a[1]) for a in asks)
    tot = bid_sz + ask_sz
    return {
        "time": int(time.time() * 1000),
        "mid": (bid + ask) / 2.0,
        "spread": ask - bid,
        "book_imbalance": (bid_sz - ask_sz) / tot if tot > 0 else 0.0,
        "open_interest": float(oi[0]["oi"]) if oi else "",
        "funding": float(fund[0]["fundingRate"]) if fund else "",
    }


def _append(sym, row):
    path = os.path.join(OUT_DIR, f"{sym}.csv")
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"recorder: {len(PAIRS)} pairs every {INTERVAL_SEC}s -> {OUT_DIR}", flush=True)
    while True:
        t0 = time.time()
        ok = 0
        for sym in PAIRS:
            try:
                row = _snapshot(_inst(sym))
            except Exception:
                row = None
            if row:
                _append(sym, row)
                ok += 1
        print(f"tick: {ok}/{len(PAIRS)} pairs", flush=True)
        time.sleep(max(1.0, INTERVAL_SEC - (time.time() - t0)))


if __name__ == "__main__":
    main()
