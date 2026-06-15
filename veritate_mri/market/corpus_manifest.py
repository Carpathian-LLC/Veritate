# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Standalone CLI. Lists the large experimental market data (raw 1m + 1s + daily pulls +
#   built byte corpus .bin) with absolute paths and sizes, and a suggested Carpathian S3
#   layout. Use it to decide what to upload to S3 and which local copies can be deleted.
# - run: python veritate_mri/market/corpus_manifest.py
# veritate_mri/market/corpus_manifest.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SOURCES = ("crypto", "crypto_1s", "stocks")

# ------------------------------------------------------------------------------------
# Functions

def _dir_stats(path, ext=".csv"):
    if not os.path.isdir(path):
        return 0, 0
    n = b = 0
    for f in os.listdir(path):
        if f.endswith(ext):
            n += 1
            b += os.path.getsize(os.path.join(path, f))
    return n, b


def _gb(b):
    return round(b / 1e9, 3)


def collect():
    out = {"raw": {}, "built": [], "total_gb": 0.0}
    for src in SOURCES:
        d = os.path.join(ROOT, "external_data", src)
        n, b = _dir_stats(d)
        if n:
            out["raw"][src] = {"path": d, "files": n, "gb": _gb(b)}
            out["total_gb"] += _gb(b)
    for src in SOURCES:
        for sp in ("train", "val"):
            p = os.path.join(ROOT, "trainers", "corpus", f"{src}_{sp}.bin")
            if os.path.isfile(p):
                out["built"].append({"path": p, "gb": _gb(os.path.getsize(p))})
                out["total_gb"] += _gb(os.path.getsize(p))
    out["total_gb"] = round(out["total_gb"], 3)
    return out


def main():
    m = collect()
    print("=" * 78)
    print("EXPERIMENTAL MARKET CORPUSES: upload candidates for Carpathian S3")
    print("=" * 78)
    print("\nRAW OHLCV (gitignored, the byte corpus builder reads these):")
    for src, v in m["raw"].items():
        print(f"  {src:8} {v['files']:>4} files  {v['gb']:>7} GB   {v['path']}")
    print("\nBUILT byte-corpus .bin (what the byte model trains on):")
    for v in m["built"]:
        print(f"  {v['gb']:>7} GB   {v['path']}")
    print(f"\nTOTAL large data: {m['total_gb']} GB")
    print("\nSuggested S3 layout:")
    print("  <base>/raw/<src>/<SYM>.csv            (crypto=1m, crypto_1s=1s, stocks=daily)")
    print("  <base>/built/<src>_train.bin , <src>_val.bin")
    if "--json" in sys.argv:
        print("\n" + json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
