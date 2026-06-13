# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Lists the large experimental market corpuses (raw pulls + built .bin + models) with
#   absolute paths and sizes, and a suggested Carpathian S3 layout. Use this to decide
#   what to upload to S3, then set `market_corpus_s3_url` in settings so the experimental
#   corpus panel can offer downloads from there (and the local copies can be deleted).
# - run: python veritate_mri/market/corpus_manifest.py
# veritate_mri/market/corpus_manifest.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

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
    out = {"raw": {}, "built": [], "models": [], "total_gb": 0.0}
    for src in ("crypto", "stocks"):
        d = os.path.join(ROOT, "external_data", src)
        n, b = _dir_stats(d)
        if n:
            out["raw"][src] = {"path": d, "files": n, "gb": _gb(b)}
            out["total_gb"] += _gb(b)
    for f in ("crypto_train.bin", "crypto_val.bin", "stocks_train.bin", "stocks_val.bin"):
        p = os.path.join(ROOT, "trainers", "corpus", f)
        if os.path.isfile(p):
            out["built"].append({"path": p, "gb": _gb(os.path.getsize(p))})
            out["total_gb"] += _gb(os.path.getsize(p))
    md = os.path.join(ROOT, "models", "market")
    if os.path.isdir(md):
        for f in sorted(os.listdir(md)):
            if f.endswith(".joblib"):
                p = os.path.join(md, f)
                out["models"].append({"path": p, "mb": round(os.path.getsize(p) / 1e6, 1)})
    out["total_gb"] = round(out["total_gb"], 3)
    return out


def main():
    m = collect()
    print("=" * 78)
    print("EXPERIMENTAL MARKET CORPUSES — upload candidates for Carpathian S3")
    print("=" * 78)
    print("\nRAW 1m OHLCV (gitignored, the GBDT platform trains directly from these):")
    for src, v in m["raw"].items():
        print(f"  {src:8} {v['files']:>4} files  {v['gb']:>7} GB   {v['path']}")
    print("\nBUILT byte-corpus .bin (for the byte-model experiment only):")
    for v in m["built"]:
        print(f"  {v['gb']:>7} GB   {v['path']}")
    print("\nTRAINED models/market (small — can ship in-repo or S3):")
    for v in m["models"]:
        print(f"  {v['mb']:>7} MB   {v['path']}")
    print(f"\nTOTAL large data: {m['total_gb']} GB")
    print("\nSuggested S3 layout (set settings.market_corpus_s3_url to the base):")
    print("  <base>/raw/crypto/<SYM>.csv          (or a single crypto_1m.tar.zst)")
    print("  <base>/raw/stocks/<SYM>.csv")
    print("  <base>/built/crypto_train.bin , crypto_val.bin")
    print("  <base>/models/1m_h{5,15,60}.joblib")
    if "--json" in sys.argv:
        print("\n" + json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
