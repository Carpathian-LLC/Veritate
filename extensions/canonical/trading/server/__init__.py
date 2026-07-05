# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - server modules for the Trading extension, one consolidated surface in three groups.
#   market: data layer (data.py), byte-model serving (veritate.py), live feed (live.py),
#   backfill (fetch.py), codec (series_codec.py), policy (policy.py), execution +
#   autotrader (execution.py, autotrader.py), corpus builder (build_series_corpus.py),
#   standalone capture/listing CLIs (recorder.py, bulk_dumps.py, corpus_manifest.py,
#   pull*.py). paper: news scraper with the shared channel registry (scraper.py),
#   sentiment scoring (sentiment.py), the news/xsmom/eqmom paper traders
#   (news_trader.py, xsmom_trader.py, eqmom_trader.py). intel: market scanner with
#   pump flags (scanner.py), local-model briefs (intel.py).
# - the registry inserts this dir onto sys.path before importing register.py, so the
#   modules import each other by bare name (import data, import scraper, ...).
# - data lives under extensions/installed/trading/data/{market,paper,intel}; the
#   platform dataset cache is extensions/installed/trading/data/extension_data.
# extensions/canonical/trading/server/__init__.py
# ------------------------------------------------------------------------------------
