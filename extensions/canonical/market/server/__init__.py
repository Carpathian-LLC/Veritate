# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - server modules for the market extension: data layer (data.py), byte-model serving
#   (veritate.py), live feed (live.py), backfill (fetch.py), codec (series_codec.py),
#   corpus builder (build_series_corpus.py), and standalone capture/listing CLIs
#   (recorder.py, bulk_dumps.py, corpus_manifest.py).
# - the registry inserts this dir onto sys.path before importing register.py, so the
#   modules import each other by bare name (import data, import veritate, ...).
# extensions/canonical/market/server/__init__.py
# ------------------------------------------------------------------------------------
