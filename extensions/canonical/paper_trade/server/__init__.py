# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Server modules for the Paper Trading extension: news/sentiment scraper (scraper.py),
#   sentiment scoring + aggregation via /teacher/complete (sentiment.py), route entry
#   point (register.py). The registry inserts this dir onto sys.path before importing
#   register.py, so modules import each other by bare name.
# extensions/canonical/paper_trade/server/__init__.py
# ------------------------------------------------------------------------------------
