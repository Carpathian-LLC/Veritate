# fetch.py: on-demand crypto backfill for the market page

Makes the `/market` page work on a fresh install with no manual data. `external_data/` is
gitignored and ships empty; when the data layer needs a crypto symbol that is not cached,
`fetch.py` pulls it from Binance and writes it in the schema `data.py` reads.

## How it works
- `ensure(symbol, source, need_1m, path)` is the entry point, called by `data.load_tail` when the
  local CSV is missing. Crypto only (returns `False` for any other source). If the file already
  exists it returns immediately, so existing local data is never re-fetched or clobbered.
- `_klines_1m(symbol, need)` pages Binance 1m klines backward (`limit=1000` per call, `endTime`
  cursor) until `need` **traded** bars or history runs out, capped at `MAX_1M` (15000, covers the
  1-week max window). Always fetches **1m** (the native resolution); coarser resolutions resample
  in `data.py`. **Zero-volume bars are dropped at parse time.** `api.binance.us` (the only
  US-reachable host) is a low-liquidity venue: ~50-75% of recent 1m bars on majors (BTC, ETH, SOL)
  are synthetic no-trade fill bars with `open==high==low==close` and `volume 0`. They carry no
  signal and would dominate any live or held-out-via-fetch eval, so they are not emitted. The
  `need` budget counts kept bars, so a backfill still returns `need` real traded bars (paging just
  reaches further back). The `endTime` cursor still steps off the earliest **raw** bar in each
  page, so dropping bars never stalls paging.
- Hosts (`HOSTS`) are tried in order per symbol: `api.binance.com` (global) then `api.binance.us`
  (US; `api.binance.com` is HTTP 451 geo-blocked in the US). The first host that answers is reused
  for the rest of the paging.
- Writes are atomic (`tmp` + `os.replace`) so concurrent fetches of the same symbol (the overlay
  and multi-instrument grid can race) never leave a half-written file.
- `fetch.MAJORS` is the default fetchable symbol list; `data.list_instruments("crypto")` unions it
  with local files so the picker is populated before anything is cached.

## Hosted fallback
When the Binance API is unreachable, `ensure` falls back to `_hosted`, which downloads a full CSV
from a URL listed in `market/market_data_catalog.json` (`{source: {SYMBOL: url}}`). The catalog
ships empty: the operator fills it (S3, CDN, GitHub release) to enable the fallback or to serve any
non-crypto data. URLs must point at the same schema (`time,open,high,low,close,volume`, epoch time).

## Pitfalls
- US reachability: `api.binance.us` is US-only; a non-US host relies on `api.binance.com`. Both
  blocked (or offline) => fetch fails and `ensure` returns `False` unless the hosted catalog covers
  the symbol. `load_tail` then returns `None` and the page shows a clean "no data" state.
- First fetch of a large window is several API calls (a 1-week 1m backfill is ~11 calls, ~1s);
  it blocks that one request, then is cached. Not a background job.
- Stocks have no public source wired; `ensure` does not fetch them. They were dropped from the UI.
- Low-liquidity venue limitation: even after dropping zero-volume bars, ~15-22% of the kept bars
  are flat (`o==h==l==c`) with **nonzero** volume — real trades that did not move price. Those are
  legitimate microstructure on `api.binance.us` and are kept. The fetched series is therefore
  sparser in wall-clock time than a continuous 1m grid (gaps where every bar in a span was
  zero-volume); the codec/resampler in `data.py` keys off timestamps, so the gaps are handled.
