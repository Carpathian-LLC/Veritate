# market live recorder

Captures the live-only crypto context signals that have no downloadable history, so a
future codec channel / v2 model can train on them once enough has accumulated.

## what it is

A standalone loop ([recorder.py](../../../extensions/canonical/market/server/recorder.py)),
a server module of the market canonical extension, that polls OKX public endpoints on a fixed
interval and appends one snapshot row per pair to `extensions/installed/market/data/live/<SYM>.csv`. `<SYM>` is
the same stem the corpus builder uses
(e.g. `DOGEUSDT`), so a later merge lines up by symbol the way `join_context` already
merges funding/sentiment.

Order-book imbalance, open interest, and live funding are the signals that actually
move short-term direction, but no vendor sells their history for free: they exist only
going forward. Funding and the fear-greed index have downloadable history (see
[series_corpus.md](../../corpus/series_corpus.md)); these three do not.

## how it works

- Pairs: `PAIRS` (10 majors incl DOGE/ETH/BTC); interval `INTERVAL_SEC` (60s).
- Per pair per tick (`_snapshot`): top-`BOOK_DEPTH` order book + open interest + current
  funding from OKX (`/market/books`, `/public/open-interest`, `/public/funding-rate`).
  Row fields: `time` (epoch-ms), `mid`, `spread`, `book_imbalance`
  (`(bid_sz - ask_sz)/total` over the top N levels, in `[-1,1]`), `open_interest`,
  `funding`.
- A failed request drops that pair for that tick (no row), never the whole loop.
- Public data only; no auth. OKX 403s the default urllib User-Agent, so a browser UA is
  sent; the bundled Python lacks CA certs, so an unverified TLS context is used (read-only
  public endpoints).
- run: `python extensions/canonical/market/server/recorder.py` from the repo root (loops until killed).

## dependencies

- OKX public REST (reachable from US IPs; Binance/Bybit are geo-blocked here).
- `extensions/installed/market/data/live/` output (gitignored, local only).

## pitfalls

- Append-only; no dedup across restarts. A merge step should dedup on `time` per symbol.
- OKX swap instId is derived as `<BASE>-USDT-SWAP`; a pair without an OKX swap yields no
  rows.
- Not yet wired into the codec: this is data capture for a future channel, not a live
  serving input. Wiring it in is a format change (a new `BAR_STRIDE` channel).
