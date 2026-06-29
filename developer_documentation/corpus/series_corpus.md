# price-series corpus

Byte-level corpus that turns numeric OHLCV history (stocks, crypto, any price
series) into an autocomplete stream the model predicts forward. The headline task
is next-bar prediction from price + volume alone, no news or semantics.

## what it is

Each bar is reduced to nine scale-free or categorical features and encoded as a
fixed-stride sequence of printable bytes. One instrument is one sequence;
instruments are concatenated, newline-separated, into a single byte stream. The
model autocompletes the next bar's bytes; the first byte of each bar is the
return bucket, so direction and magnitude fall out of the decode directly.

Channels are an additive prefix: a model serves at the stride it trained on, so
adding a channel never breaks an older model (it reads fewer chars). The format
has grown 3 -> 5 -> 7 -> 9 channels; current `BAR_STRIDE=9`.

Corpora share the identical format, one per `--source` in the builder's
`LOADERS` table ([build_series_corpus.py](../../extensions/canonical/market/server/build_series_corpus.py)).
Two loaders back every source: `load_stock_csv` (schema
`date,open,high,low,close,adjclose,volume`) and `load_crypto_csv` (schema
`time,open,high,low,close,volume`, time as numeric epoch).
- `stocks` — S&P 500 daily bars (adjusted close).
- `stocks_1m` — intraday 1-minute stock bars (stock schema). Free 1m stock history
  is not available; this source is reserved for a paid feed.
- `indices` — daily cash index levels (^GSPC, ^NDX, ...) from the Yahoo v8 chart
  feed; `adjclose` mirrors `close`. Puller: `extensions/canonical/market/server/pull_yahoo.py indices`.
- `futures` — daily Yahoo continuous front-month contracts (ES=F, CL=F, ...), the
  only free continuous series; `adjclose` mirrors `close`. Puller:
  `extensions/canonical/market/server/pull_yahoo.py futures`.
- `crypto` — major Binance USDT pairs at 1-minute resolution.
- `crypto_extra` — every other Binance Vision USDT pair (full archive minus the
  ~200 already in `crypto`). Puller: `extensions/canonical/market/server/pull_binance.py --full`.
- `crypto_1s` — the same pairs at 1-second resolution (the high-frequency tape).
  Reads `extensions/installed/market/data/crypto_1s/*.csv`; shares the crypto loader. The serving
  layer routes a `base=1s` request to this `_1s` tape and reads it raw with no
  resample ([data.py:42](../../extensions/canonical/market/server/data.py#L42)).
- `forex` — 1-minute major-pair bars resampled from Dukascopy ticks (crypto
  schema; volume is tick count). Puller: `extensions/canonical/market/server/pull_forex.py`. Writes
  per pair incrementally, flushed per completed month; `<pair>_done.json` tracks
  finished months so a killed run resumes mid-pair. Defaults to the last 3 years
  (~1h/pair/year sequential at ~0.53s/hourly fetch); `--start`/`--end` widen it.

## how it works

Codec and feature math: [series_codec.py](../../extensions/canonical/market/server/series_codec.py).
Builder: [build_series_corpus.py](../../extensions/canonical/market/server/build_series_corpus.py).

**Features per bar (9). The scale-free ones are comparable across eras; the rest
are categorical:**
- return z-score: close-to-close return divided by its trailing-window std.
- range ratio: intraday `(high-low)/prev_close` divided by its trailing mean.
- relative volume: volume divided by its trailing mean.
- realized-vol ratio: trailing return std (`RV_WINDOW`) divided by its own longer
  trailing mean (`RV_REF_WINDOW`). Measures volatility relative to recent
  volatility, so it spikes on a regime jump and settles toward 1.0. Distinct from
  the per-bar return z-score, which divides each single return by trailing std.
- session: UTC hour-of-day of the bar, encoded as bin `1..24`. Bin `0`
  (`SESSION_NONE`) is the fallback for sources without a usable timestamp, so the
  stride stays fixed-width regardless of source.
- buy pressure: taker-buy volume divided by total volume (the share of the bar's
  volume that was aggressive buying), in `[0,1]`, bucketed `1..BP_LEVELS`. Order-flow
  imbalance that price/volume alone cannot show.
- trade activity: trade count divided by its trailing mean, bucketed `1..TR_LEVELS`.
  Activity relative to recent activity.
- funding regime: the perp 8h funding rate, signed, bucketed `1..FND_LEVELS` over a
  centered clip range (`+/-FND_CLIP`). Crowded-long vs crowded-short positioning that
  price alone cannot show; an external-to-the-chart signal.
- sentiment: the crypto fear-greed index `0..100`, bucketed `1..FG_LEVELS`. Slow
  regime context (extreme-fear vs extreme-greed).

Buy pressure, trade activity, funding and sentiment each reserve bin `0`
(`CHAN_NONE`) for "input absent", so a source without taker / trade-count / funding
/ sentiment data encodes the constant fallback byte and the stride stays
fixed-width. Funding and sentiment are crypto-only context (see below).

**Timestamp + taker threading.** Five channels are input-derived beyond OHLCV:
session (timestamp), buy pressure (taker-buy volume), trade activity (trade count),
funding regime (perp funding rate), sentiment (fear-greed index). The builder reads
the bar timestamp (crypto epoch-ms `time` column, stock ISO `date` column) and, when
present, the `taker_buy` and `trades` columns of the CSV, passing them as `ts_ns` /
`tb` / `ntr` to `compute_features`. The Market LLM page passes
`market.data.index_ns(df.index)` plus the `taker_buy` / `trades` columns when the
cached CSV carries them (Binance-fetched files do; legacy OHLCV-only files do not).
Any absent input encodes its fallback bin for every bar.

**Context channels (funding, sentiment).** These two are not in the OHLCV CSV; they
are joined from separate sources by `market.data.join_context(df, symbol)`
([data.py](../../extensions/canonical/market/server/data.py)), shared by the builder and the serving
path so the on-disk and live formats never diverge. Funding is per-symbol
(`extensions/installed/market/data/funding/<SYM>.csv`, columns `time,funding`); sentiment is global
(`extensions/installed/market/data/sentiment/fng.csv`, columns `time,value`). The join is
`reindex(df.index, method="ffill")`: each bar carries the last context value at or
before its timestamp (no lookahead; never `bfill`). The join is gated to crypto
sources (`market.data.CRYPTO_SOURCES`) since funding is a perp rate and the index is
crypto-specific; forex/stock sources skip it and the channels encode the absent bin.
Coverage is partial by nature (funding history starts 2020, the index 2018), so
pre-coverage bars carry the absent bin.

**No lookahead.** Normalization at bar `t` uses strictly trailing windows
(`FEAT_WINDOW` / `RV_WINDOW` / `RV_REF_WINDOW`, bars before `t` only) —
[series_codec.py compute_features](../../extensions/canonical/market/server/series_codec.py).
The session and buy-pressure bytes are derived only from bar `t`'s own input; trade
activity uses a strictly trailing `FEAT_WINDOW` mean. Guaranteed by
`test_no_lookahead` (which threads the taker inputs) and
`test_realized_vol_channel_no_lookahead` in
[test_series_codec.py](../../extensions/canonical/market/tests/test_series_codec.py).

**Quantization.** Each numeric feature maps to a bucket over a fixed clip range
(`RET_BINS=33` centered, `RNG_BINS=16`, `VOL_BINS=16`, `RV_BINS=16`,
`BP_LEVELS=16`, `TR_LEVELS=16`, `FND_LEVELS=16` over `+/-FND_CLIP`, `FG_LEVELS=16`
over `0..100`), then to one char of a 64-char `ALPHABET`. The session, buy-pressure,
trade-activity, funding and sentiment channels map their pre-computed bin index
straight to a char (`0` = absent). A bar is `BAR_STRIDE=9` chars; position in the
stride is the feature id (0 return, 1 range, 2 volume, 3 realized-vol, 4 session,
5 buy pressure, 6 trade activity, 7 funding, 8 sentiment). Return bucket
`RET_CENTER` is flat; above is up, below is down.

**Encoding is anonymous.** No ticker label, so the model learns one
instrument-agnostic tape dynamic rather than memorizing a symbol.

**Time split.** Per instrument, the oldest `1-val_ratio` of bars are train, the
newest are val — the val set is the held-out future. The codec contract is shared
with the Market LLM page, so the live tester encodes a window and decodes the
model's continuation through the same functions.

## build

Raw data lives in the gitignored `extensions/installed/market/data/{stocks,crypto,crypto_1s}/`. Then:

```
python extensions/canonical/market/server/build_series_corpus.py --source stocks
python extensions/canonical/market/server/build_series_corpus.py --source crypto
python extensions/canonical/market/server/build_series_corpus.py --source crypto_1s
```

Output: `trainers/corpus/<source>_train.bin` + `_val.bin`
(`crypto_1s_*` is the largest, the 1-second tape).

**Coarser horizons (`--horizon`).** `--horizon {1m,5m,15m,1h,4h,1d}` resamples the
1m source bars to the named horizon before encoding (crypto sources;
`market.data.resample`, right-edge, no lookahead) and writes a distinct stem
`<source>_<horizon>_*.bin`. Coarser bars trade data volume for a stronger
magnitude/funding signal per bar; mix horizons to recover volume:

```
python extensions/canonical/market/server/build_series_corpus.py --source crypto_of --horizon 15m
python extensions/canonical/market/server/build_series_corpus.py --source crypto_of --horizon 5m
python extensions/canonical/market/server/build_series_corpus.py --source crypto_of --horizon 1h
```

Train on the mix with explicit weights (val = first stem): e.g. `--corpus
"crypto_of_15m:0.45,crypto_of_5m:0.35,crypto_of_1h:0.20"`.

**Order-flow corpus (real buy-pressure + trade-activity channels).** Legacy
`extensions/installed/market/data/crypto/` CSVs are OHLCV-only, so those two channels encode the
absent-fallback byte. To build a corpus where they carry real signal, first pull
Binance 1m bars *with* the `taker_buy` / `trades` fields, then build that source:

```
python extensions/canonical/market/server/fetch.py BTCUSDT ETHUSDT SOLUSDT --bars 400000 --source crypto_extra
python extensions/canonical/market/server/build_series_corpus.py --source crypto_extra
```

A before/after on whether order flow helps is two corpora, identical except
for those channels: `crypto` (fallback bytes) vs `crypto_extra` (real values).

**Context-channel corpus (funding + sentiment).** `extensions/installed/market/data/crypto_of/` carries
the 40 majors with order-flow columns; pulling funding
(`extensions/installed/market/data/funding/<SYM>.csv`, Binance Vision monthly dumps 2020+) and the
fear-greed index (`extensions/installed/market/data/sentiment/fng.csv`, alternative.me, 2018+) lets the
builder populate channels 7-8 for those pairs. Any crypto build then auto-joins them
via `join_context`; pairs/eras without data encode the absent bin.

## planned: cross-asset lead-lag channel (design, not implemented)

A sixth channel would encode each follower bar's recent co-movement with a
*leader* series (BTC for alts, SPY/sector for a stock). This is NOT built: the
builder encodes every instrument independently and anonymously, and lead-lag needs
a timestamp-aligned leader, which breaks that independence. Sketch for when it is
worth building:

- **Leader resolution.** Add a per-source leader map (follower symbol -> leader
  symbol) at builder scope. A leader is loaded once per source and reused across
  followers; a follower that is its own leader (BTC, SPY) gets the `SESSION_NONE`-
  style fallback bin.
- **Alignment.** Load both series as timestamp-indexed frames (reuse
  `market.data.load_1m`, which already returns a UTC `DatetimeIndex`).
  `reindex` the leader onto the follower's index with `method="ffill"` so each
  follower bar at time `t` sees the leader's last bar at-or-before `t`. Never
  interpolate forward.
- **Feature.** Trailing-window correlation (or lagged sign agreement) between
  follower returns and leader returns over `FEAT_WINDOW`, bucketed scale-free like
  the other channels.
- **No lookahead (the gating part).** The leader value used at follower bar `t`
  must be the leader bar at-or-before `t` (ffill, never bfill), and the correlation
  window is strictly trailing. The follower's own `ts_ns` is already threaded; the
  leader must be sliced to `<= t` before the window reduce. A truncation test
  identical to `test_no_lookahead` (drop late follower bars, leader frame
  unchanged, earlier bytes bitwise stable) is the acceptance gate.
- **Builder changes.** `build()` would resolve and cache the leader per source,
  pass an aligned leader-return array into `compute_features`, and the codec gains
  one more bucketed channel (`BAR_STRIDE` -> 10, another format change). The leader
  cache keeps memory flat; the alignment is one `reindex` per follower (the same
  `reindex(method="ffill")` shape `join_context` already uses for funding/sentiment).

The hard cost is correctness: a single bfill or an off-by-one in the alignment
silently leaks the future. Build only with the truncation gate in place.

## dependencies

- `extensions/installed/market/data/` raw CSVs (gitignored, local only).
- `numpy` for feature math.
- `market.data.normalize_time` / `index_ns` for timestamp conversion (session
  channel).
- Consumed by the corpus loader (training) and the Market LLM page (eval).

## pitfalls

- **Format change (7 -> 9 bytes/bar), serving is non-breaking.** `BAR_STRIDE` is now 9
  (added funding regime + sentiment). Each model records the stride it trained
  against (`bar_stride`, stamped by `training/save.py`), and the Market LLM page encodes
  every model at its own stride (`encode_sequence(..., stride=k)` emits the first `k`
  channels). So an existing stride-5 or stride-7 model keeps serving correctly against
  the stride-9 codec: it simply never sees the new channels. Models saved before
  `bar_stride` existed default to `LEGACY_STRIDE=5`.
- **Stamp = codec stride at save time.** Build the corpus and train in the same checkout
  (the standard rebuild-then-train workflow); training on a stale lower-stride `.bin`
  would mis-stamp the model. To train a model that USES the funding/sentiment channels,
  build a crypto corpus after pulling `extensions/installed/market/data/funding/` + `extensions/installed/market/data/sentiment/`
  (the context recipe above); pairs/eras without that data encode the absent bin.
- These corpora are built from local gitignored data; they are NOT in the shared
  `corpus_catalog.json` and other machines will not have them.
- Directional base rate is ~50% by construction (market efficiency). Of non-flat
  bars the up-rate is ~51.6% stocks / ~50.2% crypto — that drift is the baseline a
  model must beat. The learnable signal is move magnitude, not direction.
- Stooq is no longer scriptable (JS anti-bot). Stocks come from the Yahoo v8 chart
  endpoint with `period1/period2` epoch params; `range=max` silently downsamples
  to monthly and must not be used.
- Crypto CSVs arrive newest-first and must be sorted ascending before encoding;
  the builder does this.
