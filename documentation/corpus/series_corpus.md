# price-series corpus

Byte-level corpus that turns numeric OHLCV history (stocks, crypto, any price
series) into an autocomplete stream the model predicts forward. The headline task
is next-bar prediction from price + volume alone, no news or semantics.

## what it is

Each daily bar is reduced to three scale-free, quantized features and encoded as a
fixed-stride sequence of printable bytes. One instrument is one sequence;
instruments are concatenated, newline-separated, into a single byte stream. The
model autocompletes the next bar's bytes; the first byte of each bar is the
return bucket, so direction and magnitude fall out of the decode directly.

Two corpora share the identical format:
- `stocks` — S&P 500 daily bars (adjusted close).
- `crypto` — major Binance USDT pairs, daily.

## how it works

Codec and feature math: [series_codec.py](../../veritate_mri/tools/series_codec.py).
Builder: [build_series_corpus.py](../../veritate_mri/tools/build_series_corpus.py).

**Features per bar (3), all scale-free so a 1990 bar and a 2024 bar are
comparable:**
- return z-score: close-to-close return divided by its trailing-window std.
- range ratio: intraday `(high-low)/prev_close` divided by its trailing mean.
- relative volume: volume divided by its trailing mean.

**No lookahead.** Normalization at bar `t` uses a strictly trailing window
(`FEAT_WINDOW`, bars before `t` only) — [series_codec.py compute_features](../../veritate_mri/tools/series_codec.py#L48).
Guaranteed by `test_no_lookahead` in [test_series_codec.py](../../tests/mri/test_series_codec.py).

**Quantization.** Each feature maps to a bucket over a fixed clip range
(`RET_BINS=33` centered, `RNG_BINS=16`, `VOL_BINS=16`), then to one char of a
64-char `ALPHABET`. A bar is `BAR_STRIDE=3` chars; position in the stride is the
feature id. Return bucket `RET_CENTER` is flat; above is up, below is down.

**Encoding is anonymous.** No ticker label, so the model learns one
instrument-agnostic tape dynamic rather than memorizing a symbol.

**Time split.** Per instrument, the oldest `1-val_ratio` of bars are train, the
newest are val — the val set is the held-out future. The codec contract is shared
with the standalone predict page, so the live tester encodes a window and decodes
the model's continuation through the same functions.

## build

Raw data is pulled into the gitignored `external_data/{stocks,crypto}/` by
`external_data/pull.py` (Yahoo v8 chart for stocks, cryptodatadownload for
crypto). Then:

```
python veritate_mri/tools/build_series_corpus.py --source stocks
python veritate_mri/tools/build_series_corpus.py --source crypto
```

Output: `trainers/corpus/<source>_train.bin` + `_val.bin`.

## dependencies

- `external_data/` raw CSVs (gitignored, local only).
- `numpy` for feature math.
- Consumed by the corpus loader (training) and the standalone predict page (eval).

## pitfalls

- These corpora are built from local gitignored data; they are NOT in the shared
  `corpus_catalog.json` and other machines will not have them. The predict page
  lists them by scanning `trainers/corpus/` on disk.
- Directional base rate is ~50% by construction (market efficiency). Of non-flat
  bars the up-rate is ~51.6% stocks / ~50.2% crypto — that drift is the baseline a
  model must beat. The learnable signal is move magnitude, not direction.
- Stooq is no longer scriptable (JS anti-bot). Stocks come from the Yahoo v8 chart
  endpoint with `period1/period2` epoch params; `range=max` silently downsamples
  to monthly and must not be used.
- Crypto CSVs arrive newest-first and must be sorted ascending before encoding;
  the builder does this.
