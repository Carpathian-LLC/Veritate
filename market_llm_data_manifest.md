# Market LLM: data manifest (for S3 hosting)

Paths to every Market LLM data artifact: raw OHLCV, the built byte corpuses (training
sets), and trained models. Host the large files on an S3 bucket and pull them to the paths
below by hand. There is no in-dashboard S3 download feature. Paths are repo-relative unless
absolute. Regenerate a machine-readable copy any time with:

```
python veritate_mri/market/corpus_manifest.py --json
```

## Raw OHLCV (source data, gitignored, GB-scale)

| path | symbols | detail | size |
|------|--------:|--------|-----:|
| `external_data/crypto/` | 200 | 1m bars, Binance Vision archive; one CSV per symbol (`time,open,high,low,close,volume`) | ~34 GB |
| `external_data/stocks/` | 501 | daily/coarse bars | ~0.45 GB |

## Built byte corpuses (the training sets) -> `trainers/corpus/`

3 bytes per bar via `series_codec` (return bucket, range, volume). Bytes are the tokens.
Per-instrument time split (oldest `1 - val_ratio` train, newest val). No pair or bar caps.

| file | bytes | ~size | ~tokens | checksum |
|------|------:|------:|--------:|----------|
| `crypto_train.bin` | 1,308,686,738 | 1.31 GB | 1.31 B | `crypto_train.bin.sha256` |
| `crypto_val.bin` | 145,410,113 | 0.15 GB | 0.15 B | `crypto_val.bin.sha256` |
| `stocks_train.bin` | ~11,780,000 | 11.8 MB | 0.012 B | — |
| `stocks_val.bin` | ~1,310,000 | 1.31 MB | — | — |

Rebuild:

```
python veritate_mri/tools/build_series_corpus.py --source crypto
python veritate_mri/tools/build_series_corpus.py --source stocks
```

## Trained models -> `models/market/`

| path | what |
|------|------|
| `1m_h5.joblib`, `1m_h15.joblib`, `1m_h60.joblib` | GBDT volatility baselines (~4.45 MB each) |
| `summary.json` | GBDT certification summary |
| `marketllm_200m/` | byte-level Veritate 200M (training); checkpoints under `hooks/step_<N>/` |

## Suggested S3 layout

```
<bucket>/market-llm/raw/crypto/<SYM>.csv          (or one crypto_1m.tar.zst)
<bucket>/market-llm/raw/stocks/<SYM>.csv
<bucket>/market-llm/corpus/crypto_train.bin , crypto_val.bin
<bucket>/market-llm/corpus/stocks_train.bin , stocks_val.bin
<bucket>/market-llm/models/marketllm_200m/...
```

Mirror the local paths above under `<bucket>/market-llm`, then download by hand to those
same paths when provisioning a fresh machine.
