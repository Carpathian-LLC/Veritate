# trading extension (consolidated architecture)

The Trading canonical extension (`extensions/canonical/trading/`, id `trading`) is the
one surface for everything market-related: byte-model serving, paper-trading strategies,
and the surveillance radar. It replaced the three separate extensions `market`,
`paper_trade`, and `market_intel` (one page, one API prefix, one data root). Simulated
capital only: no broker, no keys.

## layout

- `manifest.json` : page route `/ext/trading` (`page/index.html`), `api_prefix
  /ext/trading`, `register server/register.py`, experimental.
- `page/index.html` : the five-tab UI ([../frontend/trading_page.md](../frontend/trading_page.md)).
- `page/models.html` : model analytics, served at `/ext/trading/models`
  ([../frontend/trading_models_page.md](../frontend/trading_models_page.md)).
- `server/` : all modules, imported by bare name (the registry puts `server/` on
  `sys.path`). One `register.py` mounts every route.
- `tests/` : the extension's pytest suite (route registration, settings/channels, data
  layer, codec, policy, execution, traders, scanner, intel).

## route groups (register.py)

- `/ext/trading/market/*` : byte-model serving (`veritate.py` over `data.py` +
  `series_codec.py` + `live.py`/`fetch.py`), policy backtests (`policy.py`), instruments,
  data report. Contract detail: [market_routes.md](market_routes.md),
  [market_policy.md](market_policy.md), [market_fetch.md](market_fetch.md).
- `/ext/trading/paper/*` : the news-sentiment trader (`news_trader.py` over `scraper.py`
  + `sentiment.py`), the N-arm experiment, the two momentum books
  (`xsmom_trader.py`, `eqmom_trader.py`), and the ML weekly book
  (`ml7_trader.py` over `ml7_features.py`: `/ext/trading/paper/ml7/start|stop|status|accounts`).
  Contract detail: [paper_trade_sentiment.md](paper_trade_sentiment.md),
  [paper_trade_xsmom.md](paper_trade_xsmom.md), [paper_trade_eqmom.md](paper_trade_eqmom.md),
  [trading_ml7.md](trading_ml7.md).
- `/ext/trading/intel/*` : the scanner + pump radar (`scanner.py`) and local-model briefs
  (`intel.py`). Contract detail: [market_intel.md](market_intel.md).
- `/ext/trading/settings` (GET/POST) : settings + the shared scraper channel registry,
  owned by `scraper.py` (`load_settings`/`save_settings`,
  `extensions/installed/trading/data/settings.json`).
- `/ext/trading/system` (GET), `/system/stop` (POST, body `{id}`), `/system/stop_all`
  (POST) : one view of every runnable (news main, experiment arms, xsmom base/gated,
  eqmom, ml7, intel watch) with running state + started timestamp, and stop controls
  (`register._system_view` / `_system_stop`).

## control policy

Nothing auto-starts at boot. `register(app)` ends by calling `xsmom_trader.resume()`,
`eqmom_trader.resume()`, `ml7_trader.resume()`, and `scanner.resume()`, which restart ONLY runs whose ledger
(`auto: true`) or state file (`watch.on`) says the user had them running: explicit user
intent persisting across restarts, never a default-on. `recorder.py` (OKX context
capture) and `autotrader.py` are CLI-only and never started by the server.

## data root

`extensions/installed/trading/data/` (gitignored, survives install/uninstall):
- `market/` : OHLCV serving sources (`crypto_of/`, `forex/`, ...), `funding/`,
  `sentiment/fng.csv`, `live/` (recorder output). Constants in `data.py`,
  `bulk_dumps.py`, `recorder.py`, `pull*.py`, `corpus_manifest.py`, `fetch.py`.
- `paper/` : the strategy ledgers (`account.json`, `account_<label>.json`,
  `account_xsmom_*.json`, `account_eqmom_main.json`, `account_ml7_main.json`),
  `experiment.json`, `equity_universe.json`, plus the ml7 artifacts
  (`ml7_model.joblib`, `ml7_model_manifest.json`, `ml7_history/`). Constant:
  `news_trader.LEDGER_PATH`.
- `intel/` : `snapshots/`, `events.jsonl`, `state.json`, `cg_cache.json`. Constant:
  `scanner.DATA_DIR`.
- `settings.json` : settings + channel registry (`scraper.SETTINGS_PATH`).
- `extension_data/` : the platform-owned downloadable-dataset cache
  (`extensions/data.py` contract; declared in `data_catalog.json`).

## dependencies

- Platform routes only: `POST /teacher/complete` (sentiment + briefs),
  `POST /teacher/models` (model pickers). Model checkpoints are read via
  `readers.checkpoints`/`readers.models` read-only.
- Extra deps (`requirements.txt`): `certifi` (TLS for live feeds), `scikit-learn` +
  `joblib` (the ml7 frozen ensemble).

## pitfalls

- Server modules import each other by bare name; anything importing `data`, `scraper`,
  `scanner`, etc. outside this extension is a bug (extensions are self-contained).
- `news_trader.DECIDE_URL` calls the extension's own `/ext/trading/market/paper_decide`
  over HTTP on port 8001; renaming routes must update it in the same change.
- Don't run a trader CLI and its managed in-process thread on the same ledger (they race).
