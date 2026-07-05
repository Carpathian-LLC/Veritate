# trading market intel group (server)

Surveillance radar over free public crypto data: trending coins, meme tokens, and
pump-pattern detection, explained in one line by a local model. It detects, warns,
and records; it never emits trade signals. The intel group of the Trading extension
(`extensions/canonical/trading/server/scanner.py` + `intel.py`; routes
`/ext/trading/intel/*`; UI: the Market Intel tab of the `/ext/trading` page).

## why a radar, not a signal

The 2026-06-26 pump probe (`overnight_run_log.md`) showed buying detected pumps
MEAN-REVERTS: vol-z>3 flagged spikes lost gross before fees at every horizon
(-0.43% h1 to -1.32% h24) and every public spike (volume, search, social) is
visible to everyone at once, so the actor arriving on the signal is the exit
liquidity. What that research lacked was a forward-recorded event dataset. This
extension supplies it: every flagged event is appended live to `events.jsonl`
with its metrics, timestamped at detection, no hindsight reconstruction.

## how it works

- `server/scanner.py` : the data layer, free + keyless.
  - One OKX call for all spot USDT pairs (`OKX_TICKERS_URL`, `okx_rows()`
    `scanner.py:101`), filtered to `vol_usd >= MIN_VOL_USD` (thin books are
    uninvestable noise, same probe).
  - CoinGecko trending searches + meme-token board (`cg_boards()`
    `scanner.py:124`), disk-cached `CG_TTL_S` because the free tier allows only
    ~10-30 req/min: at most one refresh per scan.
  - Each scan persists one snapshot file under `data/snapshots/`
    (`_save_snap()` `scanner.py:159`, rolling `SNAP_KEEP`), so z-scores use each
    coin's own trailing baseline across scans (`BASELINE_N` snapshots,
    `zscore()` `scanner.py:166`), not just exchange 24h fields.
  - Per symbol (`_score_rows()` `scanner.py:231`): `ret_1h` (vs the newest
    snapshot >= `RET1H_S` old), `ret_24h`, price z, volume z, range expansion,
    composite `anomaly_score()`. PUMP flag = price z >= `PRICE_Z_MIN` AND vol z
    >= `VOL_Z_MIN` (`pump_flag()` `scanner.py:182`, the validated probe recipe),
    with a per-symbol `COOLDOWN_S` stamped in `state.json` so one pump is one
    event. Flags append one JSON line to `data/events.jsonl` (`append_event()`
    `scanner.py:187`, rotated at `EVENTS_MAX`).
  - Watch thread (`start_thread()` `scanner.py:321`) rescans every `SCAN_SEC`,
    stamps auto-resume in `state.json`; `resume()` restarts it at boot so a
    server restart never gaps the dataset. All state under
    `extensions/installed/trading/data/intel/` (survives uninstall, same
    convention as paper_trade ledgers).
- `server/intel.py` : the LLM layer. Pulls recent Google News RSS headlines per
  coin (`headlines()` `intel.py:66`, own lean fetch: extensions never import
  across extensions), asks a local model via `POST /teacher/complete` (default
  `qwen2.5:7b-instruct`, provider ollama) for strict JSON: `why_moving`,
  `news_driven`, `meme`, `pump_risk`, `caution`. Cached by
  (symbol, headline-hash) (`brief()` `intel.py:140`): only new material hits the
  model; offline/malformed replies fall back to a metrics-only brief and are not
  cached. Belt and braces (`_enforce()` `intel.py:125`): a spike with no news is
  stamped `pump_risk high` mechanically, whatever the model says.
- `server/register.py` : routes under `/ext/trading/intel/`, all wrapped by the
  `_safe` JSON-error pattern; calls `scanner.resume()` at the end of
  `register(app)`.

## routes

| route | what |
|---|---|
| `GET /ext/trading/intel/scan` | cached scan (rows + trending + meme + new events); `?refresh=1` forces |
| `GET /ext/trading/intel/trending` | trending searches + meme board + scored movers, cached briefs attached |
| `GET /ext/trading/intel/pumps?n=` | recent events newest first, cached briefs attached, total count |
| `GET /ext/trading/intel/brief?symbol=&model=` | on-demand brief for one symbol |
| `GET /ext/trading/intel/status` | last scan ts, event count, watching, model state |
| `POST /ext/trading/intel/watch/start` | body `{interval?, model?}`; starts the background scan |
| `POST /ext/trading/intel/watch/stop` | stops it and clears the resume stamp |

## dependencies

stdlib + certifi + Flask (platform-provided). Platform surface: only
`POST /teacher/complete` over HTTP. External: OKX public tickers, CoinGecko free
API, Google News RSS: all keyless and US-reachable.

## pitfalls

- **Never turn flags into buy signals.** The exit-liquidity result is the
  project's research law here; all copy carries the caveat, and any future
  consumer of `events.jsonl` is for measuring pumps, not chasing them.
- CoinGecko free tier rate-limits hard; keep every CG read behind `cg_boards()`
  and its disk cache. Two calls per refresh maximum.
- z-scores are 0 until `BASELINE_MIN` snapshots exist (fresh install shows a
  quiet board for the first few scans) and 0 on a flat baseline by design.
- OKX `volCcy24h` is the quote (USD) volume for spot; `vol24h` is base units.
- The watch thread and request-path scans share `_LOCK`; scan work stays inside
  it so concurrent snapshot writes cannot race.
- Tests must monkeypatch the `DATA_DIR`-family module constants to `tmp_path`;
  the real data dir is machine-local forward data, never test scratch.

Tests: `extensions/canonical/trading/tests/` (scanner math/flag/cooldown/
rotation, intel parse/override/cache, route mount smoke).
