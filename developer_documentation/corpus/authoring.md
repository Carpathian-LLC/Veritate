# corpus authoring

Platform pipeline that authors an original conversational and reading corpus with the configured
teacher model and gates every record before it lands on disk. Unlike the seed-driven synth flow
(one prompt in, one sample out), one authoring call returns a JSONL batch, so corpus volume is a
byte target set on the dashboard panel rather than a hand-enumerated prompt file.

## what it is

| piece | path |
|---|---|
| editable spec (genres, prompts, ban list, gates) | `veritate_mri/data/authoring/corpus_spec.json` |
| gate + prompt planner | `veritate_mri/teacher/authoring.py` |
| job runner (shared with synth) | `veritate_mri/teacher/synth.py` |
| routes | `veritate_mri/routes/teacher_routes.py` |
| bin packer | `veritate_mri/tools/build_sft_corpus.py` |
| dashboard panel | `#authorPanel` in `veritate_mri/web/index.html`, `_author*` in `index.js` |
| tests | `tests/corpus/test_corpus_authoring.py` |

Nothing about the recipe is a code literal. Genres, voice pools, situations, briefs, the prompt
template, the banned-phrase list, the character rewrites, and every quality threshold live in
`corpus_spec.json`, resolved through `paths.authoring_spec_path()`.

## how it works

1. `plan_calls(spec, genre_ids, target_bytes)` (`authoring.py:125`) splits a byte target across the
   selected genres by each genre's `weight`, dividing by `records_per_call * est_bytes_per_record`.
2. `build_prompts(spec, calls_per_genre, seed, id_prefix)` (`authoring.py:139`) renders one prompt
   per call from `prompt_template`, rotating the voice pool with a seeded `random.Random` and the
   situation list by call index. Same spec plus same seed gives byte-identical prompts.
3. `SynthJob` runs the calls. `opts["record_gate"]` switches it to batch mode
   (`synth.py:324`): the response expands to N records, each written as
   `{"id": "<prompt>#<k>", "record": {...}}`, and gate stats ride along in `state.json`.
4. `RecordGate.__call__` (`authoring.py:228`) parses the reply line by line and applies the gates
   below in order. It runs on the SynthJob main thread only and is not thread safe.
5. `POST /teacher/authoring/import` (`teacher_routes.py::teacher_authoring_import_route`) is the
   entry point for externally-authored (non-teacher, e.g. agent-written) records: a directory of
   `*.jsonl` files where each line is a bare `{genre, voice, turns|text}` record with no `id` or
   `record` wrapper. It seeds a `RecordGate` from any existing `samples.jsonl` in the target job
   (`gate.seed_from_file`), then for every record looks up its own `genre` field against the spec
   (`authoring_mod.genre_by_id`, rejecting and reporting an unknown genre rather than dropping it
   silently) and runs it through the SAME `RecordGate.__call__` teacher output goes through. Accepted
   records are appended to `samples.jsonl` as `{"id": "import_<file_stem>_<line_no>", "record": {...}}`;
   the id is deterministic from the source file and line number, so re-importing the same directory
   into the same job adds nothing (the gate's content-based dedup rejects the repeat as an exact or
   near duplicate). The response carries per-file accepted/rejected-by-reason counts plus the
   corpus-wide `ngram_ratio` / `ngram_below_floor` from `gate.stats()`.
6. `POST /teacher/authoring/build` splits accepted records into per-genre family JSONL, packs them
   with `build_sft_corpus.build()`, zips both bins at zip top level, and appends a `coming_soon`
   `zip_bundle` entry to `corpus_catalog.json` with a PLACEHOLDER `train_url`. This step is identical
   whether the job's `samples.jsonl` came from a teacher run, an import, or both.

## gates

Applied per record, first failure wins. Every rejection is counted by reason and surfaced live.

| gate | rejection reason | source |
|---|---|---|
| line parses as JSON | `invalid json` | |
| exact key set, genre match, non-empty voice, alternating roles | `schema mismatch` | `schemas` in spec |
| turn count inside the genre band | `turn count out of range` | `min_turns` / `max_turns` |
| genre marker present in the first user turn | `missing required marker` | `require_in_first_user` |
| no null byte | `null byte` | |
| length window | `too short` / `too long` | `min_chars` / `max_chars` |
| em and en dashes | `em dash` when `em_dash_policy` is `reject`, otherwise rewritten via `char_rewrites` and counted | `em_dash_chars` |
| banned phrase, word boundary, assistant side only | `banned phrase` | `banned_phrases` |
| normalized sha1 already seen | `exact duplicate` | |
| more than `opening_cap` records share a normalized opening | `repeated opening` | `opening_cap` |
| simhash within `near_dup_hamming` of an accepted record | `near duplicate` | `near_dup_hamming` |

Repetition is measured, not assumed: `distinct_ngram_ratio` recomputes the share of unique word
5-grams over a rolling window of the last `ngram_window_records` accepted records every
`ngram_recompute_every` records. `ngram_below_floor` goes true under `ngram_distinct_floor` and the
panel raises a red warning naming the number and the fix.

Banned phrases match on word boundaries (`compile_ban_re`), so `as an ai` does not fire inside
`as an airport`. They are scanned on assistant turns only for dialogue genres: the same phrase in a
user turn is natural human speech.

## genres

`conversation`, `carryover`, `grounded_read`, `instruct`, `format_constraint`, `cogito`, `jokes`,
`writing`, `news`. Two schema kinds: `turns` (`{genre, voice, turns:[{role, text}]}`) and `text`
(`{genre, voice, text}`). Four genres exist because nothing on the box covered them:

- `carryover` enforces `min_turns: 6` and briefs for later turns that depend on facts stated
  earlier, including a mid-conversation correction the assistant must honor. The turn floor is
  enforced in the pipeline; the dependency itself is prompt-enforced only.
- `grounded_read` enforces the literal `context:` marker in the first user turn, and briefs for
  roughly one honest-miss record in four ("the passage does not say"). The passage is authored
  prose, so reading data and style data are the same bytes.
- `instruct` covers direct instruction execution: compose, enumerate, transform, summarize,
  extract, translate, small arithmetic, compare, classify, define, numbered steps. The assistant
  carries the instruction out with no preamble and no clarifying question. `min_turns: 2` keeps the
  structural demand low, which is what keeps its yield high (see genre yield below).
- `format_constraint` covers explicit output constraints ("one sentence", "exactly three items").

### genre yield

Planned calls are not produced records. A genre's yield is (kept records)/(records the teacher
emitted), and it falls off a cliff with structural demand. Measured over two 22k-record jobs on
`qwen2.5:14b-instruct`:

| genre | structural demand | yield |
| --- | --- | --- |
| `jokes`, `writing` | none | 82-194% |
| `conversation` | none | 39% |
| `instruct` | 2-4 turns | 64% |
| `cogito` | 4+ turns | 15% |
| `format_constraint` | 2-4 turns, obey a stated format | 1.3% |
| `carryover` | 6-10 turns, load-bearing memory | 1.4% |
| `grounded_read` | `context:` marker, 120-400 word passage | 0.5% |

A genre weighted at 0.10 that yields 1.3% lands at 0.4% of the corpus. Read `authoring.per_genre`
and `authoring.rejects` in `state.json` (or the synth status route) against the planned call counts
before trusting a mix: a starved genre looks identical to a genre that was never requested. Raising
a starved genre's share means either a stronger teacher or a lower structural demand, not a higher
weight.

## resume

`RecordGate.seed_from_file()` rebuilds dedup state from an existing `samples.jsonl`, so a resumed
job cannot re-add a record it already wrote. `_load_done_ids` in `synth.py` splits record ids on
`#`, so a partially completed batch prompt is not re-run.

## pitfalls

- `simhash64` uses blake2b, not the builtin `hash()`, which is salted per process. Reverting that
  makes dedup non-reproducible across restarts.
- `build_sft_corpus.build()` always carves at least one conversation into val, so a single-record
  build produces an empty train bin.
- The catalog entry is written straight into the shipped `corpus_catalog.json`. Rebuilding a stem
  that is already published breaks its recorded sha256; ship a new stem instead.
- `est_bytes_per_record` is an estimate, so the target size is a plan, not a guarantee. Read the
  live MB counter for what was actually produced.
