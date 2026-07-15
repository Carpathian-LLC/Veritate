# code corpus builder

Builds the code-model corpora: `py_code_*`, `js_code_*`, `html_code_*`, `css_code_*`,
`code_qa_*`, and the `mixed_code_*` experimental variants. One module owns the
pipeline: [`veritate_mri/tools/build_code_corpus.py`](../../veritate_mri/tools/build_code_corpus.py).

## what it is

Two-phase builder. `stage` pulls source documents into a local jsonl cache
(network); `build` assembles byte-level bins from that cache (offline,
deterministic from a fixed seed). Code families emit autocomplete mode (raw
files joined by `<|endoftext|>`); `code_qa` emits ChatML chat mode. See
[framing.md](framing.md).

## sources (pinned)

| family | source | pin |
|---|---|---|
| py, js | `HuggingFaceTB/stack-edu` metadata (education-classifier scored Stack v2), content fetched anonymously from the public Software Heritage S3 bucket | `STACK_EDU_REVISION`, build_code_corpus.py:52 |
| html, css | curated permissive-license GitHub repos (MDN learning-area, web-platform-tests, bootstrap, template sets), tarballs by tag or commit | `REPOS`, build_code_corpus.py:69 |
| qa | `donfu/oa-stackexchange`, stackoverflow rows only, coding tags, vote thresholds | `QA_*`, build_code_corpus.py:58 |

Every stage run records revisions and tarball sha256s to `sources.json` in the
staging dir (default `~/.cache/veritate/code_corpus_staging`).

## cleaning

- Size caps per family (`MIN_DOC_BYTES`/`MAX_DOC_BYTES`).
- Python must pass `ast.parse` (`accept_py`). JS rejects minified/bundled files
  by line-length and bundle markers (`accept_js`). HTML requires structural tags
  and caps data-URI density (`accept_html`). CSS rejects newline-free minified
  output (`accept_css`).
- Auto-generated files rejected by header markers (`AUTOGEN_MARKERS`).
- Dedup: exact sha1 plus simhash near-dedup with banded LSH (`Deduper`).
- Stack-edu score gate: staging keeps `int_score >= 3`; standalone family builds
  default to `int_score >= 4` (`BUILD_MIN_SCORE`).

## mixed_code variants (corpus-style ablation)

Same target size and seed, one axis changed per variant:

| variant | filters | score gate | composition |
|---|---|---|---|
| `mixed_raw` | size caps + exact dedup only | none | py .40 js .40 html .10 css .10 |
| `mixed_files` | full filters + near-dedup | >=3 | same |
| `mixed_edu` | full filters + near-dedup | >=4 | same |
| `mixed_qa` | full filters + near-dedup | >=3 | code .50 (same ratios) + qa .50 |

Train each on the same 200M recipe, same steps, and compare val loss on a held
clean split plus code evals; the winning style sets the mix for larger runs.

## val split

Per-document stable hash bucket (`_is_val`, 1 in 50). Identical documents cannot
straddle train/val because dedup runs first.

## dependencies

`datasets` (streaming), `requests`. No platform imports; standalone tool like
the other `build_*_corpus.py` builders.

## pitfalls

- Software Heritage fetches are per-blob HTTP gets: staging GB-scale is
  hours, not minutes. Stage once, build many tiers from the cache.
- `stage --family web` writes both `html.jsonl` and `css.jsonl` in one pass.
- A `build` whose pool underruns the target prints `WARNING: pool underran
  target` and writes the short bin anyway; pick the tier size from the actual
  staged yield.
- Tarball refs are tags; if an upstream repo moves a tag the sha256 recorded in
  `sources.json` changes. Compare against the manifest before shipping.

## tests

[`tests/mri/test_build_code_corpus.py`](../../tests/mri/test_build_code_corpus.py):
filters, dedup, ChatML framing, deterministic assembly, EOT separators.
