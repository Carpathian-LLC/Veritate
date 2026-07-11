# Model capabilities

## What it is

Per-model declaration of which generation modes a checkpoint supports. Three
additive tiers gate the Generation-tab mode picker: a model may only be driven in
a mode it was trained for. The block lives inside `models/<name>/config.json`
under the `capabilities` key and is owned by
[capabilities.py](../../../veritate_mri/readers/capabilities.py).

## The on-disk block

Versioned, HuggingFace-aligned:

```json
"capabilities": {
  "schema": 1,
  "pipeline_tag": "text-generation",
  "tasks": {
    "autocomplete": { "status": "trained" },
    "chat":         { "status": "trained", "trainer": "veritate_200m", "step": 24400, "completed_at": "2026-07-10T04:39:02Z" },
    "agent":        { "status": "untrained" }
  }
}
```

- **Tiers** (`TIERS`, capabilities.py:28): `autocomplete` < `chat` < `agent`.
  Additive: agent implies chat implies autocomplete.
- **Status** (`STATUSES`, capabilities.py:35): `untrained`, `in_progress`,
  `trained`, `failed`. Rank order in `STATUS_RANK` (capabilities.py:37).
- **pipeline_tag** (capabilities.py:49): always `text-generation`. Present for
  parity with the HF `pipeline_tag` convention; the platform does not branch on
  it yet.
- Per-entry metadata carried through reads (`PRESERVED_KEYS`, capabilities.py:55):
  `trainer`, `step`, `completed_at`, `legacy` (synthesized fallback), `implied`
  (set by additivity, not a direct training result).

### Backward compatibility

`read()` and `_normalize()` (capabilities.py:85) accept **both** the versioned
block above and the **legacy flat** form written before the schema landed:

```json
"capabilities": { "autocomplete": {"status": "trained"}, "chat": {"status": "trained"}, "agent": {"status": "untrained"} }
```

`_tasks_of()` (capabilities.py:72) picks `tasks` when present, else treats the
block as the flat tier map. No migration pass exists or is needed: existing
configs (chat200m, chat80m, market models) keep reading correctly, and are
rewritten into the versioned form the next time `mark()` touches them.

`read()` always returns the **flat tier map** (`{tier: {status, ...}}`), never the
wrapper. That return contract is stable: `/meta` and `/pytorch-models` embed it
verbatim.

## How it works

- **Write path.** `mark(name, tier, status, ...)` (capabilities.py:190) is the
  single writer. It normalizes the current block, updates one tier, applies
  additivity, and persists via `_wrap()` (capabilities.py:80). Guards: a
  `failed` mark never regresses a tier already `trained`; an `in_progress` mark
  with `step >= total_steps` is coerced to `trained`.
- **Additivity.** `_apply_additivity()` (capabilities.py:175) lifts every lower
  tier to at least the marked status when a tier reaches `in_progress`/`trained`,
  tagging the lifted entries `implied: true`. It never downgrades a tier already
  ranked higher, so a pretrained-then-chat model keeps its real per-tier metadata.
- **Corpus to tier.** `modes_for_corpus(spec)` (capabilities.py:152) resolves a
  corpus spec (single stem or multicorpus mix) to the tiers its catalog entries
  declare via `trained_modes` in
  [corpus_catalog.json](../../../veritate_mri/training/sync/corpus_catalog.json).
  Returns a tuple in `TIERS` order, or empty for custom corpora not in the
  catalog.
- **Pipeline propagation.** `save.py::_sync_capabilities` (save.py:272) runs on
  every checkpoint: it reads the run's corpus (`_corpus_spec`, save.py:260),
  resolves tiers via `modes_for_corpus`, falls back to the trainer manifest's
  `teaches` tier when the corpus is custom, and marks each tier `in_progress`
  (or `trained` on the final step). Training on a catalog chat corpus marks
  `chat` with no hand-editing.
- **Legacy fallback.** A checkpoint with no `capabilities` key reads as
  autocomplete-only (`_legacy_block`, capabilities.py:64): any byte-level model
  can autocomplete.

## Who reads / gates

- `/pytorch-models` (models_routes.py:96) embeds each model's block, keyed by
  name, for the picker.
- `/meta` (backends_routes.py:673) reports the loaded C model's and pytorch
  model's blocks (`c_model_capabilities`, `pytorch_capabilities`).
- Frontend gate: `_applyModeAvailability()` (web/index.js:1773) enables a mode
  only when its tier is `trained`. `_activeCapabilities()` (web/index.js:1723)
  prefers the **selected dropdown model's** own caps (`_selectedModelCaps`, from
  the `/pytorch-models` cache `_pytorchModelCaps`), then merges the two backend
  blocks (`_mergeCaps`, best status per tier wins), then the legacy autocomplete
  fallback.

## Dependencies

- Corpus tier vocabulary comes from `trained_modes` in the corpus catalog, loaded
  through `corpus_sync._load_local_catalog()`. Multicorpus specs are split by
  `multicorpus.parse_spec`.
- Path resolution via [paths.py](../../../veritate_mri/readers/paths.py)
  (`config_path`).

## Pitfalls

- Do not hand-edit `capabilities` in config.json to grant a mode. The next
  `save()` re-derives tiers from the corpus; a hand-set tier not backed by the
  corpus (or a trainer `teaches` key) is not re-asserted, though additivity and
  the trained-not-regressed guard mean an already-trained tier is not cleared.
- Custom corpora (stems absent from the catalog) resolve to no tiers; capability
  then falls back to the trainer's `teaches` (default `autocomplete`). To drive a
  mode from a custom corpus, add the stem to the catalog with `trained_modes` or
  set `teaches` in the trainer manifest.
- `read()` returns the flat tier map, not the on-disk wrapper. Consumers that
  need `schema`/`pipeline_tag` must read config.json directly.
