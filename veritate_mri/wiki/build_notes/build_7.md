---
title: "Build 7: HTTP updater, decode-path wins (KV cache, MTP head, constrained), v1 base versions"
date: 2026-05-11
tags: [build, updater, decode, kv_cache, mtp, constrained, mri]
summary: Updater swapped from git fetch + reset to HTTPS tarball + in-place copy; dirty-tree gate is gone. New veritate_mri/decode/ promotes the KV cache, MTP, and constrained-decoding modules out of experiments/ and into the platform. /generate gains fast=kv|mtp and constrained=json|vocab:*|stop:* request params, with matching chat-tab toggles. All component versions rebase to v1.

---

## versions

- build: 7
- engine: v1.2.0
- mri: v1.3.0
- format: v1.3.0
- plugins: v1.1.1

## what changed

### updater

- `app_sync.py` rewritten to fetch the channel's GitHub source tarball (`<repo>/archive/refs/heads/<branch>.tar.gz`) and overwrite tracked source in place. No `git` on PATH, no working-tree / staging-area concept, no dirty-tree gate, no diverging-branch failure mode.
- Skip dirs preserved across update: `data/`, `models/`, `plugins/`, `experiments/`, `.git/`, `.venv/`.
- Requires `VERITATE_REPO_URL` env var (e.g. `https://github.com/<owner>/<repo>`). With it unset the module imports cleanly and surfaces a useful error at update time.
- Update-button frontend gate dropped — it only checks `update_available` now.

### decode path (new module: `veritate_mri/decode/`)

- `kv_cache.py` — `KVCachedDecoder`. Per-step decode goes O(T²) → O(T) by caching K/V. Auto-detects three model variants (`abs_pos` / `rope_no_mtp` / `rope_mtp`) and routes byte-0 logits through the right output head for each.
- `mtp_decode.py` — `MTPDecoder`. Multi-byte-prediction head decoding for Veritate800M (`head0_only` / `accept_all` / `verify` modes). The verify mode is byte-exact equivalent to single-byte decoding.
- `constrained.py` + `constraints.py` — `ConstrainedDecoder`, `JSONConstraint`, `VocabConstraint`, `StopOnConstraint`, `CombineConstraint`. Pure-Python logit masking; composes on top of any decode mode.

### model classes

- New `veritate/model_rope.py::VeritateRoPE`. RoPE-only sibling of canonical `Veritate` and the 800M plugin's `Veritate800M` (no MTP head, tied lm_head). The Brain backend now dispatches three ways: canonical (learned `pos_emb`), RoPE-only (`VeritateRoPE`), RoPE+MTP (`Veritate800M`).
- Brain checkpoint loader now reads shape from either `args` or `config` in the saved dict, so the v2 experimental checkpoints (rope_85m etc.) load through the same path as canonical and 800M checkpoints.

### `/generate` API

Two new query parameters on the PyTorch backend path:

- `fast=kv|mtp` — switches to `Brain.stream_fast`. KV mode is byte-exact identical to the default path. MTP mode requires a model with an MTP head (currently the 800M). Both skip the per-byte brain-scan telemetry and emit `{kind: "fast_byte", byte, ms_per_byte, head?, k?}` events instead. A `{kind: "prefill", prefill_ms, tokens}` event fires once when the KV cache is filled.
- `constrained=<spec>` — applies a -inf logit mask to the output. Supported spec values:
  - `json` — grammar-valid JSON value
  - `vocab:ascii` / `vocab:alpha` / `vocab:lower` / `vocab:upper` / `vocab:alnum` / `vocab:digits`
  - `stop:newline` / `stop:double_newline` / `stop:eos` / `stop:text:<literal>`

The default `/generate` path (no `fast`, no `constrained`) is unchanged; the rich-telemetry dashboard panels keep working byte-for-byte.

### chat tab UI

- New `decode` row under the prompt with two dropdowns: `fast` (off / kv / mtp) and `constrained` (off / json / vocab presets / stop presets).
- Chat-tab JS now handles three new SSE event kinds: `prefill`, `fast_byte`, `stop`. Brain-scan panels stay populated by the default path; fast mode shows a slimmer live-stats row (bytes, b/s, ms/byte, optional `head h/k` pill).

### version scheme

All component versions rebased to `v1.X.Y`. The build counter (`build`) and the minor (`X`) are what move; `v1` is the platform-stable base across the board.

## what the user has to do

### required: reload python after updating

Click **reload python** once after the update lands so the new `app_sync` module + the `veritate_mri/decode/` package are imported.

### required: set the repo URL env var (if not already set)

The new HTTP updater needs to know where to fetch from. Set this once in your shell or `run.py` launcher:

```sh
export VERITATE_REPO_URL=https://github.com/<owner>/<repo>
```

Without it, the **Update** button surfaces `VERITATE_REPO_URL is not set` and stops cleanly. Nothing crashes.

### optional: try the fast-decode toggle

1. **Generation** tab → set backend to PyTorch (the C engine path is unchanged in this build).
2. Pick `fast = kv` and run a generation. Output is byte-for-byte identical to the default path; the time-per-byte drops once the cache is filled.
3. With the 800M loaded, pick `fast = mtp` to use the multi-byte head. Each forward emits up to 4 bytes; the live-stats row labels them by head index.

### optional: try a constrained generation

Set `constrained = json` and prompt `"{ \"name\": \"`. Every emitted byte is a legal continuation of the JSON grammar; the run stops the moment a complete top-level value closes. The same toggle accepts vocab/stop presets for "only lowercase letters" or "stop at the next blank line."

## known limitations

- Fast mode bypasses the per-byte brain-scan telemetry (no FFN activations, no attention weights, no logit-lens per byte). Use the default mode when you need those panels.
- `fast=mtp` emits bytes from heads 1..K-1 greedily and does NOT run the second-pass verify. The byte-exact verify path still lives in `veritate_mri/decode/mtp_decode.py::MTPDecoder._decode_verify` for offline benchmarks.
- Composing `fast=kv` with `fast=mtp` is not yet wired (the underlying modules can compose, but the streaming bridge needs to track both caches). Filed for build 8.
- KV cache uses model param dtype for the K/V tensors. For an 800M at seq=1024 in fp32 that's ~353 MiB; bf16 brings it to ~176 MiB. Both are negligible against the trunk.
- `VocabConstraint` is byte-level: `vocab:lower` means lowercase ASCII (0x61–0x7A). Non-ASCII letters do not pass; UTF-8 lead bytes are filtered. Use `vocab:ascii` for printable + tab/LF/CR.
