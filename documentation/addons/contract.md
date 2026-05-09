# addons contract

Inference-time addons biases the model's next-byte logits without touching the model. Every addon ships as a self-contained directory under `veritate_mri/addons/<id>/` with a `manifest.json` and an `addon.py`. The platform discovers them at runtime, the dashboard lists them on the Generation tab, and the PyTorch backend pipes the chosen ones through the sampler before each byte is drawn.

This is a versioned contract. Adding, removing, or changing the signature of any function or field below requires updating this file in the same commit. Same rule as [plugins/contract.md](../plugins/contract.md) and [hooks/contract.md](../hooks/contract.md).

## scope

A plugin trains a model. An addon runs alongside an already-trained model at decode time. Addons read raw logits, apply a bias, and pass logits forward. They never load weights, never train, never write to the checkpoint, never touch the engine. The base model and its `.pt` are immutable from the addon's point of view.

The slot table addon is the canonical example. It tracks recent named entities, gendered anchors, and the rolling byte window. It biases logits to suppress doc-boundary collapse, repetition loops, n-gram echoes, and wrong-gender pronoun completions; it boosts already-seen named entities at word-start.

## file layout

```
veritate_mri/addons/<id>/
  manifest.json
  addon.py
```

`<id>` is the directory name. It is the URL-safe identifier the dashboard sends in the `addons` query param. snake_case, no spaces.

The platform discovers an addon if and only if the directory contains both files. A folder missing one is silently skipped.

## addon.py

Exports one class named `Addon` with three methods:

```python
class Addon:
    def __init__(self, **params): ...
    def reset(self): ...
    def observe(self, byte_int: int) -> None: ...
    def bias_logits(self, logits) -> torch.Tensor: ...
```

| method | when called | what it does |
|---|---|---|
| `__init__(**params)` | once per generation | constructed with kwargs from `manifest.json::params` plus any per-request overrides. |
| `reset()` | start of each new prompt | clear all internal state. |
| `observe(byte_int)` | once per byte (prompt + sampled) | update internal state with the byte that just landed. |
| `bias_logits(logits)` | once per step, before sampling | return a same-shape tensor with the addon's bias applied. The chain pipes outputs through addons in declaration order. |

`logits` is a 1-D tensor of length 256 (the byte vocab). Returning a clone is fine. `bias_logits` must be a pure function of the addon's state plus the input logits. No side effects.

`observe` runs after each byte is committed to the output stream. The addon does not see un-sampled candidates.

The addon does not pick the byte. The chain hands biased logits back to the platform's sampler.

### import rules

Addons may import:

- the standard library
- `torch` and its functional API
- modules within their own folder (e.g. helpers split into a sibling file)

Addons may not import from `veritate_mri/` or `veritate_engine/` directly. If an addon needs a platform helper that does not exist on this contract, propose adding it here first.

## manifest.json

```json
{
  "name":        "Human-Friendly Addon Name",
  "description": "One-line summary shown in the addons panel.",
  "kind":        "decoder",
  "params":      { "<key>": { "default": <value>, "description": "..." }, ... }
}
```

| field | type | meaning |
|---|---|---|
| `name` | string | display name in the Generation tab's addons panel. |
| `description` | string | one-line summary; appears in the panel tooltip. |
| `kind` | string | `"decoder"` is the only kind today. Reserved for future categories (e.g. `"observer"` for read-only telemetry-only addons). |
| `params` | object | per-param spec. Keys are `__init__` kwargs. Values are objects with at least `default`; `description` is shown in the dashboard. The dashboard renders one control per declared param when the addon's drawer is expanded. |

A param's `default` may be a number, string, bool, or array. Future param types (range constraints, dropdown choices) are added through the [update obligation](#update-obligation), not invented per addon.

The dashboard treats undeclared params as forbidden. To add a new constructor kwarg, declare it in `manifest.json`.

## chain composition

The platform composes selected addons in `Chain` (`veritate_mri/addons/__init__.py`):

```python
from veritate_mri import addons
chain = addons.build_chain(["slot_table", {"id": "other", "params": {"foo": 0.3}}])
chain.reset()
chain.observe_bytes(prompt_bytes)
for step in range(max_new):
    logits = model_forward(...)
    biased = chain.bias_logits(logits)
    nxt    = sample(biased)
    chain.observe(nxt)
```

The chain pipes `bias_logits` through addons in selection order. `observe` and `reset` are broadcast to every addon. Addons do not see each other; they only see the logits the chain hands them. Two addons that produce conflicting bias on the same byte simply add their biases.

## sample integration points

Addons currently apply to:

| backend | path | wiring |
|---|---|---|
| PyTorch | `veritate_mri/backends/pytorch.py::Brain.stream` | `addons_chain` kwarg. The dashboard's `/generate?addons=<csv>` builds a chain from the registry and passes it through. |
| Diagnostic sampler | `veritate_mri/tools/sample_diverse.py` | `--addons <csv>` and `--list_addons`. Same chain, no streaming. |

The C engine path does not yet support addons. Addon checkboxes have no effect when `backend=c`. Porting an addon to the engine is a separate workstream: the same three-method contract is mirrored as a C function-pointer table.

## API endpoints

`GET /addons` returns the discovered registry:

```json
{
  "addons": [
    { "id": "slot_table", "manifest": { ... } },
    ...
  ]
}
```

`GET /generate?...&addons=<id1>,<id2>` enables the listed addons in selection order. Param overrides are not yet supported on this endpoint; defaults from `manifest.json` are used. Adding override support is a future contract bump.

## what addons must not do

- Do not import from `veritate_mri.*` or `veritate_engine.*` directly.
- Do not write outside the addon folder. No checkpoint writes, no log writes.
- Do not retain state across `reset()`.
- Do not read tensors other than the one passed to `bias_logits`.
- Do not introduce dependencies that are not already part of the platform's import graph (no new `pip install` requirements without an explicit contract update).
- Do not invent new manifest fields.

## stability

Method names, signatures, manifest fields, and registry behavior are stable across patch releases. A signature change or rename is a minor-version bump and requires this file's update obligation to be honored in the same commit. Addons targeting one version are expected to keep working until the next minor bump.

## update obligation

Adding, removing, or renaming any field or function above requires:

1. The implementation in `veritate_mri/addons/__init__.py` is updated.
2. The PyTorch backend integration in `veritate_mri/backends/pytorch.py::Brain.stream` is updated.
3. The diagnostic sampler in `veritate_mri/tools/sample_diverse.py` is updated.
4. The dashboard wire-up (`/addons` endpoint in `veritate_mri/app.py`, the addons panel in `veritate_mri/static/index.html` and `index.js`) is updated.
5. This file's tables and examples are updated in the same commit.
6. Any addon shipped with the platform (`veritate_mri/addons/slot_table/`, etc.) is updated and verified in the same commit.
