# multimind contract

Spec for the MultiMind plugin. A "multimind" model partitions its FFN into named regions and accepts a per-step per-region gate bias from an external source. The plugin bundles probe-driven bias, sleep-time region adapters, and persistence, and attaches to any model class that exposes the contract surface below.

This is a versioned contract. Adding, removing, or changing the signature of any field or method below requires updating this file in the same commit. Same rule as [trainers/contract.md](../trainers/contract.md), [addons/contract.md](../addons/contract.md), and [hooks/contract.md](../hooks/contract.md).

## scope

A trainer trains a model. An addon biases decode-time logits. The MultiMind plugin is neither: it installs a gate-bias provider on the model's MoE router and manages per-region LoRA adapters trained on a short "sleep" buffer. The plugin reads the model's contract surface and never branches on a specific model class.

The reference compatible model is `veritate_core.model_mtm.VeritateMultimind`. Any other Veritate model class can become multimind-compatible by exposing the contract surface; the plugin code does not import the reference class.

## model contract surface

A multimind-compatible model exposes:

| name | kind | meaning |
|---|---|---|
| `region_names` | `tuple[str, ...]` instance attribute | ordered region identifiers. snake_case. length is the expert count. |
| `gate_g` | `nn.Parameter | None` instance attribute | per-region bias vector of shape `(n_experts,)`, or `None` when no learned global bias exists. |
| `blocks` | `nn.ModuleList` instance attribute | per-layer MoE blocks. each block's `forward(x, gate_bias=...)` accepts an optional `(B, n_experts)` bias. |
| `set_gate_bias_provider(fn)` | instance method | install a callable `(tokens: LongTensor[B, T]) -> Tensor[B, n_experts]` consulted per forward when `sentiment` is None. Passing `None` clears. Provider precedence: a non-None provider overrides any legacy `sentiment` scalar path. |
| `forward(tokens, targets=None, sentiment=None)` | instance method | unchanged; when `sentiment is None` and a provider is installed, the provider supplies the gate bias for this forward. |

The model owns the routing; the plugin only supplies bias values. Adding a new compatible variant must never require touching plugin code.

## probe contract

```python
class Probe:
    def __call__(self, tokens: LongTensor[B, T]) -> Tensor[B] | Tensor[B, dims]: ...
    def valence(self, tokens: LongTensor[B, T]) -> Tensor[B] | Tensor[B, dims]: ...  # optional
```

A probe maps a window of bytes to a valence scalar (1D affect) or an affect vector (2D affect: valence, arousal, etc.). The plugin calls `probe.valence(tokens)` when present, else `probe(tokens)`. The probe is frozen at inference; the plugin sets `requires_grad_(False)` on every probe parameter after load. The probe ships as a pickled `torch.nn.Module` loadable via `torch.load`.

When the probe returns a scalar of shape `(B,)`, the plugin computes `gate_bias = model.gate_g.unsqueeze(0) * valence.unsqueeze(-1)` (rank-1 expansion). When the probe returns `(B, dims)` and `dims == n_experts`, the plugin uses the probe output directly as per-region bias.

## sleep adapter contract

Per-region LoRA adapters apply a small additive correction to the residual stream after each MoE block. Rank is caller-chosen. The plugin owns install/uninstall lifecycle.

| call | effect |
|---|---|
| `plugin.sleep(model, buffer, lr, steps, rank, save_dir=None)` | install per-region LoRAs (rank `rank`), freeze base params, train LoRAs on `buffer` for `steps` epochs at `lr`. Optionally persist to `save_dir`. |
| `plugin.wake(model, save_dir, rank)` | install LoRAs of `rank` and load saved tensors from `save_dir`. Missing files leave that region at init (A zero, residual delta zero). |
| `plugin.attach(model, probe_path)` | validate contract surface, load probe, install gate-bias provider. Idempotent: a second call replaces the provider with one of identical behavior. |
| `plugin.detach(model)` | clear the gate-bias provider. |

`buffer` is an iterable of `(input_tokens, target_tokens)` tensor pairs. The model's `forward(inp, targets=tgt)` returns `(logits, loss)`; the plugin backprops `loss` only into LoRA params.

## file format

Adapter sidecars live alongside the model checkpoint:

```
models/<name>/multimind_adapter_<region>.pt
```

One file per region. Each file is `{"A": Tensor[hidden, rank], "B": Tensor[rank, hidden]}`. Region names match `model.region_names`. The plugin never writes inside the canonical `.pt` checkpoint; it only writes sidecars next to it. Probes live at the caller's chosen path; the plugin does not impose a probe location.

## dashboard hook fields

The plugin emits these fields per byte, alongside the canonical per-byte frame:

| field | type | meaning |
|---|---|---|
| `region_choice` | `int[L, top_k]` | top-k expert indices per layer at this byte. |
| `region_gate_weight` | `float[L, n_experts]` | post-softmax gate weights per layer. |
| `valence` | `float` | probe scalar in `[-1, +1]`. Present when probe returns 1D. |
| `arousal` | `float | null` | second affect dimension. Present when probe returns 2D. |
| `refractory_mask` | `float[n_experts]` | per-region penalty added to router logits by the refractory inhibition pass; zero when disabled. |

Field index reserved at hook spec version v8. Adding a field bumps the version per [hooks/contract.md](../hooks/contract.md).

## what the plugin must not do

- Do not import any specific model class. Use the contract surface only.
- Do not write to the canonical checkpoint. Sidecars only.
- Do not retrain the base model. The plugin's only training scope is the LoRA params during `sleep`.
- Do not branch on model variant. Adding a new compatible variant is a model-side change.

## stability

Method names, signatures, contract surface, and file format are stable across patch releases. A signature change is a minor-version bump and requires this file's update obligation to be honored in the same commit.

## update obligation

Adding, removing, or renaming any field or method above requires:

1. The implementation in `veritate_core/multimind/plugin.py` is updated.
2. The reference model in `veritate_core/model_mtm.py` is updated.
3. The plugin test suite in `tests/export/test_multimind_plugin.py` is updated.
4. This file's tables and examples are updated in the same commit.
