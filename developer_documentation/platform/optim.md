# optim (shared optimizer builder)

## what it is

`veritate_core/plugin/optim.py`. Muon + AdamW hybrid optimizer for trainers. Muon (torch-native Newton-Schulz orthogonalized momentum) drives 2D hidden weight matrices; AdamW drives embeddings, norms, and all 1D params.

## how it works

- `build_muon(model, args)` splits `model.named_parameters()`: `p.ndim == 2` and no `"emb"` in the name goes to Muon, everything else to AdamW (`optim.py:57`).
- Muon runs `adjust_lr_fn="match_rms_adamw"` (update RMS matched to AdamW, the Moonlight rule), so one AdamW-scale lr schedule drives both groups. The trainer writes `g["lr"]` into `opt.param_groups` exactly as with plain AdamW.
- `MuonAdamW` wrapper exposes one optimizer surface: `step`, `zero_grad`, `param_groups` (live concatenation), `state_dict`/`load_state_dict` (`{"muon": ..., "adamw": ...}`), so trainers and `save()` never branch on optimizer kind.
- Selected per run via the `optimizer` reserved flag (`trainers/common/vanilla_trainer.py::RESERVED_STR_FLAGS`), values `adamw` (default) | `muon`. Falls back to AdamW loudly when the platform module is missing.

## dependencies

torch >= 2.12 (`torch.optim.Muon`). No third-party packages.

## pitfalls

- A checkpoint saved with `optimizer=muon` cannot resume under `adamw` (state dict shape differs); resume with the same optimizer.
- Muon is skipped when NVMe optimizer paging is active for the run (paged AdamW wins that branch).
