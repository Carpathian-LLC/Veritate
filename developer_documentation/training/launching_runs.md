# Launching a training run (contract)

How a training run is started, which fields are required, and the gotchas. Applies to
every `trainers/<plugin>/trainer.py` (each calls `trainers/common/vanilla_trainer.py::run`).

## Canonical launcher

`training/trainer_runner.py::start(plugin_id, args)` is the one entry point. The dashboard
POST `/trainers/run` calls it. It spawns `trainers/<plugin>/trainer.py` as a subprocess,
sets the env below, writes `.plugin_pid.json` (so the dashboard re-discovers an in-flight
run after a server restart via `_recover_from_disk`), and streams stdout to
`.plugin_run.log`. The dashboard training tab reads `trainer_runner.state()`.

To launch headless without the dashboard server, replicate the spawn: build argv with
`trainer_runner._build_argv(plugin, args)`, set the env keys below, `Popen(..., cwd=REPO_ROOT,
stdout=open(RUN_LOG_FILE,"w"), stderr=STDOUT)`, and write `.plugin_pid.json` in the
`_read_pid_file` shape (`plugin_id, pid, started_at, args, cmd_marker`). This avoids
`start()`'s `update_defaults` call, which otherwise writes run-specific values into the
synced trainer `manifest.json` (undesirable: `trainers/` is an upstream-synced checkout).

## Required args (run crashes without them)

- `name` : model slug (final dir is `<slug>_<size>`).
- `corpus` : corpus stem, resolved to `trainers/corpus/<stem>_{train,val}.bin`. Mixes:
  `a+b` (size-weighted) or `a:0.5,b:0.5` (explicit).
- `description` : non-empty. `save.require_description` raises `ValueError` otherwise.

## model_type is MANDATORY (read this before launching anything non-text)

`model_type` is a REQUIRED per-RUN choice, not a trainer property. Choices:
`language` | `code` | `statistical` | `other`. It decides which checkpoint evaluations run
and how the dashboard labels and renders the model.

- `language` runs the FULL language-probe suite (fluency, reading, grammar, reasoning,
  concepts, writing).
- `code`, `statistical`, `other` SKIP those language probes: they are meaningless for a
  non-text model.

**ABSENT => SILENTLY TREATED AS `language` => WRONG PROBES.** If a model's config has no
`model_type`, the dashboard defaults it to `language`
(`veritate_mri/web/index.js:5783`: `(config.training_args.model_type) || "language"`, in
`applyEvalGate`) and `save.py` defaults it to `language` too
(`veritate_mri/training/save.py:500`: `(mtype or "language").lower()`). The language probe
suite then runs on and displays for a market/statistical byte model: meaningless and wrong.

**Market / byte-series models = `statistical`.** A price-series / market model that omits
`model_type` is mislabeled as a language model, accrues meaningless language scores, and
shows empty language panels. There is no auto-detection. Set it explicitly.

### The reliable way to set it: the dashboard run form

Train THROUGH the dashboard. The `/trainers/run` form has a `model_type` field
(`veritate_mri/web/index.js:7154`, `TRAINER_SCHEMA.scratch`) with exactly the four choices
above. The runner carries the chosen value to the trainer as the env var
`VERITATE_MODEL_TYPE` (`veritate_mri/training/trainer_runner.py:51`, set from
`args["model_type"]` at lines 286-288), and `save.py` stamps it into
`config.training_args.model_type` (`veritate_mri/training/save.py:430-436`). The same gate at
`save.py:497-501` skips `LANGUAGE_DUMPS` for any non-`language` type.

### The trap: manual / one-off launchers SILENTLY DROP `--model_type`

`model_type` is NOT a manifest field. The same trainer (e.g. `veritate_80m`) trains BOTH
language and statistical models, so it is chosen per run, not per trainer. Confirmed: no
trainer `manifest.json` carries a `model_type` key.

Because it is not in any manifest, a trainer's `parse_known_args()` DISCARDS a `--model_type`
CLI flag (`trainer_runner.py:48-51` documents this). `model_type` ONLY survives via the env
-> `save.py` path. A hand-rolled launcher that passes `--model_type` on the command line but
does NOT export `VERITATE_MODEL_TYPE` records NOTHING: the model ends up with `model_type`
absent => defaulted to `language` => wrong probes. If you must launch headless, set the env
var yourself (see "Canonical launcher" above); do not rely on the CLI flag.

## Recipe flags (declared unconditionally; manifest defaults override)

- bool: `use_act_ckpt`, `use_8bit_adam`, `qat_enabled`
- str: `activation` (`gelu`/`relu`/`silu`)
- float: `l1_lambda`

Plus every key in the plugin's `manifest.json` `defaults` becomes a `--<key>` flag
(`size`, `precision`, `seq`, `batch_size`, `total_steps`, `ckpt_every`, `eval_every`,
`base_lr`, `lr_schedule`, ...).

## Gotchas

- `use_8bit_adam` needs `bitsandbytes`. Not installed -> falls back to torch AdamW (fp32
  moments, ~2x optimizer memory). Fine at <=200M on M3 Ultra; do not chase bitsandbytes on
  MPS.
- Never launch through `... | tee file`: the pipe reports tee's exit code (0) and masks a
  crashed trainer. Use `> .plugin_run.log 2>&1`.
- `VERITATE_PLUGIN_ID` must be set in the child env; `save.py` reads it.
- Device: arm64 Mac auto-selects MPS. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so an unsupported
  op falls back to CPU instead of crashing an unattended run.

## Sizing

- tokens/step is approximately `batch_size x effective_seq`. Chinchilla-optimal is ~20
  tokens per parameter (200M -> ~4B tokens, 400M -> ~8B, 800M -> ~16B).
- `total_steps` sets the WSD/cosine LR horizon. Set it to the intended run length; stopping
  early leaves the LR un-annealed (checkpoints still usable).
- `save()` dumps the full hook suite into `models/<name>/hooks/step_<N>/` every `ckpt_every`
  (hooks contract: `documentation/hooks/contract.md`). Frequent checkpoints (every 500)
  produce many large dirs: watch disk.
