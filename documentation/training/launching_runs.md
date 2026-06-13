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

## model_type (drives evals AND dashboard display)

Choices: `language` | `code` | `statistical` | `other`. Rides through the env var
`VERITATE_MODEL_TYPE` (the trainer's `parse_known_args` drops `--model_type`), and is
written into `config.training_args.model_type`. Controls which checkpoint evaluations run:

- `language` (the default if unset) runs the full language-probe suite (fluency, reading,
  grammar, reasoning, concepts, writing).
- `code`, `statistical`, `other` skip the language probes because they are meaningless for
  non-text models.

A price-series / market model MUST set `model_type=statistical`: otherwise it defaults to
`language`, runs and stores meaningless language probes, and is mislabeled in the dashboard.

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
