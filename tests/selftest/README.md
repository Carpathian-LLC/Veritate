# selftest

Centralized, top-to-bottom self-test for the Veritate platform. One entry point exercises every area: core model, QAT, plugin contract, readers, runtime, training, save / export, engine binary, decoders, addons, agent loop, eval, and every cold-state MRI endpoint. Use it after a major refactor to confirm nothing burned down.

This is **separate from `pytest`**. The pytest suite under `tests/engine/`, `tests/mri/`, `tests/export/`, `tests/plugin_contract/` is unit-style and CI-friendly. `selftest/` is whole-platform smoke: each check writes a full trace to its own log file so failures are diagnosable in one place.

## run

From the repo root:

```bash
python tests/selftest/run_all.py
```

Or as a module (also works):

```bash
python -m tests.selftest.run_all
```

Options:

| flag | effect |
|---|---|
| `--list` | print discovered check names and exit. |
| `--only core_model,mri_endpoints` | run only the named checks (comma-separated, with or without the `check_` prefix). |
| `--skip engine_loads_models` | skip the named checks. |
| `--fast` | skip checks marked `SLOW = True`. |
| `--no-network` | skip checks marked `REQUIRES_NETWORK = True`. |

Exit code is `0` when no check fails (skips do not fail the run), `1` otherwise.

## logs

Each run writes to `tests/selftest/logs/<timestamp>/`:

```
logs/20260515_142301/
├── run.log               # console mirror: one line per check + aggregate
├── summary.json          # machine-readable result for every check
├── check_core_model.log  # per-check trace: stdout, stderr, exception, timing
├── check_mri_endpoints.log
├── ...
```

`*.log` is gitignored, so logs never get committed.

`summary.json` schema:

```json
{
  "run_id": "20260515_142301",
  "elapsed": 4.21,
  "counts":  { "ok": 18, "skip": 4, "fail": 0 },
  "results": [
    {
      "name": "core_model",
      "status": "ok",
      "summary": "forward shape (2, 8, 256)",
      "details": { "params": 12345 },
      "log_path": "tests/selftest/logs/.../check_core_model.log",
      "elapsed": 0.42
    },
    ...
  ]
}
```

## what each check covers

| check | area | what it verifies |
|---|---|---|
| `check_versions_json` | platform | `versions.json` parses + has every required key. |
| `check_core_model` | core | tiny `Veritate` instantiates and forwards. |
| `check_core_model_rope` | core | tiny `VeritateRoPE` instantiates and forwards. |
| `check_core_qat` | core | every `fake_quant_*` op runs; `set_qat` toggles nested flags. |
| `check_qat_triton_parity` | core | triton fused QAT equals reference. CUDA only. |
| `check_plugin_surface` | plugin_contract | `veritate_core.plugin` re-exports the documented names. |
| `check_core_plugins` | plugin_contract | built-in plugin registry imports. |
| `check_readers_paths` | platform | every `readers.paths.*` helper returns a string. |
| `check_readers_misc` | platform | every other `readers/*.py` imports. |
| `check_runtime` | platform | `runtime/*` imports + `sys_metrics.snapshot()` returns a dict. |
| `check_training_modules` | platform | every `training/*` + `training/sync/*` imports. |
| `check_save_roundtrip` | platform | `save()` writes a checkpoint that loads with matching norms. |
| `check_export` | export | tiny model export writes a VRTE-magic `.bin`. |
| `check_engine_binary` | engine | engine self-test exits 0 + prints kernel verify lines. |
| `check_engine_loads_models` | engine | every committed `.bin` benches cleanly. SLOW. |
| `check_engine_build_paths` | engine | engine source tree has the expected dirs. |
| `check_trainers_discovery` | plugins | every `trainers/<name>/` has parseable `manifest.json` + `trainer.py`. |
| `check_decode_imports` | inference | every `inference/decode/*` imports. |
| `check_backend_pytorch` | inference | `Brain` class has stream / stream_fast. |
| `check_backend_c` | inference | `CTracedSubprocess` class imports + engine binary detected. |
| `check_addons` | inference | `inference.addons.list_addons()` works. |
| `check_agent_loop` | inference | `AgentLoop` imports + default toolbox builds. |
| `check_eval` | eval | every eval suite + `run_eval` entry imports. |
| `check_app_imports` | mri | dashboard flask app imports cleanly. |
| `check_mri_endpoints` | mri | every cold read-only endpoint returns 200 + correct content-type. |
| `check_corpus_catalog` | mri | corpus library catalog endpoint responds. |
| `check_documentation` | docs | every required `documentation/<area>/contract.md` exists. |

## adding a new check

1. **Create** `tests/selftest/checks/check_<name>.py`. The file name must start with `check_` and end with `.py`. The runner discovers it on the next run.

2. **Use the standard file header** (matches `claude_preflight.md` rule 13):

```python
# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - one line: what this check verifies + the falsifier.
# tests/selftest/checks/check_<name>.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA = "<area>"   # one of: core, platform, plugin_contract, plugins, mri,
                  # inference, engine, eval, export, docs.

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """one-line docstring: what this verifies. no 'why'."""
    ...
    return _status.ok("<name>", "summary string")
```

3. **The `run(ctx)` contract:**
   - Receives a `tests.selftest._ctx.Ctx` object with `repo`, `models`, `trainers`, `mri`, `engine` paths plus `run_id`, `log_dir`, `options`.
   - Returns one of `_status.ok(name, summary, details=None)`, `_status.skip(name, summary)`, `_status.fail(name, summary, details=None)`.
   - **Skip** when a prerequisite is missing (no model, binary not built, optional dep). Skips do not fail the run.
   - **Fail** only for real regressions.
   - Print whatever you want; stdout / stderr are captured into the check's own log file.
   - Raise nothing: uncaught exceptions are caught by the runner and converted to FAIL with full traceback in the log.

4. **Module-level flags (optional):**
   - `SLOW = True`: gates the check behind `--fast` not being set.
   - `REQUIRES_NETWORK = True`: gates the check behind `--no-network` not being set.
   - `REQUIRES_MODEL = True`, `REQUIRES_ENGINE_BUILT = True`, `REQUIRES_TORCH = True`: documentation only; gate inside `run()` with a `_status.skip(...)` so the user gets a clean reason.

5. **Imports inside `run()`**: keep platform imports lazy so an import error in one module never breaks discovery for another check. The runner catches the exception and records the failure cleanly.

6. **Update this README's "what each check covers" table** with a row for your new check.

## design rules

- **No hardcoded paths.** Pull paths from `_ctx`. The runner-side constants live in `_ctx.py`; check-side constants go in the check module's `Constants` section.
- **No live network calls.** Mock or skip.
- **No state leaks.** A check that writes files cleans them up in `finally`.
- **Cold by default.** A check that loads a real model or starts a real subprocess marks itself `SLOW = True` and gates on prerequisites in `run()`.
- **One file per concern.** Don't merge two checks into one module to save imports. The point is that each check produces its own log file.

## relationship with `tests/` pytest suite

| use selftest when ... | use pytest when ... |
|---|---|
| auditing the platform end-to-end after a refactor | iterating on one area |
| diagnosing a startup or wiring regression | enforcing per-function contracts |
| producing artifacts (logs) you can attach to a bug | running CI on a PR |

Some pytest cases (engine self-test, plugin surface, QAT parity) are mirrored here intentionally so the centralized run covers them too. Drift between the two is fine: pytest stays narrow and fast; selftest stays broad and forgiving (skips, not fails, on missing prerequisites).
