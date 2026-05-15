# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Fork = "branch" a model. Copies ONLY the latest checkpoint + config.json
#   from source into a new model directory. The new model becomes an
#   independent training root that can be continued or fine-tuned on a
#   different corpus (chat, agent_json, ...) without touching the source.
# - Copied alongside the checkpoint:
#     * hooks/step_<latest>/      (per-step probe / lens / generation; these
#                                  are point-in-time snapshots, not history)
#     * train.csv                 (header + rows at step == forked_step only;
#                                  the chart shows where the model is now, not
#                                  the full history that led there)
#     * neuron_memory.json        (per-neuron training-data memories the
#                                  dashboard renders)
# - Things deliberately NOT copied:
#     * older hook step dirs (only the forked step is needed; future steps
#       dump fresh hook artifacts on every save())
#     * older checkpoints (point is to fork forward, not duplicate history)
#     * older train.csv rows (lr / loss / grad_norm / throughput history is
#       pruned so the live training panel starts clean)
#     * veritate.bin (engine-built artifact; regenerated on export)
# - The new config records `forked_from = {source, step}` so the dashboard
#   can show provenance and so agent tooling can follow the chain back.
# veritate_mri/training/fork.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil

from readers import checkpoints, config as cfg_reader, models as models_reader, paths
from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_FORK_BUDGET = 5000
DEFAULT_FORK_WARMUP = 200
FORK_DESCRIPTION_TMPL = "Forked from {source} at step {step}."

# ------------------------------------------------------------------------------------
# Errors

class ForkError(ValueError):
    pass

# ------------------------------------------------------------------------------------
# Functions

def fork_model(source, new_name):
    """Create models/<new_name>/ as a fresh training root seeded with source's
    latest checkpoint. Returns a dict suitable for JSON response.

    Raises ForkError with a user-readable message on validation failure."""
    source   = (source or "").strip()
    new_name = (new_name or "").strip()

    if not source:
        raise ForkError("missing source model")
    if not new_name:
        raise ForkError("missing new_name")
    if not models_reader.exists(source):
        raise ForkError(f"source model {source!r} does not exist")
    if not models_reader.is_valid_name(new_name):
        raise ForkError(
            f"new_name {new_name!r} is not a valid model name. "
            "Use lowercase letters, digits and underscores, ending in a "
            "size token like '_85m' or '_1b' (e.g. 'chatty_otter_85m')."
        )
    if models_reader.exists(new_name):
        raise ForkError(f"a model named {new_name!r} already exists; pick a different name")

    latest = checkpoints.latest_step(source)
    if latest is None:
        raise ForkError(f"source model {source!r} has no checkpoints to fork from")

    src_ckpt = paths.checkpoint_path(source, latest)
    if not os.path.isfile(src_ckpt):
        raise ForkError(f"latest checkpoint missing on disk: {src_ckpt}")

    src_cfg = cfg_reader.load(source)
    if src_cfg is None:
        raise ForkError(f"source model {source!r} has no config.json")

    new_dir = paths.model_dir(new_name)
    new_ckpt_dir = paths.checkpoints_dir(new_name)

    # Atomic-ish: make the new dir last so a partial fork doesn't leave a
    # half-built model behind. shutil.copyfile is per-file atomic on most
    # filesystems; we copy to a .part path and rename.
    try:
        os.makedirs(new_ckpt_dir, exist_ok=False)
    except FileExistsError:
        # Someone else created it between the exists() check and now.
        raise ForkError(f"directory for {new_name!r} appeared during fork")
    except OSError as e:
        raise ForkError(f"could not create {new_dir}: {e}")

    try:
        # Copy the checkpoint.
        dst_ckpt = paths.checkpoint_path(new_name, latest)
        tmp = dst_ckpt + ".part"
        shutil.copyfile(src_ckpt, tmp)
        os.replace(tmp, dst_ckpt)
        ckpt_bytes = os.path.getsize(dst_ckpt)

        # Copy hooks for the forked step so the dashboard's probe / lens /
        # generation panels render against the new model immediately. Future
        # steps emit fresh hook artifacts via save().
        src_hook = paths.hook_step_dir(source, latest)
        dst_hook = paths.hook_step_dir(new_name, latest)
        hook_files = 0
        if os.path.isdir(src_hook):
            os.makedirs(dst_hook, exist_ok=True)
            for entry in os.listdir(src_hook):
                sp = os.path.join(src_hook, entry)
                if not os.path.isfile(sp):
                    continue
                dp = os.path.join(dst_hook, entry)
                tmp_hp = dp + ".part"
                shutil.copyfile(sp, tmp_hp)
                os.replace(tmp_hp, dp)
                hook_files += 1

        # Prune train.csv: keep the header + only the rows at step ==
        # forked_step. The fork's purpose is to show where the model is now,
        # not where it came from; the chart panels for loss, lr, grad_norm,
        # and throughput therefore start clean.
        src_csv = paths.train_csv_path(source)
        csv_rows_kept = 0
        if os.path.isfile(src_csv):
            dst_csv = paths.train_csv_path(new_name)
            tmp_csv = dst_csv + ".part"
            with open(src_csv, "r", encoding="utf-8") as f_in, \
                 open(tmp_csv, "w", encoding="utf-8", newline="") as f_out:
                header = f_in.readline()
                f_out.write(header)
                step_str = str(int(latest))
                for line in f_in:
                    parts = line.split(",", 1)
                    if parts and parts[0].strip() == step_str:
                        f_out.write(line)
                        csv_rows_kept += 1
            os.replace(tmp_csv, dst_csv)

        # Copy neuron_memory.json so the dashboard's "what activated this
        # neuron hardest" panel works on the forked model out of the gate.
        src_mem = os.path.join(paths.model_dir(source), "neuron_memory.json")
        if os.path.isfile(src_mem):
            dst_mem = os.path.join(paths.model_dir(new_name), "neuron_memory.json")
            tmp_mem = dst_mem + ".part"
            shutil.copyfile(src_mem, tmp_mem)
            os.replace(tmp_mem, dst_mem)

        # Build the new config. Preserve architecture and most training_args
        # so the trainer can resume from this checkpoint by default. The
        # user typically swaps the corpus on the form for fine-tunes.
        new_cfg = dict(src_cfg)
        new_cfg["name"] = new_name
        # Reset step to the forked step so the resume points at the right
        # checkpoint. The trainer will continue from here.
        new_cfg["step"] = latest
        new_cfg["forked_from"] = {"source": source, "step": latest}

        ta = dict(new_cfg.get("training_args") or {})
        ta["output_dir"] = new_dir
        # Mark this as a resume-friendly starting point. The form's
        # "continue saved" picker can then point at <new_name> and the
        # trainer will pick up step_<latest>.pt.
        ta["resume"] = True
        # Reset training-budget knobs that only make sense for a from-scratch
        # run. Without this, the trainer's main loop sees an empty range and
        # exits silently after fork.
        ta["total_steps"]  = latest + DEFAULT_FORK_BUDGET
        ta["warmup_steps"] = DEFAULT_FORK_WARMUP
        ta["description"]  = FORK_DESCRIPTION_TMPL.format(source=source, step=latest)
        # Clear the source's hardcoded corpus paths. apply_resume_overrides
        # would otherwise restore them and block the corpus stem the user picks
        # on the form: the trainer resolves --corpus <stem> only when both
        # --corpus_bin and --val_bin are empty.
        ta["corpus_bin"] = ""
        ta["val_bin"]    = ""
        new_cfg["training_args"] = ta

        cfg_path = paths.config_path(new_name)
        tmp_cfg = cfg_path + ".part"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(new_cfg, f, indent=2, ensure_ascii=False)
        os.replace(tmp_cfg, cfg_path)
    except (OSError, ValueError) as e:
        # Roll back the partial directory so the user isn't left with a
        # broken model entry.
        shutil.rmtree(new_dir, ignore_errors=True)
        raise ForkError(f"fork failed during copy: {e}")

    logmod.ok("training-fork", f"forked {source}@{latest} -> {new_name} (hooks: {hook_files} files, csv rows: {csv_rows_kept})")
    return {
        "ok":             True,
        "source":         source,
        "new_name":       new_name,
        "step":           latest,
        "checkpoint_bytes": ckpt_bytes,
        "hook_files":     hook_files,
        "csv_rows_kept":  csv_rows_kept,
        "new_dir":        new_dir,
    }
