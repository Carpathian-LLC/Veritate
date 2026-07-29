# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Checkpoint retention. A long pretrain writes one .pt every ckpt_every steps and
#   never removes any: a 200M run at ckpt_every 500 lays down 1.5 GB per checkpoint,
#   which is 240 GB by the halfway mark. Retention keeps a milestone ladder plus the
#   newest few and drops the rest.
# - hooks/step_<N>/ is NEVER touched. The dump suite is the research artifact (probe,
#   lens, classroom, eval suites, generations) at a few MB a step, and it is what
#   makes a passed step analyzable after its weights are gone.
# - plan() is pure: it reads and returns what would happen. prune() is the only thing
#   that deletes, and it re-plans internally so a stale plan can never be replayed.
# veritate_mri/training/retention.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import time

from readers import checkpoints, models, paths
from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

# Initial values for the dashboard controls, not a policy: every one of these is
# user-set per prune. keep_every 0 means "no ladder, keep only the newest".
DEFAULT_KEEP_EVERY = 5000
DEFAULT_KEEP_LAST  = 4
# A checkpoint younger than this may still be mid-write by a live trainer, which
# writes step_<N>.pt.tmp and renames. Never delete inside the window.
MIN_AGE_S = 1200

# ------------------------------------------------------------------------------------
# Errors

class RetentionError(ValueError):
    pass

# ------------------------------------------------------------------------------------
# Functions

def _size(path):
    try: return os.path.getsize(path)
    except OSError: return 0


def _age_s(path, now):
    try: return max(0.0, now - os.path.getmtime(path))
    except OSError: return 0.0


def plan(name, keep_every=DEFAULT_KEEP_EVERY, keep_last=DEFAULT_KEEP_LAST,
         min_age_s=MIN_AGE_S):
    """What a prune of <name> would keep and delete. Reads only.

    A step survives if it is on the ladder (step % keep_every == 0), among the
    newest keep_last, or younger than min_age_s. Returns a dict the route hands
    straight to the dashboard so the user sees the plan before committing.
    """
    if not models.is_valid_name(name):
        raise RetentionError(f"invalid model name: {name!r}")
    if not models.exists(name):
        raise RetentionError(f"no such model: {name}")
    try:
        keep_every = max(0, int(keep_every))
        keep_last  = int(keep_last)
        min_age_s  = max(0, int(min_age_s))
    except (TypeError, ValueError) as e:
        raise RetentionError(f"keep_every, keep_last and min_age_s must be integers: {e}") from e
    if keep_last < 1:
        raise RetentionError("keep_last must be at least 1: a model always keeps its newest checkpoint")

    steps  = checkpoints.list_steps(name)
    newest = set(steps[-keep_last:])
    now    = time.time()
    keep, drop = [], []
    for step in steps:
        path = checkpoints.path_for(name, step)
        row  = {"step": step, "bytes": _size(path)}
        if step in newest:
            row["reason"] = "newest"
        elif keep_every and step % keep_every == 0:
            row["reason"] = "milestone"
        elif _age_s(path, now) < min_age_s:
            row["reason"] = "too recent to be safe"
        else:
            drop.append(row)
            continue
        keep.append(row)
    return {
        "ok":            True,
        "model":         name,
        "keep_every":    keep_every,
        "keep_last":     keep_last,
        "min_age_s":     min_age_s,
        "total":         len(steps),
        "keep":          keep,
        "delete":        drop,
        "keep_bytes":    sum(r["bytes"] for r in keep),
        "delete_bytes":  sum(r["bytes"] for r in drop),
        "hooks_kept":    len(_hook_steps(name)),
    }


def _hook_steps(name):
    d = paths.hooks_dir(name)
    if not os.path.isdir(d):
        return []
    return sorted(int(m.group(1)) for m in
                  filter(None, (paths.HOOK_STEP_RE.match(e) for e in os.listdir(d))))


def prune(name, keep_every=DEFAULT_KEEP_EVERY, keep_last=DEFAULT_KEEP_LAST,
          min_age_s=MIN_AGE_S):
    """Delete every checkpoint plan() marks for deletion. hooks/ is untouched."""
    p = plan(name, keep_every, keep_last, min_age_s)
    deleted, freed, failed = [], 0, []
    for row in p["delete"]:
        path = checkpoints.path_for(name, row["step"])
        try:
            os.remove(path)
        except OSError as e:
            failed.append({"step": row["step"], "error": str(e)})
            continue
        deleted.append(row["step"])
        freed += row["bytes"]
    logmod.ok("training-retention",
              f"pruned {name}: removed {len(deleted)} of {p['total']} checkpoints, "
              f"freed {freed / (1 << 30):.1f} GB, kept {len(p['keep'])} "
              f"(every {keep_every} + newest {keep_last}), hooks untouched")
    p.update({"deleted": deleted, "freed_bytes": freed, "failed": failed,
              "remaining": len(checkpoints.list_steps(name))})
    return p
