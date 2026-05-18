# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - per-model capability tiers. additive: agent implies chat implies autocomplete.
# - lives inside config.json under the "capabilities" key. save.py is the only
#   writer in the training pipeline; this module owns the read shape, the write
#   primitive (mark), and the legacy-block synthesis for pre-pipeline models.
# veritate_mri/readers/capabilities.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import time

from . import paths

# ------------------------------------------------------------------------------------
# Constants

TIERS = ("autocomplete", "chat", "agent")

STATUS_UNTRAINED   = "untrained"
STATUS_IN_PROGRESS = "in_progress"
STATUS_TRAINED     = "trained"
STATUS_FAILED      = "failed"

STATUSES = (STATUS_UNTRAINED, STATUS_IN_PROGRESS, STATUS_TRAINED, STATUS_FAILED)

DEFAULT_TEACHES = "autocomplete"

CONFIG_KEY = "capabilities"

# ------------------------------------------------------------------------------------
# Functions

def _empty_block():
    return {t: {"status": STATUS_UNTRAINED} for t in TIERS}


def _legacy_block():
    """Pre-pipeline models with no capabilities key are assumed to have produced
    a checkpoint, so autocomplete is trained. Higher tiers stay untrained."""
    b = _empty_block()
    b["autocomplete"] = {"status": STATUS_TRAINED, "legacy": True}
    return b


def _normalize(block):
    """Fill missing tiers, drop unknown ones, coerce unknown statuses to untrained."""
    if not isinstance(block, dict):
        return _legacy_block()
    out = {}
    for t in TIERS:
        entry = block.get(t)
        if not isinstance(entry, dict):
            out[t] = {"status": STATUS_UNTRAINED}
            continue
        status = entry.get("status")
        if status not in STATUSES:
            status = STATUS_UNTRAINED
        new_entry = {"status": status}
        for k in ("trainer", "step", "completed_at", "legacy"):
            if k in entry:
                new_entry[k] = entry[k]
        out[t] = new_entry
    return out


def _config_path(name):
    return paths.config_path(name)


def _read_raw(name):
    p = _config_path(name)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_raw(name, data):
    p = _config_path(name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def read(name):
    """Return the model's normalized capability block. Missing models or
    missing keys yield the legacy fallback (autocomplete=trained)."""
    data = _read_raw(name)
    if not isinstance(data, dict):
        return _legacy_block()
    block = data.get(CONFIG_KEY)
    if block is None:
        return _legacy_block()
    return _normalize(block)


def highest_trained(block):
    """Return the highest tier in TIERS order that has status=trained, else None."""
    for t in reversed(TIERS):
        entry = block.get(t)
        if isinstance(entry, dict) and entry.get("status") == STATUS_TRAINED:
            return t
    return None


def mark(name, tier, status, *, trainer=None, step=None, total_steps=None):
    """Update one tier's status in the model's config.json::capabilities.

    Rules:
        - status must be one of STATUSES.
        - status=trained is forced when caller passes step >= total_steps and
          status would otherwise be in_progress. Caller may also pass status
          STATUS_TRAINED directly.
        - status=failed is dropped on the floor if the prior status was
          STATUS_TRAINED (do not regress a finished tier).

    Returns the new entry dict on success, None if the config file is missing
    or unreadable (caller is expected to have run save._ensure_config first).
    """
    if tier not in TIERS:
        raise ValueError(f"unknown capability tier: {tier!r}")
    if status not in STATUSES:
        raise ValueError(f"unknown capability status: {status!r}")

    data = _read_raw(name)
    if not isinstance(data, dict):
        return None

    block = _normalize(data.get(CONFIG_KEY))
    prior = dict(block[tier])

    if status == STATUS_FAILED and prior.get("status") == STATUS_TRAINED:
        return prior

    if (status == STATUS_IN_PROGRESS
            and step is not None and total_steps is not None
            and int(step) >= int(total_steps) > 0):
        status = STATUS_TRAINED

    new_entry = {"status": status}
    if trainer:
        new_entry["trainer"] = str(trainer)
    if step is not None:
        new_entry["step"] = int(step)
    if status == STATUS_TRAINED:
        new_entry["completed_at"] = _iso_now()

    if new_entry == prior:
        return prior

    block[tier] = new_entry
    data[CONFIG_KEY] = block
    _write_raw(name, data)
    return new_entry


def _iso_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
