# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - per-model capability tiers. additive: agent implies chat implies autocomplete,
#   enforced in mark (a trained tier lifts lower tiers to at least trained).
# - on-disk under config.json "capabilities" as a versioned, HF-aligned block
#   {schema, pipeline_tag, tasks:{tier:{status}}}. read/_normalize also accept the
#   legacy flat form ({tier:{status}}) so pre-schema configs need no migration.
# - save.py is the only pipeline writer; this module owns the read shape, the
#   write primitive (mark), corpus->tier resolution (modes_for_corpus), and the
#   legacy-block synthesis for pre-pipeline models.
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

STATUS_RANK = {
    STATUS_UNTRAINED:   0,
    STATUS_FAILED:      1,
    STATUS_IN_PROGRESS: 2,
    STATUS_TRAINED:     3,
}

DEFAULT_TEACHES = "autocomplete"

CONFIG_KEY = "capabilities"

SCHEMA_VERSION   = 1
PIPELINE_TAG     = "text-generation"
SCHEMA_KEY       = "schema"
PIPELINE_TAG_KEY = "pipeline_tag"
TASKS_KEY        = "tasks"

# entry metadata carried through _normalize (status is always kept).
PRESERVED_KEYS = ("trainer", "step", "completed_at", "legacy", "implied")

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


def _tasks_of(block):
    """Extract the tier map from either on-disk form: the schema-wrapped block
    ({schema, pipeline_tag, tasks:{...}}) or the legacy flat block ({tier:{...}})."""
    if isinstance(block, dict) and isinstance(block.get(TASKS_KEY), dict):
        return block[TASKS_KEY]
    return block


def _wrap(tasks):
    """Serialize a tier map into the versioned, HF-aligned on-disk block."""
    return {SCHEMA_KEY: SCHEMA_VERSION, PIPELINE_TAG_KEY: PIPELINE_TAG, TASKS_KEY: tasks}


def _normalize(block):
    """Fill missing tiers, drop unknown ones, coerce unknown statuses to
    untrained. Accepts both on-disk forms; always returns the flat tier map."""
    tasks = _tasks_of(block)
    if not isinstance(tasks, dict):
        return _legacy_block()
    out = {}
    for t in TIERS:
        entry = tasks.get(t)
        if not isinstance(entry, dict):
            out[t] = {"status": STATUS_UNTRAINED}
            continue
        status = entry.get("status")
        if status not in STATUSES:
            status = STATUS_UNTRAINED
        new_entry = {"status": status}
        for k in PRESERVED_KEYS:
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
        with open(p, encoding="utf-8") as f:
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


def modes_for_corpus(spec):
    """Resolve a corpus spec (single stem or multicorpus mix) to the capability
    tiers its catalog entries declare via trained_modes. Returns a tuple in TIERS
    order; empty when spec is blank or no member stem is in the catalog (custom
    corpora), so the caller can fall back to the trainer's declared tier."""
    if not isinstance(spec, str) or not spec.strip():
        return ()
    from training.sync import corpus_sync

    from veritate_core.plugin import multicorpus
    by_stem = {}
    for e in corpus_sync._load_local_catalog():
        modes = e.get("trained_modes")
        if isinstance(modes, list):
            by_stem[e.get("stem")] = modes
    found = set()
    for member, _ in multicorpus.parse_spec(spec.strip()):
        stem = member.rsplit(":", 1)[-1] if ":" in member else member
        modes = by_stem.get(stem) or by_stem.get(member)
        if isinstance(modes, list):
            found.update(m for m in modes if m in TIERS)
    return tuple(t for t in TIERS if t in found)


def _apply_additivity(block, tier, status):
    """agent implies chat implies autocomplete. A tier reaching a positive status
    lifts every lower tier to at least that status (implied), never downgrading
    one already ranked higher."""
    rank = STATUS_RANK.get(status, 0)
    if rank < STATUS_RANK[STATUS_IN_PROGRESS]:
        return
    for lower in TIERS[:TIERS.index(tier)]:
        if STATUS_RANK.get(block[lower].get("status"), 0) < rank:
            entry = {"status": status, "implied": True}
            if status == STATUS_TRAINED:
                entry["completed_at"] = _iso_now()
            block[lower] = entry


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
    before = {t: dict(block[t]) for t in TIERS}
    prior = before[tier]

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

    block[tier] = new_entry
    _apply_additivity(block, tier, status)

    if block == before:
        return prior

    data[CONFIG_KEY] = _wrap(block)
    _write_raw(name, data)
    return new_entry


def _iso_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
