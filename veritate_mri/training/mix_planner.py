# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - turns a set of selected corpus stems into a weighted mix the trainers can consume.
#   plan() returns the canonical multicorpus spec string ("stem:0.45,stem:0.30") plus
#   the per-source arithmetic that produced it, so a mix is auditable and repeatable.
# - three rules shape a plan: an epoch cap (no source is drawn more than max_epochs
#   times over its own size), an intent profile (target share per catalog topic), and
#   availability (never plan bytes a source does not have).
# - profiles are DATA, not code: veritate_mri/data/corpus_mix_profiles.json, or any
#   path the user puts in the corpus_mix_profiles_path setting.
# - deterministic: sources are walked in caller order, rows are sorted by
#   (-weight, stem), no RNG, no dict-iteration dependence.
# veritate_mri/training/mix_planner.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

from readers import corpus as corpus_reader, paths
from runtime import settings as settings_mod
from training.sync import corpus_sync

from veritate_core.plugin import multicorpus

# ------------------------------------------------------------------------------------
# Constants

SETTING_MAX_EPOCHS    = "corpus_mix_max_epochs"
SETTING_PROFILE       = "corpus_mix_default_profile"
SETTING_PROFILES_PATH = "corpus_mix_profiles_path"

PROFILES_KEY   = "profiles"
PROFILE_TOPICS = "topics"
PROFILE_SHARE  = "unlisted_topic_share"
PROFILE_STEMS  = "stems"

UNKNOWN_TOPIC = ""

WEIGHT_DECIMALS      = 6
WEIGHT_SUM_TOLERANCE = 1e-6
CAP_EPSILON          = 1e-12

ENCODING = "utf-8"

# ------------------------------------------------------------------------------------
# Functions

def profiles_path():
    return settings_mod.get().get(SETTING_PROFILES_PATH) or paths.mix_profiles_path()


def load_profiles():
    with open(profiles_path(), "r", encoding=ENCODING) as f:
        return json.load(f)[PROFILES_KEY]


def _load_profile(name):
    profiles = load_profiles()
    if name not in profiles:
        raise ValueError(f"unknown mix profile: {name!r} (have: {', '.join(sorted(profiles))})")
    prof = profiles[name]
    for key in (PROFILE_TOPICS, PROFILE_SHARE):
        if key not in prof:
            raise ValueError(f"mix profile {name!r} is missing {key!r}")
    return prof


def profile_stems(name=None):
    return list(_load_profile(name or settings_mod.get()[SETTING_PROFILE]).get(PROFILE_STEMS) or [])


# ------------------------------------------------------------------------------------
# Sources

def _catalog_by_stem():
    return {c["stem"]: c for c in corpus_sync.catalog()["corpora"]}


def _available_bytes(stem, entry):
    train_path, _ = corpus_reader.resolve_paths(stem)
    if train_path:
        return os.path.getsize(train_path)
    return int(entry.get("size_train") or 0) if entry else 0


def _sources(stems, by_stem):
    out = []
    for stem in stems:
        entry = by_stem.get(stem)
        out.append({
            "stem":            stem,
            "label":           (entry.get("label") if entry else None) or stem,
            "topic":           (entry.get("topic") if entry else None) or UNKNOWN_TOPIC,
            "bytes_available": _available_bytes(stem, entry),
            "min_params":      entry.get("recommended_min_params") if entry else None,
            "max_params":      entry.get("recommended_max_params") if entry else None,
        })
    return out


# ------------------------------------------------------------------------------------
# Weighting

def _normalize(weights):
    total = sum(weights)
    if total <= 0:
        raise ValueError("selected corpora carry no data: nothing to weight")
    return [w / total for w in weights]


def _size_weights(sources):
    return _normalize([s["bytes_available"] for s in sources])


def _profile_weights(sources, prof, warnings):
    topics   = prof[PROFILE_TOPICS]
    unlisted = float(prof[PROFILE_SHARE])
    present  = []
    for s in sources:
        if s["topic"] not in present:
            present.append(s["topic"])
    shares = {t: float(topics.get(t, unlisted)) for t in present}
    for topic in sorted(topics):
        if float(topics[topic]) > 0 and topic not in present:
            warnings.append(f"profile wants {float(topics[topic]) * 100:.0f}% {topic!r} but no "
                            f"selected corpus has that topic: its share went to the topics "
                            f"you picked")
    share_total = sum(shares.values())
    if share_total <= 0:
        raise ValueError("profile assigns zero share to every selected topic")
    out = []
    for s in sources:
        peer_bytes = sum(p["bytes_available"] for p in sources if p["topic"] == s["topic"])
        out.append(0.0 if peer_bytes <= 0 else
                   (shares[s["topic"]] / share_total) * (s["bytes_available"] / peer_bytes))
    return out


def _cap_epochs(weights, sources, planned_bytes, max_epochs, warnings):
    """Clamp each source to max_epochs passes over its own bytes, then hand the
    surplus to the sources that still have headroom (repeat until stable)."""
    caps = [s["bytes_available"] * max_epochs / planned_bytes for s in sources]
    out, capped = list(weights), [False] * len(weights)
    while True:
        over = [i for i, w in enumerate(out) if not capped[i] and w > caps[i] + CAP_EPSILON]
        if not over:
            return out
        surplus = 0.0
        for i in over:
            avail = sources[i]["bytes_available"]
            if avail > 0:
                warnings.append(f"{sources[i]['stem']!r} wanted "
                                f"{out[i] * planned_bytes / avail:.1f} epochs of its {avail:,} "
                                f"bytes: capped at {max_epochs:g} and the surplus went to the "
                                f"sources with headroom")
            surplus += out[i] - caps[i]
            out[i], capped[i] = caps[i], True
        free = [i for i in range(len(out)) if not capped[i]]
        free_total = sum(out[i] for i in free)
        if not free or free_total <= 0:
            return out
        for i in free:
            out[i] += surplus * out[i] / free_total


def _round_weights(weights):
    """Round to the spec's precision, then park the rounding residual on the
    heaviest source so the emitted weights sum to exactly 1."""
    out = [round(w, WEIGHT_DECIMALS) for w in weights]
    lead = max(range(len(out)), key=lambda i: (out[i], -i))
    out[lead] = round(out[lead] + (1.0 - sum(out)), WEIGHT_DECIMALS)
    return out


# ------------------------------------------------------------------------------------
# Plan

def _suitability_warnings(sources, model_params):
    out = []
    for s in sources:
        lo, hi = s["min_params"], s["max_params"]
        if lo is not None and model_params < lo:
            out.append(f"{s['stem']!r} is recommended for models of {lo:,}+ params: "
                       f"this model has {model_params:,}")
        elif hi is not None and model_params > hi:
            out.append(f"{s['stem']!r} is recommended for models up to {hi:,} params: "
                       f"this model has {model_params:,}")
    return out


def _spec(rows):
    return multicorpus.LIST_SEP.join(
        f"{r['stem']}{multicorpus.WEIGHT_SEP}{r['weight']:g}" for r in rows if r["weight"] > 0)


def plan(stems, target_bytes, profile=None, max_epochs=None, model_params=None, weights=None):
    """Mix `stems` into `target_bytes` of training data. Explicit `weights`
    ({stem: weight}) bypass the profile but still obey the epoch cap."""
    cfg = settings_mod.get()
    max_epochs = float(cfg[SETTING_MAX_EPOCHS] if max_epochs is None else max_epochs)
    sources    = _sources(stems, _catalog_by_stem())
    warnings   = [f"{s['stem']!r} has no data on disk or in the catalog: dropped from the mix"
                  for s in sources if s["bytes_available"] <= 0]

    if weights:
        profile_used = None
        base = _normalize([float(weights[s["stem"]]) for s in sources])
    elif profile or cfg[SETTING_PROFILE]:
        profile_used = profile or cfg[SETTING_PROFILE]
        base = _normalize(_profile_weights(sources, _load_profile(profile_used), warnings))
    else:
        profile_used = None
        base = _size_weights(sources)

    capacity = sum(s["bytes_available"] for s in sources) * max_epochs
    planned  = min(float(target_bytes), capacity)
    if capacity < target_bytes:
        warnings.append(f"selected corpora yield {capacity:,.0f} bytes at {max_epochs:g} epochs, "
                        f"short of the {target_bytes:,.0f} byte target: the mix stops there")
    final = _round_weights(_normalize(_cap_epochs(base, sources, planned, max_epochs, warnings)))
    if model_params:
        warnings.extend(_suitability_warnings(sources, model_params))

    rows = []
    for s, w in zip(sources, final):
        drawn = w * planned
        rows.append({**s, "weight": w, "bytes_drawn": int(drawn),
                     "epochs": (drawn / s["bytes_available"]) if s["bytes_available"] else 0.0})
    rows.sort(key=lambda r: (-r["weight"], r["stem"]))
    total = sum(r["weight"] for r in rows)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"mix weights sum to {total}, not 1.0")
    return {
        "ok": True,
        "spec": _spec(rows),
        "sources": rows,
        "warnings": warnings,
        "bytes_planned": int(planned),
        "inputs": {"stems": list(stems), "target_bytes": int(target_bytes),
                   "profile": profile_used, "max_epochs": max_epochs,
                   "model_params": model_params,
                   "weights": dict(weights) if weights else None},
    }
