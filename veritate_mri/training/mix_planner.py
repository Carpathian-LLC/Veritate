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

import hashlib
import json
import os
import random

from readers import corpus as corpus_reader
from readers import paths
from runtime import settings as settings_mod

from training.sync import corpus_sync
from veritate_core.plugin import multicorpus

# ------------------------------------------------------------------------------------
# Constants

SETTING_MAX_EPOCHS    = "corpus_mix_max_epochs"
SETTING_PROFILE       = "corpus_mix_default_profile"
SETTING_PROFILES_PATH = "corpus_mix_profiles_path"
SETTING_CHUNK_BYTES   = "corpus_compose_chunk_bytes"
SETTING_VAL_RATIO     = "corpus_compose_val_ratio"
SETTING_COMPOSE_SEED  = "corpus_compose_seed"

HASH_READ_BYTES = 1 << 20

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
    with open(profiles_path(), encoding=ENCODING) as f:
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
    for s, w in zip(sources, final, strict=True):
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


# ------------------------------------------------------------------------------------
# Compose

def _cyclic_read(fh, region_start, region_len, offset, n):
    """n bytes from `offset` inside a region, wrapping at the region end. A source
    drawn for more than one epoch reads its own bytes again rather than running out."""
    out = bytearray()
    pos = offset % region_len
    while len(out) < n:
        fh.seek(region_start + pos)
        buf = fh.read(min(n - len(out), region_len - pos))
        if not buf:
            break
        out += buf
        pos = (pos + len(buf)) % region_len
    return bytes(out)


def _chunk_plan(sources, chunk_bytes, seed):
    """Cut every source's draw into chunk_bytes pieces and shuffle the pieces
    together, so the composed corpus interleaves its sources instead of
    concatenating them as blocks. Deterministic for a given seed."""
    pieces = []
    for i, src in enumerate(sources):
        offset = 0
        while offset < src["draw"]:
            pieces.append((i, offset, min(chunk_bytes, src["draw"] - offset)))
            offset += chunk_bytes
    random.Random(seed).shuffle(pieces)
    return pieces


def _write_split(out_path, sources, chunk_bytes, seed, progress=None):
    digest, written = hashlib.sha256(), 0
    pieces = _chunk_plan(sources, chunk_bytes, seed)
    total = sum(s["draw"] for s in sources)
    handles = {}
    try:
        with open(out_path, "wb") as out:
            for idx, offset, length in pieces:
                src = sources[idx]
                fh = handles.get(idx)
                if fh is None:
                    # long-lived handle: source handles are cached across pieces, closed in the finally.
                    fh = handles[idx] = open(src["path"], "rb")  # noqa: SIM115
                buf = _cyclic_read(fh, src["region_start"], src["region_len"], offset, length)
                out.write(buf)
                digest.update(buf)
                written += len(buf)
                if progress:
                    progress(written, total)
    finally:
        for fh in handles.values():
            fh.close()
    return {"bytes": written, "sha256": digest.hexdigest()}


def compose(stem, plan_result, val_ratio=None, seed=None, chunk_bytes=None, progress=None):
    """Materialize a plan into ONE unified corpus: <stem>_train.bin + <stem>_val.bin.

    Every source is split by byte offset into a head (train) and a tail (val)
    region first, so a composed corpus can never leak val bytes into train no
    matter how many epochs a source is drawn for."""
    cfg         = settings_mod.get()
    val_ratio   = float(cfg[SETTING_VAL_RATIO] if val_ratio is None else val_ratio)
    seed        = int(cfg[SETTING_COMPOSE_SEED] if seed is None else seed)
    chunk_bytes = int(cfg[SETTING_CHUNK_BYTES] if chunk_bytes is None else chunk_bytes)
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")

    drawn = [r["stem"] for r in plan_result["sources"] if r["bytes_drawn"] > 0]
    if stem in drawn:
        raise ValueError(f"{stem!r} is both the output and one of the sources: composing would "
                         f"overwrite the source while reading it. Give the output a new name.")

    train_sources, val_sources, rows = [], [], []
    for row in plan_result["sources"]:
        if row["bytes_drawn"] <= 0:
            continue
        path, _ = corpus_reader.resolve_paths(row["stem"])
        if not path:
            raise ValueError(f"{row['stem']!r} has no train bin on disk: install it first")
        size    = os.path.getsize(path)
        val_len = int(size * val_ratio)
        if val_ratio > 0 and val_len <= 0:
            raise ValueError(f"{row['stem']!r} is too small to hold out a val region")
        train_len  = size - val_len
        train_draw = int(row["bytes_drawn"])
        val_draw   = int(train_draw * val_ratio)
        train_sources.append({"path": path, "region_start": 0, "region_len": train_len,
                              "draw": train_draw})
        if val_draw > 0:
            val_sources.append({"path": path, "region_start": train_len,
                                "region_len": val_len, "draw": val_draw})
        rows.append({"stem": row["stem"], "weight": row["weight"],
                     "bytes_available": row["bytes_available"], "train_bytes": train_draw,
                     "val_bytes": val_draw, "epochs": round(row["epochs"], 4),
                     "source_path": path})

    if not train_sources:
        raise ValueError("plan draws no bytes: nothing to compose")

    train = _write_split(paths.corpus_train_path(stem), train_sources, chunk_bytes, seed, progress)
    val   = ({"bytes": 0, "sha256": hashlib.sha256().hexdigest()} if not val_sources else
             _write_split(paths.corpus_val_path(stem), val_sources, chunk_bytes, seed + 1))
    if not val_sources:
        with open(paths.corpus_val_path(stem), "wb"):
            pass
    return {
        "ok": True, "stem": stem,
        "train_path": paths.corpus_train_path(stem), "val_path": paths.corpus_val_path(stem),
        "train_bytes": train["bytes"], "val_bytes": val["bytes"],
        "sha256_train": train["sha256"], "sha256_val": val["sha256"],
        "sources": rows, "spec": plan_result["spec"], "warnings": plan_result["warnings"],
        "compose": {"val_ratio": val_ratio, "seed": seed, "chunk_bytes": chunk_bytes},
    }
