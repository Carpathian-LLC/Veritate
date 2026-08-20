# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Reads the seed packs the Distillation tab's interview mode draws its opening
#   questions from. One JSON file per vertical in data/authoring/seeds/, each a
#   list of named topic groups, so a corpus can be narrowed to chosen subjects
#   rather than always being one undifferentiated file.
# - Seeds are the ceiling on this whole mode. Measured 2026-08-20: one seed yields
#   roughly 150 usable openers before the distinct-5-gram ratio reaches the 0.90
#   floor, so the seed count sets the maximum corpus size. 1,520 conversation
#   seeds is about 230,000 conversations, near 500 MB.
# - PLANNED_VERTICALS exist so the dashboard can show what is coming without
#   pretending it is usable: a vertical is selectable only when its file is on
#   disk and parses. Availability is computed, never declared.
# veritate_mri/readers/seeds.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

from readers import paths as paths_mod

# ------------------------------------------------------------------------------------
# Constants

SEEDS_DIRNAME = os.path.join("authoring", "seeds")
PACK_SUFFIX = ".json"

# Verticals we intend to ship. Presence here does NOT make one selectable.
PLANNED_VERTICALS = (
    {"vertical": "conversation", "label": "Conversation and talking"},
    {"vertical": "code", "label": "Code and programming"},
    {"vertical": "technical", "label": "Technical and scientific"},
    {"vertical": "business", "label": "Business and professional"},
    {"vertical": "medical", "label": "Health and medical"},
)

# ------------------------------------------------------------------------------------
# Functions

def seeds_dir():
    return os.path.join(paths_mod.DATA_ROOT, SEEDS_DIRNAME)


def pack_path(vertical):
    return os.path.join(seeds_dir(), f"{vertical}{PACK_SUFFIX}")


def load_pack(vertical):
    """One vertical's pack, or None when it is absent or unreadable.

    Unreadable is treated as absent on purpose: a half-written pack must not make
    a vertical look selectable and then fail at generation time."""
    path = pack_path(vertical)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            pack = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(pack, dict) or not isinstance(pack.get("groups"), list):
        return None
    return pack


def _group_view(group):
    seeds = group.get("seeds") or []
    return {"id": group.get("id"), "label": group.get("label") or group.get("id"),
            "count": len(seeds)}


def list_packs():
    """Every known vertical with its groups and whether it can actually be used.

    Planned-but-unwritten verticals are listed with available=False and no groups,
    so the dashboard can grey them out instead of hiding the roadmap."""
    out = []
    listed = set()
    for planned in PLANNED_VERTICALS:
        vertical = planned["vertical"]
        listed.add(vertical)
        pack = load_pack(vertical)
        if pack is None:
            out.append({"vertical": vertical, "label": planned["label"],
                        "description": "", "available": False,
                        "groups": [], "seed_count": 0})
            continue
        groups = [_group_view(g) for g in pack["groups"]]
        out.append({
            "vertical": vertical,
            "label": pack.get("label") or planned["label"],
            "description": pack.get("description", ""),
            "available": True,
            "groups": groups,
            "seed_count": sum(g["count"] for g in groups),
        })
    # anything dropped into the directory that is not on the planned list
    if os.path.isdir(seeds_dir()):
        for name in sorted(os.listdir(seeds_dir())):
            if not name.endswith(PACK_SUFFIX):
                continue
            vertical = name[: -len(PACK_SUFFIX)]
            if vertical in listed:
                continue
            pack = load_pack(vertical)
            if pack is None:
                continue
            groups = [_group_view(g) for g in pack["groups"]]
            out.append({
                "vertical": vertical,
                "label": pack.get("label") or vertical,
                "description": pack.get("description", ""),
                "available": True,
                "groups": groups,
                "seed_count": sum(g["count"] for g in groups),
            })
    return out


def seeds_for(vertical, group_ids=None):
    """Seeds from the named groups, in pack order. No groups named means all of
    them. Unknown group ids are ignored rather than raising: a stale selection in
    a browser must not break a run."""
    pack = load_pack(vertical)
    if pack is None:
        return []
    wanted = set(group_ids) if group_ids else None
    out = []
    for group in pack["groups"]:
        if wanted is not None and group.get("id") not in wanted:
            continue
        out.extend(group.get("seeds") or [])
    return out
