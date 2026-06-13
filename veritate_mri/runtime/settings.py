# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - persistent dashboard settings. live file at data/mri_settings.json (gitignored,
#   machine-local). DEFAULTS below is the single source of truth; on first run
#   (or after a build adds new keys) _ensure_settings reconciles the live file
#   against DEFAULTS, writing only missing keys. user values are preserved.
# - DO NOT add a tracked default JSON. DEFAULTS in this file is canonical.
# veritate_mri/runtime/settings.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import random
import threading

from readers.paths import REPO_ROOT

# ------------------------------------------------------------------------------------
# Constants

SETTINGS_PATH = os.path.join(REPO_ROOT, "data", "mri_settings.json")

DEVICE_NAME_MAX_LEN = 15

# First-run device name: adjective-noun-NN, kept short so it fits DEVICE_NAME_MAX_LEN.
DEVICE_NAME_ADJECTIVES = ("brave", "calm", "bright", "swift", "keen", "bold", "lush",
                          "warm", "cool", "vivid", "quiet", "sharp", "witty", "sunny", "misty")
DEVICE_NAME_NOUNS = ("fox", "owl", "elk", "lynx", "hawk", "wolf", "otter", "crane",
                     "bison", "heron", "ibex", "koala", "raven", "tapir", "gecko")

# THE HARDCODED PUBLIC KEY MUST STAY IN THIS FILE! DO NOT REMOVE IT!
# cai_ keys are PUBLIC shared keys, intentionally committed. The public chat
# endpoint + key are NOT persisted to mri_settings.json: get() injects them live
# from the constants below, so rotating the key here reaches every existing
# install on the next load (a persisted copy would shadow the new value forever).
# Only user-entered keys (teacher_api_key, ai_api_key_user) are secret; those
# live in the gitignored data/mri_settings.json or env, never in tracked source.
PUBLIC_AI_ENDPOINT = "https://api.carpathian.ai/ai/v1/chat/completions"
PUBLIC_AI_KEY = "cai_D1swbd9sfAA6BJ8HX3yDby2J5C6ZO8zN91IKP_2iI1g"
PUBLIC_AI_BLURB = ('Adds an "ask AI" button next to selected dashboard panels. '
                   "Each click sends the panel's data to a remote model. "
                   "When disabled, no buttons render and no calls are made.")
PUBLIC_AI_DEFAULTS = {"ai_endpoint": PUBLIC_AI_ENDPOINT, "ai_api_key": PUBLIC_AI_KEY,
                      "ai_assist_blurb": PUBLIC_AI_BLURB}

DEFAULTS = {
    "pytorch_load_mode": "on_demand",
    "pytorch_idle_unload_secs": 600,
    "hud_enabled": False,
    "hud_position": "top",
    "hud_detailed": False,
    "temperature_unit": "C",
    "heartbeat_enabled": True,
    "heartbeat_send_errors": True,
    "consent_modal_seen": False,
    "analytics_advanced_enabled": False,
    "diagnostics_logs_enabled": False,
    "device_preference": "auto",
    "update_channel": "stable",
    "auto_reload_on_update": True,
    "experimental": False,
    "ai_enabled": False,
    "ai_endpoint_user": "",
    "ai_api_key_user": "",
    "last_acknowledged_build": 0,
    "device_name": "",
    "corpus_catalog_url": "",
    "corpus_user_sources": [],
    "teacher_provider": "",
    "teacher_model": "",
    "teacher_base_url": "",
    "teacher_api_key": "",
    "teacher_configs": {},
    "teacher_max_concurrency": 16,
    "teacher_max_tokens": 2048,
    "teacher_temperature": 0.7,
    "mesh_role": "off",
    "mesh_hub_address": "",
    "mesh_auth_token": "",
    "tutorial_enabled": True,
    "tutorial_completed": False,
}

VALID_TEMPERATURE_UNITS = ("C", "F", "K")

KNOWN_TEACHER_PROVIDERS = (
    "carpathian", "openai", "anthropic", "gemini", "xai", "deepseek",
    "mistral", "groq", "openrouter", "ollama", "lm_studio", "llama_cpp",
)

# Per-provider remembered config, keyed by provider id in teacher_configs.
TEACHER_CONFIG_FIELDS = ("api_key", "model", "base_url")

VALID_MESH_ROLES = ("off", "node", "hub", "both")

# Build notices surface a modal in the dashboard for breaking-build changes the
# user needs to act on. Add an entry only when a build introduces something the
# user must read/do; quiet builds add nothing here. Dismissed by setting
# last_acknowledged_build >= the highest key.
BUILD_NOTICES = {
    5: "Build 5 contains substantial engine changes. Please pull the latest source and fully restart the application; older runtime state may not be compatible. Requires a reinstall of requirements (pip install -r requirements.txt).",
    6: "Build 6 reworks the in-app updater. The 'Update' button now overwrites local tracked source to match upstream so diverging branches and dirty trees no longer block updates; user data in data/, models/, and veritate_mri/trainers/ is gitignored and untouched. If your repo is stuck from a previously failed update, delete and re-clone the Veritate repository (your data/, models/, and veritate_mri/trainers/ folders carry over). Click the yellow 'reload python' button once after this update to load the new updater logic. Build 6 also adds a Corpus library in Settings (apt-style installer for training data into veritate_mri/trainers/corpus/) shipping a local catalog of known corpora from Tiny Shakespeare up to RedPajama-V2, with one-click install and downloads above 10 GB gated by a confirm dialog. New Python dependencies are required for HuggingFace-sourced corpora: run 'pip install -r requirements.txt' to pull in datasets and pyarrow before installing any non-direct corpus.",
    7: "Build 7 renames the trainer plugins folder and bundle entry file. If you have a local 'plugins/' folder, rename it to 'trainers/', and rename each bundle's 'plugin.py' to 'trainer.py'. Example — before: plugins/veritate_85m/plugin.py — after: trainers/veritate_85m/trainer.py. Manifests and corpus/ subfolders are unchanged.",
}

_LOCK = threading.Lock()
_CACHE = None

# ------------------------------------------------------------------------------------
# Functions

def _random_device_name():
    name = f"{random.choice(DEVICE_NAME_ADJECTIVES)}-{random.choice(DEVICE_NAME_NOUNS)}-{random.randint(0, 99):02d}"
    return name[:DEVICE_NAME_MAX_LEN]


def _ensure_settings():
    if not os.path.isfile(SETTINGS_PATH):
        fresh = dict(DEFAULTS)
        fresh["device_name"] = _random_device_name()
        _write(fresh)
        return fresh
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            cur = json.load(f)
        if not isinstance(cur, dict):
            cur = {}
    except (OSError, json.JSONDecodeError):
        cur = {}
    missing = {k: v for k, v in DEFAULTS.items() if k not in cur}
    legacy = [k for k in PUBLIC_AI_DEFAULTS if k in cur]
    if missing or legacy:
        for k in legacy:
            cur.pop(k, None)
        cur = {**cur, **missing}
        _write(cur)
    return cur


def _write(data):
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, SETTINGS_PATH)


def get():
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            _CACHE = {**DEFAULTS, **_ensure_settings(), **PUBLIC_AI_DEFAULTS}
        return dict(_CACHE)


def pending_notices():
    last_ack = get().get("last_acknowledged_build", 0)
    return [
        {"build": b, "message": BUILD_NOTICES[b]}
        for b in sorted(BUILD_NOTICES)
        if b > last_ack
    ]


def _validate(patch):
    if "device_name" in patch:
        v = patch["device_name"]
        if v is None:
            patch["device_name"] = ""
        elif not isinstance(v, str):
            raise ValueError("device_name must be a string")
        else:
            stripped = v.strip()
            if len(stripped) > DEVICE_NAME_MAX_LEN:
                raise ValueError(f"device_name must be {DEVICE_NAME_MAX_LEN} characters or fewer")
            patch["device_name"] = stripped
    if "temperature_unit" in patch:
        v = patch["temperature_unit"]
        if not isinstance(v, str):
            raise ValueError("temperature_unit must be a string")
        v = v.strip().upper()
        if v not in VALID_TEMPERATURE_UNITS:
            raise ValueError(f"temperature_unit must be one of {VALID_TEMPERATURE_UNITS}")
        patch["temperature_unit"] = v
    if "corpus_catalog_url" in patch:
        v = patch["corpus_catalog_url"]
        if v is None:
            patch["corpus_catalog_url"] = ""
        elif not isinstance(v, str):
            raise ValueError("corpus_catalog_url must be a string")
        else:
            patch["corpus_catalog_url"] = v.strip()
    if "corpus_user_sources" in patch:
        v = patch["corpus_user_sources"]
        if v is None:
            patch["corpus_user_sources"] = []
        elif not isinstance(v, list):
            raise ValueError("corpus_user_sources must be a list")
        else:
            cleaned = []
            for entry in v:
                if isinstance(entry, dict) and entry.get("stem"):
                    cleaned.append(entry)
            patch["corpus_user_sources"] = cleaned
    for skey in ("teacher_provider", "teacher_model", "teacher_base_url", "teacher_api_key"):
        if skey in patch:
            v = patch[skey]
            if v is None:
                patch[skey] = ""
            elif not isinstance(v, str):
                raise ValueError(f"{skey} must be a string")
            else:
                patch[skey] = v.strip()
    if "teacher_provider" in patch:
        v = patch["teacher_provider"]
        if v and v not in KNOWN_TEACHER_PROVIDERS:
            raise ValueError(f"teacher_provider must be one of {KNOWN_TEACHER_PROVIDERS} or empty")
    if "teacher_configs" in patch:
        v = patch["teacher_configs"]
        if v is None:
            v = {}
        if not isinstance(v, dict):
            raise ValueError("teacher_configs must be a dict")
        cleaned = {}
        for pid, cfg in v.items():
            if pid not in KNOWN_TEACHER_PROVIDERS or not isinstance(cfg, dict):
                continue
            cleaned[pid] = {f: str(cfg.get(f) or "").strip() for f in TEACHER_CONFIG_FIELDS}
        patch["teacher_configs"] = cleaned
    if "teacher_max_concurrency" in patch:
        v = patch["teacher_max_concurrency"]
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("teacher_max_concurrency must be int")
        if v < 1 or v > 128:
            raise ValueError("teacher_max_concurrency must be 1-128")
    if "teacher_max_tokens" in patch:
        v = patch["teacher_max_tokens"]
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("teacher_max_tokens must be int")
        if v < 1 or v > 32768:
            raise ValueError("teacher_max_tokens must be 1-32768")
    if "teacher_temperature" in patch:
        v = patch["teacher_temperature"]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("teacher_temperature must be number")
        if v < 0.0 or v > 2.0:
            raise ValueError("teacher_temperature must be 0.0-2.0")
        patch["teacher_temperature"] = float(v)
    if "mesh_role" in patch:
        v = patch["mesh_role"]
        if not isinstance(v, str):
            raise ValueError("mesh_role must be a string")
        v = v.strip().lower()
        if v not in VALID_MESH_ROLES:
            raise ValueError(f"mesh_role must be one of {VALID_MESH_ROLES}")
        patch["mesh_role"] = v
    if "mesh_hub_address" in patch:
        v = patch["mesh_hub_address"]
        if v is None:
            patch["mesh_hub_address"] = ""
        elif not isinstance(v, str):
            raise ValueError("mesh_hub_address must be a string")
        else:
            patch["mesh_hub_address"] = v.strip().rstrip("/")
    if "mesh_auth_token" in patch:
        v = patch["mesh_auth_token"]
        if v is None:
            patch["mesh_auth_token"] = ""
        elif not isinstance(v, str):
            raise ValueError("mesh_auth_token must be a string")
        else:
            patch["mesh_auth_token"] = v.strip()
    return patch


def update(patch):
    global _CACHE
    if not isinstance(patch, dict):
        return get()
    patch = _validate(dict(patch))
    with _LOCK:
        cur = {**DEFAULTS, **_ensure_settings()}
        for k, v in patch.items():
            if k in DEFAULTS:
                cur[k] = v
        _write(cur)
        _CACHE = {**cur, **PUBLIC_AI_DEFAULTS}
        return dict(_CACHE)
