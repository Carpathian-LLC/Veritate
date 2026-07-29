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
import secrets
import threading
import time

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
    # Model names kept permanently loaded as resident C-engine subprocesses so
    # switching to them serves warm (no spawn/reload). Spawned at startup and on
    # change; see backends_routes warm pool.
    "warm_models": [],
    # Speculative prefetch. While a client is still typing, generate up to
    # speculative_bytes of the reply for the draft prompt; the real request flushes
    # the buffer instead of paying prefill. speculative_chunk_bytes is the engine
    # turn size, which bounds how long a real request waits behind a speculative
    # turn. Off by default: it spends compute on drafts that may be discarded.
    # See routes/backends_routes POST /prefetch.
    "speculative_enabled": False,
    "speculative_bytes": 1024,
    "speculative_chunk_bytes": 32,
    # How long the composer waits on a still prompt before treating it as a finished
    # question, in ms. 0 tracks the typist's live median keystroke gap; a positive
    # value is an explicit pause threshold in ms; 0 tracks the typist's live median.
    "speculative_pause_ms": 0,
    # Read-ahead: read the prompt being typed into the engine so the request that
    # carries it skips the prefill. On by default: it predicts nothing, and the work is
    # what the real request has to do anyway, so a miss costs nothing.
    # See routes/backends_routes POST /prefill.
    "read_ahead_enabled": True,
    # Whether a programmatic caller (one presenting a bearer token) may have the box
    # work ahead of its request. Separate from the dashboard's own switches: this box
    # runs a single stateful engine, so a client working ahead is a client holding it.
    # Reading ahead is work the request must do anyway; generating ahead can be
    # discarded entirely, so it is off.
    "api_read_ahead_enabled": True,
    "api_generate_ahead_enabled": False,
    "hud_enabled": False,
    "hud_position": "top",
    "hud_detailed": False,
    # Advanced: stream reduced (compact) MRI telemetry frames. The engine summarizes
    # each byte's telemetry itself (~33x smaller per-byte frame) instead of shipping
    # full-resolution arrays for the browser to reduce. The dashboard renders the
    # same; only the wire payload shrinks. Off by default so nothing changes on deploy
    # until the user opts in from Settings -> Advanced.
    "mri_compact_frames": False,
    "temperature_unit": "C",
    "heartbeat_enabled": True,
    "heartbeat_send_errors": True,
    "consent_modal_seen": False,
    "analytics_advanced_enabled": True,
    "share_current_training": True,
    "diagnostics_logs_enabled": False,
    "device_preference": "auto",
    "update_channel": "stable",
    "auto_reload_on_update": True,
    "extensions": False,
    "ai_enabled": False,
    "ai_endpoint_user": "",
    "ai_api_key_user": "",
    "last_acknowledged_build": 0,
    "device_name": "",
    "corpus_catalog_url": "",
    "corpus_user_sources": [],
    # Corpus mix planner (veritate_mri/training/mix_planner.py). max_epochs caps how
    # many times any one source may be redrawn into a mix; profiles_path empty means
    # the shipped veritate_mri/data/corpus_mix_profiles.json.
    "corpus_mix_max_epochs": 4,
    "corpus_mix_default_profile": "pretrain",
    "corpus_mix_profiles_path": "",
    # Corpus compose (mix_planner.compose). chunk_bytes is the interleave grain: a
    # plan is cut into chunks this size and shuffled, so one unified corpus is not
    # source-ordered blocks. val_ratio is held out from a disjoint tail region of
    # every source, so train and val never share bytes.
    "corpus_compose_chunk_bytes": 1048576,
    "corpus_compose_val_ratio": 0.005,
    "corpus_compose_seed": 20260727,
    # Native trainer size -> shape table. Empty means the shipped
    # veritate_mri/data/trainer_sizes.json; point it at your own file to add or
    # retune shapes without touching code.
    "trainer_sizes_path": "",
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
    "tutorial_completed": True,
    "api_key": "",
    "api_key_request_count": 0,
    "api_key_last_used_at": 0,
}

VALID_TEMPERATURE_UNITS = ("C", "F", "K")

# Speculative prefetch bounds. The byte cap matches the engine's own max_new cap so
# a draft can never be told to speculate more than a real request could generate.
SPECULATIVE_BYTES_MAX      = 4096
SPECULATIVE_CHUNK_BYTES_MAX = 256
# Calibrated pause bounds. Below the floor a draft fires between words; above the
# ceiling nothing is ever ready in time to be worth the compute.
SPECULATIVE_PAUSE_MS_MIN   = 200
SPECULATIVE_PAUSE_MS_MAX   = 5000

# Optional API-key gate for the programmatic API surface (/v1/*, /generate,
# /agent/stream). Off by default (empty api_key). Minted keys carry this prefix.
API_KEY_PREFIX = "vrt_"
API_KEY_RANDOM_BYTES = 32

KNOWN_TEACHER_PROVIDERS = (
    "carpathian", "openai", "anthropic", "gemini", "xai", "deepseek",
    "mistral", "groq", "openrouter", "ollama", "lm_studio", "llama_cpp",
)

# Per-provider remembered config, keyed by provider id in teacher_configs.
TEACHER_CONFIG_FIELDS = ("api_key", "model", "base_url")

VALID_MESH_ROLES = ("off", "node", "hub", "both")

# Build notices surface a modal in the dashboard for breaking-build changes the
# user must act on. Keyed by versions.json::build. Add an entry only when a build
# needs the user to delete, rebuild, or rerun something; quiet builds add nothing.
# Dismissed by setting last_acknowledged_build >= the highest key.
# Empty at the 1.0.0 launch: there is no pre-1.0 state to migrate from.
BUILD_NOTICES = {}

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
        with open(SETTINGS_PATH, encoding="utf-8") as f:
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
    if "corpus_mix_max_epochs" in patch:
        v = patch["corpus_mix_max_epochs"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("corpus_mix_max_epochs must be a positive number")
    for ikey in ("corpus_compose_chunk_bytes", "corpus_compose_seed"):
        if ikey in patch:
            v = patch[ikey]
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{ikey} must be a positive integer")
    if "corpus_compose_val_ratio" in patch:
        v = patch["corpus_compose_val_ratio"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1:
            raise ValueError("corpus_compose_val_ratio must be in [0, 1)")
    for skey in ("corpus_mix_default_profile", "corpus_mix_profiles_path",
                 "trainer_sizes_path"):
        if skey in patch:
            v = patch[skey]
            if v is None:
                patch[skey] = ""
            elif not isinstance(v, str):
                raise ValueError(f"{skey} must be a string")
            else:
                patch[skey] = v.strip()
    if "speculative_enabled" in patch:
        patch["speculative_enabled"] = bool(patch["speculative_enabled"])
    if "speculative_bytes" in patch:
        v = patch["speculative_bytes"]
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= SPECULATIVE_BYTES_MAX:
            raise ValueError(f"speculative_bytes must be an integer 1..{SPECULATIVE_BYTES_MAX}")
    if "speculative_chunk_bytes" in patch:
        v = patch["speculative_chunk_bytes"]
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= SPECULATIVE_CHUNK_BYTES_MAX:
            raise ValueError(f"speculative_chunk_bytes must be an integer 1..{SPECULATIVE_CHUNK_BYTES_MAX}")
    if "speculative_pause_ms" in patch:
        v = patch["speculative_pause_ms"]
        ok = (not isinstance(v, bool) and isinstance(v, int)
              and (v == 0 or SPECULATIVE_PAUSE_MS_MIN <= v <= SPECULATIVE_PAUSE_MS_MAX))
        if not ok:
            raise ValueError(f"speculative_pause_ms must be 0 (auto) or an integer "
                             f"{SPECULATIVE_PAUSE_MS_MIN}..{SPECULATIVE_PAUSE_MS_MAX}")
    for bkey in ("read_ahead_enabled", "api_read_ahead_enabled", "api_generate_ahead_enabled"):
        if bkey in patch:
            patch[bkey] = bool(patch[bkey])
    if "warm_models" in patch:
        v = patch["warm_models"]
        if v is None:
            patch["warm_models"] = []
        elif not isinstance(v, list):
            raise ValueError("warm_models must be a list")
        else:
            patch["warm_models"] = [s.strip() for s in v if isinstance(s, str) and s.strip()]
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


def generate_api_key():
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(API_KEY_RANDOM_BYTES)}"


def rotate_api_key():
    return update({"api_key": generate_api_key(), "api_key_request_count": 0,
                   "api_key_last_used_at": 0})


def clear_api_key():
    return update({"api_key": "", "api_key_request_count": 0,
                   "api_key_last_used_at": 0})


def record_api_key_use():
    global _CACHE
    with _LOCK:
        cur = {**DEFAULTS, **_ensure_settings()}
        cur["api_key_request_count"] = int(cur.get("api_key_request_count") or 0) + 1
        cur["api_key_last_used_at"] = time.time()
        _write(cur)
        _CACHE = {**cur, **PUBLIC_AI_DEFAULTS}
