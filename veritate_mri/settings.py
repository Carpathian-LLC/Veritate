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
# veritate_mri/settings.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import threading

from readers.paths import REPO_ROOT

# ------------------------------------------------------------------------------------
# Constants

SETTINGS_PATH = os.path.join(REPO_ROOT, "data", "mri_settings.json")

DEVICE_NAME_MAX_LEN = 15

DEFAULTS = {
    "pytorch_load_mode": "on_demand",
    "pytorch_idle_unload_secs": 600,
    "hud_enabled": False,
    "hud_position": "top",
    "hud_detailed": False,
    "heartbeat_enabled": True,
    "consent_modal_seen": False,
    "analytics_advanced_enabled": False,
    "update_channel": "stable",
    "auto_reload_on_update": False,
    "ai_enabled": False,
    "ai_endpoint": "https://api.carpathian.ai/ai/v1/chat/completions",
    "ai_api_key": "cai_fnNpuo53DY8AMGqjuaRoYmpT8274cx6aYCuQR_w1F7w",
    "ai_endpoint_user": "",
    "ai_api_key_user": "",
    "last_acknowledged_build": 0,
    "device_name": "",
}

# Build notices surface a modal in the dashboard for breaking-build changes the
# user needs to act on. Add an entry only when a build introduces something the
# user must read/do; quiet builds add nothing here. Dismissed by setting
# last_acknowledged_build >= the highest key.
BUILD_NOTICES = {
    5: "Build 5 contains substantial engine changes. Please pull the latest source and fully restart the application; older runtime state may not be compatible. Requires a reinstall of requirements (pip install -r requirements.txt).",
}

_LOCK = threading.Lock()
_CACHE = None

# ------------------------------------------------------------------------------------
# Functions

def _ensure_settings():
    if not os.path.isfile(SETTINGS_PATH):
        _write(dict(DEFAULTS))
        return dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            cur = json.load(f)
        if not isinstance(cur, dict):
            cur = {}
    except (OSError, json.JSONDecodeError):
        cur = {}
    missing = {k: v for k, v in DEFAULTS.items() if k not in cur}
    if missing:
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
            _CACHE = {**DEFAULTS, **_ensure_settings()}
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
        _CACHE = cur
        return dict(cur)
