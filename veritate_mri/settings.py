# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - persistent dashboard settings. live file lives at data/mri_settings.json
#   (gitignored, machine-local). a tracked baseline at data/mri_settings.default.json
#   seeds the live file on first run when missing.
# ------------------------------------------------------------------------------------

import json
import os
import shutil
import threading

from readers.paths import REPO_ROOT

SETTINGS_PATH         = os.path.join(REPO_ROOT, "data", "mri_settings.json")
SETTINGS_DEFAULT_PATH = os.path.join(REPO_ROOT, "data", "mri_settings.default.json")

DEFAULTS = {
    "pytorch_load_mode": "on_demand",
    "pytorch_idle_unload_secs": 600,
    "hud_enabled": False,
    "hud_position": "top",
    "hud_detailed": False,
    "heartbeat_enabled": True,
    "update_channel": "stable",
    "auto_reload_on_update": False,
    "ai_enabled": False,
    "ai_endpoint": "https://api.carpathian.ai/ai/v1/chat/completions",
    "ai_api_key": "cai_fnNpuo53DY8AMGqjuaRoYmpT8274cx6aYCuQR_w1F7w",
    "ai_endpoint_user": "",
    "ai_api_key_user": "",
}

_LOCK = threading.Lock()
_CACHE = None


def _read():
    if not os.path.isfile(SETTINGS_PATH) and os.path.isfile(SETTINGS_DEFAULT_PATH):
        try:
            os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
            shutil.copyfile(SETTINGS_DEFAULT_PATH, SETTINGS_PATH)
        except OSError:
            pass
    if not os.path.isfile(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


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
            _CACHE = {**DEFAULTS, **_read()}
        return dict(_CACHE)


def update(patch):
    global _CACHE
    if not isinstance(patch, dict):
        return get()
    with _LOCK:
        cur = {**DEFAULTS, **_read()}
        for k, v in patch.items():
            if k in DEFAULTS:
                cur[k] = v
        _write(cur)
        _CACHE = cur
        return dict(cur)
