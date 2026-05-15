# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - shared helpers for route modules. path sanitization, folder-open dispatch.
# veritate_mri/routes/_common.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import platform
import subprocess

# ------------------------------------------------------------------------------------
# Constants

THREADS_AUTO_MAX = 16

# ------------------------------------------------------------------------------------
# Functions

def safe_name(name):
    if not name: return False
    if ".." in name.split("/") or ".." in name.split("\\"): return False
    if name.startswith("/") or name.startswith("\\"): return False
    if ":" in name: return False
    return True


def auto_thread_count():
    n = os.cpu_count() or 1
    physical = n // 2 if n >= 2 else 1
    return max(1, min(THREADS_AUTO_MAX, physical))


def user_error(e, prefix=None):
    """Plain-language exception message for JSON returns. Drops the class name;
    keep that in `logmod.error(...)` for devs. `prefix` (e.g. 'rag retrieve')
    becomes 'rag retrieve failed: {e}'."""
    body = str(e).strip() or "unknown error"
    return f"{prefix} failed: {body}" if prefix else body


def open_folder(folder):
    os.makedirs(folder, exist_ok=True)
    sysname = platform.system()
    try:
        if sysname == "Windows":
            subprocess.Popen(["explorer.exe", folder])
        elif sysname == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except OSError as e:
        return ({"ok": False, "error": str(e), "path": folder}, 500)
    return {"ok": True, "path": folder}
