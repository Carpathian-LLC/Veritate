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

from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

THREADS_AUTO_MAX = 16

# ------------------------------------------------------------------------------------
# Functions

def safe_route(source, fn, *a, **kw):
    """Run a route handler, log any exception to the dashboard's in-memory log
    ring, return a JSON-friendly error body + 500 status so the frontend gets
    parseable bytes instead of Flask's HTML error page. WebKit reports
    HTML-where-JSON-expected as 'string did not match the expected pattern',
    which is unhelpful — using this wrapper keeps the error visible and
    diagnosable."""
    try:
        return fn(*a, **kw)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logmod.error(source, msg)
        return ({"ok": False, "error": msg}, 500)

def safe_name(name):
    if not name: return False
    if ".." in name.split("/") or ".." in name.split("\\"): return False
    if name.startswith("/") or name.startswith("\\"): return False
    if ":" in name: return False
    return True


def auto_thread_count():
    """Physical-core count capped at THREADS_AUTO_MAX. Uses psutil when available
    for an accurate physical-vs-logical split (Apple Silicon has no SMT so the
    old n//2 heuristic undercounted by 2x). Falls back to os.cpu_count // 2 only
    when psutil is missing and the system reports an even logical count
    (typical hyperthreaded Intel/AMD)."""
    physical = None
    try:
        import psutil as _ps
        physical = _ps.cpu_count(logical=False)
    except ImportError:
        physical = None
    if not physical:
        n = os.cpu_count() or 1
        physical = max(1, n // 2 if n >= 2 else 1)
    return max(1, min(THREADS_AUTO_MAX, int(physical)))


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
