# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - one git subprocess runner shared by app_sync, plugins_sync, models_sync.
# - hardens the env so git can never prompt for credentials, restricts the
#   protocol allow-list to https, suppresses the windows console flash.
# - returns (returncode, stdout, stderr). stdout/stderr are stripped strings.
# - timeout, file-not-found, and protocol failures map to deterministic
#   sentinel return codes so callers can branch without parsing stderr.
# veritate_mri/sync/git_runner.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import subprocess

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_TIMEOUT_SECS = 60

RC_NOT_FOUND = 127
RC_TIMEOUT   = 124

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

_GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ALLOW_PROTOCOL":  "https",
    "GIT_ASKPASS":         "echo",
}

# ------------------------------------------------------------------------------------
# Functions

def run_git(args, cwd, timeout=DEFAULT_TIMEOUT_SECS):
    env = {**os.environ, **_GIT_ENV}
    try:
        r = subprocess.run(
            ["git"] + list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            creationflags=_NO_WINDOW,
        )
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except FileNotFoundError:
        return RC_NOT_FOUND, "", "git executable not found on PATH"
    except subprocess.TimeoutExpired:
        return RC_TIMEOUT, "", f"git {' '.join(args)} timed out after {timeout}s"
