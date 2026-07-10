# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - registry of Veritate's system-level dependencies (OS packages) plus a secure
#   installer. Detection is `shutil.which`; installation runs the distro package
#   manager. Extend by adding a DEPS entry, not by touching call sites.
# - The install path is deliberately narrow: packages come from the registry (never
#   from request input), the command runs NON-INTERACTIVELY (sudo -n), so the
#   dashboard can only escalate when the OS already grants passwordless sudo. The
#   server holds no credentials and never prompts. The route layer additionally gates
#   install to loopback callers. Without passwordless sudo the install fails fast and
#   the UI falls back to the install_command for the user to paste.
# - Python itself and pip packages are NOT here: they are the pre-server bootstrap
#   (the dashboard cannot run without them), so there is nothing to install from the
#   running UI. This registry is OS packages only.
# veritate_mri/training/system_deps.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import shutil
import subprocess

from readers import paths
from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

# One entry per OS package. `binary` is the PATH probe; `packages` maps a package
# manager to the package name; `required` gates the "engine cannot build" banner.
DEPS = (
    {"key": "clang", "label": "C compiler (clang)", "binary": "clang", "required": True,
     "packages": {"apt-get": "clang", "dnf": "clang", "pacman": "clang"},
     "purpose": "builds the inference engine"},
    {"key": "git", "label": "git", "binary": "git", "required": False,
     "packages": {"apt-get": "git", "dnf": "git", "pacman": "git"},
     "purpose": "git-based model and trainer sync (the built-in updater works without it)"},
)

# Probe order; first present wins. INSTALL_ARGS are the verb + assume-yes flags.
PACKAGE_MANAGERS = ("apt-get", "dnf", "pacman")
INSTALL_ARGS = {
    "apt-get": ("install", "-y"),
    "dnf":     ("install", "-y"),
    "pacman":  ("-S", "--noconfirm"),
}
MACOS_MANAGER         = "xcode-select"
MACOS_INSTALL_COMMAND = "xcode-select --install"
INSTALL_TIMEOUT_S     = 600
LOG_SOURCE            = "system-deps"

# ------------------------------------------------------------------------------------
# Functions

def _package_manager():
    for pm in PACKAGE_MANAGERS:
        if shutil.which(pm):
            return pm
    return None


def _is_root():
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _install_command_str(pm, packages):
    return "sudo " + " ".join((pm,) + INSTALL_ARGS[pm] + tuple(packages))


def _dep_view(dep, os_name, pm):
    present = shutil.which(dep["binary"]) is not None
    if os_name == paths.OS_MACOS:
        command, can_auto = MACOS_INSTALL_COMMAND, False
    elif pm:
        command, can_auto = _install_command_str(pm, [dep["packages"][pm]]), not present
    else:
        command, can_auto = None, False
    return {"key": dep["key"], "label": dep["label"], "present": present,
            "required": dep["required"], "purpose": dep["purpose"],
            "install_command": None if present else command,
            "can_auto_install": can_auto}


def status():
    """Readiness of every registry dependency. `missing_required` drives the
    engine-build banner; `can_auto_install` is true when at least one missing dep
    can be installed in-place (a package manager is present and the OS is not macOS,
    whose xcode-select installer is an interactive GUI dialog)."""
    os_name = paths.current_os()
    pm = None if os_name == paths.OS_MACOS else _package_manager()
    views = [_dep_view(d, os_name, pm) for d in DEPS]
    missing = [v for v in views if not v["present"]]
    return {
        "os": os_name,
        "arch": paths.current_arch(),
        "package_manager": MACOS_MANAGER if os_name == paths.OS_MACOS else pm,
        "deps": views,
        "all_present": not missing,
        "missing_required": any(v["required"] for v in missing),
        "can_auto_install": any(v["can_auto_install"] for v in views),
    }


def _selected_missing(keys):
    by_key = {d["key"]: d for d in DEPS}
    chosen = [by_key[k] for k in (keys or list(by_key)) if k in by_key]
    return [d for d in chosen if shutil.which(d["binary"]) is None]


def install(keys=None):
    """Install missing deps (all, or the subset named in `keys`) via the package
    manager. `keys` are validated against the registry; packages are never taken
    from the caller. Streams to the log ring, returns {ok, exit_code, installed,
    error, status}. Route MUST enforce loopback. Non-interactive: fails fast when
    passwordless sudo is absent instead of hanging on a prompt."""
    os_name = paths.current_os()
    if os_name == paths.OS_MACOS:
        return {"ok": False, "error": f"macOS installs via `{MACOS_INSTALL_COMMAND}` "
                "(interactive); run it in a terminal", "status": status()}
    pm = _package_manager()
    if not pm:
        return {"ok": False, "error": "no supported package manager (apt-get/dnf/pacman)",
                "status": status()}
    missing = _selected_missing(keys)
    if not missing:
        return {"ok": True, "already_present": True, "status": status()}
    packages = [d["packages"][pm] for d in missing]
    argv = ([] if _is_root() else ["sudo", "-n"]) + [pm] + list(INSTALL_ARGS[pm]) + packages
    logmod.info(LOG_SOURCE, f"installing {' '.join(packages)} via {pm} (non-interactive)")
    ok, code, run_err = _run(argv)
    after = status()
    installed_all = all(shutil.which(d["binary"]) for d in missing)
    if ok and installed_all:
        logmod.ok(LOG_SOURCE, f"installed {' '.join(packages)}")
        return {"ok": True, "exit_code": code, "installed": packages, "status": after}
    error = run_err or "install failed: passwordless sudo may be required (run the shown command in a terminal)"
    logmod.error(LOG_SOURCE, f"install failed: exit={code}")
    return {"ok": False, "exit_code": code, "error": error, "status": after}


def _run(argv):
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logmod.info(LOG_SOURCE, line)
        proc.wait(timeout=INSTALL_TIMEOUT_S)
        return proc.returncode == 0, proc.returncode, None
    except subprocess.TimeoutExpired:
        proc.kill()
        return False, None, "install timed out"
    except FileNotFoundError as exc:
        return False, None, str(exc)
