# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for the system-dependency registry and the loopback route guard.
#   shutil.which and paths.current_os are stubbed so nothing touches the real
#   machine's PATH or OS.
# tests/mri/test_system_deps.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from readers import paths
from routes._common import is_loopback
from training import system_deps

# ------------------------------------------------------------------------------------
# Constants

PRESENT = "/usr/bin"

# ------------------------------------------------------------------------------------
# Functions

def _stub(monkeypatch, present_names, os_name):
    """Route shutil.which through a name allow-set and pin the reported OS."""
    monkeypatch.setattr(system_deps.shutil, "which",
                        lambda name: f"{PRESENT}/{name}" if name in present_names else None)
    monkeypatch.setattr(system_deps.paths, "current_os", lambda: os_name)


def _dep(st, key):
    return next(d for d in st["deps"] if d["key"] == key)


def test_loopback_accepts_localhost_and_rejects_lan():
    """is_loopback allows 127.0.0.1 and ::1, rejects a LAN address and garbage."""
    assert is_loopback("127.0.0.1") is True
    assert is_loopback("::1") is True
    assert is_loopback("192.168.0.43") is False
    assert is_loopback("not-an-ip") is False


def test_all_present_reports_ready(monkeypatch):
    """Every dep on PATH yields all_present with no install offered."""
    _stub(monkeypatch, {"clang", "git", "apt-get"}, paths.OS_LINUX)
    st = system_deps.status()
    assert st["all_present"] is True
    assert st["missing_required"] is False
    assert st["can_auto_install"] is False


def test_missing_clang_is_required_and_auto_installable(monkeypatch):
    """Missing clang on linux+apt flags missing_required and an apt install command."""
    _stub(monkeypatch, {"git", "apt-get"}, paths.OS_LINUX)
    st = system_deps.status()
    clang = _dep(st, "clang")
    assert clang["present"] is False and clang["required"] is True
    assert clang["can_auto_install"] is True
    assert clang["install_command"] == "sudo apt-get install -y clang"
    assert st["missing_required"] is True


def test_missing_git_is_optional(monkeypatch):
    """Missing git is not required, so it does not raise missing_required."""
    _stub(monkeypatch, {"clang", "apt-get"}, paths.OS_LINUX)
    st = system_deps.status()
    assert _dep(st, "git")["required"] is False
    assert st["missing_required"] is False


def test_no_package_manager_blocks_auto_install(monkeypatch):
    """Missing deps with no package manager cannot auto-install and offer no command."""
    _stub(monkeypatch, set(), paths.OS_LINUX)
    st = system_deps.status()
    assert st["can_auto_install"] is False
    assert _dep(st, "clang")["install_command"] is None


def test_macos_offers_xcode_command_but_no_auto(monkeypatch):
    """macOS reports the xcode-select command but never auto-installs."""
    _stub(monkeypatch, set(), paths.OS_MACOS)
    st = system_deps.status()
    assert st["can_auto_install"] is False
    assert _dep(st, "clang")["install_command"] == system_deps.MACOS_INSTALL_COMMAND


def test_install_refuses_when_no_package_manager(monkeypatch):
    """install() returns an error (never shells out) when no package manager exists."""
    _stub(monkeypatch, set(), paths.OS_LINUX)
    res = system_deps.install()
    assert res["ok"] is False
    assert "package manager" in res["error"]


def test_install_ignores_unknown_keys(monkeypatch):
    """An unknown dep key installs nothing even while a real dep is missing."""
    _stub(monkeypatch, {"apt-get"}, paths.OS_LINUX)  # clang + git both missing
    res = system_deps.install(["definitely-not-a-real-dep"])
    assert res.get("already_present") is True  # unknown key selected nothing, no shell-out


def test_install_builds_noninteractive_sudo_command(monkeypatch):
    """install() runs exactly `sudo -n apt-get install -y clang git` for missing deps."""
    _stub(monkeypatch, {"apt-get"}, paths.OS_LINUX)  # clang + git both missing
    monkeypatch.setattr(system_deps.os, "geteuid", lambda: 1000)  # not root
    captured = {}
    def _fake_run(argv):
        captured["argv"] = argv
        return True, 0, None
    monkeypatch.setattr(system_deps, "_run", _fake_run)
    system_deps.install()
    assert captured["argv"] == ["sudo", "-n", "apt-get", "install", "-y", "clang", "git"]
