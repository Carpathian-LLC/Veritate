# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for the C-engine warm pool (backends_routes): selecting a warm model
#   re-points the active slot without spawning, unpinning a non-active model closes
#   it, an active model is unpinned but kept live, a died warm sub is respawned on
#   select, and warm_apply reconciles the pool. subprocess is faked (no real engine).
# tests/mri/test_warm_pool.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from routes import backends_routes as br

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

class _FakeProc:
    def __init__(self, pid=1234, alive=True):
        self.pid = pid
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


class _FakeSub:
    def __init__(self, exe="/fake/exe", model_path="/fake/m.bin", alive=True):
        self.exe = exe
        self.model_path = model_path
        self.proc = _FakeProc(alive=alive)
        self.closed = False
        self.revived = 0

    def _ensure_alive(self):
        self.revived += 1
        self.proc._alive = True

    def close(self):
        self.closed = True


def _cfg(**kw):
    d = {"C_SUBPROCESS": None, "C_EXE": None, "C_MODEL": None, "C_WARM": {}}
    d.update(kw)
    return d


def test_warm_select_reuses_without_spawn():
    """A warm model is selected by re-pointing the active slot, never spawning."""
    sub = _FakeSub(model_path="/fake/m1.bin")
    cfg = _cfg(C_WARM={"m1": sub})
    assert br.warm_select(cfg, "m1") is True
    assert cfg["C_SUBPROCESS"] is sub and cfg["C_MODEL"] == "/fake/m1.bin"
    assert sub.closed is False


def test_warm_select_absent_returns_false():
    """Selecting a model not in the pool returns False and leaves the slot alone."""
    cfg = _cfg()
    assert br.warm_select(cfg, "nope") is False
    assert cfg["C_SUBPROCESS"] is None


def test_warm_drop_closes_inactive():
    """Dropping a pinned model that is not active closes and removes its subprocess."""
    sub = _FakeSub()
    cfg = _cfg(C_WARM={"m1": sub})
    br.warm_drop(cfg, "m1")
    assert sub.closed is True and "m1" not in cfg["C_WARM"]


def test_warm_drop_keeps_active_alive():
    """Dropping the currently active pinned model unpins it but leaves it running."""
    sub = _FakeSub()
    cfg = _cfg(C_SUBPROCESS=sub, C_WARM={"m1": sub})
    br.warm_drop(cfg, "m1")
    assert sub.closed is False and "m1" not in cfg["C_WARM"]
    assert cfg["C_SUBPROCESS"] is sub


def test_dead_warm_respawns_on_select():
    """A warm subprocess whose process exited is revived when it is selected."""
    sub = _FakeSub(alive=False)
    cfg = _cfg(C_WARM={"m1": sub})
    assert br.warm_select(cfg, "m1") is True
    assert sub.revived == 1 and sub.proc.poll() is None


def test_warm_is_pinned_identity():
    """warm_is_pinned reports membership by object identity in the pool."""
    pinned = _FakeSub()
    other = _FakeSub()
    cfg = _cfg(C_WARM={"m1": pinned})
    assert br.warm_is_pinned(cfg, pinned) is True
    assert br.warm_is_pinned(cfg, other) is False


def test_warm_spawn_adopts_active_subprocess(monkeypatch):
    """Pinning the model already active in the single slot adopts it, not a duplicate."""
    spawned = []

    def _fake_ctor(exe, model_bin):
        s = _FakeSub(exe=exe, model_path=model_bin)
        spawned.append(s)
        return s

    monkeypatch.setattr(br, "CTracedSubprocess", _fake_ctor)
    monkeypatch.setattr(br.binr, "exists", lambda name: True)
    monkeypatch.setattr(br.paths, "engine_binary_path", lambda: "/fake/exe")
    monkeypatch.setattr(br.paths, "bin_path", lambda name: os.path.abspath(f"/fake/{name}.bin"))
    monkeypatch.setattr(br.os.path, "isfile", lambda p: True)

    active = _FakeSub(model_path=os.path.abspath("/fake/m1.bin"))
    cfg = _cfg(C_SUBPROCESS=active)
    assert br.warm_spawn(cfg, "m1") is True
    assert cfg["C_WARM"]["m1"] is active and spawned == []


def test_warm_apply_spawns_new_and_drops_removed(monkeypatch):
    """warm_apply spawns models newly added to the list and closes models removed from it."""
    spawned = {}

    def _fake_ctor(exe, model_bin):
        s = _FakeSub(exe=exe, model_path=model_bin)
        spawned[model_bin] = s
        return s

    monkeypatch.setattr(br, "CTracedSubprocess", _fake_ctor)
    monkeypatch.setattr(br.binr, "exists", lambda name: True)
    monkeypatch.setattr(br.paths, "engine_binary_path", lambda: "/fake/exe")
    monkeypatch.setattr(br.paths, "bin_path", lambda name: f"/fake/{name}.bin")
    monkeypatch.setattr(br.os.path, "isfile", lambda p: True)

    cfg = _cfg()
    br.warm_apply(cfg, ["m1"])
    assert "m1" in cfg["C_WARM"]
    m1 = cfg["C_WARM"]["m1"]
    br.warm_apply(cfg, ["m2"])
    assert "m2" in cfg["C_WARM"] and "m1" not in cfg["C_WARM"]
    assert m1.closed is True
