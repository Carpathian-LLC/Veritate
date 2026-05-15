# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - cross-platform process + gpu telemetry for the dashboard hud and logs panel.
# - cpu / rss come from psutil. gpu is best-effort per platform; adapters with no
#   accessible utilization counter are reported with load_pct=null.
# - design rule: no subprocess call may block an http thread. adapter discovery
#   (slow: system_profiler / Get-CimInstance) is background-warmed; live load
#   queries (nvidia-smi / ioreg) are memoed for 1s.
# veritate_mri/runtime/sys_metrics.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import platform
import re
import subprocess
import sys
import threading
import time

from readers.paths import REPO_ROOT

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

# ------------------------------------------------------------------------------------
# Constants

_PROC = psutil.Process(os.getpid()) if _PSUTIL_OK else None
if _PROC is not None:
    _PROC.cpu_percent(None)

_CPU_COUNT = (psutil.cpu_count(logical=True) if _PSUTIL_OK else 1) or 1

_ADAPTERS = []
_ADAPTERS_TS = 0.0
_ADAPTERS_REFRESHING = False
_ADAPTER_TTL = 60.0

_LIVE_TTL = 1.0
_NV_CACHE = (0.0, [])
_MAC_LOAD_CACHE = (0.0, None)
_CPU_TEMP_CACHE = (0.0, None)
_LHM_TEMP_CACHE = (0.0, None)
_LHM_NAMESPACE = None

_INSTALLED_RAM = None  # bytes; one-shot at startup, doesn't change at runtime
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# ------------------------------------------------------------------------------------
# Functions

def _run(cmd, timeout=2.0):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False,
                             creationflags=_NO_WINDOW)
        return out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _nvidia_query():
    global _NV_CACHE
    now = time.time()
    if (now - _NV_CACHE[0]) < _LIVE_TTL:
        return _NV_CACHE[1]
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    if out:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                temp = None
                if len(parts) >= 5 and parts[4] not in ("", "N/A", "[N/A]"):
                    try:
                        temp = float(parts[4])
                    except ValueError:
                        temp = None
                rows.append({
                    "name": parts[0],
                    "vendor": "NVIDIA",
                    "integrated": False,
                    "load_pct": float(parts[1]),
                    "vram_used": int(parts[2]) * 1024 * 1024,
                    "vram_total": int(parts[3]) * 1024 * 1024,
                    "temp_c": temp,
                })
            except ValueError:
                continue
    _NV_CACHE = (now, rows)
    return rows


def _mac_adapters():
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=4.0)
    if not out:
        return []
    try:
        blob = json.loads(out)
    except json.JSONDecodeError:
        return []
    rows = []
    for d in blob.get("SPDisplaysDataType", []) or []:
        name = d.get("sppci_model") or d.get("_name") or "GPU"
        vendor = d.get("spdisplays_vendor") or ""
        is_integrated = "Apple" in name or "integrated" in (d.get("sppci_bus") or "").lower()
        rows.append({
            "name": name,
            "vendor": vendor.replace("sppci_vendor_", "").upper() or "?",
            "integrated": is_integrated,
            "load_pct": None,
            "vram_used": None,
            "vram_total": None,
            "temp_c": None,
        })
    return rows


_IOREG_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')


def _psutil_cpu_temp():
    """psutil.sensors_temperatures. Works on Linux (coretemp/k10temp/cpu_thermal)
    and some macOS builds. Returns Celsius or None. Empty on Windows."""
    if not _PSUTIL_OK or not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        temps = psutil.sensors_temperatures() or {}
    except (AttributeError, OSError):
        return None
    for key in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz"):
        for e in temps.get(key) or []:
            if e.current and e.current > 0:
                return float(e.current)
    for entries in temps.values():
        for e in entries or []:
            label = (getattr(e, "label", "") or "").lower()
            if "cpu" in label or "package" in label or "tdie" in label or "tctl" in label:
                if e.current and e.current > 0:
                    return float(e.current)
    return None


def _lhm_sensors():
    """Query the LibreHardwareMonitor / OpenHardwareMonitor WMI namespace if
    running. Returns dict {'cpu': float|None, 'gpus': [{'name': str, 'temp_c': float}]}
    or None when no provider is reachable. 1s memoized. No-op off Windows."""
    global _LHM_TEMP_CACHE, _LHM_NAMESPACE
    if not sys.platform.startswith("win"):
        return None
    now = time.time()
    if (now - _LHM_TEMP_CACHE[0]) < _LIVE_TTL:
        return _LHM_TEMP_CACHE[1]
    namespaces = [_LHM_NAMESPACE] if _LHM_NAMESPACE else ["root/LibreHardwareMonitor", "root/OpenHardwareMonitor"]
    result = None
    for ns in namespaces:
        out = _run([
            "powershell", "-NoProfile", "-Command",
            f"Get-CimInstance -Namespace '{ns}' -ClassName Sensor -ErrorAction Stop | "
            "Where-Object { $_.SensorType -eq 'Temperature' } | "
            "Select-Object Name,Parent,Value | ConvertTo-Json -Compress",
        ], timeout=2.0)
        if not out or not out.strip():
            continue
        try:
            blob = json.loads(out)
        except json.JSONDecodeError:
            continue
        if isinstance(blob, dict):
            blob = [blob]
        cpu_val = None
        gpus = []
        for s in blob:
            name = (s.get("Name") or "").strip()
            parent = (s.get("Parent") or "").lower()
            val = s.get("Value")
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            if val <= 0 or val > 150:
                continue
            low = name.lower()
            if "cpu" in parent or "cpu" in low or "package" in low or "tctl" in low or "tdie" in low:
                if cpu_val is None or (("package" in low) and val > 0):
                    cpu_val = val
            elif "gpu" in parent or "gpu" in low:
                gpus.append({"name": name, "temp_c": val})
        if cpu_val is not None or gpus:
            _LHM_NAMESPACE = ns
            result = {"cpu": cpu_val, "gpus": gpus}
            break
    _LHM_TEMP_CACHE = (now, result)
    return result


def _cpu_temp():
    """Best-effort CPU package temperature in Celsius. Tries psutil first
    (Linux/Mac); on Windows falls back to LibreHardwareMonitor / OpenHardware-
    Monitor WMI namespaces if either daemon is running. None when unreadable.
    1s memoized."""
    global _CPU_TEMP_CACHE
    now = time.time()
    if (now - _CPU_TEMP_CACHE[0]) < _LIVE_TTL:
        return _CPU_TEMP_CACHE[1]
    val = _psutil_cpu_temp()
    if val is None:
        lhm = _lhm_sensors()
        if lhm and lhm.get("cpu") is not None:
            val = lhm["cpu"]
    _CPU_TEMP_CACHE = (now, val)
    return val


def _mac_apple_gpu_load():
    global _MAC_LOAD_CACHE
    now = time.time()
    if (now - _MAC_LOAD_CACHE[0]) < _LIVE_TTL:
        return _MAC_LOAD_CACHE[1]
    out = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], timeout=2.0)
    val = None
    if out:
        m = _IOREG_UTIL_RE.search(out)
        if m:
            try:
                val = float(m.group(1))
            except ValueError:
                pass
    _MAC_LOAD_CACHE = (now, val)
    return val


def _win_adapters():
    out = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
    ], timeout=4.0)
    if not out:
        return []
    try:
        blob = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(blob, dict):
        blob = [blob]
    rows = []
    for d in blob:
        name = (d.get("Name") or "GPU").strip()
        ram = d.get("AdapterRAM")
        try:
            ram_int = int(ram) if ram is not None else None
        except (TypeError, ValueError):
            ram_int = None
        vendor = "?"
        low = name.lower()
        if "nvidia" in low or "geforce" in low or "rtx" in low or "gtx" in low:
            vendor = "NVIDIA"
        elif "amd" in low or "radeon" in low:
            vendor = "AMD"
        elif "intel" in low:
            vendor = "Intel"
        integrated = vendor == "Intel" or "integrated" in low or "vega" in low
        rows.append({
            "name": name,
            "vendor": vendor,
            "integrated": integrated,
            "load_pct": None,
            "vram_used": None,
            "vram_total": ram_int,
            "temp_c": None,
        })
    return rows


def _linux_adapters():
    rows = []
    drm = "/sys/class/drm"
    if not os.path.isdir(drm):
        return rows
    for entry in sorted(os.listdir(drm)):
        if not re.match(r"^card\d+$", entry):
            continue
        dev = os.path.join(drm, entry, "device")
        vendor = "?"
        try:
            with open(os.path.join(dev, "vendor"), "r") as f:
                vid = f.read().strip()
            vendor = {"0x1002": "AMD", "0x10de": "NVIDIA", "0x8086": "Intel"}.get(vid, vid)
        except OSError:
            pass
        load = None
        try:
            with open(os.path.join(dev, "gpu_busy_percent"), "r") as f:
                load = float(f.read().strip())
        except OSError:
            pass
        rows.append({
            "name": entry,
            "vendor": vendor,
            "integrated": vendor == "Intel",
            "load_pct": load,
            "vram_used": None,
            "vram_total": None,
            "temp_c": None,
        })
    return rows


def _refresh_adapters():
    global _ADAPTERS, _ADAPTERS_TS, _ADAPTERS_REFRESHING
    plat = sys.platform
    if plat == "darwin":
        rows = _mac_adapters()
    elif plat.startswith("win"):
        rows = _win_adapters()
    else:
        rows = _linux_adapters()
    _ADAPTERS = rows
    _ADAPTERS_TS = time.time()
    _ADAPTERS_REFRESHING = False


def _adapters():
    """Never blocks. Returns the last-known adapter list (possibly stale or empty
    if not yet warmed). Triggers a background refresh when stale."""
    global _ADAPTERS_REFRESHING
    if (time.time() - _ADAPTERS_TS) >= _ADAPTER_TTL and not _ADAPTERS_REFRESHING:
        _ADAPTERS_REFRESHING = True
        threading.Thread(target=_refresh_adapters, name="sys-adapters", daemon=True).start()
    return _ADAPTERS


def _installed_ram_bytes():
    """Physical RAM installed in the machine. Differs from psutil's vm.total on
    Windows by the hardware-reserved region (typically 0.5-1 GB), Task Manager
    shows installed; psutil shows OS-visible. We use installed so HUD numbers
    match what users see. Cached forever; RAM doesn't change at runtime."""
    global _INSTALLED_RAM
    if _INSTALLED_RAM is not None:
        return _INSTALLED_RAM
    if sys.platform.startswith("win"):
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum",
        ], timeout=4.0)
        if out:
            try:
                _INSTALLED_RAM = int(out.strip())
                return _INSTALLED_RAM
            except ValueError:
                pass
    if _PSUTIL_OK:
        _INSTALLED_RAM = psutil.virtual_memory().total
    else:
        _INSTALLED_RAM = 0
    return _INSTALLED_RAM


def warm():
    """Call once at startup so the first /sys_metrics request doesn't see an
    empty adapter list, and so the WMI installed-RAM lookup is done off the
    request thread."""
    global _ADAPTERS_REFRESHING
    if not _ADAPTERS_REFRESHING:
        _ADAPTERS_REFRESHING = True
        threading.Thread(target=_refresh_adapters, name="sys-adapters-warm", daemon=True).start()
    threading.Thread(target=_installed_ram_bytes, name="sys-ram-warm", daemon=True).start()


def _gpus():
    """Merge cached adapter list with fresh utilization. NVIDIA cards get filled
    by nvidia-smi (cross-platform). On macOS, the integrated Apple GPU gets
    ioreg-derived load. Windows non-NVIDIA and Linux non-AMD adapters report
    load_pct=null, vendor SDKs needed for telemetry, not worth the per-poll
    cost."""
    adapters = [dict(a) for a in _adapters()]
    nvidia = _nvidia_query()
    if nvidia:
        for a in adapters:
            key = a["name"].lower()
            for nv in nvidia:
                nk = nv["name"].lower()
                if nk in key or key in nk:
                    a["load_pct"]   = nv["load_pct"]
                    a["vram_used"]  = nv["vram_used"]
                    a["vram_total"] = nv["vram_total"]
                    a["temp_c"]     = nv.get("temp_c")
                    break
    if sys.platform == "darwin":
        load = _mac_apple_gpu_load()
        if load is not None:
            for a in adapters:
                if "apple" in a["name"].lower() or a["integrated"]:
                    a["load_pct"] = load
                    break
    if sys.platform.startswith("win"):
        lhm = _lhm_sensors()
        if lhm and lhm.get("gpus"):
            for a in adapters:
                if a.get("temp_c") is not None:
                    continue
                key = a["name"].lower()
                for g in lhm["gpus"]:
                    nk = g["name"].lower()
                    if any(tok and tok in key for tok in nk.split()):
                        a["temp_c"] = g["temp_c"]
                        break
                else:
                    if lhm["gpus"]:
                        a["temp_c"] = lhm["gpus"][0]["temp_c"]
    return adapters


def snapshot():
    """One-shot telemetry. CPU% is per-core-normalized (sums to ~100% × ncores).
    Returns null fields when psutil isn't installed or a counter can't be read."""
    if not _PSUTIL_OK:
        return {
            "available": False,
            "reason": "psutil not installed (pip install psutil)",
            "cpu_pct": None, "rss_bytes": None, "sys_mem_total": None,
            "cpu_temp_c": None,
            "gpus": [],
        }
    cpu_pct = _PROC.cpu_percent(None)
    rss     = _PROC.memory_info().rss
    vm      = psutil.virtual_memory()
    installed = _installed_ram_bytes() or vm.total
    return {
        "available": True,
        "cpu_pct":          round(cpu_pct, 1),
        "cpu_count":        _CPU_COUNT,
        "cpu_temp_c":       _cpu_temp(),
        "rss_bytes":        int(rss),
        "sys_mem_total":      int(installed),
        "sys_mem_total_os":   int(vm.total),
        "sys_mem_used":       int(installed - vm.available),
        "sys_mem_available":  int(vm.available),
        "gpus": _gpus(),
        "ts": time.time(),
    }


# ------------------------------------------------------------------------------------
# Saved system spec file. Captured on demand from settings; lives under data/.

SPECS_PATH = os.path.join(REPO_ROOT, "data", "system_specs.json")


def detect_specs():
    """Cross-platform machine spec snapshot for the saved specs file.
    Builds on snapshot() and adds static platform info (OS name+version,
    Python version, CPU brand). Forces a synchronous adapter refresh so
    GPU info is fresh when the user clicks 'detect'."""
    _refresh_adapters()
    snap = snapshot()
    return {
        "captured_at": int(time.time()),
        "platform": {
            "system":   platform.system() or "",
            "release":  platform.release() or "",
            "version":  platform.version() or "",
            "machine":  platform.machine() or "",
            "processor": platform.processor() or "",
            "python":   platform.python_version(),
        },
        "cpu": {
            "count_logical": int(_CPU_COUNT),
            "count_physical": int(psutil.cpu_count(logical=False)) if _PSUTIL_OK else None,
        },
        "memory": {
            "total_bytes":     int(snap.get("sys_mem_total") or 0) or None,
            "available_bytes": int(snap.get("sys_mem_available") or 0) or None,
        },
        "gpus": snap.get("gpus") or [],
    }


def save_specs(specs):
    os.makedirs(os.path.dirname(SPECS_PATH), exist_ok=True)
    tmp = SPECS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)
    os.replace(tmp, SPECS_PATH)


def load_specs():
    if not os.path.isfile(SPECS_PATH):
        return None
    try:
        with open(SPECS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def detect_and_save():
    specs = detect_specs()
    save_specs(specs)
    return specs
