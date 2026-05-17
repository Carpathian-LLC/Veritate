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


def _parse_size_str(s):
    """Parse '3 GB' / '1536 MB' / '512 KB' to bytes; None if unparseable."""
    if not s or not isinstance(s, str):
        return None
    parts = s.strip().split()
    if len(parts) < 2:
        return None
    try:
        n = float(parts[0])
    except ValueError:
        return None
    u = parts[1].upper()
    if u.startswith("GB"): return int(n * (1024 ** 3))
    if u.startswith("MB"): return int(n * (1024 ** 2))
    if u.startswith("KB"): return int(n * 1024)
    return None


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
        vram_str = d.get("spdisplays_vram") or d.get("spdisplays_vram_shared")
        rows.append({
            "name": name,
            "vendor": vendor.replace("sppci_vendor_", "").upper() or "?",
            "integrated": is_integrated,
            "load_pct": None,
            "vram_used": None,
            "vram_total": _parse_size_str(vram_str),
            "temp_c": None,
            "metal_family": d.get("spdisplays_metalfamily") or d.get("spdisplays_mtlgpufamilysupport"),
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


# macmon has a ~8s startup cost when stdout is piped, so per-poll subprocess
# calls would freeze the HUD. We run it once as a persistent streaming process
# and read JSON samples from a background thread.
_MAC_MACMON_PROC   = None
_MAC_MACMON_LATEST = None  # {"cpu": float|None, "gpu": float|None, "ts": float}
_MAC_MACMON_LOCK   = threading.Lock()
_MAC_MACMON_STALE_S = 5.0


def _mac_macmon_reader(proc):
    global _MAC_MACMON_LATEST
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                blob = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            temp = blob.get("temp") or {}
            cpu = temp.get("cpu_temp_avg")
            gpu = temp.get("gpu_temp_avg")
            with _MAC_MACMON_LOCK:
                _MAC_MACMON_LATEST = {
                    "cpu": float(cpu) if isinstance(cpu, (int, float)) and 0 < cpu < 150 else None,
                    "gpu": float(gpu) if isinstance(gpu, (int, float)) and 0 < gpu < 150 else None,
                    "ts":  time.time(),
                }
    except (OSError, ValueError):
        pass


def _mac_macmon_start():
    global _MAC_MACMON_PROC
    if sys.platform != "darwin":
        return
    if _MAC_MACMON_PROC is not None and _MAC_MACMON_PROC.poll() is None:
        return
    # Reap any orphan macmons from a prior dashboard instance whose os._exit
    # bypassed atexit cleanup. Only kills processes owned by the current user.
    try:
        subprocess.run(["pkill", "-U", str(os.getuid()), "-x", "macmon"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        proc = subprocess.Popen(
            ["macmon", "pipe", "-i", "1000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, text=True, bufsize=1,
        )
    except (FileNotFoundError, OSError):
        return
    _MAC_MACMON_PROC = proc
    threading.Thread(target=_mac_macmon_reader, args=(proc,),
                     name="macmon-reader", daemon=True).start()


def stop():
    """Terminate any background subprocesses (currently: macmon).
    Lifecycle calls this before os._exit so we don't orphan helpers."""
    global _MAC_MACMON_PROC
    p = _MAC_MACMON_PROC
    _MAC_MACMON_PROC = None
    if p is None:
        return
    try:
        if p.poll() is None:
            p.terminate()
    except OSError:
        pass


def _mac_macmon_sample():
    if sys.platform != "darwin":
        return None
    if _MAC_MACMON_PROC is None:
        _mac_macmon_start()
        return None
    with _MAC_MACMON_LOCK:
        if _MAC_MACMON_LATEST is None:
            return None
        if time.time() - _MAC_MACMON_LATEST["ts"] > _MAC_MACMON_STALE_S:
            return None
        return {"cpu": _MAC_MACMON_LATEST["cpu"], "gpu": _MAC_MACMON_LATEST["gpu"]}


def _mac_cpu_temp():
    s = _mac_macmon_sample()
    if s and s.get("cpu") is not None:
        return s["cpu"]
    out = _run(["osx-cpu-temp", "-c"], timeout=1.5)
    if out:
        m = re.search(r"(\d+(?:\.\d+)?)", out)
        if m:
            try:
                v = float(m.group(1))
                if 0 < v < 150:
                    return v
            except ValueError:
                pass
    return None


def _mac_gpu_temp():
    s = _mac_macmon_sample()
    return s.get("gpu") if s else None


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
    if val is None and sys.platform == "darwin":
        val = _mac_cpu_temp()
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
    _mac_macmon_start()
    import atexit as _atexit
    _atexit.register(stop)


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
        gpu_t = _mac_gpu_temp()
        if gpu_t is not None:
            for a in adapters:
                if a.get("temp_c") is None and ("apple" in a["name"].lower() or a["integrated"]):
                    a["temp_c"] = gpu_t
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
    # cpu_pct is system-wide across all cores (0-100). This is what users
    # expect the HUD to spike on — trainers run in subprocesses, so the
    # dashboard's own process_cpu_pct stays near zero even when the box is
    # pegged. psutil.cpu_percent uses delta-since-last-call when interval=None;
    # the HUD polls regularly so the first reading after launch may be 0.
    sys_cpu_pct     = psutil.cpu_percent(interval=None)
    process_cpu_pct = _PROC.cpu_percent(None)
    rss     = _PROC.memory_info().rss
    vm      = psutil.virtual_memory()
    installed = _installed_ram_bytes() or vm.total
    return {
        "available": True,
        "cpu_pct":          round(sys_cpu_pct, 1),
        "process_cpu_pct":  round(process_cpu_pct, 1),
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
# Hardware capability probes. Cross-platform. Used by detect_specs() and the
# settings-tab "what we collect" panel.

# NEON/ASIMD are mandatory on ARMv8 so Apple doesn't expose a separate sysctl
# for them on Apple Silicon. We treat them as always-present on arm64 macOS.
_MAC_ARM_IMPLIED_FEATURES = ("NEON", "ASIMD")
_MAC_ARM_FEATURE_PROBES = (
    # (sysctl key, feature name we report). Apple Silicon ships these as
    # hw.optional.arm.FEAT_*, each returning "1" when supported.
    ("hw.optional.arm.FEAT_DotProd",       "ASIMDDP"),
    ("hw.optional.arm.FEAT_FP16",          "FP16"),
    ("hw.optional.arm.FEAT_BF16",          "BF16"),
    ("hw.optional.arm.FEAT_I8MM",          "I8MM"),
    ("hw.optional.arm.FEAT_FHM",           "ASIMDFHM"),
    ("hw.optional.arm.FEAT_SHA512",        "SHA512"),
    ("hw.optional.arm.FEAT_SHA3",          "SHA3"),
    ("hw.optional.arm.FEAT_AES",           "AES"),
)


def _cpu_features_macos():
    """Returns {brand, vendor, features:set, freq_max_hz}. Empty fields if a
    probe fails — never raises. On Intel Macs reads machdep.cpu.{features,
    leaf7_features}; on Apple Silicon probes hw.optional.* per-feature."""
    out = {"brand": None, "vendor": None, "features": set(), "freq_max_hz": None}
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=1.0)
    if brand: out["brand"] = brand.strip()
    vendor = _run(["sysctl", "-n", "machdep.cpu.vendor"], timeout=1.0)
    if vendor and vendor.strip():
        out["vendor"] = vendor.strip()
    elif (platform.machine() or "").lower() == "arm64":
        out["vendor"] = "Apple"

    f1 = _run(["sysctl", "-n", "machdep.cpu.features"], timeout=1.0) or ""
    f2 = _run(["sysctl", "-n", "machdep.cpu.leaf7_features"], timeout=1.0) or ""
    for tok in (f1 + " " + f2).split():
        out["features"].add(tok.upper().replace(".", "_"))

    if (platform.machine() or "").lower() == "arm64":
        for name in _MAC_ARM_IMPLIED_FEATURES:
            out["features"].add(name)
        for key, name in _MAC_ARM_FEATURE_PROBES:
            v = _run(["sysctl", "-n", key], timeout=1.0)
            if v and v.strip() == "1":
                out["features"].add(name)

    freq = _run(["sysctl", "-n", "hw.cpufrequency_max"], timeout=1.0)
    if freq:
        try: out["freq_max_hz"] = int(freq.strip())
        except ValueError: pass
    return out


def _cpu_features_linux():
    out = {"brand": None, "vendor": None, "features": set(), "freq_max_hz": None}
    try:
        with open("/proc/cpuinfo", "r") as f:
            for raw in f:
                if ":" not in raw: continue
                k, _, v = raw.partition(":")
                k = k.strip(); v = v.strip()
                if k == "model name" and out["brand"] is None:
                    out["brand"] = v
                elif k == "vendor_id" and out["vendor"] is None:
                    out["vendor"] = v
                elif k == "flags" and not out["features"]:
                    out["features"] = {tok.upper() for tok in v.split()}
                elif k == "cpu MHz" and out["freq_max_hz"] is None:
                    try: out["freq_max_hz"] = int(float(v) * 1_000_000)
                    except ValueError: pass
    except OSError:
        pass
    return out


def _cpu_features_windows():
    """Windows: CPU brand via WMI. Feature flags aren't exposed by WMI; we
    infer the obvious ones (AVX, AVX2) from the brand string when we can,
    leaving the set possibly incomplete."""
    out = {"brand": None, "vendor": None, "features": set(), "freq_max_hz": None}
    j = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Processor | "
        "Select-Object Name,Manufacturer,MaxClockSpeed | ConvertTo-Json -Compress",
    ], timeout=4.0)
    if j:
        try:
            blob = json.loads(j)
            if isinstance(blob, list) and blob:
                blob = blob[0]
            if isinstance(blob, dict):
                out["brand"] = (blob.get("Name") or "").strip() or None
                out["vendor"] = (blob.get("Manufacturer") or "").strip() or None
                mhz = blob.get("MaxClockSpeed")
                if mhz:
                    try: out["freq_max_hz"] = int(mhz) * 1_000_000
                    except (TypeError, ValueError): pass
        except (ValueError, json.JSONDecodeError):
            pass
    return out


def _cpu_features():
    """Cross-platform CPU brand + feature set. Always returns the same shape;
    empty when nothing could be probed."""
    if sys.platform == "darwin":   return _cpu_features_macos()
    if sys.platform.startswith("linux"): return _cpu_features_linux()
    if sys.platform.startswith("win"):   return _cpu_features_windows()
    return {"brand": None, "vendor": None, "features": set(), "freq_max_hz": None}


def _os_version():
    """Product-level OS version string. macOS: `sw_vers -productVersion`.
    Linux: best-effort from /etc/os-release. Windows: platform.win32_ver."""
    if sys.platform == "darwin":
        v = _run(["sw_vers", "-productVersion"], timeout=1.0)
        b = _run(["sw_vers", "-buildVersion"],   timeout=1.0)
        return {
            "product": (v.strip() if v else None),
            "build":   (b.strip() if b else None),
        }
    if sys.platform.startswith("linux"):
        try:
            with open("/etc/os-release", "r") as f:
                kv = {}
                for line in f:
                    if "=" not in line: continue
                    k, _, val = line.partition("=")
                    kv[k.strip()] = val.strip().strip('"')
            return {
                "product": kv.get("PRETTY_NAME") or kv.get("NAME"),
                "build":   kv.get("BUILD_ID") or kv.get("VERSION_ID"),
            }
        except OSError:
            return {"product": None, "build": None}
    if sys.platform.startswith("win"):
        rel, ver, csd, ptype = platform.win32_ver()
        return {"product": rel or None, "build": ver or None}
    return {"product": None, "build": None}


def _disk_free_at_repo():
    try:
        s = os.statvfs(REPO_ROOT) if hasattr(os, "statvfs") else None
        if s is not None:
            return int(s.f_bavail) * int(s.f_frsize)
        import shutil as _sh
        return int(_sh.disk_usage(REPO_ROOT).free)
    except (OSError, AttributeError):
        return None


# A stable, ordered list of CPU feature flag names that downstream code keys
# off when picking kernels or warning the user. The presence/absence of each
# is reported as a bool so consumers don't have to canonicalize a giant set.
CPU_FEATURES_OF_INTEREST = (
    "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2",
    "AVX1_0", "AVX2", "AVX512F", "AVX512BW", "AVX512VL", "AVX512VNNI",
    "FMA", "F16C", "BMI1", "BMI2", "POPCNT", "AES", "PCLMULQDQ", "RDRAND",
    "NEON", "ASIMD", "ASIMDDP", "ASIMDFHM", "FP16", "BF16", "I8MM",
)


# ------------------------------------------------------------------------------------
# Saved system spec file. Captured on demand from settings; lives under data/.

SPECS_PATH = os.path.join(REPO_ROOT, "data", "system_specs.json")


def detect_specs():
    """Cross-platform machine spec snapshot for the saved specs file. Includes
    raw OS/CPU/GPU/memory details PLUS pre-derived `capabilities` booleans so
    the dashboard doesn't have to re-derive them from feature flag strings.
    Everything in this dict is what the heartbeat sends when the user opts
    into hardware analytics."""
    _refresh_adapters()
    snap = snapshot()
    cpu_info = _cpu_features()
    feats = cpu_info.get("features") or set()
    features_present = {name: (name in feats) for name in CPU_FEATURES_OF_INTEREST}
    has_any_nvidia = any((g.get("vendor") or "").upper() == "NVIDIA" for g in (snap.get("gpus") or []))
    is_apple_silicon = (sys.platform == "darwin" and (platform.machine() or "").lower() == "arm64")
    os_v = _os_version()
    return {
        "captured_at": int(time.time()),
        "platform": {
            "system":   platform.system() or "",
            "release":  platform.release() or "",
            "version":  platform.version() or "",
            "machine":  platform.machine() or "",
            "processor": platform.processor() or "",
            "python":   platform.python_version(),
            "os_product": os_v.get("product"),
            "os_build":   os_v.get("build"),
        },
        "cpu": {
            "brand":          cpu_info.get("brand"),
            "vendor":         cpu_info.get("vendor"),
            "count_logical":  int(_CPU_COUNT),
            "count_physical": int(psutil.cpu_count(logical=False)) if _PSUTIL_OK else None,
            "freq_max_hz":    cpu_info.get("freq_max_hz"),
            "features":       sorted(feats),
            "features_present": features_present,
        },
        "memory": {
            "total_bytes":     int(snap.get("sys_mem_total") or 0) or None,
            "available_bytes": int(snap.get("sys_mem_available") or 0) or None,
        },
        "disk": {
            "repo_free_bytes": _disk_free_at_repo(),
        },
        "gpus": snap.get("gpus") or [],
        "capabilities": {
            "has_sse42":        features_present.get("SSE4_2", False),
            "has_avx1":         features_present.get("AVX1_0", False) or features_present.get("AVX", False),
            "has_avx2":         features_present.get("AVX2", False),
            "has_avx512f":      features_present.get("AVX512F", False),
            "has_avx512vnni":   features_present.get("AVX512VNNI", False),
            "has_fma":          features_present.get("FMA", False),
            "has_f16c":         features_present.get("F16C", False),
            "is_apple_silicon": is_apple_silicon,
            "can_use_cuda":     has_any_nvidia,
            "can_use_mps":      is_apple_silicon,
            "can_use_metal":    sys.platform == "darwin",
        },
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
