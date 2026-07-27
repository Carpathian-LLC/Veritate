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
import uuid

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
    psutil.cpu_percent(None)   # prime system-wide baseline so the first poll reads real load, not 0

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

# macmon has a ~8s startup cost when stdout is piped, so per-poll subprocess
# calls would freeze the HUD. Run it once as a persistent streaming process and
# read JSON samples from a background thread.
_MAC_MACMON_PROC   = None
_MAC_MACMON_LATEST = None  # {"cpu": float|None, "gpu": float|None, "ts": float}
_MAC_MACMON_LOCK   = threading.Lock()
_MAC_MACMON_STALE_S = 5.0

# Subprocess budgets by call class: one-shot probes, live per-poll queries,
# slow adapter discovery.
_PROBE_TIMEOUT_S     = 1.0
_MAC_TEMP_TIMEOUT_S  = 1.5
_LIVE_TIMEOUT_S      = 2.0
_DISCOVERY_TIMEOUT_S = 4.0

_KIB = 1024
_MIB = _KIB * 1024
_GIB = _MIB * 1024

# Plausible die-temperature window. Readings outside it are sensor noise.
_TEMP_MIN_C = 0.0
_TEMP_MAX_C = 150.0

SPECS_PATH = os.path.join(REPO_ROOT, "data", "system_specs.json")

# ------------------------------------------------------------------------------------
# Per-OS probe commands. One table per platform so an arch fix lands in one
# place (rule 34c). Every subprocess argv in this file comes from here.

_NVIDIA_QUERY_ARGS = (
    "nvidia-smi",
    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
)
_NVIDIA_NA_VALUES = ("", "N/A", "[N/A]")

_MAC_MACMON_BIN           = "macmon"
_MAC_MACMON_INTERVAL_MS   = 1000
_MAC_MACMON_ARGS          = (_MAC_MACMON_BIN, "pipe", "-i", str(_MAC_MACMON_INTERVAL_MS))
_MAC_SYSTEM_PROFILER_ARGS = ("system_profiler", "SPDisplaysDataType", "-json")
_MAC_IOREG_ACCEL_ARGS     = ("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator")
_MAC_IOREG_PLATFORM_ARGS  = ("ioreg", "-rd1", "-c", "IOPlatformExpertDevice")
_MAC_CPU_TEMP_ARGS        = ("osx-cpu-temp", "-c")
_MAC_SW_VERS_PRODUCT_ARGS = ("sw_vers", "-productVersion")
_MAC_SW_VERS_BUILD_ARGS   = ("sw_vers", "-buildVersion")
_MAC_SYSCTL_ARGS          = ("sysctl", "-n")
_MAC_SYSCTL_BRAND         = "machdep.cpu.brand_string"
_MAC_SYSCTL_VENDOR        = "machdep.cpu.vendor"
_MAC_SYSCTL_FEATURE_KEYS  = ("machdep.cpu.features", "machdep.cpu.leaf7_features")
_MAC_SYSCTL_FREQ_MAX      = "hw.cpufrequency_max"
_MAC_ARM_MACHINE          = "arm64"
_MAC_ARM_VENDOR           = "Apple"
_MAC_APPLE_GPU_TOKEN      = "apple"
_SYSCTL_TRUE              = "1"
_MAC_UUID_RE  = re.compile(r'"IOPlatformUUID"\s*=\s*"([^"]+)"')
_IOREG_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")
# NEON/ASIMD are mandatory on ARMv8 so Apple exposes no separate sysctl for
# them. Treat as always-present on arm64 macOS.
_MAC_ARM_IMPLIED_FEATURES = ("NEON", "ASIMD")
# (sysctl key, feature name reported). Apple Silicon ships these as
# hw.optional.arm.FEAT_*, each returning "1" when supported.
_MAC_ARM_FEATURE_PROBES = (
    ("hw.optional.arm.FEAT_DotProd",       "ASIMDDP"),
    ("hw.optional.arm.FEAT_FP16",          "FP16"),
    ("hw.optional.arm.FEAT_BF16",          "BF16"),
    ("hw.optional.arm.FEAT_I8MM",          "I8MM"),
    ("hw.optional.arm.FEAT_FHM",           "ASIMDFHM"),
    ("hw.optional.arm.FEAT_SHA512",        "SHA512"),
    ("hw.optional.arm.FEAT_SHA3",          "SHA3"),
    ("hw.optional.arm.FEAT_AES",           "AES"),
)

_LINUX_DRM_PATH        = "/sys/class/drm"
_LINUX_DEVICE_DIR      = "device"
_LINUX_VENDOR_FILE     = "vendor"
_LINUX_GPU_BUSY_FILE   = "gpu_busy_percent"
_LINUX_CPUINFO_PATH    = "/proc/cpuinfo"
_LINUX_OS_RELEASE_PATH = "/etc/os-release"
_LINUX_MACHINE_ID_PATHS = ("/etc/machine-id", "/var/lib/dbus/machine-id")
_LSPCI_ARGS             = ("lspci", "-D")

_PS_ARGS = ("powershell", "-NoProfile", "-Command")
_PS_LHM_SENSORS_FMT = (
    "Get-CimInstance -Namespace '{ns}' -ClassName Sensor -ErrorAction Stop | "
    "Where-Object {{ $_.SensorType -eq 'Temperature' }} | "
    "Select-Object Name,Parent,Value | ConvertTo-Json -Compress"
)
_PS_VIDEO_CONTROLLERS = (
    "Get-CimInstance Win32_VideoController | "
    "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
)
_PS_PHYSICAL_MEMORY_SUM = "(Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum"
_PS_PROCESSOR = (
    "Get-CimInstance Win32_Processor | "
    "Select-Object Name,Manufacturer,MaxClockSpeed | ConvertTo-Json -Compress"
)
_LHM_NAMESPACES = ("root/LibreHardwareMonitor", "root/OpenHardwareMonitor")
_WIN_MACHINE_GUID_ARGS = (
    "reg", "query", r"HKLM\SOFTWARE\Microsoft\Cryptography", "/v", "MachineGuid",
)
_WIN_GUID_RE = re.compile(r"MachineGuid\s+REG_SZ\s+(\S+)")

# psutil sensor buckets and label tokens that carry a CPU package temperature.
_PSUTIL_TEMP_KEYS = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz")
_PSUTIL_TEMP_LABEL_TOKENS = ("cpu", "package", "tdie", "tctl")
_LHM_CPU_TOKENS      = ("cpu", "package", "tctl", "tdie")
_LHM_CPU_PARENT_TOKEN = "cpu"
_LHM_GPU_TOKEN       = "gpu"
_LHM_PACKAGE_TOKEN   = "package"

# Substring patterns that identify non-hardware display adapters hidden from
# the HUD: remote-session viewers (RDP), fallback software rasterizers, VM
# guest drivers, USB "indirect display" adapters, virtual-monitor tools.
# Matched case-insensitively; discrete and integrated GPUs are unaffected.
_VIRTUAL_ADAPTER_TOKENS = (
    "remote display",       # Microsoft Remote Display Adapter (RDP session)
    "basic display",        # Microsoft Basic Display Adapter (fallback WDDM)
    "basic render",         # Microsoft Basic Render Driver
    "mirror",               # Legacy mirror drivers
    "virtual display",      # Generic virtual-monitor drivers
    "virtual monitor",
    "virtualbox",           # Oracle VirtualBox guest
    "vmware",               # VMware SVGA
    "hyper-v",              # Microsoft Hyper-V Video
    "parsec",               # Parsec virtual display
    "citrix",               # Citrix Display Only Adapter
    "displaylink",          # USB display adapters (indirect display)
    "indirect display",
    "idd driver",
    "meta virtual",         # Meta Quest / Oculus virtual display
    "moonlight",            # Moonlight streaming virtual display
    "software adapter",
    "llvmpipe",             # Mesa software rasterizer
    "swrast",
    "virtio",               # KVM/QEMU virtio-gpu
    "qxl",                  # Spice/QEMU QXL
    "cirrus logic",         # QEMU legacy VGA
)

# (canonical vendor, name substrings). Shared by the Windows adapter reader and
# the lspci name cleaner so vendor detection has one owner (rule 20).
_GPU_VENDOR_TOKENS = (
    ("Intel",  ("intel",)),
    ("NVIDIA", ("nvidia", "geforce", "rtx", "gtx")),
    ("AMD",    ("amd", "radeon", "advanced micro")),
)
_INTEGRATED_TOKENS = ("integrated", "vega")
_INTEGRATED_VENDOR = "Intel"
_UNKNOWN_VENDOR    = "?"

_GPU_CLASS_RE = re.compile(
    r"^(\S+)\s+(?:VGA compatible controller|3D controller|Display controller):\s*(.+)$",
    re.IGNORECASE,
)
_GPU_MODEL_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_GPU_REV_RE = re.compile(r"\s*\(rev [0-9a-fA-F]+\)\s*$")
_DRM_CARD_RE = re.compile(r"^card\d+$")
_PCI_VENDOR_NAMES = {"0x1002": "AMD", "0x10de": "NVIDIA", "0x8086": "Intel"}

# Stable, ordered CPU feature flags downstream code keys off when picking
# kernels or warning the user. Reported as bools so consumers never
# canonicalize a raw feature set.
CPU_FEATURES_OF_INTEREST = (
    "SSE2", "SSE3", "SSSE3", "SSE4_1", "SSE4_2",
    "AVX1_0", "AVX2", "AVX512F", "AVX512BW", "AVX512VL", "AVX512VNNI",
    "FMA", "F16C", "BMI1", "BMI2", "POPCNT", "AES", "PCLMULQDQ", "RDRAND",
    "NEON", "ASIMD", "ASIMDDP", "ASIMDFHM", "FP16", "BF16", "I8MM",
)

# ------------------------------------------------------------------------------------
# Functions

def _run(cmd, timeout=_LIVE_TIMEOUT_S):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False,
                             creationflags=_NO_WINDOW)
        return out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _in_temp_range(v):
    return _TEMP_MIN_C < v < _TEMP_MAX_C


def _vendor_from_name(name):
    """Canonical GPU vendor for an adapter/controller name. None when unknown."""
    low = (name or "").lower()
    for vendor, tokens in _GPU_VENDOR_TOKENS:
        if any(tok in low for tok in tokens):
            return vendor
    return None


def _is_virtual_adapter(name):
    if not name:
        return False
    low = name.lower()
    return any(tok in low for tok in _VIRTUAL_ADAPTER_TOKENS)


def _nvidia_query():
    global _NV_CACHE
    now = time.time()
    if (now - _NV_CACHE[0]) < _LIVE_TTL:
        return _NV_CACHE[1]
    out = _run(_NVIDIA_QUERY_ARGS)
    rows = []
    if out:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                temp = None
                if len(parts) >= 5 and parts[4] not in _NVIDIA_NA_VALUES:
                    try:
                        temp = float(parts[4])
                    except ValueError:
                        temp = None
                rows.append({
                    "name": parts[0],
                    "vendor": "NVIDIA",
                    "integrated": False,
                    "load_pct": float(parts[1]),
                    "vram_used": int(parts[2]) * _MIB,
                    "vram_total": int(parts[3]) * _MIB,
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
    if u.startswith("GB"): return int(n * _GIB)
    if u.startswith("MB"): return int(n * _MIB)
    if u.startswith("KB"): return int(n * _KIB)
    return None


def _mac_adapters():
    out = _run(_MAC_SYSTEM_PROFILER_ARGS, timeout=_DISCOVERY_TIMEOUT_S)
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


def _psutil_cpu_temp():
    """psutil.sensors_temperatures. Works on Linux (coretemp/k10temp/cpu_thermal)
    and some macOS builds. Returns Celsius or None. Empty on Windows."""
    if not _PSUTIL_OK or not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        temps = psutil.sensors_temperatures() or {}
    except (AttributeError, OSError):
        return None
    for key in _PSUTIL_TEMP_KEYS:
        for e in temps.get(key) or []:
            if e.current and e.current > _TEMP_MIN_C:
                return float(e.current)
    for entries in temps.values():
        for e in entries or []:
            label = (getattr(e, "label", "") or "").lower()
            if any(tok in label for tok in _PSUTIL_TEMP_LABEL_TOKENS) and e.current and e.current > _TEMP_MIN_C:
                return float(e.current)
    return None


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
                    "cpu": float(cpu) if isinstance(cpu, (int, float)) and _in_temp_range(cpu) else None,
                    "gpu": float(gpu) if isinstance(gpu, (int, float)) and _in_temp_range(gpu) else None,
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
        subprocess.run(["pkill", "-U", str(os.getuid()), "-x", _MAC_MACMON_BIN],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        proc = subprocess.Popen(
            _MAC_MACMON_ARGS,
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
    out = _run(_MAC_CPU_TEMP_ARGS, timeout=_MAC_TEMP_TIMEOUT_S)
    if out:
        m = _FLOAT_RE.search(out)
        if m:
            try:
                v = float(m.group(1))
                if _in_temp_range(v):
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
    namespaces = (_LHM_NAMESPACE,) if _LHM_NAMESPACE else _LHM_NAMESPACES
    result = None
    for ns in namespaces:
        out = _run([*_PS_ARGS, _PS_LHM_SENSORS_FMT.format(ns=ns)], timeout=_LIVE_TIMEOUT_S)
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
            if not _in_temp_range(val):
                continue
            low = name.lower()
            if _LHM_CPU_PARENT_TOKEN in parent or any(tok in low for tok in _LHM_CPU_TOKENS):
                if cpu_val is None or _LHM_PACKAGE_TOKEN in low:
                    cpu_val = val
            elif _LHM_GPU_TOKEN in parent or _LHM_GPU_TOKEN in low:
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
    out = _run(_MAC_IOREG_ACCEL_ARGS, timeout=_LIVE_TIMEOUT_S)
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
    out = _run([*_PS_ARGS, _PS_VIDEO_CONTROLLERS], timeout=_DISCOVERY_TIMEOUT_S)
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
        low = name.lower()
        vendor = _vendor_from_name(name) or _UNKNOWN_VENDOR
        integrated = vendor == _INTEGRATED_VENDOR or any(tok in low for tok in _INTEGRATED_TOKENS)
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


def _clean_gpu_name(desc):
    """Short model name from an lspci controller description.
    'Intel Corporation CoffeeLake-S GT2 [UHD Graphics 630] (rev 02)' ->
    'Intel UHD Graphics 630'. Falls back to the trimmed description."""
    desc = _GPU_REV_RE.sub("", desc).strip()
    brackets = _GPU_MODEL_BRACKET_RE.findall(desc)
    model = brackets[-1].strip() if brackets else desc
    vendor = _vendor_from_name(desc)
    if vendor and vendor.lower() not in model.lower():
        return f"{vendor} {model}"
    return model


def _lspci_gpu_names():
    """Map PCI slot (domain:bus:dev.fn) -> model name via lspci. {} if lspci absent."""
    out = _run(_LSPCI_ARGS, timeout=_LIVE_TIMEOUT_S)
    names = {}
    if not out:
        return names
    for line in out.splitlines():
        m = _GPU_CLASS_RE.match(line.strip())
        if m:
            names[m.group(1)] = _clean_gpu_name(m.group(2))
    return names


def _linux_adapters():
    rows = []
    if not os.path.isdir(_LINUX_DRM_PATH):
        return rows
    names = _lspci_gpu_names()
    for entry in sorted(os.listdir(_LINUX_DRM_PATH)):
        if not _DRM_CARD_RE.match(entry):
            continue
        dev = os.path.join(_LINUX_DRM_PATH, entry, _LINUX_DEVICE_DIR)
        vendor = _UNKNOWN_VENDOR
        try:
            with open(os.path.join(dev, _LINUX_VENDOR_FILE)) as f:
                vid = f.read().strip()
            vendor = _PCI_VENDOR_NAMES.get(vid, vid)
        except OSError:
            pass
        slot = os.path.basename(os.path.realpath(dev))
        name = names.get(slot) or (f"{vendor} GPU" if vendor not in (_UNKNOWN_VENDOR, "") else entry)
        load = None
        try:
            with open(os.path.join(dev, _LINUX_GPU_BUSY_FILE)) as f:
                load = float(f.read().strip())
        except OSError:
            pass
        rows.append({
            "name": name,
            "vendor": vendor,
            "integrated": vendor == _INTEGRATED_VENDOR,
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
    rows = [r for r in rows if not _is_virtual_adapter(r.get("name"))]
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
        out = _run([*_PS_ARGS, _PS_PHYSICAL_MEMORY_SUM], timeout=_DISCOVERY_TIMEOUT_S)
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
                if _MAC_APPLE_GPU_TOKEN in a["name"].lower() or a["integrated"]:
                    a["load_pct"] = load
                    break
        gpu_t = _mac_gpu_temp()
        if gpu_t is not None:
            for a in adapters:
                if a.get("temp_c") is None and (_MAC_APPLE_GPU_TOKEN in a["name"].lower() or a["integrated"]):
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
    # expect the HUD to spike on: trainers run in subprocesses, so the
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

def _sysctl(key):
    return _run([*_MAC_SYSCTL_ARGS, key], timeout=_PROBE_TIMEOUT_S)


def _is_mac_arm():
    return (platform.machine() or "").lower() == _MAC_ARM_MACHINE


def _cpu_features_macos():
    """Returns {brand, vendor, features:set, freq_max_hz}. Empty fields if a
    probe fails: never raises. On Intel Macs reads machdep.cpu.{features,
    leaf7_features}; on Apple Silicon probes hw.optional.* per-feature."""
    out = {"brand": None, "vendor": None, "features": set(), "freq_max_hz": None}
    brand = _sysctl(_MAC_SYSCTL_BRAND)
    if brand: out["brand"] = brand.strip()
    vendor = _sysctl(_MAC_SYSCTL_VENDOR)
    if vendor and vendor.strip():
        out["vendor"] = vendor.strip()
    elif _is_mac_arm():
        out["vendor"] = _MAC_ARM_VENDOR

    for key in _MAC_SYSCTL_FEATURE_KEYS:
        for tok in (_sysctl(key) or "").split():
            out["features"].add(tok.upper().replace(".", "_"))

    if _is_mac_arm():
        out["features"].update(_MAC_ARM_IMPLIED_FEATURES)
        for key, name in _MAC_ARM_FEATURE_PROBES:
            v = _sysctl(key)
            if v and v.strip() == _SYSCTL_TRUE:
                out["features"].add(name)

    freq = _sysctl(_MAC_SYSCTL_FREQ_MAX)
    if freq:
        try: out["freq_max_hz"] = int(freq.strip())
        except ValueError: pass
    return out


def _cpu_features_linux():
    out = {"brand": None, "vendor": None, "features": set(), "freq_max_hz": None}
    try:
        with open(_LINUX_CPUINFO_PATH) as f:
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
    j = _run([*_PS_ARGS, _PS_PROCESSOR], timeout=_DISCOVERY_TIMEOUT_S)
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
        v = _run(_MAC_SW_VERS_PRODUCT_ARGS, timeout=_PROBE_TIMEOUT_S)
        b = _run(_MAC_SW_VERS_BUILD_ARGS,   timeout=_PROBE_TIMEOUT_S)
        return {
            "product": (v.strip() if v else None),
            "build":   (b.strip() if b else None),
        }
    if sys.platform.startswith("linux"):
        try:
            with open(_LINUX_OS_RELEASE_PATH) as f:
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
        rel, ver, _, _ = platform.win32_ver()
        return {"product": rel or None, "build": ver or None}
    return {"product": None, "build": None}


def stable_machine_key():
    """Per-machine identifier that survives reboots but differs across machines.
    Linux: `/etc/machine-id`. macOS: `IOPlatformUUID`. Windows: registry
    `MachineGuid`. Falls back to hostname + NIC MAC. Lets the heartbeat bind its
    device id to this box so a copied install re-derives its own instead of
    colliding."""
    if sys.platform.startswith("linux"):
        for p in _LINUX_MACHINE_ID_PATHS:
            try:
                with open(p) as f:
                    v = f.read().strip()
                if v:
                    return v
            except OSError:
                pass
    elif sys.platform == "darwin":
        m = _MAC_UUID_RE.search(_run(_MAC_IOREG_PLATFORM_ARGS, timeout=_LIVE_TIMEOUT_S) or "")
        if m:
            return m.group(1)
    elif sys.platform.startswith("win"):
        m = _WIN_GUID_RE.search(_run(_WIN_MACHINE_GUID_ARGS, timeout=_LIVE_TIMEOUT_S) or "")
        if m:
            return m.group(1)
    return f"{platform.node() or ''}|{uuid.getnode()}"


def _disk_free_at_repo():
    try:
        s = os.statvfs(REPO_ROOT) if hasattr(os, "statvfs") else None
        if s is not None:
            return int(s.f_bavail) * int(s.f_frsize)
        import shutil as _sh
        return int(_sh.disk_usage(REPO_ROOT).free)
    except (OSError, AttributeError):
        return None


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
    is_apple_silicon = (sys.platform == "darwin" and _is_mac_arm())
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
        with open(SPECS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def detect_and_save():
    specs = detect_specs()
    prev = load_specs()
    if prev and isinstance(prev.get("measured"), dict):
        specs["measured"] = prev["measured"]
    save_specs(specs)
    from runtime import heartbeat
    heartbeat.reconcile_identity()
    return specs


def save_measured(measured):
    """Merge an auto-tune probe result into the saved specs. `measured` is the
    summary dict the probe emits (device, max_batch, mem_ceiling_gb, tok_per_s).

    Per-device caching: each call is filed under `measured.per_device[device]`
    so a later probe on a *different* device (e.g. CPU then CUDA after
    installing the CUDA torch build) doesn't overwrite the previous one. The
    top-level `measured` still holds the most recent probe for backwards
    compatibility with older consumers that read specs["measured"] directly.
    Rule 34c: distinguish device+memory kind at every layer: the per-device
    map is the storage half of that."""
    if not isinstance(measured, dict):
        return None
    specs = load_specs() or {}
    device = str(measured.get("device") or "cpu")
    prev = specs.get("measured")
    per_device = {}
    # Migrate an older single-blob `measured` into the per_device map so we
    # don't lose the previous probe on the first per-device save.
    if isinstance(prev, dict):
        if isinstance(prev.get("per_device"), dict):
            per_device.update(prev["per_device"])
        pdev = prev.get("device")
        if pdev and pdev not in per_device:
            per_device[str(pdev)] = {k: v for k, v in prev.items() if k != "per_device"}
    per_device[device] = dict(measured)
    merged = dict(measured)
    merged["per_device"] = per_device
    specs["measured"] = merged
    save_specs(specs)
    return specs
