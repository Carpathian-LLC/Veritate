# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Hardware benchmark suite that runs alongside the trainer-specific bench.py:
#   disk write speed, CPU FLOPs + memory bandwidth, GPU FLOPs on every detected
#   accelerator, and RAM headroom. Cross-platform (x86, arm64, AMD via CUDA when
#   present). Result is a compact JSON dict written to data/system_specs.json
#   under the "probes" key so bench + planner can consume without re-probing.
# - Independent of the model: no weights are built, no corpus is touched. Runs
#   short synthetic workloads (~5-15 s total) so Auto tune stays snappy.
# veritate_core/plugin/sysprobe.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import platform
import shutil
import sys
import tempfile
import time

# ------------------------------------------------------------------------------------
# Constants

MB = 1024 * 1024
GB = 1024 ** 3

DISK_PAYLOAD_BYTES = 64 * MB
DISK_CHUNK_BYTES   = 4 * MB
DISK_RANDOM_IOS    = 256
DISK_RANDOM_BYTES  = 4 * 1024

CPU_MATMUL_N       = 1024
CPU_MATMUL_ITERS   = 6
CPU_BANDWIDTH_MB   = 128

GPU_MATMUL_N       = 2048
GPU_MATMUL_ITERS   = 8
GPU_WARMUP_ITERS   = 2

# ------------------------------------------------------------------------------------
# Functions


def _probe_disk(dir_path):
    """Sequential + random write MB/s under `dir_path`. Uses os.fsync so cached
    writes don't inflate the measurement. Cleans up after itself."""
    os.makedirs(dir_path, exist_ok=True)
    seq_path = os.path.join(dir_path, "sysprobe_seq.tmp")
    rand_path = os.path.join(dir_path, "sysprobe_rand.tmp")
    payload = os.urandom(DISK_CHUNK_BYTES)
    chunks = DISK_PAYLOAD_BYTES // DISK_CHUNK_BYTES
    seq_mb_s = 0.0
    rand_mb_s = 0.0
    try:
        t0 = time.perf_counter()
        with open(seq_path, "wb", buffering=0) as f:
            for _ in range(chunks):
                f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        seq_elapsed = time.perf_counter() - t0
        if seq_elapsed > 0:
            seq_mb_s = (DISK_PAYLOAD_BYTES / MB) / seq_elapsed

        small = os.urandom(DISK_RANDOM_BYTES)
        with open(rand_path, "wb", buffering=0) as f:
            f.write(b"\0" * DISK_PAYLOAD_BYTES)
            f.flush()
            os.fsync(f.fileno())
        with open(rand_path, "r+b", buffering=0) as f:
            fsize = DISK_PAYLOAD_BYTES - DISK_RANDOM_BYTES
            offsets = [(i * 65537) % fsize for i in range(DISK_RANDOM_IOS)]
            t0 = time.perf_counter()
            for off in offsets:
                f.seek(off)
                f.write(small)
            f.flush()
            os.fsync(f.fileno())
            rand_elapsed = time.perf_counter() - t0
        if rand_elapsed > 0:
            total = DISK_RANDOM_IOS * DISK_RANDOM_BYTES
            rand_mb_s = (total / MB) / rand_elapsed
    finally:
        for p in (seq_path, rand_path):
            try:
                os.remove(p)
            except OSError:
                pass
    free = shutil.disk_usage(dir_path).free
    return {
        "path": dir_path,
        "seq_write_mb_s":  round(seq_mb_s, 1),
        "rand_write_mb_s": round(rand_mb_s, 1),
        "free_gb": round(free / GB, 2),
    }


def _probe_cpu():
    """Two numbers: dense matmul GFLOP/s (compute) and memory-copy GB/s
    (bandwidth). Uses numpy so no torch import cost when GPU-only callers want
    just the disk/RAM probes later."""
    import numpy as np
    n = CPU_MATMUL_N
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    a @ b
    t0 = time.perf_counter()
    for _ in range(CPU_MATMUL_ITERS):
        a @ b
    elapsed = time.perf_counter() - t0
    flops = 2.0 * (n ** 3) * CPU_MATMUL_ITERS
    gflop_s = (flops / 1e9) / elapsed if elapsed > 0 else 0.0

    buf_bytes = CPU_BANDWIDTH_MB * MB
    src = np.zeros(buf_bytes // 4, dtype=np.float32)
    src.fill(1.0)
    src.copy()
    t0 = time.perf_counter()
    for _ in range(3):
        src.copy()
    elapsed = time.perf_counter() - t0
    gb_s = (3 * buf_bytes / GB) / elapsed if elapsed > 0 else 0.0

    from veritate_core.plugin import hardware
    return {
        "physical_cores": hardware.physical_cores(),
        "brand":          platform.processor() or platform.machine() or "cpu",
        "arch":           platform.machine() or "",
        "matmul_gflops":  round(gflop_s, 1),
        "copy_gb_s":      round(gb_s, 2),
    }


def _torch_devices():
    """Enumerate every accelerator torch can dispatch to on this box. CUDA can
    report multiple GPUs (dual-card rigs); MPS is Apple-Silicon-only. Skips
    devices whose backend errored during import."""
    devs = []
    try:
        import torch
    except ImportError:
        return devs
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devs.append(("cuda", i, torch.cuda.get_device_name(i)))
    except (RuntimeError, AssertionError):
        pass
    try:
        from veritate_core.plugin import hardware
        if hardware.mps_supported():
            devs.append(("mps", 0, "Apple GPU (MPS)"))
    except (RuntimeError, ImportError):
        pass
    return devs


def _probe_gpu(kind, index, name):
    """FP32 matmul TFLOP/s on one accelerator. `kind` is 'cuda' or 'mps'; index
    picks the CUDA card. VRAM total is reported so the planner can cross-check."""
    import torch
    n = GPU_MATMUL_N
    dev = torch.device(f"{kind}:{index}" if kind == "cuda" else kind)
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)
    for _ in range(GPU_WARMUP_ITERS):
        (a @ b).sum().item()
    if kind == "cuda":
        torch.cuda.synchronize(dev)
    else:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    c = a
    for _ in range(GPU_MATMUL_ITERS):
        c = a @ b
    c.sum().item()
    if kind == "cuda":
        torch.cuda.synchronize(dev)
    else:
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    flops = 2.0 * (n ** 3) * GPU_MATMUL_ITERS
    tflop_s = (flops / 1e12) / elapsed if elapsed > 0 else 0.0
    vram_total = None
    if kind == "cuda":
        vram_total = int(torch.cuda.get_device_properties(index).total_memory)
    return {
        "device":     f"{kind}:{index}",
        "name":       name,
        "matmul_tflops": round(tflop_s, 2),
        "vram_total_gb": round(vram_total / GB, 2) if vram_total else None,
    }


def _probe_ram():
    """Installed vs available RAM + swap. Swap presence flips the "safe budget"
    down: paging into swap during training is what triggers the SIGKILL guard."""
    try:
        import psutil
    except ImportError:
        return {"total_gb": None, "available_gb": None, "swap_total_gb": None,
                "swap_used_gb": None}
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "total_gb":       round(vm.total / GB, 2),
        "available_gb":   round(vm.available / GB, 2),
        "swap_total_gb":  round(sw.total / GB, 2),
        "swap_used_gb":   round(sw.used / GB, 2),
    }


def run(disk_dir=None, on_progress=None):
    """Execute every probe once and return a compact dict. `disk_dir` overrides
    the disk probe location (defaults to system temp so we don't stomp the
    repo). `on_progress(str)` receives one line per stage for the Auto tune
    modal to tail."""
    emit = on_progress or (lambda _line: None)
    disk_dir = disk_dir or tempfile.gettempdir()
    t0 = time.perf_counter()

    emit("probing disk write speed...")
    disk = _probe_disk(disk_dir)
    emit(f"disk: seq {disk['seq_write_mb_s']} MB/s, rand {disk['rand_write_mb_s']} MB/s "
         f"({disk['free_gb']} GB free)")

    emit("probing CPU compute + bandwidth...")
    cpu = _probe_cpu()
    emit(f"cpu: {cpu['physical_cores']} cores, matmul {cpu['matmul_gflops']} GFLOP/s, "
         f"copy {cpu['copy_gb_s']} GB/s")

    devs = _torch_devices()
    gpus = []
    for kind, idx, name in devs:
        emit(f"probing {kind}:{idx} ({name})...")
        try:
            gpu = _probe_gpu(kind, idx, name)
        except Exception as exc:
            emit(f"{kind}:{idx} probe failed: {type(exc).__name__}: {exc}")
            continue
        gpus.append(gpu)
        emit(f"{gpu['device']}: {gpu['matmul_tflops']} TFLOP/s"
             + (f", {gpu['vram_total_gb']} GB VRAM" if gpu["vram_total_gb"] else ""))
    if not devs:
        emit("no torch accelerator detected (cpu-only box)")

    ram = _probe_ram()
    emit(f"ram: {ram['total_gb']} GB total, {ram['available_gb']} GB available, "
         f"swap {ram['swap_used_gb']}/{ram['swap_total_gb']} GB")

    elapsed = time.perf_counter() - t0
    emit(f"sysprobe done in {elapsed:.1f}s")
    return {
        "captured_at": int(time.time()),
        "elapsed_s":   round(elapsed, 2),
        "os":          sys.platform,
        "arch":        platform.machine() or "",
        "disk": disk,
        "cpu":  cpu,
        "gpus": gpus,
        "ram":  ram,
    }
