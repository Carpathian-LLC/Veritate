# sysprobe

## What it is

Cross-platform hardware benchmark suite at [veritate_core/plugin/sysprobe.py](../../../veritate_core/plugin/sysprobe.py). Runs disk write, CPU FLOPs + memory bandwidth, GPU FLOPs on every detected torch accelerator, and RAM headroom. Independent of any model: no weights are built, no corpus is touched. Used by the dashboard's Auto tune modal as the pre-stage before the trainer-specific [bench](../../platform/bench.md).

## How it works

- `sysprobe.run(disk_dir=None, on_progress=None)` → result dict. Total wall-clock ~1-3 s on modern boxes.
- **Disk** ([`_probe_disk`](../../../veritate_core/plugin/sysprobe.py#L51)) — writes `DISK_PAYLOAD_BYTES` (64 MB) sequentially, then does `DISK_RANDOM_IOS` random-offset 4 KB writes. Both use `os.fsync` so the OS page cache doesn't inflate the numbers. Reports `seq_write_mb_s`, `rand_write_mb_s`, `free_gb`.
- **CPU** ([`_probe_cpu`](../../../veritate_core/plugin/sysprobe.py#L92)) — dense `CPU_MATMUL_N × CPU_MATMUL_N` fp32 matmul via numpy (BLAS on x86 = MKL/OpenBLAS, on arm64 = Accelerate/OpenBLAS) for GFLOP/s, plus a `numpy.copy` on a 128 MB buffer for bandwidth. Reports `physical_cores`, `matmul_gflops`, `copy_gb_s`.
- **GPU** ([`_probe_gpu`](../../../veritate_core/plugin/sysprobe.py#L131)) — iterates every device from [`_torch_devices`](../../../veritate_core/plugin/sysprobe.py#L114) (all CUDA cards + MPS when arm64-guarded via [hardware.mps_supported](../../../veritate_core/plugin/hardware.py#L47)) and runs a 2048×2048 fp32 matmul for TFLOP/s. Reports per-device `matmul_tflops` and `vram_total_gb`. A device that raises during probe is skipped, not fatal.
- **RAM** ([`_probe_ram`](../../../veritate_core/plugin/sysprobe.py#L165)) — `psutil.virtual_memory` + `swap_memory`. Reports `total_gb`, `available_gb`, `swap_total_gb`, `swap_used_gb`. Swap presence during training is the trigger the SIGKILL guard in [bench](../../platform/bench.md) is protecting against.

Progress callback (`on_progress`) receives one line per stage; the dashboard's Auto tune modal renders them live above the bench log.

## Dependencies

- `numpy`, `psutil` — always present.
- `torch` — optional; when missing, the GPU stage is skipped and the CPU/disk/RAM probes still run.
- [veritate_core/plugin/hardware.py](../../../veritate_core/plugin/hardware.py) — MPS arm64 guard and `physical_cores()`.

## Consumers

- [`POST /trainers/sysprobe`](../../../veritate_mri/routes/trainers_routes.py#L86) — the runtime endpoint. The Auto tune modal calls it first (before launching bench) so the hardware summary is on screen while the trainer subprocess is still spinning up, and so the same data is available for the `bench_report` upload regardless of whether bench itself completes.
- Result payload is passed to `POST /trainers/tune_defaults` as the `sysprobe` field, which forwards it to [`heartbeat.send_bench_report`](heartbeat.md) alongside the bench summary — Carpathian aggregates real-machine tuning data across the fleet under `analytics_advanced_enabled` consent.

## Pitfalls

- **Disk probe writes to `tempfile.gettempdir()` by default.** On a machine with a slow system drive but a fast NVMe scratch disk, the reported `seq_write_mb_s` will not reflect what the paged optimizer actually gets: pass `disk_dir` matching the paging state dir when calling from a context that knows where paging lands.
- **CPU numbers depend on the numpy BLAS backend.** Anaconda ships MKL by default; a bare `pip install numpy` on x86 uses OpenBLAS. Cross-machine comparisons only make sense inside the same distribution.
- **GPU matmul is fp32.** A card that lists a TF32/bf16-adjusted spec sheet will look slower here than the vendor's number; that's intentional — the training loop is dominated by fp32 gradient math, so an fp32 probe predicts training tok/s better than a marketing FLOPs figure.
