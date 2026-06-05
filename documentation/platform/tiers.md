# Hardware tiers

The launcher detects the host and dispatches per-tier dependency pins. Veritate's mission requires runnability on older consumer hardware, so we don't bail on Intel Macs etc. — we just install the right torch for what each tier can run.

## Tier matrix

| Tier | OS × arch | Python | Torch | Compute |
|---|---|---|---|---|
| `mac_arm` | macOS · arm64 (M-series) | 3.10–3.13 | `~=2.11` | MPS + CPU |
| `mac_intel` | macOS · x86_64 | 3.10–3.11 | `~=2.2` | CPU only (no MPS) |
| `linux_x86` | Linux · x86_64 | 3.10–3.13 | `~=2.11` | CUDA + CPU |
| `linux_arm` | Linux · arm64/aarch64 | 3.10–3.13 | `~=2.11` | CPU (CUDA on Jetson) |
| `windows_x86` | Windows · x86_64 | 3.10–3.13 | `~=2.11` | CUDA + CPU |

## Where it's wired

- **Detection:** [veritate.py:_detect_tier](../../veritate.py) — returns the tier label from `sys.platform` and `platform.machine()`.
- **Python gate:** [veritate.py:_ensure_venv_and_deps](../../veritate.py) — refuses to proceed if `sys.version_info` is outside the tier's `TIER_PYTHON_RANGE`. Prints the exact `brew install python@X.Y` (or distro equivalent) command to fix.
- **Dependency pin:** [requirements.txt](../../requirements.txt) — uses pip environment markers (`sys_platform`, `platform_machine`) so the right torch and numpy lines activate per host.
- **Runtime tier flag:** the launcher exports `VERITATE_TIER` into the re-exec env. Runtime code that needs to feature-gate reads `os.environ.get("VERITATE_TIER")`.

## Why `mac_intel` is special

PyTorch dropped Intel macOS wheels at torch 2.3. The last build with `macosx_*_x86_64.whl` was **torch 2.2.2** (March 2024). So Intel Mac is pinned to `torch~=2.2` forever.

torch 2.2 lacks two non-essential niceties: the `torch.amp.autocast` unified namespace (2.3+) and the maturity gains in `torch.compile` (2.4+). The code uses the long-standing `torch.autocast` spelling everywhere so 2.2 and 2.11+ share one path, and `torch.compile` is gated on `torch.__version__ >= (2, 4)`.

What's lost on `mac_intel`:
- **No MPS.** Apple's MPS backend only works on Apple Silicon. Intel Macs run entirely on CPU.
- **No flash / mem-efficient SDPA kernels.** `F.scaled_dot_product_attention` falls back to the math kernel on CPU in torch 2.2. Correctness preserved, throughput is much lower.
- **No `torch.compile`.** Hard-disabled on this tier.

Net: Intel Macs are useful for **inference of small models** and **running the dashboard**. Training a model larger than ~10M params is impractically slow.

## Adding a new tier

1. Add the label constant in `veritate.py`.
2. Add the Python range to `TIER_PYTHON_RANGE`.
3. Add the install hint in `_tier_install_hint`.
4. Add the platform branch to `_detect_tier`.
5. Add per-tier requirement lines (if pin differs) using pip env markers.
6. Run the torch-API audit (see `dev_documentation/platform/` for the audit prompt) before relaxing pins downward.
