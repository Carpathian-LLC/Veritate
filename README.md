<div align="center">

<br/>

# V&nbsp;·&nbsp;E&nbsp;·&nbsp;R&nbsp;·&nbsp;I&nbsp;·&nbsp;T&nbsp;·&nbsp;A&nbsp;·&nbsp;T&nbsp;·&nbsp;E

***"truth"** in Latin*

A hand-coded INT8 / ternary transformer inference engine in C and assembly.<br/>
Every internal activation is tap-able at zero runtime cost.<br/>
Ternary (BitNet b1.58) weights ship 5×&nbsp;smaller `.bin` files at full INT8 throughput.

<br/>

[**Get started**](#get-started) &nbsp;·&nbsp; [**Dashboard**](#the-dashboard) &nbsp;·&nbsp; [**Components**](#components--whats-what) &nbsp;·&nbsp; [**Hardware**](#supported-hardware) &nbsp;·&nbsp; [**Index**](#index)

<br/>

</div>

---

<br/>

## Get started

**One launcher per OS, double-click to run.** The launcher creates its own Python virtual environment, installs every dependency on first launch, and starts the dashboard. Subsequent launches are instant. Everything else (building the C engine, pulling updates, switching release channels, training your first model) is done from the GUI.

<br/>

### Prerequisites

| | Requirement | Notes |
|---|---|---|
| **1** | **Git** | Already on macOS (via Xcode CLT) and most Linux distros. On Windows install [Git for Windows](https://git-scm.com/download/win) or run `winget install Git.Git`. |
| **2** | **Python 3.10+** | The launcher handles everything else (venv, pip install, dashboard launch). On Windows: `winget install Python.Python.3.12`. |
| **3** | **A C compiler** *(optional)* | Only needed for the hand-tuned C engine. The PyTorch backend works without it. See [Compiling the C engine](#compiling-the-c-engine-optional) below. |
| **4** | **CUDA toolkit** *(optional)* | Only for GPU training on NVIDIA. CPU and Apple Silicon work without it. |

[`models/`](./) is gitignored. A fresh clone ships **no weights**. You'll train your first model from the dashboard's [**Training**](#tabs) tab in step 3.

<br/>

---

<br/>

## Step 1 &nbsp;·&nbsp; Clone

```sh
git clone https://github.com/Carpathian-LLC/Veritate.git
cd Veritate
```

That's the only manual install step. The launcher in step 2 handles the rest.

<br/>

---

<br/>

## Step 2 &nbsp;·&nbsp; Launch the dashboard

Double-click the launcher for your OS, or run it from a terminal. First launch creates the venv and installs all Python dependencies (this can take a few minutes — `torch` is ~2 GB). Re-launches are instant.

| OS | Double-click | From terminal |
|---|---|---|
| **Windows** | [`start.bat`](start.bat) | `start.bat` |
| **macOS**   | [`start.command`](start.command) | `./start.command` |
| **Linux**   | *(no GUI launcher)* | `python veritate.py` |

The dashboard opens automatically at **[http://localhost:8001/](http://localhost:8001/)**.

<br/>

To pass flags (custom port, preload a model, skip the engine build) invoke the central entry point directly:

```sh
python veritate.py --port 9000 --threads 8
```

<br/>

[`veritate.py`](veritate.py) is the single entry point. It auto-builds the C engine in the background for your OS and arch, then serves the MRI dashboard. PyTorch is available immediately; the C backend lights up when the build finishes. Watch the [**Logs**](#tabs) tab to see build progress.

<br/>

**Available flags**

| Flag | Description |
|---|---|
| `--model <name>` | Preload a model on startup. Defaults to `auto` (latest in [`models/`](./)). |
| `--port <n>` | HTTP port. Default `8001`. |
| `--threads <n>` | PyTorch CPU threads. `0` auto-picks physical cores capped at 16. |
| `--skip-build` | Skip auto-building the engine. PyTorch only. |
| `--no-browser` | Don't auto-open the dashboard URL in a web browser. |

<br/>

---

<br/>

## Compiling the C engine (optional)

The dashboard auto-builds the C engine in the background once a compiler is installed. PyTorch is available immediately; the C backend lights up when the build finishes. Skip this section if you only want the PyTorch backend.

<br/>

<details>
<summary><h3 style="display:inline">&nbsp;&nbsp; macOS &nbsp;<sub><em>(click to expand)</em></sub></h3></summary>

<br/>

Apple ships clang as part of the Xcode Command Line Tools.

```sh
xcode-select --install
```

Or run the bundled checker, which verifies the toolchain is in place:

```sh
bash veritate_engine/v1/build/setup.sh
```

See [`veritate_engine/v1/build/setup.sh`](veritate_engine/v1/build/setup.sh).

</details>

<br/>

<details>
<summary><h3 style="display:inline">&nbsp;&nbsp; Windows &nbsp;<sub><em>(click to expand)</em></sub></h3></summary>

<br/>

The bundled installer uses `winget` to install LLVM (clang) and NASM if they're missing. Run it from the repo root in Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File veritate_engine\v1\build\setup.ps1
```

If you have PowerShell 7+ installed, this also works:

```powershell
pwsh -ExecutionPolicy Bypass -File veritate_engine\v1\build\setup.ps1
```

After it finishes, **open a new terminal** so the updated `PATH` takes effect, then re-launch the dashboard. See [`veritate_engine/v1/build/setup.ps1`](veritate_engine/v1/build/setup.ps1).

</details>

<br/>

<details>
<summary><h3 style="display:inline">&nbsp;&nbsp; Linux &nbsp;<sub><em>(click to expand)</em></sub></h3></summary>

<br/>

The bundled script auto-detects your package manager (`apt`, `dnf`, `pacman`) and installs clang.

```sh
bash veritate_engine/v1/build/setup.sh
```

See [`veritate_engine/v1/build/setup.sh`](veritate_engine/v1/build/setup.sh).

</details>

<br/>

---

<br/>

## Step 3 &nbsp;·&nbsp; The rest is in the dashboard

From the [**Settings**](#tabs) tab you can:

- **Switch release channel.** Choose Stable (`main`), Experimental, or Development (`dev`). Pull updates with one click. See [Release channels](#release-channels) below.
- **Sync plugins and models.** Independent git remotes for trainers and barebones model definitions.
- **Toggle PyTorch backend mode.** On-demand (default) or always loaded.
- **Re-trigger the engine build** if it failed.

From the [**Training**](#tabs) tab you can:

- Pick a plugin (e.g. `example_plugin`), configure the run, and click **Start**.
- Training writes checkpoints + a per-step CSV to [`models/<name>/`](./).

See [`trainers/readme.md`](trainers/readme.md) for the plugin contract and how to author your own.

<br/>

---

<br/>

## The dashboard

Six tabs at **[http://localhost:8001/](http://localhost:8001/)**.

### Tabs

| Tab | What it does |
|---|---|
| **Generation** | Chat with the model. Watch every byte's activations live across all layers. |
| **Models** | Scrub across saved training checkpoints. Watch the model organise itself over training. Confidence evolution, lens-consistency, grade evaluations. |
| **Training** | Pick a plugin, configure a run, click start. Loss curve, plateau detection, throughput, gradient norm. All live from the trainer's CSV. |
| **Wiki** | Built-in concept docs and build notes. See [`veritate_mri/wiki/`](veritate_mri/wiki/). |
| **Logs** | Engine build output, route errors, runtime status. In-memory ring buffer (latest 1000 entries). |
| **Settings** | Release channels, repo sync, PyTorch backend mode, engine rebuild. |

Tab state persists in the URL hash. Refresh keeps you on the same tab.

<br/>

### Release channels

The Settings tab lets you switch the platform between three release channels. Pick one and click **pull**:

| Channel | Branch | What you get |
|---|---|---|
| **Stable** | [`main`](https://github.com/Carpathian-LLC/Veritate/tree/main) | Canonical baseline. Fast-forward only. |
| **Experimental** | [`experimental`](https://github.com/Carpathian-LLC/Veritate/tree/experimental) | Divergent fork with in-flight features. Versions show `(E)` next to them. |
| **Development** | [`dev`](https://github.com/Carpathian-LLC/Veritate/tree/dev) | Active development line. Newest changes; expect rough edges. |

Plugins and models are tracked as **separate git remotes** with their own sync buttons in the same Settings panel. Pulling the platform doesn't touch your trainers or trained weights.

<br/>

---

<br/>

## Running the C engine directly

Skip this section if you're using the dashboard. [`veritate.py`](veritate.py) already drives the engine. These commands are for benchmarking or scripted inference.

If you didn't let `veritate.py` auto-build, build manually.

<br/>

<details>
<summary><strong>&nbsp;&nbsp; macOS &nbsp;<sub><em>(click to expand)</em></sub></strong></summary>

<br/>

```sh
bash veritate_engine/v1/build/build.sh
```

Binary lands at `veritate_engine/v1/bin/macos/arm64/veritate`. Examples:

```sh
veritate_engine/v1/bin/macos/arm64/veritate chat
veritate_engine/v1/bin/macos/arm64/veritate bench 50 200
```

See [`veritate_engine/v1/build/build.sh`](veritate_engine/v1/build/build.sh).

</details>

<br/>

<details>
<summary><strong>&nbsp;&nbsp; Windows &nbsp;<sub><em>(click to expand)</em></sub></strong></summary>

<br/>

```powershell
veritate_engine\v1\build\build.bat
```

Binary lands at `veritate_engine\v1\bin\windows\x86_64\veritate.exe`:

```powershell
veritate_engine\v1\bin\windows\x86_64\veritate.exe chat
veritate_engine\v1\bin\windows\x86_64\veritate.exe bench 50 200
```

See [`veritate_engine/v1/build/build.bat`](veritate_engine/v1/build/build.bat).

</details>

<br/>

<details>
<summary><strong>&nbsp;&nbsp; Linux &nbsp;<sub><em>(click to expand)</em></sub></strong></summary>

<br/>

```sh
bash veritate_engine/v1/build/build.sh
```

Binary lands at `veritate_engine/v1/bin/linux/x86_64/veritate`:

```sh
veritate_engine/v1/bin/linux/x86_64/veritate chat
veritate_engine/v1/bin/linux/x86_64/veritate bench 50 200
```

See [`veritate_engine/v1/build/build.sh`](veritate_engine/v1/build/build.sh).

</details>

<br/>

---

<br/>

## Components &nbsp;·&nbsp; what's what

Three independent pieces.

| Piece | What it is | Language | Runs on | Output |
|---|---|---|---|---|
| **[Plugins](trainers/readme.md)** | Training scripts + manifests. Each plugin trains, fine-tunes, or distills a model. | PyTorch | GPU | `models/<name>/checkpoints/` |
| **[Inference engine](veritate_engine/)** | Loads converted INT8 or ternary weights, generates text. Hand-written C + architecture-specific assembly. | C + asm | CPU | tokens via stdin/stdout, sub-ms |
| **[Project MRI](veritate_mri/)** | Web app to watch the model think while it generates. Visualization + debugging tool. | Flask + JS | CPU | live UI on [http://localhost:8001](http://localhost:8001) |

<br/>

**The model itself** (the trained weights) is the same regardless of which runtime executes it. PyTorch and the C engine are two different *interpreters* of the same neural network. The C engine is the production target; PyTorch is used for training and for the MRI app's live-streaming fallback.

<br/>

**Models live in [`models/<name>/`](./).** Each model folder is self-contained:

- `config.json`: shape + training hyperparams + canonical step
- `checkpoints/`: PyTorch `.pt` snapshots at every saved step
- `veritate.bin`: exported INT8 or ternary (`.bin` v9 / v11) weights of the canonical step (loaded by the C engine)

**Naming convention:** `<corpus>_<size>_<precision>_<version>` &nbsp;·&nbsp; e.g. `children_classics_80m_bf16_v1`.

Tools take `--model <name>` and resolve paths from `config.json`.

<br/>

---

<br/>

## Independence model

Each subsystem is standalone. None of them launches another. They communicate through files on disk only:

- **[Plugins](trainers/)** write checkpoints + the per-step CSV to `models/<name>/`.
- **[`veritate_mri/`](veritate_mri/)** reads checkpoints + the per-step CSV from `models/<name>/`.
- **[`veritate_engine/`](veritate_engine/)** reads exported `.bin` files from `models/<name>/`.

You can run any combination concurrently. The web server does not start training. Training does not start the web server. Each is its own process.

<br/>

---

<br/>

## What's always on

- **CSV training log.** Every training run writes per-step rows to `models/<name>/train.csv`. Always. There is no flag to disable it. The [**Training**](#tabs) tab polls this file.
- **MRI activation capture.** The web server's PyTorch hooks fire on every forward pass during chat generation. Always. The capture cost is hooked at the framework level, so production has the visibility built in.
- **Run-level logging.** Every run tees its stdout + stderr to a per-run log while still printing live. The first line of every run prints the log path so you can find it later.

<br/>

---

<br/>

## Supported hardware

The launcher detects the host (OS + arch + CPU features) and dispatches per-tier dependency pins, Python version checks, and engine build flags. The C engine compiles every kernel into one binary and selects at runtime via CPUID, so a binary built on an AVX-512 host still runs on an Ivy Bridge CPU — shared TUs are pinned to SSE4.2 baseline so non-kernel code never emits an instruction the host can't run.

| Tier | Hardware | Python | Torch | Training | C engine |
|---|---|---|---|---|---|
| **mac_arm** | Apple Silicon (M1–M4) | 3.10–3.13 | `~=2.11` | MPS + CPU | ARM64 NEON SDOT |
| **mac_intel** | Intel Mac, AVX2+ (Haswell+) | 3.10–3.11 | `~=2.2` (last x86 macOS wheels) | CPU only | AVX2 / VNNI / AVX-512 kernels dispatched at runtime |
| **mac_intel** | Intel Mac, AVX1 only (Ivy Bridge, Mac Pro 2013) | 3.10–3.11 | `~=2.2` | CPU only | scalar kernels + AVX1 (dispatched) |
| **linux_x86** | x86_64 Linux, AVX-512+VNNI | 3.10–3.13 | `~=2.11` | CUDA + CPU | full kernel set |
| **linux_x86** | x86_64 Linux, AVX2 | 3.10–3.13 | `~=2.11` | CUDA + CPU | AVX2 kernels dispatched |
| **linux_arm** | ARM64 Linux (Jetson, Graviton, Pi 4) | 3.10–3.13 | `~=2.11` | CPU (CUDA on Jetson) | NEON SDOT kernels |
| **windows_x86** | x86_64 Windows | 3.10–3.13 | `~=2.11` | CUDA + CPU | full kernel set |
| **GPU (NVIDIA)** | CUDA on any tier above | — | `torch.cuda` | bfloat16 AMP | N/A (CPU engine) |
| **GPU (AMD on macOS)** | Discrete AMD on Intel Mac | — | not supported by torch on macOS | CPU only | planned Metal compute path |

Capability flags (`has_avx2`, `has_avx512vnni`, `can_use_mps`, `can_use_cuda`, etc.) are exposed in the hardware dump for any code that needs to gate features at runtime. See [`dev_documentation/platform/tiers.md`](dev_documentation/platform/tiers.md) for how the tier dispatch works.

<br/>

---

<br/>

## Cross-platform plan

[`build.bat`](veritate_engine/v1/build/build.bat) (Windows) and [`build.sh`](veritate_engine/v1/build/build.sh) (POSIX) compile shared translation units at the architecture's safe baseline (SSE4.2 on x86_64, ARMv8 on arm64) and apply per-kernel ISA flags only to the kernels that need them. `src/dispatch.c` picks at runtime via CPUID so a binary works on every CPU that meets the baseline, no matter how high the compile-time kernel ISA was. All kernels live under [`veritate_engine/v1/kernels/<arch>/`](veritate_engine/v1/kernels/) and [`veritate_engine/v1/kernels/scalar/`](veritate_engine/v1/kernels/scalar/). One binary per OS+arch. No fat universal binary.

| Kernel tier | Hardware | Status |
|---|---|---|
| **scalar C** | Anything | ![done](https://img.shields.io/badge/done-brightgreen) Always built; correctness oracle |
| **x86_64 + AVX2** | Haswell+ (2013+ consumer), every x86 Mac with AVX2 | ![done](https://img.shields.io/badge/done-brightgreen) Matmul, INT4, ternary |
| **x86_64 + AVX-512 + VNNI** | Zen 4+, Sapphire Rapids+, Mac Pro 2019 Xeon W | ![done](https://img.shields.io/badge/done-brightgreen) Full set |
| **x86_64 + AVX1 only** | Ivy Bridge / Sandy Bridge / Mac Pro 2013 | ![partial](https://img.shields.io/badge/partial-yellow) Scalar fallback (AVX1-specific kernels not yet) |
| **ARM64 + NEON SDOT** | Apple M1+, Cortex-A76+, ARMv8.2+dotprod | ![done](https://img.shields.io/badge/done-brightgreen) |
| **ARM64 + NEON only** | Pi 4, older Android | ![partial](https://img.shields.io/badge/partial-yellow) Matmul only |
| **ARM64 + AMX** | M-series Mac | ![planned](https://img.shields.io/badge/planned-lightgrey) Stretch goal |
| **Metal compute (AMD GPU on macOS)** | Mac Pro 2013/2019 FirePro, AMD eGPU | ![planned](https://img.shields.io/badge/planned-lightgrey) Investigating |

<br/>

---

<br/>

## License & ownership

A project by **[Carpathian, LLC](https://carpathian.ai/veritate)**. **Distribution not authorized.**

<br/>

---

<br/>

## Index

| folder | what |
|---|---|
| [`veritate_engine/`](veritate_engine/) | C inference engine. Versioned subtrees (currently [`v1/`](veritate_engine/v1/)) hold `src/`, `kernels/<arch>/`, `build/`, `bin/<os>/<arch>/`, and `engine_versions.json`. |
| [`veritate_mri/`](veritate_mri/) | MRI server, dashboard, [`save.py`](veritate_mri/save.py) (checkpoint + dump suite), [`readers/`](veritate_mri/readers/), [`atlas.py`](veritate_mri/atlas.py). |
| [`veritate_core/`](veritate_core/) | Python package. `veritate_core.plugin` is the only surface plugins may import. |
| [`trainers/`](trainers/) | Plugin implementations. [`common/`](trainers/common/) for shared helpers, [`corpus/`](trainers/corpus/) for `.bin` training data. See [`trainers/readme.md`](trainers/readme.md). |
| `models/` | One self-contained subdir per model (gitignored). |
| [`documentation/`](documentation/) | Current platform contracts (committed). Subfolders: [`hooks/`](documentation/hooks/), [`kernels/`](documentation/kernels/). |
