<div align="center">

<br/>

# &nbsp; V &nbsp;·&nbsp; E &nbsp;·&nbsp; R &nbsp;·&nbsp; I &nbsp;·&nbsp; T &nbsp;·&nbsp; A &nbsp;·&nbsp; T &nbsp;·&nbsp; E &nbsp;

***"truth"** in Latin*

A hand-coded INT8 transformer inference engine in C and assembly.<br/>
Every internal activation is tap-able at zero runtime cost.

<br/>

[**Get started**](#get-started) &nbsp;·&nbsp; [**Dashboard**](#the-dashboard) &nbsp;·&nbsp; [**Components**](#components--whats-what) &nbsp;·&nbsp; [**Hardware**](#supported-hardware) &nbsp;·&nbsp; [**Index**](#index)

<br/>

</div>

---

<br/>

## Get started

**Setup happens in the dashboard.** Two short CLI commands to install Python and start the server. Everything else (building the C engine, pulling updates, switching release channels, training your first model) is done from the GUI.

<br/>

### Prerequisites

| | Requirement | Notes |
|---|---|---|
| **1** | **Python 3.10+** | For the dashboard, training plugins, and PyTorch fallback. |
| **2** | **A C compiler** | The dashboard builds the C engine for you. You just need `clang` (macOS / Linux) or MSVC / `clang` (Windows) installed. See the per-OS install steps below. |
| **3** | **CUDA toolkit** *(optional)* | Only for GPU training on NVIDIA. CPU and Apple Silicon work without it. |

[`models/`](./) is gitignored. A fresh clone ships **no weights**. You'll train your first model from the dashboard's [**Training**](#tabs) tab in step 4.

<br/>

---

<br/>

## Step 1 &nbsp;·&nbsp; Install Python deps

Same on every platform.

```sh
git clone https://github.com/Carpathian-LLC/Veritate.git
cd Veritate
pip install -r requirements.txt
```

<br/>

---

<br/>

## Steps 2 &amp; 3 &nbsp;·&nbsp; Install and run &nbsp;<sub>*(per platform)*</sub>

**Click your platform below to expand its install steps.** Each block contains the C compiler install plus the dashboard launch command for that OS.

<br/>

<details>
<summary><h3 style="display:inline">&nbsp;&nbsp; macOS &nbsp;<sub><em>(click to expand)</em></sub></h3></summary>

<br/>

#### Step 2 &nbsp;·&nbsp; Install a C compiler &nbsp;<sub>*(one-time)*</sub>

Apple ships clang as part of the Xcode Command Line Tools.

```sh
xcode-select --install
```

Or run the bundled checker, which verifies the toolchain is in place:

```sh
bash veritate_engine/build/setup.sh
```

See [`veritate_engine/build/setup.sh`](veritate_engine/build/setup.sh).

<br/>

#### Step 3 &nbsp;·&nbsp; Start the dashboard

```sh
python run.py --port 8001 --threads 8
```

Then open **[http://localhost:8001/](http://localhost:8001/)** in your browser.

</details>

<br/>

<details>
<summary><h3 style="display:inline">&nbsp;&nbsp; Windows &nbsp;<sub><em>(click to expand)</em></sub></h3></summary>

<br/>

#### Step 2 &nbsp;·&nbsp; Install a C compiler &nbsp;<sub>*(one-time)*</sub>

The bundled installer checks for and installs the toolchain:

```powershell
pwsh veritate_engine/build/setup.ps1
```

See [`veritate_engine/build/setup.ps1`](veritate_engine/build/setup.ps1).

<br/>

#### Step 3 &nbsp;·&nbsp; Start the dashboard

```powershell
py run.py --port 8001 --threads 8
```

Then open **[http://localhost:8001/](http://localhost:8001/)** in your browser.

</details>

<br/>

<details>
<summary><h3 style="display:inline">&nbsp;&nbsp; Linux &nbsp;<sub><em>(click to expand)</em></sub></h3></summary>

<br/>

#### Step 2 &nbsp;·&nbsp; Install a C compiler &nbsp;<sub>*(one-time)*</sub>

The bundled script auto-detects your package manager (`apt`, `dnf`, `pacman`) and installs clang.

```sh
bash veritate_engine/build/setup.sh
```

See [`veritate_engine/build/setup.sh`](veritate_engine/build/setup.sh).

<br/>

#### Step 3 &nbsp;·&nbsp; Start the dashboard

```sh
python run.py --port 8001 --threads 8
```

Then open **[http://localhost:8001/](http://localhost:8001/)** in your browser.

</details>

<br/>

**Skip the C compiler** if you only want the PyTorch backend. The dashboard still works; the C engine just won't be available.

<br/>

[`run.py`](run.py) auto-builds the C engine in the background for your OS and arch, then serves the MRI dashboard. PyTorch is available immediately; the C backend lights up when the build finishes. Watch the [**Logs**](#tabs) tab to see the build progress.

<br/>

**Available flags**

| Flag | Description |
|---|---|
| `--model <name>` | Preload a model on startup. Defaults to `auto` (latest in [`models/`](./)). |
| `--port <n>` | HTTP port. Default `8001`. |
| `--threads <n>` | PyTorch CPU threads. `0` auto-picks physical cores capped at 16. |
| `--skip-build` | Skip auto-building the engine. PyTorch only. |

<br/>

---

<br/>

## Step 4 &nbsp;·&nbsp; The rest is in the dashboard

From the [**Settings**](#tabs) tab you can:

- **Switch release channel.** Choose Stable (`main`), Experimental, or Development (`dev`). Pull updates with one click. See [Release channels](#release-channels) below.
- **Sync plugins and models.** Independent git remotes for trainers and barebones model definitions.
- **Toggle PyTorch backend mode.** On-demand (default) or always loaded.
- **Re-trigger the engine build** if it failed.

From the [**Training**](#tabs) tab you can:

- Pick a plugin (e.g. `example_plugin`), configure the run, and click **Start**.
- Training writes checkpoints + a per-step CSV to [`models/<name>/`](./).

See [`plugins/readme.md`](plugins/readme.md) for the plugin contract and how to author your own.

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

Skip this section if you're using the dashboard. [`run.py`](run.py) already drives the engine. These commands are for benchmarking or scripted inference.

If you didn't let `run.py` auto-build, build manually.

<br/>

<details>
<summary><strong>&nbsp;&nbsp; macOS &nbsp;<sub><em>(click to expand)</em></sub></strong></summary>

<br/>

```sh
bash veritate_engine/build/build.sh
```

Binary lands at `veritate_engine/bin/macos/arm64/veritate`. Examples:

```sh
veritate_engine/bin/macos/arm64/veritate chat
veritate_engine/bin/macos/arm64/veritate bench 50 200
```

See [`veritate_engine/build/build.sh`](veritate_engine/build/build.sh).

</details>

<br/>

<details>
<summary><strong>&nbsp;&nbsp; Windows &nbsp;<sub><em>(click to expand)</em></sub></strong></summary>

<br/>

```powershell
veritate_engine\build\build.bat
```

Binary lands at `veritate_engine\bin\windows\x86_64\veritate.exe`:

```powershell
veritate_engine\bin\windows\x86_64\veritate.exe chat
veritate_engine\bin\windows\x86_64\veritate.exe bench 50 200
```

See [`veritate_engine/build/build.bat`](veritate_engine/build/build.bat).

</details>

<br/>

<details>
<summary><strong>&nbsp;&nbsp; Linux &nbsp;<sub><em>(click to expand)</em></sub></strong></summary>

<br/>

```sh
bash veritate_engine/build/build.sh
```

Binary lands at `veritate_engine/bin/linux/x86_64/veritate`:

```sh
veritate_engine/bin/linux/x86_64/veritate chat
veritate_engine/bin/linux/x86_64/veritate bench 50 200
```

See [`veritate_engine/build/build.sh`](veritate_engine/build/build.sh).

</details>

<br/>

---

<br/>

## Components &nbsp;·&nbsp; what's what

Three independent pieces.

| Piece | What it is | Language | Runs on | Output |
|---|---|---|---|---|
| **[Plugins](plugins/readme.md)** | Training scripts + manifests. Each plugin trains, fine-tunes, or distills a model. | PyTorch | GPU | `models/<name>/checkpoints/` |
| **[Inference engine](veritate_engine/)** | Loads converted INT8 weights, generates text. Hand-written C + architecture-specific assembly. | C + asm | CPU | tokens via stdin/stdout, sub-ms |
| **[Project MRI](veritate_mri/)** | Web app to watch the model think while it generates. Visualization + debugging tool. | Flask + JS | CPU | live UI on [http://localhost:8001](http://localhost:8001) |

<br/>

**The model itself** (the trained weights) is the same regardless of which runtime executes it. PyTorch and the C engine are two different *interpreters* of the same neural network. The C engine is the production target; PyTorch is used for training and for the MRI app's live-streaming fallback.

<br/>

**Models live in [`models/<name>/`](./).** Each model folder is self-contained:

- `config.json`: shape + training hyperparams + canonical step
- `checkpoints/`: PyTorch `.pt` snapshots at every saved step
- `veritate.bin`: exported INT8 weights of the canonical step (loaded by the C engine)

**Naming convention:** `<corpus>_<size>_<precision>_<version>` &nbsp;·&nbsp; e.g. `children_classics_80m_bf16_v1`.

Tools take `--model <name>` and resolve paths from `config.json`.

<br/>

---

<br/>

## Independence model

Each subsystem is standalone. None of them launches another. They communicate through files on disk only:

- **[Plugins](plugins/)** write checkpoints + the per-step CSV to `models/<name>/`.
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

| Hardware | Training (PyTorch) | Inference (C engine) |
|---|---|---|
| **NVIDIA GPU** (CUDA) | yes (CUDA + bfloat16 AMP) | N/A (CPU engine) |
| **Apple M-series** (MPS) | yes (fp32 only, MPS backend, no AMP) | ARM64 NEON SDOT (M1+); AMX empty |
| **x86_64 CPU** (AVX-512 + VNNI) | yes (CPU fallback) | primary target (Zen 4+, Sapphire Rapids+) |
| **x86_64 CPU** (AVX2) | yes (CPU fallback) | matmul done; transformer hot-path in progress |
| **ARM64 CPU** (NEON only) | yes (CPU fallback) | matmul only |

<br/>

---

<br/>

## Cross-platform plan

Windows is the primary platform today. The structure is set up to add macOS and Linux as ports land. See [`documentation/kernels/platforms.md`](documentation/kernels/platforms.md) for the tier matrix, per-platform bench targets, and the function-pointer contract.

| Tier | Hardware | Status |
|---|---|---|
| **x86_64 + AVX-512 + VNNI** | Ryzen 9800X3D, Zen 4+, Sapphire Rapids+ | ![done](https://img.shields.io/badge/done-brightgreen) Primary target |
| **x86_64 + AVX2** | Intel Haswell through Ice Lake, 2013+ consumer x86 | ![partial](https://img.shields.io/badge/partial-yellow) Matmul done; transformer hot-path not yet |
| **ARM64 + NEON SDOT** | Apple M1+, Cortex-A76+ | ![done](https://img.shields.io/badge/done-brightgreen) Initial port landed; bench TBD |
| **ARM64 + NEON only** | Pi 4, older Android | ![partial](https://img.shields.io/badge/partial-yellow) Matmul only |
| **ARM64 + AMX** | M-series Mac | ![planned](https://img.shields.io/badge/planned-lightgrey) Stretch goal |
| **scalar C** | Anything | ![done](https://img.shields.io/badge/done-brightgreen) Correctness oracle for all other tiers |

[`build.bat`](veritate_engine/build/build.bat) (Windows) and [`build.sh`](veritate_engine/build/build.sh) (POSIX) drive identical compile commands per OS. `build.sh` detects host via `uname -s/-m` and picks the matching kernel translation units. All kernels live under [`veritate_engine/kernels/<arch>/`](veritate_engine/kernels/) and [`veritate_engine/kernels/scalar/`](veritate_engine/kernels/scalar/). One binary per OS+arch. No fat universal binary.

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
| [`veritate_engine/`](veritate_engine/) | C inference engine. `src/`, `kernels/<arch>/`, `build/`, `bin/<os>/<arch>/`, `engine_versions.json`. |
| [`veritate_mri/`](veritate_mri/) | MRI server, dashboard, [`save.py`](veritate_mri/save.py) (checkpoint + dump suite), [`readers/`](veritate_mri/readers/), [`atlas.py`](veritate_mri/atlas.py). |
| [`veritate/`](veritate/) | Python package. `veritate.plugin` is the only surface plugins may import. |
| [`plugins/`](plugins/) | Plugin implementations. [`common/`](plugins/common/) for shared helpers, [`corpus/`](plugins/corpus/) for `.bin` training data. See [`plugins/readme.md`](plugins/readme.md). |
| `models/` | One self-contained subdir per model (gitignored). |
| [`documentation/`](documentation/) | Current platform contracts (committed). Subfolders: [`hooks/`](documentation/hooks/), [`kernels/`](documentation/kernels/). |
