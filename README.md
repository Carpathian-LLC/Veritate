# Veritate

> *"truth"* in Latin

A hand-coded INT8 transformer inference engine in C and assembly. Every internal activation is tap-able at zero runtime cost.

---

## Components: what's what

Three independent pieces.

| Piece                | What it is                                                                                          | Language    | Runs on  | Output                            |
|----------------------|-----------------------------------------------------------------------------------------------------|-------------|----------|-----------------------------------|
| **Plugins**          | Training scripts + manifests. Each plugin trains, fine-tunes, or distills a model.                 | PyTorch     | GPU      | `models/<name>/checkpoints/`      |
| **Inference engine** | Loads converted INT8 weights, generates text. Hand-written C + architecture-specific assembly.      | C + asm     | CPU      | tokens via stdin/stdout, sub-ms   |
| **Project MRI**      | Web app to watch the model think while it generates. Visualization + debugging tool.                | Flask + JS  | CPU      | live UI on `http://localhost:8001`|

**The model itself** (the trained weights) is the same regardless of which runtime executes it. PyTorch and the C engine are two different *interpreters* of the same neural network. The C engine is the production target; PyTorch is used for training and for the MRI app's live-streaming fallback.

**Models live in `models/<name>/`.** Each model folder is self-contained: a `config.json` (shape + training hyperparams + canonical step), a `checkpoints/` directory with PyTorch `.pt` snapshots at every saved step, and a single `veritate.bin` (exported INT8 weights of the canonical step) that the C engine loads. Naming convention: `<corpus>_<size>_<precision>_<version>`, e.g. `children_classics_80m_bf16_v1`. Tools take `--model <name>` and resolve paths from `config.json`.

---

## Independence model

Each subsystem is standalone. None of them launches another. They communicate through files on disk only:

- Plugins write checkpoints + the per-step CSV to `models/<name>/`.
- `veritate_mri/` reads checkpoints + the per-step CSV from `models/<name>/`.
- `veritate_engine/` reads exported `.bin` files from `models/<name>/`.

You can run any combination concurrently. The web server does not start training. Training does not start the web server. Each is its own process.

---

## What's always on

- **CSV training log.** Every training run writes per-step rows to `models/<name>/train.csv`. Always. There is no flag to disable it. The Live Training tab polls this file.
- **MRI activation capture.** The web server's PyTorch hooks fire on every forward pass during chat generation. Always. The capture cost is hooked at the framework level, so production has the visibility built in.
- **Run-level logging.** Every run tees its stdout + stderr to a per-run log while still printing live. The first line of every run prints the log path so you can find it later.

---

## Quickstart

```powershell
# one-time setup (Windows)
pwsh veritate_engine/build/setup.ps1

# start the MRI dashboard (auto-builds the engine in the background, then serves)
py run.py --model <name> --port 8001 --threads 8

# build the C engine manually (Windows)
veritate_engine\build\build.bat

# build the C engine manually (POSIX)
bash veritate_engine/build/build.sh

# run the C engine directly (chat, bench)
veritate_engine\bin\windows\x86_64\veritate.exe
veritate_engine\bin\windows\x86_64\veritate.exe bench 50 200
veritate_engine\bin\windows\x86_64\veritate.exe chat
```

Open `http://localhost:8001/` in your browser. Three tabs:

- **Generation**: chat with the model, watch every byte's activations live.
- **Learning**: scrub across saved training checkpoints, watch the model organise itself.
- **Live Training**: poll the trainer's CSV in real time. Loss curve, plateau detection, throughput, gradient norm.

Tab state persists in the URL hash, so refresh keeps you on the same tab.

Training is plugin-driven. Open the Training tab in the dashboard, pick a plugin, configure the run, and click start. See `plugins/readme.md` for the plugin contract and how to write one.

---

## Supported hardware

| Hardware | Training (PyTorch) | Inference (C engine) |
|---|---|---|
| NVIDIA GPU (CUDA) | yes — CUDA + bfloat16 AMP | N/A (CPU engine) |
| Apple M-series (MPS) | yes — fp32 only, MPS backend, no AMP | ARM64 NEON SDOT (M1+); AMX empty |
| x86_64 CPU (AVX-512 + VNNI) | yes — CPU fallback | primary target (Zen 4+, Sapphire Rapids+) |
| x86_64 CPU (AVX2) | yes — CPU fallback | matmul done; transformer hot-path in progress |
| ARM64 CPU (NEON only) | yes — CPU fallback | matmul only |

---

## Cross-platform plan

Windows is the primary platform today. The structure is set up to add macOS and Linux as ports land. See `documentation/kernels/platforms.md` for the tier matrix, per-platform bench targets, and the function-pointer contract.

| Tier | Hardware | Status |
|---|---|---|
| x86_64 + AVX-512 + VNNI | Ryzen 9800X3D, Zen 4+, Sapphire Rapids+ | Done. Primary target. |
| x86_64 + AVX2 | Intel Haswell through Ice Lake, 2013+ consumer x86 | Matmul done; transformer hot-path not yet. |
| ARM64 + NEON SDOT | Apple M1+, Cortex-A76+ | Initial port landed (matmul + transformer hot-path). Bench TBD. |
| ARM64 + NEON only | Pi 4, older Android | Matmul only; transformer rides NEON SDOT translation units. |
| ARM64 + AMX | M-series Mac | Empty. Stretch goal. |
| scalar C | Anything | Matmul done; correctness oracle for all other tiers. |

`build.bat` (Windows) and `build.sh` (POSIX) drive identical compile commands per OS. `build.sh` detects host via `uname -s/-m` and picks the matching kernel translation units. All kernels live under `veritate_engine/kernels/<arch>/` and `veritate_engine/kernels/scalar/`. One binary per OS+arch; no fat universal binary.

---

## License & ownership

A project by [Carpathian, LLC](https://carpathian.ai/veritate). Distribution not authorized.

---

## Index

| folder | what |
|---|---|
| `veritate_engine/` | C inference engine. `src/`, `kernels/<arch>/`, `build/`, `bin/<os>/<arch>/`, `engine_versions.json`. |
| `veritate_mri/` | MRI server, dashboard, `save.py` (checkpoint + dump suite), `readers/`, `atlas.py`. |
| `veritate/` | Python package. `veritate.plugin` is the only surface plugins may import. |
| `plugins/` | Plugin implementations. `common/` for shared helpers, `corpus/` for `.bin` training data. See `plugins/readme.md`. |
| `models/` | One self-contained subdir per model (gitignored). See `docs/training/storage.md`. |
| `documentation/` | Current platform contracts (committed). Subfolders: `hooks/`, `kernels/`, `plugins/`. |
| `docs/` | Papers, plans, results, notes, workbook, agents (gitignored). See `docs/index.md`. |
