# engine

Veritate's inference engine. Hand-coded INT8 transformer forward and decode in C, with per-arch SIMD kernels (AVX-512 + VNNI on x86_64, NEON SDOT on ARM64). Byte-level vocabulary (256 tokens, raw bytes in and out). Glass-box trace protocol that exposes every per-token activation to the MRI dashboard at zero runtime cost.

## layout

```
veritate_engine/
├── v1/                                  # primary engine
│   ├── src/{main,model,dispatch,...}.c
│   ├── kernels/{scalar,arm64,x86_64,inline}/
│   ├── build/{build.sh,build.bat,setup.*}
│   └── bin/<os>/<arch>/veritate
└── v2/                                  # empty scratchpad (see v2.md)
```

`v1/` is the only engine that builds and ships. `v2/` is reserved for future experiments. See `v1.md` for engine details and `v2.md` for the scratchpad note.

## supported `.bin` formats

The engine reads any `VRTE` `.bin` whose version is in this table. Format selection is automatic from the magic + version bytes; no flag needed.

| version | tag | weights | scale | notes |
|---|---|---|---|---|
| v3 | `VERITATE_MODEL_VERSION` | INT8 per tensor | one `scale_q24` per tensor | original |
| v4 | `VERITATE_MODEL_VERSION_INT4` | INT4 packed (2 weights/byte) | per-row `row_q24` | QuaRot-rotated path |
| v5 | `VERITATE_MODEL_VERSION_PERCOL` | INT8 | per output column | finer-grained scale |
| v6 | `VERITATE_MODEL_VERSION_MOD` | INT8 per col | per col | adds per-block MoD gate |
| v8 | `VERITATE_MODEL_VERSION_NORM` | INT8 per col | per col | adds final `n_out` RMSNorm |
| v9 | `VERITATE_MODEL_VERSION_BOOST` | INT8 per col | per col | adds `act_boost` (residual scale 1/2/4). default for INT8 non-MoE `export_checkpoint` |
| v11 | `VERITATE_MODEL_VERSION_QAT` | INT8 per tensor *or* ternary (5 trits/byte) | `scale_q24` (INT8) or per-tensor `gamma_q24` (ternary) | unified post-merge format. header carries `act_boost` + `quant_mode` (INT8 / INT4 / TERNARY) + `n_experts` + `router_topk`. covers ternary-FFN and MoE-routed checkpoints under one version. (v10 was retired: dev assigned it to MoE, experimental assigned it to ternary; v11 supersedes both.) |

## build

```bash
bash veritate_engine/v1/build/build.sh
# -> veritate_engine/v1/bin/<os>/<arch>/veritate
```

The script detects `uname -s/-m` and selects the matching kernel translation units. Apple M1 gets `-mcpu=apple-m1` (NEON + SDOT implied). x86_64 builds with AVX2 + AVX-512 + VNNI. Linux ARM64 uses `-march=armv8.2-a+dotprod`.

## run

```bash
export VERITATE_MODEL_PATH=$(pwd)/models/<name>/veritate.bin
veritate_engine/v1/bin/<os>/<arch>/veritate chat_greedy 200
```

Subcommands: `chat`, `chat_greedy`, `trace`, `bench`. Optional environment overrides: `VERITATE_ACT_BOOST=N`, `VERITATE_MAX_LAYERS=N`. See `v1.md` for the full reference.

## resolving the binary path

```python
from veritate_mri.readers import paths
paths.engine_binary_path()
# -> .../veritate_engine/v1/bin/<os>/<arch>/veritate
```

`paths.engine_binary_path()` is the single source of truth; the dashboard, build runner, and any tool that invokes the engine read from it.
