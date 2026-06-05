# Multi-Mind (MtM) Benchmark — 2026-05-22

## Hardware

- GPU: NVIDIA GeForce RTX 5070
- VRAM total: 11.94 GB
- CPU count: 16
- RAM total: 31.16 GB
- torch: 2.11.0+cu128  | seed: 0

## 1. Training throughput (batch=32, seq=512, bias_mode=True, 50 steps)

| hidden | layers | ffn  | params (M) | median ms/step | tokens/sec | note |
|--------|--------|------|------------|---------------:|-----------:|------|
| 128 | 2 | 256 | 1.02 | 46.19 | 354,701 | ok |
| 256 | 4 | 512 | 7.55 | 110.75 | 147,933 | ok |
| 512 | 6 | 1024 | 44.46 | 296.01 | 55,350 | ok |

_Interpretation: best throughput is (128,2,256) at 354,701 tok/s; cost scales near-linearly with params._

## 2. Inference latency — greedy 1-byte decode on (256,4,512), 100 trials

| metric | ms/byte |
|--------|--------:|
| median | 17.48 |
| p95    | 25.07 |

_Interpretation: single-byte decode lands at 17.5 ms median; MoE loop + full-context recompute is the cost driver (no KV cache in this path)._

## 3. Affect probe latency — 256 bytes, 100 trials

| metric | ms/sample |
|--------|----------:|
| median | 0.37 |

_Interpretation: probe is 0.37 ms — cheap enough to run per sample without gating MtM decode._

## 4. End-to-end pipeline (probe + MtM + argmax), 50 trials

| metric | ms |
|--------|---:|
| median | 17.97 |
| p95    | 36.14 |

_Interpretation: e2e median 18.0 ms = probe + MtM + ~0.1 ms glue; probe contributes ~2% of pipeline._

## 5. Memory — (256,4,512), batch=32, seq=512

| metric | value |
|--------|------:|
| peak VRAM (1 train step) | 1426.9 MB |
| peak VRAM (1 inference)  | 77.5 MB |
| state-dict on disk       | 28.8 MB |

_Interpretation: training peak (1427 MB) is dominated by activations + AdamW state; disk footprint (28.8 MB) tracks param count._

## 6. Refractory cost — (256,4,512), batch=32, 30 trials

| refractory_steps | median ms/step |
|------------------|---------------:|
| 0 | 108.95 |
| 4 | 110.38 |

_Interpretation: refractory_steps=4 adds +1.3% per step — within budget for eval-time use, would dominate at long-seq training._

## SHARP take

- Best training config: (128,2,256) at 354,701 tok/s (46.2 ms/step, 1.0M params).
- Best inference config: (256,4,512) greedy decode at 17.5 ms/byte median (p95 25.1); probe overhead is negligible (0.37 ms).
- Biggest perf risk before scale-up: Python MoE expert loop (n_experts * top_k Python branches per FFN) — sets the inference floor and will not amortize at scale; rewrite to grouped-gemm or fused MoE before pushing past ~50M params.
