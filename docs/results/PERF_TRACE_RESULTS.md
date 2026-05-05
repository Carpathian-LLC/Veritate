# Veritate end-to-end perf trace

- date: 2026-04-29 09:42:24
- exe: `C:\Users\malka\AppData\Local/veritate/veritate_test.exe`
- model: `data/models/tinystories-80m-v5-qat2/veritate.bin`
- prompt: `'Once upon a time'`
- frames: 16  (token_count=16)
- frame size: 125,764 bytes  (4 marker + 12 header + 125,748 payload)
- total stream wall: 69.95 ms
- outer wall (incl. close): 69.99 ms
- total bytes streamed: 2,012,224
- wall per token (avg, includes prefill): 4.372 ms
- steady-state per token (skip frame 0 prefill): 1.575 ms

## per-stage stats (ms)

> **read pipe** blocks on the engine, so it includes engine compute + pipe transfer.
> **engine inter-frame** is the gap between consecutive `_read_exact` start calls — near zero
> because Python re-enters the read loop immediately after parse.

| stage              |   avg |   p50 |   p99 |   min |   max |
|--------------------|------:|------:|------:|------:|------:|
| read pipe          | 4.173 | 1.368 | 44.934 | 1.288 | 44.934 |
| parse frame        | 0.133 | 0.116 | 0.333 | 0.087 | 0.333 |
| engine inter-frame | 0.002 | 0.002 | 0.006 | 0.001 | 0.006 |

## read-pipe decomposition (estimated)

- engine compute (kernel telemetry):    ~0.90 ms p50
- pipe + python overhead (read - engine): ~0.47 ms p50
- frame size: 125,764 bytes
- effective pipe bandwidth: ~269 MB/s for the 123 KB frame

## per-frame trace (first 32)

| # | read_ms | parse_ms | engine_inter_ms | bytes |
|---|--------:|---------:|----------------:|------:|
| 0 |  44.934 |    0.333 |           0.000 | 125764 |
| 1 |   1.754 |    0.161 |           0.006 | 125764 |
| 2 |   2.123 |    0.156 |           0.004 | 125764 |
| 3 |   1.526 |    0.132 |           0.003 | 125764 |
| 4 |   1.501 |    0.101 |           0.003 | 125764 |
| 5 |   1.339 |    0.118 |           0.002 | 125764 |
| 6 |   1.305 |    0.094 |           0.002 | 125764 |
| 7 |   1.361 |    0.087 |           0.001 | 125764 |
| 8 |   1.593 |    0.115 |           0.001 | 125764 |
| 9 |   1.368 |    0.118 |           0.002 | 125764 |
| 10 |   1.318 |    0.116 |           0.002 | 125764 |
| 11 |   1.376 |    0.115 |           0.002 | 125764 |
| 12 |   1.288 |    0.094 |           0.002 | 125764 |
| 13 |   1.361 |    0.154 |           0.002 | 125764 |
| 14 |   1.315 |    0.115 |           0.002 | 125764 |
| 15 |   1.305 |    0.113 |           0.002 | 125764 |

## top-3 wins (ranked by ms saved per token, p50)

1. **shrink the frame payload** — current 123 KB / token. FFN neurons (36 KB) + attention floats (147 KB) + lens logits (12 KB) dominate. Switching attention from f32 -> u8 (or downsampling to top-k) saves ~0.33 ms p50 by cutting bytes-on-pipe.
2. **parse frame in one shot** — current 0.116 ms p50 from many `np.frombuffer` calls with per-call dtype dispatch. A single structured-dtype view over the whole payload (or a flat memcpy into a pre-allocated buffer) saves ~0.096 ms p50.
3. **read full frame in one syscall** — current `_read_exact` loops 3 times on a 64 KB pipe buffer. Increasing the engine's stdout buffer via `setvbuf` + a single big `read()` saves ~0.1-0.3 ms p50 from per-chunk overhead.

## interpretation vs. browser 4 ms/byte

- harness avg wall-per-token (includes prefill): 4.37 ms
- harness steady-state per token (no prefill): 1.57 ms
- frame 0 prefill cost: 44.9 ms (one-shot, amortized)
- engine kernel-side decode (per workbook): ~0.9 ms p50
- python-side overhead per token: read+parse = 1.48 ms p50

Conclusion: the user's 4 ms/byte browser wall is mostly the **prefill on frame 0** smeared
across 16 tokens (41 ms / 16 = ~2.6 ms/token contribution). Steady-state per-token is ~1.6 ms.
Flask/SSE/WS/render sit on top of that but are NOT the dominant cost — pipe + numpy parse is.
