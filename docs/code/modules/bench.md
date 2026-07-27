# Benchmark

Single-file harness: `ds4_bench.c` — no header.

## Purpose

Measure inference throughput across a sweep of context lengths. Reports prefill speed, decode speed, and time-to-first-token per context frontier.

## How It Works (Frontier Walk)

The benchmark walks a single fixed token sequence through configurable context frontiers, measuring only the newest prefill interval at each step.

1. **Tokenize** a long prompt (must be ≥ `--ctx-max` tokens).
2. **Start** at `--ctx-start` tokens.
3. **Prefill** the prompt prefix up to the current frontier → measure prefill time.
4. **Generate** `--gen-tokens` greedily via `ds4_session_argmax_excluding` (argmax, EOS excluded) → measure decode time.
5. **Snapshot** the live session in memory (if payload ≤ 1 GiB default threshold).
6. **Restore** snapshot or replay prefix to reset state for the next frontier.
7. **Advance** to the next frontier via `--step-incr` (additive) or `--step-mul` (multiplicative).
8. **Repeat** until `--ctx-max` is reached.

Key design points:

- **Only the newest prefill interval** is measured at each step — `prefill_tokens = frontier - previous`. The old prefix is reused.
- **Snapshot save/restore time** is intentionally outside both timing windows.
- **Single pass per frontier** — no repeated trials, no statistics.
- **Snapshot-based state restore** when payload ≤ snapshot max bytes; falls back to prefix replay otherwise.
- **Distributed mode** always replays prefix (no snapshot/restore).
- **Timing** via `CLOCK_MONOTONIC` (`clock_gettime`), not wall-clock.
- **Backend-agnostic** — same harness for Metal, CUDA, ROCm, CPU.

### Frontier Step Logic

```c
// If step_mul == 1.0:  next = cur + step_incr
// If step_mul > 1.0:   next = ceil(cur * step_mul)
// Clamped to ctx_max.
```

## What It Measures (CSV Columns)

Header: `ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes`

| Column | Unit | Description |
|---|---|---|
| `ctx_tokens` | tokens | Total context length at this frontier |
| `prefill_tokens` | tokens | New tokens prefilled this step (`frontier - previous`) |
| `prefill_tps` | tok/s | Prefill throughput (`prefill_tokens / prefill_time`) |
| `gen_tokens` | tokens | Tokens generated this step |
| `gen_tps` | tok/s | Overall generation throughput including first token |
| `gen_first_ms` | ms | Time-to-first-token (first decode step latency) |
| `gen_steady_tokens` | tokens | Tokens after first (`gen_tokens - 1`, or 0 if only 1) |
| `gen_steady_tps` | tok/s | Steady-state decode throughput excluding first token |
| `kvcache_bytes` | bytes | Snapshot payload size at this frontier (0 if no snapshot taken) |

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `-m`, `--model` | `ds4flash.gguf` | Model file path |
| `--prompt-file` | — | Plain-text prompt file (mutually exclusive with `--chat-prompt-file`) |
| `--chat-prompt-file` | — | Chat-style prompt file, processed through chat template (mutually exclusive with `--prompt-file`) |
| `-sys`, `--system` | `"You are a helpful assistant."` | System prompt for chat mode |
| `--ctx-start` | `2048` | Starting context length (tokens) |
| `--ctx-max` | `32768` | Maximum context length (tokens) |
| `--ctx-alloc` | `ctx_max + gen_tokens + 1` | Allocated session context size (must exceed `ctx_max + gen_tokens`) |
| `--step-incr` | `2048` | Additive frontier step (tokens); used when `--step-mul` is 1.0 |
| `--step-mul` | `1.0` | Multiplicative frontier step factor (≥ 1.0); overrides `--step-incr` when > 1.0 |
| `--gen-tokens`, `--tokens`, `-n` | `128` | Tokens to generate per frontier (0 to skip generation) |
| `--csv` | stdout | Output CSV file path |
| `--dump-frontier-logits-dir` | — | Directory for per-frontier logit JSON dumps |
| `--expert-profile` | — | Expert profile file path |
| `-t`, `--threads` | — | CPU thread count |
| `--backend` | auto | Backend name: `metal`, `cuda`, `rocm`, `cpu` |
| `--metal` | — | Select Metal backend |
| `--cuda` / `--rocm` | — | Select CUDA or ROCm backend |
| `--cpu` | — | Select CPU backend |
| `--gpu-vram` | — | GPU VRAM allocation (e.g., `auto`, `<N>GB`) |
| `--gpu-devices` | — | GPU device list |
| `--cuda-tensor-parallel` | — | Enable CUDA tensor parallelism |
| `--quality` | — | Quality mode (higher precision) |
| `--ssd-streaming` | — | Enable SSD streaming |
| `--ssd-streaming-cold` | — | Cold-start SSD streaming (no cache pre-warming) |
| `--ssd-streaming-cache-experts` | — | Expert cache size: count or `<N>GB` |
| `--ssd-streaming-full-layers` | — | Number of full layers to keep in memory |
| `--ssd-streaming-preload-experts` | — | Number of experts to preload |
| `--simulate-used-memory` | — | Simulate used memory (e.g., `64GB`) |
| `--prefill-chunk` | — | Prefill chunk size (tokens) |
| `--power` | — | Power limit percent (1–100) |
| `--warm-weights` | — | Warm model weights before benchmark |
| `--show-output` | — | Print decoded generation text to stderr |
| `-h`, `--help` | — | Show help (optional topic filter as argument) |
| *(distributed flags)* | — | See `ds4_dist_parse_cli_arg` for `--dist-*` flags |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DS4_BENCH_SNAPSHOT_MAX_BYTES` | `1<<30` (1 GiB) | Max session payload for snapshot. Set to `"unlimited"` or `"inf"` to allow any size. |
| `DS4_BENCH_DISABLE_SNAPSHOT` | unset | If set, always replay prefix instead of snapshot/restore |
| `DS4_BENCH_FORCE_SNAPSHOT` | unset | If set, snapshot even if payload exceeds the max bytes limit |

## Limitations

- **Single pass per frontier** — no variance or statistics reported.
- **No warmup runs** — first prefill at each frontier includes cold-start overhead.
- **Snapshot size threshold** (1 GiB default) may prevent snapshot on large models, falling back to slower prefix replay.
- **Only greedy argmax generation** — no sampling, top-k, top-p, temperature.
- **Prompt length requirement** — prompt must be long enough to reach `--ctx-max`; benchmark fails if prompt is shorter.
- **Prefill chunk pipeline overlap** only visible when `--step-incr` ≥ prefill chunk size (relevant for distributed mode).

## See Also

- [Engine API](../engine/engine-api.md)
- [Session Snapshots](../engine/session-snapshots.md)
- [Model Shape Detection](../concepts/model-shape-detection.md)

[← Back to Index](../README.md)
