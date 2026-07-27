# SSD Streaming

## Files

- `ds4_ssd.c` — SSD streaming config parsing, auto-cache planning, memory lock (simulate-used-memory)
- `ds4_ssd.h` — public API, config structs, memory lock types
- `ds4.c` — expert locality profiler, Metal/ROCm streaming decode backend, cache budget wiring, engine init/teardown

## Purpose

Load routed expert weights from SSD on demand instead of keeping all in RAM. Enables running models larger than available memory.

## Dependencies

- **Imports from**: `ds4_ssd.h` types consumed by `ds4.c` (engine options, cache plan), `ds4_cli.c` (parse flags), `ds4_help.c` (help text)
- **Exports to**: `ds4.c` — engine init calls `ds4_ssd_memory_lock_acquire()` / `ds4_ssd_memory_lock_release()`; auto-cache plan feeds `ssd_streaming_cache_experts` / `ssd_streaming_cache_bytes` fields on `ds4_engine`. All CLI entry points (`ds4_cli.c`, `ds4_agent.c`, `ds4_server.c`, `ds4_eval.c`, `ds4_bench.c`) call `ds4_parse_streaming_cache_experts_arg()`.
- **Init order**: memory lock acquired early in `ds4_engine_load()` (ds4.c:55233) before model load. Cache plan resolved after model weights loaded (ds4.c:53253+). Lock released in `ds4_engine_close()` (ds4.c:56297).

## Streaming Model

Shared layers always resident in RAM. Routed experts reside on SSD and are loaded per-layer during decode. An expert locality profiler (`ds4_expert_profile_cap_candidates[]` in `ds4.c:1209`) simulates per-layer latest-N unique expert caches at sizes 1, 2, 4, 8, 16, 32, 64, 128, 256, 384 and records hit rates to guide cache sizing.

See [gpu-expert-streaming-cache.md](../concepts/gpu-expert-streaming-cache.md) for expert prefetch model and cache eviction policy.

## Key Types

| Type | Role |
|---|---|
| `ds4_ssd_memory_lock` | Opaque handle for a simulated-memory locked region (ptr + bytes) |
| `ds4_ssd_cache_plan` | Output of auto-cache planning: model target bytes, cache bytes, effective cache bytes, cache expert count |

## API Surface

### Memory Lock (ds4_ssd.c:129-208)

- `ds4_ssd_memory_lock_acquire(lock, bytes)` — mmap + mlock `bytes` of anonymous RAM to simulate a smaller-memory machine. Touches each page with a byte write, then locks in 256 MiB chunks so partial failures are diagnosable and macOS uninterruptible VM work stays bounded. Called from `ds4_engine_load()` (ds4.c:55233) when `--simulate-used-memory NGB` is set. Returns false on any mmap or mlock failure; partial mlock is rolled back via munlock+munmap.
- `ds4_ssd_memory_lock_release(lock)` — munlock + munmap the locked region. Called from `ds4_engine_close()` (ds4.c:56297). No-op if lock is NULL or already released.

### Cache Planning (ds4_ssd.c:54-127)

- `ds4_ssd_cache_experts_for_byte_budget(bytes, per_expert_bytes)` — floor-divide byte budget by per-expert weight size. Returns 0 if either arg is 0 or result exceeds UINT32_MAX.
- `ds4_ssd_auto_cache_plan(recommended_bytes, non_routed_bytes, per_expert_bytes, max_model_experts, &out)` — auto-cache planner. Computes `model_target_bytes = recommended_bytes × pct / 100`, subtracts `non_routed_bytes` to get cache bytes, then derives expert count. Caps at `max_model_experts`. Always ensures at least 1 expert. Returns false if inputs invalid or result is 0 experts.

### Config Parsing (ds4_ssd.c:25-52)

- `ds4_parse_gib_arg(s, &bytes)` — parse a GiB suffix string (e.g. "64GB") to bytes. Returns false on malformed input or overflow.
- `ds4_parse_streaming_cache_experts_arg(s, &experts, &bytes)` — parse `--ssd-streaming-cache-experts` argument. If string ends with GB/gb suffix, delegates to `ds4_parse_gib_arg` and sets bytes (experts = 0). Otherwise parses as integer expert count (bytes = 0).

### Query / Introspection

- `ds4_engine_dynamic_expert_cache_bytes(e)` (ds4.c:35290+) — returns effective expert cache in bytes: `ssd_streaming_cache_bytes` if set, else `ssd_streaming_cache_experts × per_expert_bytes`. Used for transient guard calculation.

## Data Flow

```
CLI flags
  │
  ├──ssd-streaming-cache-experts N|NGB ──→ ds4_parse_streaming_cache_experts_arg()
  ├──simulate-used-memory NGB    ─────────→ ds4_ssd_memory_lock_acquire()
  └──DS4_SSD_AUTO_CACHE_PCT (env) ───────→ ds4_ssd_auto_cache_plan()
                                              │
                                              ▼
                                    ds4_engine.ssd_streaming_cache_experts
                                    ds4_engine.ssd_streaming_cache_bytes
                                              │
                                              ▼
                                    ds4_gpu_set_streaming_expert_cache_budget()
                                              │
                                              ▼
                                    GPU backends: Metal/ROCm expert prefetch
```

## Auto-Cache Planner

`ds4_ssd_auto_cache_plan()` computes a cache budget when neither `--ssd-streaming-cache-experts` nor `--ssd-streaming-cache-bytes` is explicitly set.

### DS4_SSD_AUTO_CACHE_PCT

Environment variable, range 50..95, default 80. Fraction (as percent) of the model's recommended working set bytes to target for total RAM usage. Read once per plan via `getenv("DS4_SSD_AUTO_CACHE_PCT")`. If unset or invalid, falls back to 80.

Default 80 rationale (ds4_ssd.c:93-103): Decode on ROCm streaming path is SSD-bandwidth bound — every routed expert miss is a random NVMe read, so larger resident cache is the biggest throughput lever. But expert cache shares physical RAM with the OS on unified-memory APUs; transient spikes (pinned read/upload staging, prefill headroom, cache growth) ride on top. 80% measured safe; opt into more only deliberately via env var.

### Budget Calculation

```
model_target_bytes = recommended_bytes * pct / 100
cache_bytes        = model_target_bytes - non_routed_bytes  (clamped to 0)
cache_experts      = cache_bytes / per_expert_bytes         (min 1, cap at max_model_experts)
effective_bytes    = cache_experts * per_expert_bytes
```

### Decision Tree

```
Is ssd_streaming_cache_bytes != 0?
  └─Yes→ use exact byte budget
          └─Also clamp to safe_cache_bytes (ds4.c:55408-55423)
Is ssd_streaming_cache_experts != 0?
  └─Yes→ use exact expert count
          └─Warn if < 2× min_experts per layer (ds4.c:55730-55737)
Neither set?
  └─Call ds4_ssd_auto_cache_plan()
      ├─Read DS4_SSD_AUTO_CACHE_PCT (env, default 80)
      ├─Compute model_target_bytes = recommended * pct / 100
      ├─Subtract non_routed_bytes → cache_bytes
      ├─Divide by per_expert_bytes → cache_experts
      └─Cap at max_model_experts, min 1
```

## CLI Options

All flags parsed in `ds4_cli.c` (parse_options, ~line 1879) and documented in `ds4_help.c` (line 170-175). Also accepted by `ds4_agent`, `ds4_server`, `ds4_eval`, `ds4_bench`.

| Flag | Default | Description |
|---|---|---|
| `--ssd-streaming` | off | Opt in to SSD-backed model streaming instead of full residency |
| `--ssd-streaming-cold` | off | Skip default popularity-based expert-cache preload |
| `--ssd-streaming-cache-experts N\|NGB` | 0 (auto) | N = exact dynamic expert count; NGB = routed memory budget that also reserves two full prefill layers. Auto: 80% working set minus non-routed weights; GLM Metal caps lower |
| `--ssd-streaming-full-layers N` | auto from NGB budget | GLM Metal streaming: keep first N routed layers fully resident. 0 to disable |
| `--ssd-streaming-preload-experts N` | auto (DeepSeek seeds by popularity) | Upfront popularity preload count. GLM demand-fills unless N explicit |
| `--simulate-used-memory NGB` | 0 (no lock) | Diagnostic: lock N GiB before model load to simulate a smaller-memory machine |

### Environment

| Variable | Range | Default | Description |
|---|---|---|---|
| `DS4_SSD_AUTO_CACHE_PCT` | 50..95 | 80 | Fraction (percent) of recommended model bytes to target for expert cache |

## Cache Policy

| Config | Mechanism |
|---|---|
| `--ssd-streaming-cache-experts N` | Pin N most-recently-used experts in RAM |
| `--ssd-streaming-cache-experts NGB` | Pin up to N GiB of expert weights in RAM (also reserves two full prefill layers) |
| Neither set | Auto mode via `ds4_ssd_auto_cache_plan()` using recommended bytes × `DS4_SSD_AUTO_CACHE_PCT` / 100 |
| Cache bytes = 0, experts = 0 | Full streaming — no cache, load every expert from SSD on demand |

## Invariants

- Only routed expert weights streamed; shared layers always in RAM.
- SSD streaming compatible with Metal and ROCm backends.
- Profiler overhead negligible when disabled (no-op hooks).
- Memory lock (mmap+mlock) must succeed before model load or engine init aborts.
- Cache plan resolved after model weights loaded (per_expert_bytes known).
- Auto cache always produces at least 1 expert (avoids degenerate 0-cache streaming).
- Cache bytes clamped to `safe_cache_bytes` in `ds4_engine_load()` (ds4.c:55408-55423) to avoid OOM.

## Notes

- `ds4_ssd.c` handles budget parsing and auto-cache planning. Actual streaming read is in `ds4.c` and GPU backends (Metal, ROCm).
- Profiler location: `ds4_expert_profile_cap_candidates[]` in `ds4.c:1209`, not in `ds4_ssd.c`.
- Memory lock is a diagnostic tool (simulate smaller RAM), not part of the streaming data path.
- Default 80% for auto-cache measured on unified-memory ROCm APU; tuning for discrete GPU setups may differ.

## See Also

- [GPU Expert Streaming Cache](../concepts/gpu-expert-streaming-cache.md)
- [MoE FFN](../engine/moe-ffn.md)

[← Back to Index](../README.md)
