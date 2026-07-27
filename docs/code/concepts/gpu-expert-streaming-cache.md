# GPU Expert Streaming Cache

## Definition

Resident LRU cache (Metal) / single-use compact buffer (CUDA) holding frequently-routed MoE expert weights in GPU VRAM. Under SSD streaming (`ssd_streaming`) when experts exceed VRAM, cache intercepts expert loads: hits skip SSD read, misses trigger read + evict. Shared across sessions, persists between requests on Metal; single-shot on CUDA.

## Why It Exists

Large MoE models (Pro: 384 experts, 16 routed) exceed GPU VRAM. Without cache, every decode step reads from SSD — latency kills throughput. Cache keeps hot experts GPU-resident, reducing SSD reads 90%+ on typical workloads. CUDA variant avoids LRU complexity: copies weights on-demand into compact per-batch buffer, releases after use.

## Where It Appears

| File | Symbol | Role |
|---|---|---|
| `ds4_gpu.h` | `ds4_gpu_stream_expert_table` | Per-layer table with model map, byte sizes, layer index, expert counts, gate/up/down offsets |
| `ds4_gpu.h` | `ds4_gpu_set_streaming_expert_cache_budget` | Configure expert slot count |
| `ds4_gpu.h` | `ds4_gpu_set_streaming_expert_cache_expert_bytes` | Set per-expert byte size for budget calc |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_configured_count` | Query configured max entries |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_current_count` | Query resident entry count |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_budget_for_expert_size` | Compute max cacheable entries from weight sizes |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_seed_selected` | Post-router pre-warm: hint cache which experts selected |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_begin_selected_load` | Prefetch selected experts not yet resident |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_seed_experts` | Batch seed multiple experts with priority weights |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_reset_route_hotness` | Zero hotness counters between sessions |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_release_resident` | (CUDA/ROCm) Free all cached expert GPU buffers |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_note_service_thread` | (Metal) Register worker thread so cache paths skip wait on its command buffers |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_prepare_selected_batch` | (ROCm) Batch prepare selected experts |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_load_layer` | (ROCm) Load entire layer's experts |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_seed_from_layer_selected` | (ROCm) Seed from layer-selected tensor |
| `ds4_gpu.h` | `ds4_gpu_stream_expert_cache_release_layer_cache` | (ROCm) Free per-layer GPU resources |
| `ds4_gpu.h` | `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor` | (GLM) Read selected expert IDs from GPU tensor, begin load |
| `ds4.c` | `graph_stream_expert_table_make` | Construct per-layer table from model + layer weights |
| `ds4.c` | `metal_graph_seed_streaming_expert_cache_from_hotlist` | Prefill: seed cache from hotlist at startup |
| `ds4.c` | `metal_graph_seed_streaming_expert_cache_from_router` | Prefill: seed per token-row after router runs |
| `ds4.c` | `metal_graph_decode_cpu_router` | Decode: route + call `begin_selected_load` to prefetch |
| `ds4.c` | `metal_graph_selected_async_load_worker_main` | Async load worker thread: calls `begin_selected_load`, waits on service thread |
| `ds4_metal.m` | `g_stream_expert_cache` | 2D array `[layer][expert]` of cache entries |
| `ds4_metal.m` | `g_stream_expert_cache_route_hotness` | 2D array `[layer][expert]` of `uint32_t` hotness counters |
| `ds4_metal.m` | `g_stream_expert_cache_layer_count` | Per-layer resident expert count |
| `ds4_metal.m` | `g_stream_expert_cache_layer_hits/misses/evictions/pread_bytes/pread_ms` | Per-layer counters + last-snapshot for delta logging |
| `ds4_metal.m` | `ds4_gpu_stream_expert_cache_entry` | Struct: model map, gate/up/down offsets, byte sizes, last_used, use_count, Metal buffer inner lengths, inflight_seq, slab_slot, valid/slab_backed flags |
| `ds4_metal.m` | `ds4_gpu_stream_expert_cache_prune_layer` | Evict from single layer when count exceeds effective cap |
| `ds4_metal.m` | `ds4_gpu_stream_expert_cache_prune_global` | Evict across all layers when total entries exceed budget |
| `ds4_metal.m` | `ds4_gpu_stream_expert_cache_clear_entry` | Free GPU buffers, mark entry invalid |
| `ds4_metal.m` | `ds4_gpu_stream_expert_cache_take_reusable` | Search entries with matching byte sizes to steal buffers |
| `ds4_metal.m` | `ds4_gpu_stream_expert_cache_note_route_hotness` | Increment hotness when expert selected |
| `ds4_metal.m` | `ds4_gpu_stream_expert_cache_maybe_decay_route_hotness` | Halve all hotness counters every 16 decode tokens |
| `ds4_metal.m` | `ds4_gpu_stream_expert_evict_dontneed_range` | POSIX_MADV_DONTNEED on model map range (gated by env var) |
| `ds4_cuda.cu` | `g_stream_selected_cache` | Single-use compact buffer: gate/up/down ptrs, slot_selected_tensor |
| `ds4_cuda.cu` | `cuda_stream_selected_cache_begin_load` | Copy expert weights from model file map to GPU compact buffer |
| `ds4.h` | `ds4_engine_options::ssd_streaming_cache_experts` | Fixed expert slot count |
| `ds4.h` | `ds4_engine_options::ssd_streaming_cache_bytes` | Byte budget for cache (auto-compute slots) |
| `ds4.h` | `ds4_engine_options::ssd_streaming_full_layers` | Layers to keep fully resident |
| `ds4.h` | `ds4_engine_options::ssd_streaming_preload_experts` | Experts to preload at startup |
| `ds4.h` | `ds4_engine_options::ssd_streaming_cold` | Cold-start flag (skip hotlist seeding) |

## Cache Architecture

### Table Construction

`graph_stream_expert_table_make` populates `ds4_gpu_stream_expert_table` per MoE layer from model mmap base, layer offsets, and per-expert byte sizes. Each entry stores gate/up/down absolute file offsets and the number of experts in the layer.

### Metal: Full LRU with Hotness Tracking

Three core structures:

- `g_stream_expert_cache[layer][expert]` — 2D array of `ds4_gpu_stream_expert_cache_entry`
- `g_stream_expert_cache_route_hotness[layer][expert]` — `uint32_t` counters
- `g_stream_expert_cache_layer_count[layer]` — current slot count per layer

#### Prefill Phase

```
hotlist seed at startup → seed_experts() for each hot expert → prune_layer + prune_global
router seed per token-row → seed_selected(table, selected_ids, n_selected) → prune
```

Hotlist seeds high-priority experts before inference. Router-based seeding warm-caches per token-row during prefill.

#### Decode Phase

```
metal_graph_decode_cpu_router → begin_selected_load(table, selected_ids, n_selected)
  → on hit: use resident GPU buffer
  → on miss: read from SSD → load to GPU buffer → prune both
```

Async load worker runs on a service thread. `note_service_thread` registers it so cache load paths skip waiting on that thread's command buffers. If load fails (buffer unavailable on that thread), caller retries synchronously on main thread.

Each cache entry stores:
- `model_map` + gate/up/down absolute offsets for on-demand load
- `gate/up/down_inner` — Metal buffer inner lengths (bytes ÷ element stride)
- `last_used` — timestamp for LRU tiebreak
- `use_count` — cumulative use count
- `inflight_seq` — nonzero while async load in flight (eviction skips)
- `slab_slot` — index into GPU slab allocator (256 max slabs)
- `valid`, `slab_backed` — entry state flags

Hotness counters decay every 16 decode tokens. `reset_route_hotness` zeros all counters between sessions. `note_route_hotness` increments when router selects an expert.

### CUDA: Single-Use Compact Buffer

No LRU. `g_stream_selected_cache` holds one compact GPU buffer with gate/up/down device pointers and a `slot_selected_tensor` for selected expert IDs.

`cuda_stream_selected_cache_begin_load`:
1. Invalidates previous load
2. Copies gate/up/down weight rows from model file map to GPU
3. Writes selected IDs to `slot_selected_tensor`
4. Released after FFN eval

`seed_selected` is no-op on CUDA — no benefit from pre-warming a single-use buffer.

### GLM: Tensor-Driven Load

`ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor` reads selected expert IDs from a GPU tensor back to host (device-to-host copy, i.e. CPU round-trip), then dispatches load from IDs now on host. Follows same compact-buffer pattern as CUDA but reads routing IDs from GPU tensor rather than host pointer. Compatible with tensor parallelism (both ranks mirror identical tensor read).

### Configuration

Engine options in `ds4_engine_options`:

| Field | Type | Default | Effect |
|---|---|---|---|
| `ssd_streaming_cache_experts` | `uint32_t` | 0 | Fixed expert slot count. Mutually exclusive with `cache_bytes`. |
| `ssd_streaming_cache_bytes` | `uint64_t` | 0 | Byte budget. Auto-computes slot count from per-expert byte size. |
| `ssd_streaming_full_layers` | `uint32_t` | 0 | Layers to keep fully resident (experts + dense weights). |
| `ssd_streaming_preload_experts` | `uint32_t` | 0 | Experts to preload from hotlist at engine start. |
| `ssd_streaming_cold` | `bool` | false | Skip hotlist seeding; start with empty cache. |

Budget is set at engine open: either from explicit expert count or computed from byte budget via `budget_for_expert_size`, which fits entries into available VRAM after reserving space for KV cache, model weights, and scratch buffers.

Environment variable `DS4_METAL_ENABLE_STREAMING_EXPERT_EVICT_DONTNEED` enables POSIX_MADV_DONTNEED on evicted expert map ranges.

### Backend Comparison

| Aspect | Metal | CUDA | GLM |
|---|---|---|---|
| Cache duration | LRU, persists across sessions | Single-use per batch | Single-use per batch |
| Eviction | Per-layer + global LRU | None (invalidate prev) | None |
| Hotness tracking | Yes, half-life decay | No | No |
| Router interaction | CPU router (`seed_selected`) | `seed_selected` no-op | Tensor-driven (D2H copy for routing IDs) |
| Async load worker | Yes, service thread | No | No |
| Buffer reuse | Yes (`take_reusable`) | No | No |
| DONTNEED hinting | Yes (optional) | No | No |
| `note_service_thread` | Yes | N/A | N/A |

## LRU Eviction

Metal only. CUDA and GLM use single-shot buffers with no eviction.

### Victim Selection

Same policy for per-layer and global pruning:
1. Lowest `route_hotness[layer][expert]`
2. Tiebreak: oldest `last_used`
3. Skip entries with `inflight_seq != 0` (load in progress)
4. Skip protected entries

### Per-Layer Pruning

When `layer_count[layer] > effective_cap`, scan layer's entry row, pick victim via hotness + last_used, free GPU buffers via `clear_entry`. Called after each `seed_selected` / `begin_selected_load` on that layer.

### Global Pruning

When total entry count exceeds budget, scan all layers × experts, pick victim across entire cache, free via `clear_entry`. Called after per-layer prune. Ensures total resident entries stay under budget.

### Buffer Reuse

`take_reusable` searches for entries with matching `gate_expert_bytes` + `down_expert_bytes`. On match, steals the Metal buffers (__bridge transfer) instead of freeing and reallocating. Reduces GPU allocation churn.

### DONTNEED Hinting

`ds4_gpu_stream_expert_evict_dontneed_range` calls `POSIX_MADV_DONTNEED` on the model map range of evicted expert weights. Gated by env var `DS4_METAL_ENABLE_STREAMING_EXPERT_EVICT_DONTNEED`. Tells kernel pages are reclaimable under memory pressure.

### Hotness Decay

Every 16 decode tokens, all hotness counters are right-shifted by 1 (divide by 2). Half-life decay prevents stale hotness from dominating. `reset_route_hotness` does a full memset zero between sessions, giving each session clean counters.

### Hit/Miss Counters

Per-layer tracking: hits, misses, evictions, pread_bytes, pread_ms. Each has a last-snapshot field for delta logging between observation points.

## SSD Read Policy

On Metal cache miss: weights copied from SSD → host pinned memory → GPU async via Metal blit encoder. Cache is GPU-device-resident — not host memory.

On CUDA: every decode step copies from model file mmap to GPU compact buffer regardless of prior state (no reuse between steps).

On GLM: same as CUDA — routing IDs read via D2H copy, then fresh weights copy from mmap to GPU.

In all backends, the model file map (`model->map`) is the single source of weight data. Cache intercepts only on Metal; CUDA and GLM always read from mmap.

## Code Pattern

### Metal: Full LRU Lifecycle (`ds4_metal.m`)

```c
// Configuration — set at engine open (ds4.c L55687-L55697)
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts);         // L3568: fixed slot count
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);     // L3576: per-expert byte size
uint32_t ds4_gpu_stream_expert_cache_configured_count(void);              // L10465: query max entries
uint32_t ds4_gpu_stream_expert_cache_current_count(void);                 // L10473: query resident entries
uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(              // L10479: compute max from byte budget
    uint64_t gate_expert_bytes, uint64_t down_expert_bytes);

// Prefill — warm cache before decode (ds4.c L23446, L29549)
int ds4_gpu_stream_expert_cache_seed_selected(                            // L14808: per-token-row post-router
    const ds4_gpu_stream_expert_table *table,
    const int32_t *selected_ids, uint32_t n_selected);
int ds4_gpu_stream_expert_cache_seed_experts(                             // L14889: batch seed with priorities
    const ds4_gpu_stream_expert_table *table,
    const int32_t *expert_ids,
    const uint32_t *expert_priorities, uint32_t n_experts);

// Load — async prefetch from SSD to GPU on cache miss (ds4.c L20519-L20960)
int ds4_gpu_stream_expert_cache_begin_selected_load(                      // L13567
    const ds4_gpu_stream_expert_table *table,
    const int32_t *selected_ids, uint32_t n_selected);
    // → on hit: use resident GPU buffer
    // → on miss: read from SSD → alloc GPU buffer → prune

// Eviction — LRU with hotness tiebreak (Metal only)
static void ds4_gpu_stream_expert_cache_prune_layer(                      // L12542
    uint32_t layer, uint32_t n_total_expert,
    uint32_t n_selected,
    const int32_t *protect_ids, uint32_t n_protect);
    // Per-layer: evict when layer_count > effective_cap
static void ds4_gpu_stream_expert_cache_prune_global(                     // L13025
    uint32_t protect_layer,
    const int32_t *protect_ids, uint32_t n_protect);
    // Global: evict when total entries > budget, scan all layers

// Victim selection policy: lowest route_hotness, then oldest last_used
static void ds4_gpu_stream_expert_cache_note_route_hotness(               // L11551
    uint32_t layer, uint32_t expert, uint32_t amount);
    // Incremented on each router selection; halved every 16 decode tokens

// Release — free GPU buffers, mark invalid (per-entry, not a global flush)
static void ds4_gpu_stream_expert_cache_clear_entry(                      // L12405
    uint32_t layer, uint32_t expert, int count_eviction);
    // calls clear_entry_internal(L12318) with slab recycle + optional buffer reuse

// Buffer reuse — steal buffers from evicted entry with matching byte sizes
static int ds4_gpu_stream_expert_cache_take_reusable(                     // L12616
    int force_reuse, uint32_t protect_layer,
    const int32_t *protect_ids, uint32_t n_protect,
    uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
    ds4_gpu_stream_expert_reusable_buffers *reuse);

// DONTNEED hinting — optional POSIX_MADV_DONTNEED on evicted map range
void ds4_gpu_stream_expert_evict_dontneed_range(                          // ds4_metal.m L...
    const void *map, uint64_t map_size,
    uint64_t abs_offset, uint64_t bytes);
    // Gated by env var DS4_METAL_ENABLE_STREAMING_EXPERT_EVICT_DONTNEED

// [P0] No ds4_gpu_stream_expert_cache_release_resident on Metal.
// Metal uses per-entry clear_entry + LRU eviction instead.
```

### CUDA: Single-Use Compact Buffer (`ds4_cuda.cu`)

```c
// Configuration — no-ops on CUDA (budget always 0)
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts);         // L27381: (void)experts
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);     // L27385: (void)bytes

// Load — copy selected expert weights from mmap to GPU compact buffer
int ds4_gpu_stream_expert_cache_begin_selected_load(                      // L27253: public wrapper
    const ds4_gpu_stream_expert_table *table,
    const int32_t *selected_ids, uint32_t n_selected);
static int cuda_stream_selected_cache_begin_load(                         // L22954: internal impl
    const ds4_gpu_stream_expert_table *table,
    const int32_t *selected_ids, uint32_t slot_count);
    // 1. cuda_stream_selected_cache_invalidate() — free previous buffers
    // 2. Deduplicate expert IDs → compact_ids
    // 3. cudaMemcpy gate/up/down rows from model mmap to device
    // 4. Write slot_selected_tensor (compact slot → expert mapping)

// Pre-warm — no-op on CUDA (single-use buffer has no reuse across steps)
int ds4_gpu_stream_expert_cache_seed_selected(                            // L27405: (void)args, return 1
    const ds4_gpu_stream_expert_table *table,
    const int32_t *selected_ids, uint32_t n_selected);

// Release — free all GPU buffers for the compact cache
void ds4_gpu_stream_expert_cache_release_resident(void) {                 // L27401
    cuda_stream_selected_cache_release();                                  // L157: cudaFree gate/up/down/slot ptrs
}
```

### GLM: Tensor-Driven Load (both backends)

```c
// Metal: ds4_metal.m L14254
// CUDA: ds4_cuda.cu L26696
int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
    const ds4_gpu_stream_expert_table *table,
    const ds4_gpu_tensor              *selected,     // GPU tensor with expert IDs
    uint32_t                           n_selected);
    // 1. Sync GPU (Metal: end_commands; CUDA: implicit via cudaMemcpy)
    // 2. D2H copy: read selected IDs from GPU tensor to host array
    //    Metal: ds4_gpu_tensor_read()       — L14285
    //    CUDA:  cudaMemcpy(DeviceToHost)    — L26710
    // 3. Call begin_selected_load(table, ids, n_selected)
    //
    // [P0] CPU round-trip for routing IDs: both Metal and CUDA read GPU
    // tensor back to host before dispatching load. No true GPU-only path.
```

### Callers (`ds4.c`)

```c
// Engine open — set budget from engine options
// ds4.c L55687: ds4_gpu_set_streaming_expert_cache_budget(e->ssd_streaming_cache_experts);
// ds4.c L55697: ds4_gpu_set_streaming_expert_cache_expert_bytes(slab_expert_bytes);

// Decode — route → seed → load
// ds4.c L29549: seed_selected(table, selected_ids, n_selected)   — prefill warm
// ds4.c L20519: begin_selected_load(table, selected_ids, n_selected)  — decode load
// ds4.c L32956: ds4_gpu_stream_expert_cache_release_resident()   — CUDA-only cleanup

// GLM — tensor-driven load
// ds4.c L40102: glm_begin_selected_load_tensor(table, selected_tensor, n_selected)
```

## Relationship

- **Depends on**: SSD streaming (data source via model file mmap), [gpu-tensor-primitives.md](gpu-tensor-primitives.md) (Metal buffer allocation, CUDA device memory), slab allocator (Metal slab slots).
- **Used by**: engine eval path — MoE FFN dispatch checks cache before SSD read. [moe-routing.md](moe-routing.md) (router functions seed/load cache as side effect of expert selection).
- **Alternatives**: full-model GPU residency (experts always in VRAM, limited by capacity), no cache (every expert read from SSD), single-use compact buffers (CUDA approach, lower complexity but no reuse across decode steps).
- **Not used by**: CPU backend (no GPU cache needed), non-streaming mode (all experts in VRAM).

## Notes

- ROCm build has additional `load_layer`, `seed_from_layer_selected`, `release_layer_cache` for per-layer GPU resource management. These allocate/free GPU memory for an entire layer's experts at once rather than per-expert LRU.
- `release_resident` is CUDA/ROCm only — not implemented on Metal. Metal relies on per-entry `clear_entry` and prunes via LRU eviction instead.
- Eviction via `clear_entry` frees Metal buffers (`releaseBuffer`). If buffers are reusable (matching byte sizes), `take_reusable` transfers ownership instead of freeing. Avoids GPU allocation churn on hot paths.

[← Back to Index](../README.md)
