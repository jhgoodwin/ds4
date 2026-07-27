# Layer Pack

## Files

- `ds4_layer_pack.c` — monotonic-contiguous greedy first-fit placement
- `ds4_layer_pack.h` — public API, config struct, constants

## Purpose

Assign each model entry (embedding pseudo-layer, transformer layers, output-head pseudo-layer) to a GPU or CPU device using a greedy first-fit algorithm constrained by per-device byte budgets. Used in multi-GPU (wave-2) pipeline parallelism where layers are distributed across devices within a single machine.

Entry byte footprints are computed by `engine_compute_entry_bytes` and include tensor weights plus per-layer KV cache estimate. Per-device budgets come from `ds4_gpu_config` and are adjusted before packing (per-tier graph scratch, safety margin, cuBLAS workspace subtracted).

The packer does NOT handle MoE expert sharding — that is the tensor-parallelism module's responsibility via the `engine_cuda_tp_ep_requested` override path.

## Dependencies

- **Imports from**: engine memory estimation (`engine_compute_entry_bytes`, KV planners), `ds4_gpu_config` (budget source).
- **Exports to**: `ds4_engine_create_with_gpu_config`, `engine_classify_multi_tier`, CLI (`--gpu-vram`, `--tensor-split`).
- **Init order**: called during `engine_classify_multi_tier` after model weights are bound and entry bytes computed, before GPU device init.
- **Differentiated from**: tensor parallelism (splits MoE experts within a layer), distributed pipeline (multi-machine, network), manual layer-to-device (`--load-layer-start/end`).

## Key Types

| Type | Role |
|---|---|
| `ds4_layer_pack_config` | Per-device budgets (bytes, net of overhead) + GPU count |
| `int device_for_entry[]` | Output array: target device index per entry (`DS4_LAYER_PACK_CPU` = -1) |

### `ds4_layer_pack_config`

```c
typedef struct {
    size_t gpu_budget_bytes[DS4_LAYER_PACK_MAX_GPUS];  /* net of per-device overhead */
    int    n_gpus;
} ds4_layer_pack_config;
```

Budgets are pre-adjusted by the caller: per-tier graph scratch, safety margin, cuBLAS workspace, and (for CUDA TP/EP) output shard reservations are subtracted before the struct reaches the packer. The packer treats budgets as raw capacity — any entry exceeding a budget advances to the next device.

## API Surface

### Core Operations

#### `ds4_compute_layer_placement`

```c
int ds4_compute_layer_placement(const size_t *entry_bytes,
                                int n_entries,
                                const ds4_layer_pack_config *cfg,
                                int *device_for_entry);
```

Compute a monotonic-contiguous layer placement.

**Inputs:**
- `entry_bytes[]` — per-entry byte footprint, in forward order: entry 0 = embedding pseudo-layer, entry 1..n_layers = transformer layers, entry n_layers+1 = output-head pseudo-layer.
- `n_entries` — total number of entries (typically `DS4_N_LAYER + 2`, i.e. `g_ds4_shape.n_layer + 2`).
- `cfg` — per-device budgets (already net of per-device overhead).

**Output:**
- `device_for_entry[]` — caller-owned `int` array of length `n_entries`. Each slot is a GPU device index (0..n_gpus-1) or `DS4_LAYER_PACK_CPU` (-1). Consumed by multi-GPU pipeline code for per-device selective caching.

**Returns:**
- `0` on success.
- `1` if any input pointer is NULL.
- `2` if `n_entries < 0`.
- `3` if `cfg->n_gpus` is negative or exceeds `DS4_LAYER_PACK_MAX_GPUS`.

**Error contract:** An entry that exceeds every GPU budget spills to CPU; by the monotonicity rule every entry after it also goes to CPU. There is no error path for "entry too large" — the CPU tier is always available.

**Complexity:** O(n×g) where n = n_entries, g = n_gpus. Each entry advances the device cursor monotonically; worst case scans all devices per entry.

#### `ds4_layer_pack_print`

```c
void ds4_layer_pack_print(FILE *out,
                          const int *device_for_entry,
                          int n_entries,
                          int n_layers,
                          const size_t *entry_bytes,
                          const size_t *gpu_used_bytes,
                          const size_t *gpu_budget_bytes,
                          int n_gpus);
```

Print a human-readable layout summary.

**Output format:**
```
multi-GPU layout:
  GPU0: layers 0-21 + embedding   (38.4 / 40.0 GB)
  GPU1: layers 22-31              (11.7 / 12.0 GB)
  CPU : layers 32-42 + output head
```

**Entry naming convention:** entry 0 → "embedding", entry i in 1..n_layers → "layer \<i-1\>", entry n_layers+1 → "output head".

**Parameters:** `gpu_used_bytes[]` and `gpu_budget_bytes[]` may be NULL — the "(used / budget)" line is omitted for GPU lines.

**No return value.** No-op on NULL `out` or `device_for_entry`.

## Data Flow

```
ds4_gpu_config
    │
    ▼
engine_per_tier_graph_overhead_bytes() ──► subtract per-tier scratch
safety_margin_bytes + cublas_workspace ──► subtract from budget
    │
    ▼
ds4_layer_pack_config { gpu_budget_bytes[], n_gpus }
    │
    ▼
engine_compute_entry_bytes() ──► entry_bytes[] (weights + KV cache)
    │
    ▼
ds4_compute_layer_placement(entry_bytes, n_entries, cfg, device_for_entry[])
    │
    ├── GPU 0..n_gpus-1  ──► consumed by multi-GPU pipeline
    └── DS4_LAYER_PACK_CPU ──► CPU fallback path
```

Entry bytes computed once before packing. Budgets are caller-adjusted and copied locally inside the packer (the caller's config is not mutated).

## Placement Algorithm

Monotonic-contiguous greedy first-fit. Walks entries in forward order; for each entry, advances device `d` until the entry fits within the remaining budget. Once all GPUs are exhausted, remaining entries spill to CPU.

```
for each entry e (0..n_entries-1):
    while d < n_gpus AND entry_bytes[e] > budget[d]:
        d++
    if d < n_gpus:
        device_for_entry[e] = d
        budget[d] -= entry_bytes[e]
    else:
        device_for_entry[e] = DS4_LAYER_PACK_CPU
        // d stays at n_gpus → subsequent entries also CPU
```

Returns 0 on success. Nonzero on configuration errors (null pointer, negative n_entries, n_gpus out of range).

## Entry Types

| Entry Index | Name | Components |
|---|---|---|
| 0 | Embedding pseudo-layer | `token_embd.weight` + any unrecognized tensor |
| 1..n_layers | Transformer layers | Layer `i-1` weights (attn, FFN, norms) |
| n_layers+1 | Output-head pseudo-layer | `output.weight`, `output_norm.weight`, `output_hc_*`, `mtp.*` |

The total entry count is `n_placement_entries = g_ds4_shape.n_layer + 2` — a runtime value, not a compile-time constant. The `DS4_MAX_LAYER` constant (79) bounds the stack-allocated `entry_bytes[]` buffer in the caller.

## Invariants

- **Monotonicity**: `tier(e) ≤ tier(e+1)`. No backtracking; CPU is the max tier.
- **Strict `>` budget check**: exact fits stay on current device (`entry_bytes[e] > budget[d]`, not `>=`).
- **CPU-spill terminal**: once an entry lands on CPU, every subsequent entry also goes to CPU (the device cursor stays at `n_gpus`).
- **Zero-budget skip**: a device with budget 0 is effectively skipped (every entry exceeds 0).
- **No "entry too large" error**: entries exceeding all budgets simply spill to CPU; CPU tier is always available.
- **Pure C99**: no CUDA, no platform-specific code. Used by both CUDA and Metal/CPU builds.

## Configuration

- **No environment variables** read by this module.
- **No compile-time flags** specific to this module.
- `DS4_LAYER_PACK_MAX_GPUS` (16): hard upper bound on GPU count, enforced by return code 3.

## Notes

- Budgets are caller-adjusted before packing: per-tier graph scratch, safety margin, cuBLAS workspace (64 MiB), and CUDA EP output shard reservations are subtracted by `engine_classify_multi_tier`. The packer never sees the raw `ds4_gpu_config.vram_bytes`.
- The packer copies budgets into a local array so the caller's config is not mutated.
- The CUDA TP/EP override path (`engine_compute_cuda_ep_placement`) replaces this function entirely for tensor-parallel + expert-parallel deployments. The two paths share the same `device_for_entry[]` output contract.
- `ds4_layer_pack_print`'s "(no transformer layers)" stub is emitted when a tier owns no layers but has pseudo-layer tags; the tag text is appended regardless.

## See Also

- [Layer Packing Engine](../concepts/layer-packing-engine.md)
- [Multi-GPU Pipeline](../concepts/multi-gpu-pipeline.md)

[← Back to Index](../README.md)
