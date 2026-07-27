# Layer Packing Engine

## Definition

Layer packing engine assigns each model entry (embedding, transformer layers, output head) to GPU or CPU device using monotonic-contiguous greedy first-fit algorithm. Entry byte footprints computed from tensor weights plus per-layer KV cache estimate. Entries exceeding every GPU budget spill to CPU. Result stored in `ds4_engine::placement[]`.

See [multi-gpu-pipeline.md](multi-gpu-pipeline.md) for GPU pipeline architecture and device assignment strategy.

## Why It Exists

Without packer, multi-GPU placement requires manual layer-to-device mapping. Packer automates split: given VRAM budgets (from `ds4_gpu_config` populated by CLI `--gpu-vram` or auto-detection), produces monotonic-contiguous allocation maximizing GPU residency while respecting per-device capacity.

## Where It Appears

- **`ds4_layer_pack.h`** — `ds4_layer_pack_config` struct, `ds4_compute_layer_placement` API, `ds4_layer_pack_print` API.
- **`ds4_layer_pack.c`** — placement algorithm + print implementation.
- **`ds4_gpu_mgpu.h`** — `ds4_gpu_config` struct (device_indices, vram_bytes, safety_margin_bytes, n_gpus) — budget source consumed by packer.
- **`ds4.c`** — engine-side helpers: `tensor_to_entry`, `engine_compute_entry_bytes`, `engine_per_layer_kv_bytes_planner`, `engine_glm_per_layer_kv_bytes_planner`, `engine_classify_multi_tier` (packer invocation steps).

### Key Data Structures

```c
#define DS4_LAYER_PACK_MAX_GPUS 16
#define DS4_LAYER_PACK_CPU      (-1)

typedef struct {
    size_t gpu_budget_bytes[DS4_LAYER_PACK_MAX_GPUS];
    int    n_gpus;
} ds4_layer_pack_config;

typedef struct ds4_gpu_config {
    int    device_indices[DS4_MAX_GPUS];
    size_t vram_bytes[DS4_MAX_GPUS];
    size_t safety_margin_bytes;
    int    n_gpus;
} ds4_gpu_config;
```

| `ds4_engine` field | Type | Purpose |
|---|---|---|
| `placement[DS4_MAX_LAYER + 2]` | `int[]` | Per-entry device assignment (GPU index or `DS4_LAYER_PACK_CPU`) |
| `n_placement_entries` | `int` | `g_ds4_shape.n_layer + 2` after classification |
| `multi_tier` | `int` | Gate flag: 1 when placement spans multiple devices or includes CPU spill |

### Entry Mapping

Entry footprint computed by `tensor_to_entry`:

| Entry Index | Match Rule | Components |
|---|---|---|
| 0 | Default (no prefix match) | `token_embd.weight` + any unrecognized tensor |
| 1..n_layers | `blk.<digits>.` prefix → entry `il + 1` | transformer layer `il` weights (attn, FFN, norms) |
| n_layers+1 | `output.weight`, `output_norm.weight`, `output_hc_*`, `mtp.*` | output head + optional MTP tensors |

KV cache estimated per-layer via `placement_ctx_hint` (default 4096). Per-tier scratch accounted in `engine_per_tier_graph_overhead_bytes` — pre-subtracted from device budgets, NOT double-counted in entry bytes.

### Budget Adjustments

Before packing, budgets are adjusted from `ds4_gpu_config`:

1. Per-tier graph overhead subtracted (conservative pre-subtract — unused tiers still reserve overhead, preventing OOM if packer assigns to tier whose scratch not reserved. Over-charge is few MB total.)
2. Safety margin subtracted per-GPU
3. 64 MiB cuBLAS workspace reserve subtracted
4. Configs with `total_budget == 0` rejected (caller must populate budgets — auto-detect is CLI's job, not engine's)

### Packer Invocation

`engine_classify_multi_tier` (called from `ds4_engine_open_internal` after model load + weight bind, pure CPU, no GPU init):

1. Pre-subtract `engine_per_tier_graph_overhead_bytes` from every GPU budget
2. Compute entry bytes via `engine_compute_entry_bytes`
3. Build packer config: post-subtract budgets from `ds4_gpu_config`, subtract `safety_margin_bytes` + 64 MiB cuBLAS workspace reserve
4. Run `ds4_compute_layer_placement(entry_bytes, g_ds4_shape.n_layer + 2, &pcfg, e->placement)`
5. Set `e->n_placement_entries = g_ds4_shape.n_layer + 2`, compute `e->multi_tier` from placement uniformity

## Packing Algorithm

### Core Loop (`ds4_compute_layer_placement`)

Monotonic-contiguous greedy first-fit. Walks entries forward; for each entry, advances device `d` until `entry_bytes[e] > budget[d]` fails (strict greater-than, not >=). Once all GPUs exhausted, remaining entries spill to CPU and all subsequent entries also land on CPU (monotonic property).

```
for each entry e:
  while d < n_gpus AND entry_bytes[e] > budget[d]:
    d++
  if d < n_gpus:
    device_for_entry[e] = d
    budget[d] -= entry_bytes[e]
  else:
    device_for_entry[e] = DS4_LAYER_PACK_CPU
    # d stays at n_gpus → subsequent entries also CPU
```

Returns 0 on success; nonzero on null pointer, negative n_entries, or n_gpus out of range.

### Output Format (`ds4_layer_pack_print`)

```
multi-GPU layout:
  GPU0: layers 0-21 + embedding   (38.4 / 40.0 GB)
  GPU1: layers 22-31              (11.7 / 12.0 GB)
  CPU : layers 32-42 + output head
```

Entry-naming convention: entry 0 → "embedding", entry i in 1..n_layers → transformer layer (numbered i-1), entry n_layers+1 → "output head". Usage/budget lines omitted when pointers NULL. Budget bytes shown are post-adjustment.

## Monotonic Placement

Monotonic-contiguous means entries are placed on devices in increasing order with no backtracking. Once entry `e` lands on device `d`, no later entry goes to a device earlier than `d`. Once any entry spills to CPU, all subsequent entries also land on CPU.

Not optimal for all weight distributions (bin-packing could pack more onto GPUs) but guarantees deterministic, debuggable layout with O(n_entries × n_gpus) worst case.

Budget check uses strict greater-than (`>`), not `>=`. Exact fits stay on current device.

Zero-budget GPU is effectively skipped — entry cannot fit on device with zero remaining budget, so loop advances past it.

## GPU/CPU Split

When all GPU budgets exhausted, remaining entries assign to `DS4_LAYER_PACK_CPU`. CPU spill is terminal — once entry lands on CPU, all subsequent entries also CPU. Deliberate: mixing GPU/CPU per-layer routing adds complexity beyond scope.

Gate flag `ds4_engine::multi_tier` set to 1 when placement spans multiple devices or includes CPU spill. Zero means single-GPU, no placement needed.

Packer does not handle expert sharding — that is Tensor Parallelism responsibility via `engine_cuda_tp_ep_requested` override path.

## Relationship to Other Concepts

- **Depends on**: `ds4_gpu_config` (budget source), engine memory estimation (`engine_compute_entry_bytes`, KV planners).
- **Used by**: [multi-gpu-pipeline.md](multi-gpu-pipeline.md) (consumer of `placement[]`), CLI (`--gpu-vram`, `--tensor-split`), CUDA tensor parallelism (placement determines TP tiers), CUDA expert-parallel (override placement path).
- **Alternatives**: manual layer-to-device with `--load-layer-start/end`, single-GPU (`multi_tier == 0`, no placement).

[← Back to Index](../README.md)
