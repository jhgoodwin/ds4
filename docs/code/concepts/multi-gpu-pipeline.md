# Multi-GPU Pipeline Parallelism (Wave 2)

## Definition

Multi-GPU pipeline parallelism distributes transformer layers across multiple GPU devices within a single machine. Each GPU owns a contiguous slice of layers (monotonic-contiguous placement). Activations flow device-to-device via cross-device copy (xdev). The engine manages one logical session whose per-layer compute is routed to the owning GPU.

Wave 2 ships GPU-only multi-tier: placement planning, per-device init, selective caches, refusal on CPU-spill. CPU-spill refused with stderr naming the follow-up.

## Why It Exists

Single-GPU VRAM limits model sizes at given quantization. Pipeline parallelism across multiple GPUs (4x 24 GiB = 96 GiB, 8x 80 GiB = 640 GiB) fits larger models without network hops. Unlike distributed (multiple machines), multi-GPU uses GPU peer-to-peer DMA or host bounce — no network stack.

## Where It Appears

| File | Symbol | Role |
|---|---|---|
| `ds4_gpu_mgpu.h` | `ds4_gpu_config` | Device indices, per-device VRAM budgets, safety margin |
| `ds4_gpu_mgpu.h` | `ds4_gpu_ctx` | Per-device runtime context: stream, cuBLAS handle, scratch, budget tracking, boundary event |
| `ds4_gpu_mgpu.h` | `g_gpu[]` | Global array of per-device contexts (`ds4_gpu_ctx`) |
| `ds4_gpu_mgpu.h` | `g_n_gpus` | Global count of initialized devices |
| `ds4_gpu_mgpu.h` | `g_gpu_peer_ok[][]` | Peer access matrix: DIRECT or BOUNCE per device pair |
| `ds4_gpu_mgpu.h` | `ds4_gpu_init_multi` | Multi-device init — populates `g_gpu[]`, streams, cuBLAS handles |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_alloc_on` | Allocate on specific CUDA device |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_alloc_ptr_on` | Heap-alloc tensor on specified logical tier |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_alloc_managed_on` | Managed-memory tensor on specified logical tier |
| `ds4_gpu_mgpu.h` | `ds4_gpu_set_current_device` | Set current CUDA device by logical tier index |
| `ds4_gpu_mgpu.h` | `ds4_gpu_set_current_device_fenced` | Fenced variant: syncs before switching |
| `ds4_gpu_mgpu.h` | `ds4_gpu_register_model_map_no_copy` | Register host model map without whole-model copy |
| `ds4_gpu_mgpu.h` | `ds4_gpu_lookup_cache_strict` | Per-device strict cache lookup (no fallback) |
| `ds4_gpu_mgpu.h` | `ds4_gpu_device_cache_tensors` | Per-device selective tensor cache install |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_copy_xdev` | Single-buffer cross-device copy (peer or host-bounce) |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_copy_xdev3` | Three-buffer grouped cross-device copy |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_copy_xdev_ordered` | Destination-order sync for pipelined prefill |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_copy_xdev_default` | Default-stream fallback copy |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_copy_xdev3_default_dst` | Grouped default-stream handoff copy |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_wait_xdev` | Order peer read without data copy |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_wait_xdev_default` | Default-stream peer-read ordering |
| `ds4_gpu_mgpu.h` | `ds4_gpu_add_xdev_tensor` | Cross-device float add (tensor-parallel reduction) |
| `ds4_gpu_mgpu.h` | `ds4_gpu_tensor_copy_async` | Device-local async copy |
| `ds4_gpu.h` | `ds4_tensor_range` | Model-file slice: `source_offset`, `bytes`, `target_device` |
| `ds4_layer_pack.h` | `ds4_layer_pack_config` | Per-device byte budgets for the packer |
| `ds4_layer_pack.h` | `ds4_compute_layer_placement` | Greedy monotonic-contiguous placement algorithm |
| `ds4_layer_pack.h` | `ds4_layer_pack_print` | Human-readable layout printer |
| `ds4.c` | `engine_classify_multi_tier` | Runs placement, pre-subtracts overhead, sets `e->multi_tier` |
| `ds4.c` | `engine_install_gpu_placement` | Prints layout, rejects CPU-spill, installs per-device caches |
| `ds4.c` | `engine_compute_entry_bytes` | Computes per-entry byte footprint (including KV per layer) |
| `ds4.c` | `engine_per_tier_graph_overhead_bytes` | Per-tier scratch reservation (indexer scores, comp mask, etc.) |
| `ds4.c` | `engine_install_per_device_caches` | Builds `ds4_tensor_range` lists per tier, calls `ds4_gpu_device_cache_tensors` |
| `ds4.c` | `ds4_engine_create_with_gpu_config` | Public entry point — delegates to `ds4_engine_open_internal` |
| `ds4.h` | `ds4_engine_options::placement_ctx_hint` | Max-context hint for KV cache budget estimation |
| `ds4.h` | `ds4_engine_options::share_session_prefill_workspace` | Prefill workspace sharing toggle |

## Pipeline Architecture

### Key Types

| Type | Fields | Role |
|---|---|---|
| `ds4_gpu_config` | `device_indices[]`, `vram_bytes[]`, `n_gpus`, `safety_margin_bytes` | Caller-supplied GPU configuration: devices, VRAM budgets, per-device reserve |
| `ds4_gpu_ctx` | `device_id`, `stream`, `cublas`, `scratch`, `budget_bytes`, `used_bytes`, `boundary_event` | Per-device runtime state. `boundary_event` for cross-device handoff sync |
| `ds4_tensor_range` | `source_offset`, `bytes`, `target_device` | One span of model-file data cached on a specific device |
| `ds4_layer_pack_config` | `gpu_budget_bytes[]`, `n_gpus` | Per-device budgets net of overhead, fed to packer. `DS4_MAX_GPUS = 16` |

### Memory Planning

```
engine_classify_multi_tier():
  1. Copy caller config → e->gpu_cfg
  2. Pre-subtract per-tier overhead from each vram_bytes[d]
  3. Per GPU: budget = vram_bytes - safety_margin - cuBLAS_workspace (64 MiB)
  4. Fill ds4_layer_pack_config.gpu_budget_bytes[]
  5. Run ds4_compute_layer_placement or engine_compute_cuda_ep_placement
  6. Detect multi_tier: any entry on different device, or any CPU spill
```

Per-device budgets net of: cuBLAS workspace, safety margin, per-tier graph scratch. Budget zero after subtraction pushes placement to CPU.

Per-tier graph scratch (indexer scores, comp mask, chunked-prefill batch buffers) reserved by `engine_per_tier_graph_overhead_bytes`, pre-subtracted to avoid double-count in packer.

### Entry Footprint

- Entry 0 — embedding pseudo-layer
- Entries 1..n_layers — transformer layers (priced per layer via `placement_ctx_hint`)
- Entry n_layers+1 — output head

KV cache priced per layer via `engine_per_layer_kv_bytes_planner`, accounts for raw sliding-window capacity, compressed cache (indexer), prefill chunk.

### Lifecycle

```
caller builds ds4_gpu_config (devices, VRAM budgets)
→ ds4_engine_create_with_gpu_config(e, opt, &cfg)
  → ds4_engine_open_internal(out, opt, cfg)
    → engine_classify_multi_tier(e, cfg)
      → pre-subtract per-tier overhead
      → build layer_pack_config
      → ds4_compute_layer_placement() → e->placement[], e->multi_tier
    → if ssd_streaming && multi_tier: refuse (incompatible)
    → if CPU-spill: print layout, detailed stats, refuse
    → [graph_backend && multi_tier]:
      → ds4_gpu_init_multi(gpu_cfg)     // init all devices, streams
      → engine_install_gpu_placement(e)
        → engine_print_layout(e)         // layout + peer matrix
        → if cpu_spill: print refusal, return -1
        → ds4_gpu_register_model_map_no_copy()
        → engine_install_per_device_caches(e)
          → ds4_gpu_device_cache_tensors() per tier
      → engine_install_dspark_support_cache(e)
      → return (GPU-only multi-tier ready)
```

GPU-only multi-tier completes. CPU-spill placements rejected with diagnostic: per-GPU used/budget stats, total unaccommodated bytes, hints to adjust context or budgets.

## Stage Assignment

### Placement Algorithm

```
ds4_compute_layer_placement(entry_bytes[], n_entries, cfg, device_for_entry[]):
  for each entry i:
    if entry_bytes[i] <= budget[current_device]:
      assign to current_device
    else:
      advance current_device
      assign to current_device
    if devices exhausted: assign DS4_LAYER_PACK_CPU (-1)
```

Greedy monotonic-contiguous: once entry moves to GPU N, no earlier GPU receives entries after. CPU spill terminal — everything after first CPU entry stays CPU. No "entry too large" error; CPU tier always available.

### Selective Cache

Each device caches only its assigned tensors via per-device `ds4_tensor_range[]` lists built in `engine_install_per_device_caches`.

`ds4_gpu_register_model_map_no_copy` registers the host mmap'd model pointer without device-side copy. Bypasses `DS4_CUDA_COPY_MODEL` — essential for multi-tier where each device caches only its assigned spans.

`ds4_gpu_lookup_cache_strict` returns 1 only if covering cache entry exists for exact physical device. No fallback to different-device or host pointer. Used by multi-tier kernel-dispatch resolvers.

`ds4_gpu_device_cache_tensors` selectively caches model-file spans on one device. Called once per tier.

### Device Routing

| Function | Purpose |
|---|---|
| `ds4_gpu_set_current_device(tier)` | Canonical shim: `g_gpu[tier].device_id`, `cudaSetDevice` |
| `ds4_gpu_set_current_device_fenced(tier)` | Implicitly syncs before device switch |
| `ds4_gpu_tensor_alloc_on(device_id)` | Allocate on physical CUDA device |
| `ds4_gpu_tensor_alloc_ptr_on(tier)` | Heap-alloc on logical tier (0..g_n_gpus-1) |
| `ds4_gpu_tensor_alloc_managed_on(tier)` | Managed memory on logical tier; stamps tier for free accounting |

## Communication

### Cross-Device Transfer (XDEV)

| Scenario | Mechanism |
|---|---|
| Same device | `ds4_gpu_tensor_copy_async` → `cudaMemcpyAsync` |
| Peer-capable cross-device | `ds4_gpu_tensor_copy_xdev` → `cudaMemcpyPeerAsync` + event sync |
| Non-peer cross-device | Pinned host bounce buffer (src→host→dst) |
| Test override | `DS4_FORCE_HOST_BOUNCE=1` forces non-peer path |

`ds4_gpu_tensor_copy_xdev3` groups three buffers under one event pair for tiny activation handoffs where per-copy event plumbing dominates.

`ds4_gpu_tensor_copy_xdev3_default_dst` groups copies on destination default stream: source records readiness, destination waits, performs all three, then naturally orders following kernels. Allows source work to overlap handoff.

`ds4_gpu_tensor_copy_xdev_ordered` orders against prior work on destination stream before overwriting dst. Needed for pipelined prefill slots where next producer reuses consumed buffer.

`ds4_gpu_tensor_wait_xdev` orders peer read without copying: records readiness event on src stream, makes dst tier's stream wait.

`ds4_gpu_add_xdev_tensor` performs cross-device float add for CUDA tensor-parallel reductions.

### Peer Access

`g_gpu_peer_ok[i][j]` populated during `ds4_gpu_init_multi`:

```
ds4: peer access matrix (validated): 0->1 DIRECT 1->0 DIRECT
ds4: peer access matrix (validated): 0->1 BOUNCE 1->0 BOUNCE
```

`DS4_FORCE_HOST_BOUNCE=1` forces host-bounce path even when CUDA reports peer capability. Useful for testing non-peer path on peer-capable hardware.

## Relationship

- **Depends on**: [layer-packing-engine.md](layer-packing-engine.md) (`ds4_compute_layer_placement`), [gpu-tensor-primitives.md](gpu-tensor-primitives.md) (`ds4_gpu_mgpu.h`), CUDA backend, selective cache (`ds4_gpu_device_cache_tensors`)
- **Used by**: CLI (`--gpu-vram`), server (multi-GPU config)
- **Alternatives**: [distributed-protocol.md](distributed-protocol.md) (multiple machines, network), tensor parallelism (splits experts across GPUs within one layer)
- **Differs from distributed**: no network, GPU peer DMA, single process, CUDA-only
- `ds4_engine_create_with_gpu_config` is ABI-extensible: NULL config produces bit-equivalent engine to `ds4_engine_open`

[← Back to Index](../README.md)
