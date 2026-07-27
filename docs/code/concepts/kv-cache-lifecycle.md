# KV Cache Lifecycle

## Definition

KV cache stores key-value pairs from previous tokens so attention can reference them.  DS4 uses three-tier cache: raw sliding window, compressed rows, and indexer mask.

## Why It Exists

Autoregressive decoding needs all previous KV pairs.  Without caching, each token would recompute all previous attention.  Compression reduces cache size ~100× for long contexts.

## Where It Appears

| File | Role |
|---|---|
| `ds4.c` | cache management, compression ratios, E4M3 compression |
| `ds4.c` (session-snapshots) | disk serialization |

## Lifecycle Stages

```
session_create → allocate cache at max context size
session_sync → populate cache (prefill)
session_eval → append one token, compress if ratio boundary reached
session_snapshot → serialize cache to disk
session_restore → deserialize cache from disk
session_free → free cache
```

## Raw KV Cache

128-row buffer per layer.  CPU path uses sliding window (memmove shifts rows left on overflow).  GPU path uses circular buffer (modulo indexing, `(pos + t) % raw_cap`).  Values are E4M3-quantized then fp16-round-tripped (stored as float).  Overwritten oldest-first when full.

## GPU Managed KV Cache Lifecycle

GPU KV cache mirrors the three-tier CPU structure (raw, compressed, indexer) with per-layer buffers allocated during graph construction in `metal_graph_alloc_raw_cap`.  CUDA uses `cudaMallocManaged` for large caches (`ds4_gpu_should_use_managed_kv_cache` thresholds at 8 GiB KV or tight free VRAM); Metal always uses regular buffer allocation (`should_use_managed_kv_cache` returns 0).

### Allocation

Each layer gets:
- **Raw cache** (`layer_raw_cache[il]`): `raw_cap` × `head_dim` × f32.  Capacity = `DS4_N_SWA` (128) clamped to context size.
- **Compressor state** (`layer_attn_state_kv/score[il]`): `coff * ratio` rows × `coff * head_dim` — a small ring pool for the weighted-merge compressor.
- **Compressed cache** (`layer_attn_comp_cache[il]`): `comp_cap` (`ctx_size / ratio + 2`) × `head_dim`.  Stores pooled/roped/normed rows.  On CUDA, optionally stored as f16 (`DS4_GPU_ATTN_COMP_CACHE_F16`).
- **Indexer state** (`layer_index_state_kv/score[il]`): same shape as compressor state but projected to `DS4_N_INDEXER_HEAD_DIM`.  Ratio-4 layers only.
- **Indexer compressed cache** (`layer_index_comp_cache[il]`): same count as compressed cache but `head_dim = DS4_N_INDEXER_HEAD_DIM`.  Ratio-4 layers only.

Multi-GPU CUDA: allocation routes through `metal_graph_alloc_kv_cache_tensor_on` which dispatches to `ds4_gpu_tensor_alloc_managed_on(tier)` (managed memory) or `ds4_gpu_tensor_alloc_ptr_on(tier)` (device-only).  Tensor-parallel configs allocate per-layer buffers on the layer's assigned tier; optional TP-attn duplicates raw+comp caches on the partner tier.

### Raw Store (GPU)

- **CUDA**: `store_raw_kv_batch_kernel` writes `__half2float(__float2half(kv[t * head_dim + d]))` at row `(pos0 + t) % raw_cap`.  Circular buffer — oldest row overwritten when window advances past capacity.
- **Metal**: `ds4_gpu_store_raw_kv_batch_tensor` uses explicit f16 round-trip pipeline: `ds4_gpu_encode_f16_round_copy_for_raw_store` (f32→f16 blit, then f16→f32 blit via `g_f16_round_scratch_buffer`/`g_raw_store_round_buffer`), then `ds4_gpu_encode_set_rows_f32_i32` writes modulo-indexed rows.  Same `(pos0 + t) % raw_cap` circular behavior.

Both paths guarantee E4M3 round-trip (P0 invariant).

### GPU Compression

Driven by `compressor_decode_one` on CPU; on GPU, split into explicit kernel stages:

1. **Store** (`compressor_store_kernel`): copies incoming KV+score into state buffer at `pos % ratio` slot, adding APE bias.  CUDA: single kernel launch.  Metal: equivalent `kernel_compressor_store`.
2. **Pool** (`compressor_prefill_pool_kernel` / `compressor_update_pool_kernel`): softmax-weighted merge of candidates → compressed row.  Prefill processes `n_tokens / ratio` compressed rows; decode processes one row when `(pos + 1) % ratio == 0`.
3. **Post-process**: RMS norm (via `ds4_gpu_rms_norm_weight_rows_tensor`), RoPE tail rotation (via `rope_tail_kernel`), optional fp8 quantization (`ds4_gpu_dsv4_fp8_kv_quantize_tensor` → E4M3 encoding, **not stored** — only round-tripped for attention compute).

Ratio-4 layers use a dual-buffer state scheme: `coff=2` splits K and V halves.  `compressor_shift_ratio4_kernel` shifts the state window on decode emit.

See [attention-compression.md](attention-compression.md) for ratio details and merge scheme.

### GPU Indexer Interaction

Ratio-4 layers maintain a separate indexer compressed cache (`layer_index_comp_cache[il]`).  During prefill, `ds4_gpu_compressor_prefill_tensor` writes compressed rows to both attn and indexer caches.  During decode:

1. Compressed row emitted → stored in indexer compressed cache (monotonic append).
2. `ds4_gpu_dsv4_indexer_qat_tensor` quantizes (QAT-aware) the row before indexer scoring.
3. `ds4_gpu_glm_indexer_score_one_tensor` / `ds4_gpu_glm_indexer_scores_batch_tensor` reads from `index_comp` tensor (the indexer compressed cache) to produce scores per compressed row.
4. GLM models also keep an **indexer key cache** (`ds4_gpu_glm_store_indexer_k_tensor`): raw K → normed/roped key stored in a separate circular buffer (`indexer_key_cache`), read by `ds4_gpu_glm_indexer_score_one_tensor` via the `indexer_key_cache` argument.

See [indexer-subsystem.md](indexer-subsystem.md) for indexer architecture.

### Eviction

No eviction on GPU for KV cache itself:
- Raw cache is fixed-capacity circular (overwrites oldest, never freed until session end).
- Compressed and indexer caches grow monotonically until context end.
- Metal maintains `ds4_gpu_model_buffer_cache_maybe_evict` and streaming expert cache eviction (`g_stream_expert_cache_evictions`), but these manage model weights, not KV cache.

### CUDA vs Metal Differences

| Aspect | CUDA | Metal |
|---|---|---|
| Managed memory | `cudaMallocManaged` for KV >8 GiB or tight VRAM | Always regular allocation (metal heap) |
| E4M3 round-trip | Inline `__half2float(__float2half(...))` in kernel | Explicit f32→f16→f32 blit pipeline via `g_f16_round_scratch_buffer` |
| Multi-tier | Full TP support per-layer on tier | Single-device (multi-GPU through separate graphs) |
| Compressor kernels | `__global__` CUDA kernels | Metal Shading Language kernels via pipeline cache |
| Flash ring buffer | N/A | `g_flash_attn_ring_buffer` for Metal flash attention |
| Compiler fence | Implicit kernel launch ordering | Explicit `MTLCommandBuffer` serialization |

## Compression Trigger

See [attention-compression.md](attention-compression.md) for compression scheme details (CSA/HCA ratios, E4M3 encoding).

## Compressed KV Cache

Grows monotonically as tokens process.  Each compressed row represents N raw rows (ratio-dependent).  Values stored as f32 but round-tripped through E4M3 encoding — the container is f32, each value carries only E4M3 precision (~3 mantissa bits + 4 exponent bits, ~8-bit effective).  Not compact storage: each element occupies 4 bytes (f32) but encodes only E4M3-range information.

Memory figures in this doc are stale and incorrect — do not rely on them.  Recalculate with actual head_dim=512 and per-model n_layer (43 Flash, 61 Pro).  Raw window per layer: n_swa (128) × head_dim (512) × 4 B = 256 KB.

## Indexer Update

See [indexer-subsystem.md](indexer-subsystem.md) for indexer architecture and search algorithm.

## Relationship

- **Used by**: [attention-compression.md](attention-compression.md) (compression scheme), attention pipeline (reads from cache), session snapshots (serializes cache)

[← Back to Index](../README.md)
