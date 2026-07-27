# Metal Backend

## Files

- `ds4_metal.m` — Objective-C Metal runtime: device setup, buffer management, kernel dispatch, graph orchestration
- `metal/*.metal` — GPU compute kernels (attention, FFN, MoE, RoPE, quant, etc.)

## Purpose

Primary production backend. Apple Metal GPU inference via custom compute kernels. Weights stay in mmap; Metal wraps slices as no-copy MTLBuffers.

## Kernel Categories

Compute kernels organized by model pipeline stage. Fused kernels combine adjacent operations to reduce dispatch overhead (e.g., Q+K+V projection fused into one kernel).

See [metal-graph.md](../engine/metal-graph.md) for Metal decode pipeline and graph construction.

## Command Buffer

One command buffer per layer or per group of layers. Commits to a serial MTLCommandQueue. No inter‑layer synchronization needed — Metal guarantees ordered execution within a queue. Full-buffer readback during diagnostic dumps (env `DS4_METAL_GRAPH_DUMP_PREFIX`). Per-step selective readbacks during normal inference — see GPU Readback below.

## Memory Model

GGUF mmap'd with `MAP_SHARED`. Metal wraps mmap'd slices via `newBufferWithBytesNoCopy` with `MTLResourceStorageModeShared` — zero‑copy for weights: GPU accesses mmap'd pages through a Metal buffer wrapper, not the OS page cache directly. All tensor dimensions known at compile time (profile‑specific constants). Persistent activation tensors allocated once at graph setup. Transient buffers (attention masks, group IDs, RoPE positions) and scratch buffer growth may allocate at runtime.

### `g_model_buffer_cache`

`NSMutableDictionary<NSString*, id<MTLBuffer>>` that caches MTLBuffer wrappers around mmap'd model weight regions. Keyed by `"%p:%llu:%llu:%llu"` (base pointer, model size, page-aligned offset, page-aligned length). On first access to a model weight range, `ds4_gpu_wrap_model_exact_range` creates a zero-copy `MTLBuffer` via `newBufferWithBytesNoCopy` and inserts it into the cache. Subsequent accesses to the same range hit the cache, avoiding redundant Metal buffer creation. Cache eviction (via `ds4_gpu_model_buffer_cache_clear` / `ds4_gpu_model_buffer_cache_maybe_evict`) is triggered when total cached bytes exceed the limit set by `DS4_METAL_EXACT_VIEW_CACHE_LIMIT` environment variable. Eviction removes all entries and increments `g_model_buffer_cache_evictions`.

## Activation Tensors

Persistent activation tensors (`g->cur`, `g->next`, attention/FFN intermediates, KV cache, MTP scratch, dspark capture, etc.) are allocated via `ds4_gpu_tensor_alloc()` at graph setup time, in `metal_graph_alloc()` and `glm_graph_verify_ws_init()`. These use `MTLResourceStorageModeShared` buffers allocated with `newBufferWithLength`.

Transient MTLBuffers (attention masks, attention output group IDs, set-rows indices, RoPE positions) are allocated per‑step via `ds4_gpu_new_transient_buffer()` with `MTLResourceStorageModeShared`. Scratch buffers (`ds4_gpu_ensure_scratch_buffer()`) use shared mode by default, falling back to private on M5 when CPU access is not needed.

## GPU Readback

Readback during normal (non‑diagnostic) inference occurs in two cases:

- **CPU router fallback** (`ds4_gpu_cpu_router_run()` at ds4.c:~20539): reads back FFN norm tensor via `ds4_gpu_tensor_read()`, computes expert selection on CPU, writes results back via `ds4_gpu_tensor_write()`.
- **Streaming prefill selected‑ID readahead** (`ds4_gpu_stream_prefill_selected_readback()` at ds4.c:~18053): reads back router selected expert IDs via `ds4_gpu_tensor_read()` to determine which expert weights to prefetch.
- **Selected‑ID overlap** (`ds4_gpu_signal_selected_readback_ready()` / `ds4_gpu_commit_and_wait_selected_readback()`: uses `MTLSharedEvent` to pipeline expert cache with the decode loop without a full GPU sync.

These readbacks synchronize only the relevant subset, not the entire GPU pipeline.

## GPU Graph

Graph state tracked in `ds4.c` (metal-graph section). Decode/prefill orchestration managed through command buffer lifecycle: encode kernels, commit, wait, recycle.

See [gpu-tensor-primitives.md](../concepts/gpu-tensor-primitives.md) for GPU tensor operation primitives.

## Invariants

- All tensor dimensions compile‑time constants (per profile).
- Persistent activation tensors allocated once at graph setup. Transient allocations (attention masks, group IDs) may occur per‑step. Scratch buffers grow lazily.
- Command buffer commit per layer or layer group.
- GPU readback on diagnostic dump, CPU router fallback, and streaming prefill selected-ID readahead.

## See Also

- [Metal Graph](../engine/metal-graph.md)
- [Quantization Kernels](../engine/quant-kernels.md)
- [KV Cache](../engine/kv-cache.md)
- [Attention](../engine/attention.md)
- [MoE FFN](../engine/moe-ffn.md)
- [GPU Tensor Primitives](../concepts/gpu-tensor-primitives.md)

[← Back to Index](../README.md)
