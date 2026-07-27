# GPU Tensor Primitives & Command Buffer

## Definition

GPU tensor layer abstracts device memory management, kernel submission, and synchronization behind a command-buffer model. Every backend exposes the same API: allocate tensors, begin command sequence, enqueue operations, flush, commit, wait. Backend-specific differences hidden behind uniform signatures.

The core data type is `ds4_gpu_tensor` (defined in `ds4_gpu_mgpu.h`):

```c
struct ds4_gpu_tensor {
    void    *ptr;        /* device pointer (cudaMalloc ptr, MTLBuffer backing) */
    uint64_t bytes;      /* allocated size in bytes */
    int      owner;      /* owner ID for tracking */
    int      device_id;  /* -1 = legacy/untagged → device 0 */
};
```

Backend differences at the tensor level:
- **Metal**: `ptr` is a `MTLBuffer` (shared memory), `device_id` unused (single-device)
- **CUDA**: `ptr` is a `cudaMalloc` device pointer, `device_id` targets a specific GPU
- **ROCm**: `ptr` is a `hipMalloc` device pointer (CUDA-compatible surface)

## Why It Exists

Decouples graph construction (metal_graph in `ds4.c`) from backend specifics. Engine writes one graph encoding pass; GPU tensor layer translates to MTLCommandBuffer / CUDA stream operations. Enables cross-backend correctness without backend-specific graph code.

## Where It Appears

| File | Symbol | Role |
|---|---|---|
| `ds4_gpu.h` | `ds4_gpu_tensor` (typedef, opaque) | Forward declaration of device tensor handle |
| `ds4_gpu_mgpu.h` | `struct ds4_gpu_tensor` | Fields: `ptr`, `bytes`, `owner`, `device_id` |
| `ds4_gpu.h` | `ds4_gpu_tensor_alloc` | Allocate device-local tensor (default device) |
| `ds4_gpu.h` | `ds4_gpu_tensor_alloc_managed` | Allocate managed/unified memory tensor |
| `ds4_gpu.h` | `ds4_gpu_tensor_view` | Sub-range view without new allocation |
| `ds4_gpu.h` | `ds4_gpu_tensor_free` | Release tensor and tracking |
| `ds4_gpu.h` | `ds4_gpu_tensor_bytes` | Query tensor byte size |
| `ds4_gpu.h` | `ds4_gpu_tensor_contents` | CPU-mapped pointer for shared memory |
| `ds4_gpu.h` | `ds4_gpu_tensor_write` | CPU → GPU data transfer |
| `ds4_gpu.h` | `ds4_gpu_tensor_read` | GPU → CPU data transfer |
| `ds4_gpu.h` | `ds4_gpu_tensor_copy` | GPU → GPU transfer via blit encoder |
| `ds4_gpu.h` | `ds4_gpu_tensor_copy_f32_to_f16` | Format conversion copy |
| `ds4_gpu.h` | `ds4_gpu_tensor_fill_f32` | Fill tensor region with scalar float |
| `ds4_gpu.h` | `ds4_gpu_begin_commands` | Open command encoder / begin command recording |
| `ds4_gpu.h` | `ds4_gpu_flush_encoder` | Close encoder, keep command buffer open |
| `ds4_gpu.h` | `ds4_gpu_flush_commands` | Commit command buffer + device sync, open new |
| `ds4_gpu.h` | `ds4_gpu_commands_active` | Check whether command buffer is open |
| `ds4_gpu.h` | `ds4_gpu_end_commands` | Close + submit + wait for completion |
| `ds4_gpu.h` | `ds4_gpu_synchronize` | Wait for all pending GPU work |
| `ds4_gpu.h` | `ds4_gpu_signal_selected_readback_ready` | Record sync event in current command buffer |
| `ds4_gpu.h` | `ds4_gpu_commit_and_wait_selected_readback` | Commit + wait on event |
| `ds4_gpu.h` | `ds4_gpu_wait_selected_readback_ready` | Wait on event only |
| `ds4_gpu.h` | `ds4_gpu_set_model_map` | Register full host model mapping |
| `ds4_gpu.h` | `ds4_gpu_cache_model_range` | Cache weight range to device memory |
| `ds4_gpu.h` | `ds4_gpu_cache_q8_f16_range` | Pre-quantize weight range to Q8_0 on device |
| `ds4_gpu.h` | `ds4_gpu_print_memory_report` | Diagnostic memory report to stderr |
| `ds4_gpu.h` | `ds4_gpu_recommended_working_set_size` | Query VRAM budget recommendation |
| `ds4_metal.m` | `ds4_gpu_tensor_alloc` (impl) | MTLBuffer with `MTLResourceStorageModeShared` |
| `ds4_metal.m` | `ds4_gpu_begin_commands` (impl) | Creates MTLCommandBuffer via `[g_device newCommandBuffer]` |
| `ds4_metal.m` | `ds4_gpu_flush_commands` (impl) | Close encoder, commit CB to pending list, create new CB |
| `ds4_metal.m` | `ds4_gpu_tensor_contents` (impl) | `[obj.buffer contents] + obj.offset` |
| `ds4_cuda.cu` | `ds4_gpu_tensor_alloc` (impl) | `cudaMalloc` on device 0 |
| `ds4_cuda.cu` | `ds4_gpu_tensor_alloc_managed` (impl) | `cudaMallocManaged` on device 0 |
| `ds4_cuda.cu` | `ds4_gpu_begin_commands` (impl) | No-op (returns 1); CUDA streams are implicit |
| `ds4_cuda.cu` | `ds4_gpu_flush_commands` (impl) | `cudaDeviceSynchronize()` |
| `ds4_cuda.cu` | `ds4_gpu_tensor_contents` (impl) | `cudaDeviceSynchronize() + t->ptr` |
| `ds4_cuda.cu` | `ds4_gpu_end_commands` (impl) | `cudaStreamSynchronize(0)` or `cudaDeviceSynchronize()` |

## Lifecycle

Command-buffer lifecycle spans tensor create → command sequence → sync → teardown.

```
tensor alloc → begin commands → [alloc/ view/ enqueue kernels] → flush encoder
  → [signal readback/ more kernels] → flush commands → ... → synchronize
  → [read data] → tensor free
```

### Create / Destroy Tensors

| Phase | Functions | Metal | CUDA |
|---|---|---|---|
| Allocate | `ds4_gpu_tensor_alloc(bytes)` | `MTLResourceStorageModeShared` buffer | `cudaMalloc` on device 0 |
| Allocate managed | `ds4_gpu_tensor_alloc_managed(bytes)` | Delegates to alloc (Metal always shared) | `cudaMallocManaged` on device 0 |
| Sub-range view | `ds4_gpu_tensor_view(base, offset, bytes)` | Wraps same `MTLBuffer` with offset | Wraps same `ptr` with offset |
| Destroy | `ds4_gpu_tensor_free(tensor)` | Release `MTLBuffer`, untrack | `cudaFree` + free struct |

### Allocate / Free Buffers

Tensor allocation is the buffer allocation — no separate buffer step. `ds4_gpu_tensor_alloc` creates the device memory. `ds4_gpu_tensor_free` releases it. Model weight caching adds a secondary path:

- `ds4_gpu_cache_model_range` — page a host model range to device memory
- `ds4_gpu_cache_q8_f16_range` — pre-quantize to Q8_0 on device
- `ds4_gpu_lookup_cache` — query cached range, returns device pointer

### Map / Unmap (CPU Access)

`ds4_gpu_tensor_contents(tensor)` returns a CPU-mapped pointer:

- **Metal**: `(uint8_t *)[obj.buffer contents] + obj.offset` — direct shared-memory pointer into the MTLBuffer, adjusted for tensor view offset.
- **CUDA**: `cudaDeviceSynchronize(); return t->ptr` — full device sync then returns device pointer (managed memory assumed for CPU access).
- **ROCm**: mirrors CUDA with `hipDeviceSynchronize()`.

Explicit unmap not required — Metal shared memory is always mapped; CUDA managed memory pages on access.

### Sync Points

| Function | Metal | CUDA / ROCm |
|---|---|---|
| `ds4_gpu_flush_commands()` | Close encoder, commit CB to pending list, create new CB | `cudaDeviceSynchronize()` |
| `ds4_gpu_end_commands()` | Close + submit + wait for completion | `cudaStreamSynchronize(0)` or `cudaDeviceSynchronize()` |
| `ds4_gpu_synchronize()` | Wait for all pending command buffers | `cudaDeviceSynchronize()` |
| `ds4_gpu_signal_selected_readback_ready(v)` | `[cb encodeSignalEvent:g_event value:v]` | `cudaDeviceSynchronize()` |
| `ds4_gpu_commit_and_wait_selected_readback(v,l)` | Commit CB, wait on shared event value | `cudaDeviceSynchronize()` |
| `ds4_gpu_wait_selected_readback_ready(v,l)` | Wait on shared event value only | `cudaDeviceSynchronize()` |

Metal uses `MTLSharedEvent` for fine-grained CPU/GPU synchronization without full drain. CUDA/ROCm map these to `cudaDeviceSynchronize()` because CUDA streams execute in order — no split possible.

## Variants

| Variant | When Used | Key Difference |
|---|---|---|
| **Metal** | Apple Silicon (macOS/iOS) | Shared memory (`MTLResourceStorageModeShared`); command buffers via `[g_device newCommandBuffer]`; `MTLSharedEvent` for async readback signaling |
| **CUDA** | NVIDIA GPUs | Explicit `cudaMalloc`/`cudaFree`; `cudaDeviceSynchronize()` for all sync; `ds4_gpu_tensor_contents` includes sync before returning pointer |
| **ROCm** | AMD GPUs (HIP compat) | CUDA-compatible surface: `hipMalloc`/`hipFree`; sync maps to `hipDeviceSynchronize()`; `ds4_gpu_tensor_read_after_selected_event` (ROCm-only) combines event wait + readback in one call |
| **Multi-GPU (CUDA)** | 2+ NVIDIA GPUs | `ds4_gpu_tensor_alloc_on` targets specific device; `g_gpu[]` array, peer-to-peer via `g_gpu_peer_ok[]`; tier-based VRAM queries (`ds4_gpu_tier_free_vram`) |

### Per-Backend `ds4_gpu_tensor_contents` Detail

P0 fix — the content function differs critically by backend:

- **Metal**: `(uint8_t *)[obj.buffer contents] + obj.offset` — uses the tensor *object*'s buffer and applies the view offset. Without `obj.offset`, tensor views return wrong base address.
- **CUDA**: calls `cudaDeviceSynchronize()` before returning `tensor->ptr`. Without this sync, CPU reads from the returned pointer may see stale data.
- **ROCm**: equivalent to CUDA: `hipDeviceSynchronize()` + `t->ptr`.

## Code Pattern

Minimal snippet showing tensor create → contents → use → free:

```c
/* Allocate 1 MiB tensor */
ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(1 << 20);
if (!t) abort();

/* Begin command sequence */
ds4_gpu_begin_commands();

/* CPU write via shared-memory pointer (Metal) or managed ptr (CUDA) */
float *cpu = (float *)ds4_gpu_tensor_contents(t);
for (int i = 0; i < 256; i++) cpu[i] = 1.0f;

/* Enqueue a kernel (e.g. fill) */
ds4_gpu_tensor_fill_f32(t, 0.5f, 256);

/* Flush and sync */
ds4_gpu_flush_commands();

/* Read back */
float result[256];
ds4_gpu_tensor_read(t, 0, result, sizeof(result));

/* Teardown */
ds4_gpu_tensor_free(t);
```

For a full prefill/decode loop, tensors stay allocated across the entire sequence:

```c
/* Allocate once */
ds4_gpu_tensor *activations = ds4_gpu_tensor_alloc(act_bytes);
ds4_gpu_tensor *kv_cache    = ds4_gpu_tensor_alloc(kv_bytes);

for (int pos = 0; pos < n_tokens; pos++) {
    ds4_gpu_begin_commands();
    ds4_gpu_matmul_q8_0_tensor(...);
    ds4_gpu_rms_norm_weight_tensor(...);
    ds4_gpu_rope_tail_tensor(...);
    ds4_gpu_flush_commands();
}

/* Free once */
ds4_gpu_tensor_free(activations);
ds4_gpu_tensor_free(kv_cache);
```

## Relationship

- **Depends on**: backend-specific kernel libraries (Metal Shading Language, CUDA, ROCm / HIP).
- **Used by**: metal-graph allocation (`ds4.c` metal_graph functions), model weight loading (cache/paging), [multi-gpu-pipeline.md](multi-gpu-pipeline.md) (placement planner).
- **Alternatives**: backend-implicit kernel dispatch (not portable), plain CUDA/HIP streams without tensor abstraction.

[← Back to Index](../README.md)
