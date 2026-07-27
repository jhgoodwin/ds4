# Imatrix Collection

## Files

- `ds4.c`

## Purpose

Collect importance matrix data for quantization. Observes activation statistics during Metal prefill and writes llama.cpp-compatible `.dat` format consumed by `gguf-tools/deepseek4-quantize.c`.

See [imatrix/README.md](../../../gguf-tools/imatrix/README.md) for operational usage (CLI flags, dataset preparation, running imatrix).

## Collector Struct

```c
typedef struct {
    float *gate_up_sum2;       // [n_layer][n_expert][n_embd]
    float *down_sum2;          // [n_layer][n_expert][n_ff_exp]
    uint32_t gate_up_count[DS4_MAX_LAYER][DS4_MAX_EXPERT];  // activation count per expert
    uint32_t down_count[DS4_MAX_LAYER][DS4_MAX_EXPERT];     // activation count per expert
    float *ffn_norm_buf;       // [cap_tokens][n_embd] — GPU readback target for FFN norm
    float *routed_mid_buf;     // [cap_tokens][n_expert_used][n_ff_exp] — GPU readback target
    uint16_t *routed_mid_f16_buf; // same, fp16 variant
    int   *selected_buf;       // [cap_tokens][n_expert_used] — expert selection indices
    float *sq_tmp;             // [n_embd] — scratch for per-expert sum-of-squares
    uint32_t cap_tokens;       // max tokens per batch (from prefill chunk size)
    uint64_t observed_tokens;  // total tokens processed across all chunks
    uint64_t observed_routes;  // total routed expert invocations (sum over experts)
    uint32_t chunks;           // number of batch calls (chunks)
    const char *dataset_path;  // source dataset path, embedded in output metadata
} ds4_imatrix_collector;
```

Importance values accumulate across all prompted tokens. Two independent sums-of-squares per layer per expert:

- **Gate/up** (`gate_up_sum2`): `n_embd` dims. Squared FFN input norms, accumulated per expert based on router selection. Captures how strongly each expert's input varies across tokens.
- **Down** (`down_sum2`): `n_ff_exp` dims. Squared routed mid activations, accumulated per expert per token. Captures activation variance in the expert's feed-forward output.

Readback buffers (`ffn_norm_buf`, `routed_mid_buf`, `routed_mid_f16_buf`, `selected_buf`) are sized to `cap_tokens` and reused across layers within a batch. The fp16 variant of `routed_mid_buf` is used when the Metal graph materializes routed mid in fp16 to save bandwidth.

Counters track observations per expert for normalization at save time. Experts with zero observations emit neutral (1.0f) importance.

`imatrix_gate_up_ptr` / `imatrix_down_ptr` compute flat offsets into `gate_up_sum2` / `down_sum2` via `(layer * n_expert + expert) * dim`.

## Buffer Allocation

`imatrix_collector_init` allocates:

| Buffer | Size | Allocation | Purpose |
|--------|------|------------|---------|
| `gate_up_sum2` | `n_layer * n_expert * n_embd` floats | `xcalloc` | Accumulated gate+up sum-of-squares |
| `down_sum2` | `n_layer * n_expert * n_ff_exp` floats | `xcalloc` | Accumulated down sum-of-squares |
| `ffn_norm_buf` | `cap_tokens * n_embd` floats | `xmalloc` | GPU readback: FFN input norms |
| `routed_mid_buf` | `cap_tokens * n_expert_used * n_ff_exp` floats | `xmalloc` | GPU readback: expert mid activations |
| `routed_mid_f16_buf` | same size in fp16 | `xmalloc` | GPU readback variant for fp16 mid |
| `selected_buf` | `cap_tokens * n_expert_used` ints | `xmalloc` | GPU readback: expert selection |
| `sq_tmp` | `n_embd` floats | `xmalloc` | Per-expert temp reduction |

Accumulators use `xcalloc` (zero-initialized) since they accumulate via `+=`. Readback buffers use `xmalloc` (uninitialized) — they're filled by GPU reads before CPU consumption. `cap_tokens` defaults to 1 if zero is passed (guard against degenerate configs).

Gate/up and down accumulators are flat arrays indexed by `(layer * n_expert + expert) * dim` via `imatrix_gate_up_ptr` / `imatrix_down_ptr`.

## GPU Readback Flow

`imatrix_collect_layer_batch` is called once per layer, after the GPU command buffer for that layer has completed execution. The GPU must be synchronized before reads — this happens implicitly because `ds4_gpu_end_commands` (called after each layer's encode) commits the command buffer, and `ds4_gpu_tensor_read` operates on the already-committed buffer's `contents` pointer (Metal) or issues a synchronous `cudaMemcpyDeviceToHost` (CUDA).

Three GPU tensors are read back per layer:

1. **FFN norm** (`batch_ffn_norm`): `n_tokens * n_embd` floats. Each token's norm vector is element-wise squared into `sq_tmp`, then accumulated into `gate_up_sum2` at the expert slot identified by `batch_router_selected` for that token.
2. **Expert selection** (`batch_router_selected`): `n_tokens * n_expert_used` ints. Identifies which experts were activated per token per MoE slot. Validated (expert index in range `[0, n_expert)`). Used to dispatch both norm contributions and mid activations to the correct expert's accumulator.
3. **Routed mid** (`batch_routed_mid`): `n_tokens * n_expert_used * n_ff_exp` floats or fp16. Read into `routed_mid_buf` or `routed_mid_f16_buf` depending on `batch_routed_mid_is_f16`. Each value is squared and accumulated into `down_sum2` per expert.

### Sync Points

| Backend | Mechanism |
|---------|-----------|
| **Metal** | `ds4_gpu_end_commands` commits the MTLCommandBuffer. `ds4_gpu_tensor_read` calls `memcpy` from `[MTLBuffer contents]` — the buffer's GPU work is implicitly completed because the command buffer was committed before the read. No explicit `waitUntilCompleted` needed per layer; the sequential encode → commit → read cycle provides ordering. |
| **CUDA** | `ds4_gpu_tensor_read` issues `cudaMemcpyDeviceToHost` which is synchronous (blocks until GPU work completes). `ds4_gpu_synchronize` calls `cudaDeviceSynchronize` for explicit sync at error-recovery points. |

Counters (`gate_up_count`, `down_count`) track how many observations per expert to enable correct normalization at save time.

## Chunked Prefill Interaction

`ds4_engine_collect_imatrix` allocates a single `ds4_imatrix_collector` and passes it through the prefill pipeline:

- **Short prompts** (`prompt.len <= prefill_cap`): collector passed directly to `metal_graph_prefill_layer_major` as the `imatrix` parameter. Inside, it's called per layer via `imatrix_collect_layer_batch` after each layer's GPU commands complete.
- **Long prompts** (`prompt.len > prefill_cap`): collector passed to `metal_graph_prefill_chunked_range`, which iterates over chunks and forwards the same collector pointer to each `metal_graph_prefill_layer_major` call. The collector is also present in the non-streaming path (chunked prefill disables the streaming decode shortcut when `imatrix != NULL`).

### Persistence Across Chunks

The accumulator buffers (`gate_up_sum2`, `down_sum2`, counters) persist across all chunks of all prompts. Each `imatrix_collect_layer_batch` call adds to the running sums. This means:

- A 100k-token prompt split into 10 chunks produces the same accumulated importance as a single-batch 100k-token prefill.
- Multiple prompts in the dataset file accumulate into the same buffers — the collector is initialized once before the prompt loop and saved once after.
- `observed_tokens`, `observed_routes`, and `chunks` counters provide progress reporting and metadata for the output file.

### Layer-Major Dispatch

`metal_graph_prefill_layer_major` processes all layers for a batch. When `imatrix` is non-NULL, the `split_commands` path is forced. This ensures each layer gets its own command buffer commit, making the GPU tensor data available for CPU readback before the next layer starts encoding. Without split commands, all layers share one command buffer and intermediate tensors would be overwritten.

## Format Writer

`imatrix_collector_save` writes llama.cpp legacy `.dat` format:

### Binary Layout

```
int32_t  entry_count          // n_layer * 3
--- per entry (3 per layer: gate, up, down) ---
  int32_t  name_len           // length of tensor name string
  char[]   name               // tensor name (e.g. "blk.0.ffn_gate_exps.weight")
  int32_t  ncall              // always 1
  int32_t  nval               // n_expert * n_col
  float[]  values             // n_expert blocks of n_col normalized floats
--- trailing metadata ---
int32_t  chunks               // total batch calls across all prompts
int32_t  dataset_len          // length of dataset path string
char[]   dataset_path         // source dataset path (may be empty)
```

### Entry Generation

Three entries per layer: gate, up, down. Gate and up share the same `gate_up_sum2` data (and `gate_up_count`) since both projections receive the same FFN input norm. Down uses `down_sum2` / `down_count`. Tensor names are extracted from `ds4_layer_weights` (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`).

### Normalization

Each expert's `n_col` accumulated sum-of-squares values are divided by the expert's observation count (`gate_up_count[il][expert]` or `down_count[il][expert]`). Experts with zero observations emit 1.0f for all values (neutral importance — no bias toward or against that expert during quantization).

### Trailing Metadata

After all tensor entries, the file stores `chunks` (int32) and `dataset_path` (length-prefixed string). These are diagnostic fields not consumed by the quantizer, but useful for verifying which dataset produced a given imatrix file.

### Output

Output path is the `output_path` argument to `ds4_engine_collect_imatrix`. Written via `fopen`/`fwrite`; failure calls `ds4_die`.

## Quantization Impact

Outputs one tensor entry per expert with `n_expert * n_columns` floats. Quantizer (`gguf-tools/deepseek4-quantize.c`) slices the vector per expert for importance-weighted quantization — experts with higher activation variance retain more precision. Accessed via `imatrix_load` → `imatrix_find` during quantization metadata setup.

## Performance

Collection adds no inference overhead when disabled (no-op hooks — the `imatrix` pointer is NULL in all prefill paths). When enabled, overhead components:

- **GPU readback bandwidth**: `ffn_norm_buf` (n_tokens × n_embd × 4 bytes), `selected_buf` (n_tokens × n_expert_used × 4 bytes), `routed_mid_buf` (n_tokens × n_expert_used × n_ff_exp × 4 or 2 bytes) per layer per batch. For 61 layers, 4096 tokens, this is ~8 GB of reads per batch.
- **CPU reduction**: element-wise squaring and accumulation into per-expert sums. O(n_tokens × n_embd + n_tokens × n_expert_used × n_ff_exp) per layer.
- **Command buffer splitting**: forcing split_commands when imatrix is enabled adds per-layer command buffer commit overhead vs a single monolithic buffer.

Readbacks exploit already-materialized tensors — no additional inference passes beyond standard prefill. The tensors (`batch_ffn_norm`, `batch_routed_mid`, `batch_router_selected`) are produced as intermediates during normal MoE prefill.

## See Also

- [gguf-format.md](../concepts/gguf-format.md) — quant format specifications relevant to importance-weighted quantization
- [gpu-tensor-primitives.md](../concepts/gpu-tensor-primitives.md) — GPU tensor readback primitives used during collection

[← Back to Index](../README.md)
