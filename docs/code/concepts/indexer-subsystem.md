# Compressed Attention Indexer

## Definition

Indexer selects which compressed KV rows are visible to each token during compressed attention. For ratio-4 layers, the key cache carries an additional indexer key per row. Indexer scores each compressed row against the query, picks top-K visible rows, masks rest. Limits O(n²) attention scope to a fixed window of compressed history.

## Why It Exists

Without indexer, compressed attention would attend over all compressed rows — O(n) per token even after compression. Indexer reduces this to constant K (default DS4_N_INDEXER_TOP_K=512). 1M-context attention becomes compute-bounded by compressed window size, not full history.

## Where It Appears

| File | Symbol | Role |
|---|---|---|
| `ds4_gpu.h` | `ds4_gpu_indexer_score_one_tensor` | Single-token decode: score each compressed row vs query. Metal, CUDA, ROCm. |
| `ds4_gpu.h` | `ds4_gpu_indexer_scores_prefill_tensor` | Prefill (n tokens, pos0=0). Delegates to internal `_batch_tensor`. |
| `ds4_gpu.h` | `ds4_gpu_indexer_scores_decode_batch_tensor` | Batch decode (n tokens, non-zero pos0). Delegates to internal `_batch_tensor`. |
| `ds4_gpu.h` | `ds4_gpu_indexer_topk_tensor` | Global top-K across all compressed rows per token. |
| `ds4_gpu.h` | `ds4_gpu_indexer_top1_value_tensor` | Argmax with value extraction (alternative fused path). |
| `ds4_gpu.h` | `ds4_gpu_matmul_q8_0_top1_tensor` | Fused Q8_0 matmul + argmax (alt output projection). |
| `ds4_gpu.h` | `ds4_gpu_dsv4_topk_mask_tensor` | Produce dense attend/no-attend mask (-inf for masked, 0 for selected). |
| `ds4_gpu.h` | `ds4_gpu_dsv4_indexer_qat_tensor` | Indexer QAT: Hadamard-128 rotation + FP4 activation simulation. |
| `ds4_gpu.h` | `ds4_gpu_glm_indexer_rope_tail_tensor` | Tail-only RoPE for GLM indexer query. |
| `ds4_gpu.h` | `ds4_gpu_glm_indexer_score_one_tensor` | Single-token GLM indexer score. Handles f16/f32 cache format; direct 32x128 path. |
| `ds4_gpu.h` | `ds4_gpu_glm_indexer_scores_batch_tensor` | Batch GLM indexer scores, tiled WMMA kernel. |
| `ds4.c` | `g_ds4_shape.n_indexer_head` | DS4_N_INDEXER_HEAD (def=64) — number of indexer heads. |
| `ds4.c` | `g_ds4_shape.n_indexer_head_dim` | DS4_N_INDEXER_HEAD_DIM (def=128) — indexer head dimension. |
| `ds4.c` | `g_ds4_shape.n_indexer_top_k` | DS4_N_INDEXER_TOP_K (def=512) — top-K rows retained. |
| `ds4.c` | `g_ds4_shape.n_rot` | DS4_N_ROT (def=64) — RoPE dimension. |
| `ds4.c` | `layer_index_comp_cache[il]` | GPU tensor: compressed indexer keys. Size: `comp_cap * N_INDEXER_HEAD_DIM * sizeof(float)`. |
| `ds4.c` | `layer_index_state_kv[il]` | GPU tensor: compressor state KV. Size: `index_width * index_rows * sizeof(float)`. |
| `ds4.c` | `layer_index_state_score[il]` | GPU tensor: compressor state score. Init to DS4_NEG_INF. |
| `ds4.c` | `layer_n_index_comp[il]` | Running count of compressed indexer rows stored. |

## Indexer Architecture

### Decode Path (single token)

```
attn_norm
  → ds4_gpu_matmul_f16_tensor (indexer_proj: DS4_N_EMBD × DS4_N_INDEXER_HEAD)
  → ds4_gpu_dsv4_indexer_qat_tensor (Hadamard-128 + FP4)
  → ds4_gpu_matmul_f16_tensor (indexer weights: DS4_N_INDEXER_HEAD × DS4_N_INDEXER_HEAD_DIM)
  → ds4_gpu_indexer_score_one_tensor (Q · indexer_key[0..n_comp])
  → ds4_gpu_indexer_topk_tensor (top-K scores)
  → ds4_gpu_attention_indexed_mixed_batch_heads_tensor (indexed compressed attention)
```

Scoring and topk stages have profiling boundaries via `metal_graph_indexer_stage_profile_boundary`.

### Batch Decode Path

```
batch_attn_norm
  → ds4_gpu_dsv4_indexer_qat_tensor
  → ds4_gpu_matmul_f16_tensor (indexer weights)
  → ds4_gpu_indexer_scores_decode_batch_tensor (n tokens, pos0 > 0)
  → ds4_gpu_indexer_topk_tensor
  → ds4_gpu_attention_indexed_mixed_batch_heads_tensor
```

TP split: score/topk replicated across ranks, attention row-split via `tp_row0` offset.

### Prefill Path

```
ds4_gpu_indexer_scores_prefill_tensor (n tokens, pos0=0)
  → ds4_gpu_indexer_topk_tensor
  → ds4_gpu_attention_indexed_mixed_batch_heads_tensor
```

Prefill scores run only when `ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K`.

### Speculative Decode Path (per-token, n_comp > top_k)

```
score_one (per-token row-view from batch tensors)
  → topk
  → ds4_gpu_dsv4_topk_mask_tensor (comp_mask)
  → ds4_gpu_attention_indexed_mixed_batch_heads_tensor (with comp_mask)
```

Falls back to non-indexed `ds4_gpu_attention_decode_heads_tensor` when n_comp <= top_k or mask produces no compression.

### KV Cache Allocation

Per ratio-4 layer during Metal graph init:

```c
g->layer_index_comp_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
    managed_kv_cache, layer_tier,
    (uint64_t)g->layer_comp_cap[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float));

g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc_ptr_on(
    layer_tier, index_width * index_rows * sizeof(float));
g->layer_index_state_score[il] = ds4_gpu_tensor_alloc_ptr_on(
    layer_tier, index_width * index_rows * sizeof(float));
// index_width = coff * DS4_N_INDEXER_HEAD_DIM, index_rows = coff * ratio
```

Frontier snapshot (speculative decode) allocates per-layer `spec_index_state_kv[il]`, `spec_index_state_score[il]` (and optional `spec_prefix1_*`).

### KV Cache Population (Compression Phase)

During prefill with `zero_prefix && ratio == 4`:
1. `ds4_gpu_compressor_prefill_tensor`: APE + norm → produces compressed indexer keys into `layer_index_comp_cache[il]`.
2. `ds4_gpu_dsv4_indexer_qat_tensor`: quantize the compressed cache.
3. `metal_graph_refresh_ratio4_compressor_state`: update `layer_index_state_kv[il]` and `layer_index_state_score[il]`.

During decode (per-token, non-zero_prefix):
1. `ds4_gpu_compressor_update_tensor`: incrementally update state and produce new compressed row.
2. `ds4_gpu_dsv4_indexer_qat_tensor` on the new row.

`layer_n_index_comp[il]` tracks how many compressed rows exist.

### KV Cache Layout

```
layer_index_comp_cache[il]: [n_comp][DS4_N_INDEXER_HEAD_DIM] float
  → stored on managed KV cache (tiered for GPU memory pressure)

layer_index_state_kv[il]: [index_rows][index_width] float
  → index_width = coff * DS4_N_INDEXER_HEAD_DIM
  → index_rows = coff * ratio
  → state for incremental compressor (not consumed by indexer directly)

layer_index_state_score[il]: [index_rows][index_width] float
  → initialized to DS4_NEG_INF
  → updated by compressor to track compression quality
```

## Score Computation

Single-token score:

```
ds4_gpu_indexer_score_one_tensor(scores, q, weights, index_comp,
                                  n_comp, n_head, head_dim, scale)
```

Scale: `1.0f / sqrtf(head_dim * n_head)`. One token scored against all compressed rows. Metal dispatch: `kernel_dsv4_indexer_score_one_direct`. CUDA: `indexer_score_one_direct_kernel`. ROCm: `indexer_score_one_direct_kernel`.

Prefill score:

```
ds4_gpu_indexer_scores_prefill_tensor(scores, q, weights, index_comp,
                                       n_comp, n_tokens, n_head,
                                       head_dim, ratio, scale)
```

pos0=0. CUDA and Metal delegate to internal `_batch_tensor`.

Batch decode score:

```
ds4_gpu_indexer_scores_decode_batch_tensor(scores, q, weights, index_comp,
                                            n_comp, n_tokens, pos0,
                                            n_head, head_dim, ratio, scale)
```

pos0 > 0. CUDA and Metal delegate to internal `_batch_tensor`.

**Internal `ds4_gpu_indexer_scores_batch_tensor`** (Metal, static) is shared by prefill and decode batch. Three dispatch paths based on batch size + quality mode:

- **NAX** (`kernel_dsv4_indexer_scores_nax`): when `n_tokens >= 16` and MPP available. Tiled fp16 MMA. Threadgroup: 128 threads, shared memory `2*32*32 * sizeof(uint16_t) + 32*32 * sizeof(float)`.
- **Tiled f32** (`kernel_dsv4_indexer_scores_tiled_f32`): quality mode. Tile: 8 queries × 32 keys. Shared: `(8*128 + 32*128 + 8*32) * sizeof(float)`.
- **Tiled fp16** (`kernel_dsv4_indexer_scores_tiled`): default. Tile: 8 queries × 32 keys. Shared: `(8*128 + 32*128) * sizeof(uint16_t) + 8*32 * sizeof(float)`.

Scoring formula: `dot(Q, K[i]) * scale` where `scale = 1/sqrt(head_dim * n_head)`. Q and K are already processed through QAT.

### GLM Variants

```
ds4_gpu_glm_indexer_rope_tail_tensor(x, n_tokens, n_head, head_dim,
                                      rot_dim, pos0, n_ctx_orig,
                                      freq_base, freq_scale,
                                      ext_factor, attn_factor,
                                      beta_fast, beta_slow)
```

Tail-only RoPE for GLM indexer query. Uses GLM-specific NTK-aware scaling parameters.

```
ds4_gpu_glm_indexer_score_one_tensor(scores, q, weights,
                                      indexer_key_cache,
                                      n_rows, n_head, head_dim,
                                      scale, cache_f16)
```

Single-token GLM indexer score. Handles f16/f32 cache format. Direct 32×128 path when head_dim=128.

```
ds4_gpu_glm_indexer_scores_batch_tensor(scores, q, weights,
                                         indexer_key_cache,
                                         n_rows, n_tokens, pos0,
                                         n_head, head_dim,
                                         scale, cache_f16)
```

Batch GLM indexer scores. Metal: tiled WMMA kernel `kernel_glm_indexer_scores_batch`. ROCm: `glm_indexer_scores_batch_kernel` with WMMA tile.

## Top-K Selection

```
ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k)
```

Metal: partial argsort (`kernel_argsort_f32_i32_desc`) + iterative merge (`kernel_argsort_merge_f32_i32_desc`). Multi-pass when `n_comp` exceeds threadgroup width.

CUDA: specialized kernels for common sizes:
- `top_k == 1`: `indexer_top1_kernel` — fast path.
- `top_k <= 1024` and `n_comp <= 2048`: `indexer_topk_1024_kernel` or `indexer_topk_pow2_kernel<2048>`.
- `top_k <= 4096` and `n_comp <= 4096`: `indexer_topk_pow2_kernel<4096>` or `indexer_topk_8192_cub_kernel` (CUB radix sort).
- `n_comp > 4096` (wide mode): chunk into 4096-element blocks, `indexer_topk_chunk_pow2_kernel<4096>` per chunk, tree-merge via `indexer_topk_tree_merge_pow2_kernel<4096>`, then `indexer_topk_merge_pow2_kernel<4096>`.
- Generic fallback for other sizes.

```
ds4_gpu_indexer_top1_value_tensor(selected, values, scores,
                                   n_comp, n_tokens, index_offset)
```

Argmax with value extraction (fused alternative to separate top1+copy).

### QAT (Quantize-Aware Transform)

```
ds4_gpu_dsv4_indexer_qat_tensor(x, n_rows, head_dim)
```

Applies to both indexer Q and indexer compressor KV. Sequence:
1. `dsv4_hadamard128_inplace` — 128-wide Hadamard transform.
2. `dsv4_fp4_act_quantize_row_inplace` — E2M1 FP4 simulation (8 values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} times scale).

Without QAT, top-K compressed-row selection diverges from model's training graph.

### Fused Q8_0 Matmul + Top-1

```
ds4_gpu_matmul_q8_0_top1_tensor(selected, values, model_map,
                                  model_size, weight_offset,
                                  in_dim, out_dim, x, index_offset)
```

CUDA: `quantize_q8_0_f32` → `matmul_q8_0_top1_preq_warp8` → unpack result. Metal: stub returning 0 (not implemented).

## Indexer Mask Application

```
ds4_gpu_dsv4_topk_mask_tensor(mask, topk, n_comp, n_tokens, top_k)
```

Two ordered Metal dispatches:
1. `kernel_dsv4_topk_mask`: fill entire `[n_comp][n_tokens]` mask with `-INFINITY`.
2. `kernel_dsv4_topk_mask_scatter`: set selected rows (from topk indices) to `0.0f`.

Design: fill -inf (256-wide), scatter 0 (top_k-wide). Replaced a prior O(n_comp × n_tokens × top_k) membership test. Total work O(n_comp + top_k) per token.

Also exposes `kernel_dsv4_sort_i32_rows_asc`: sorts selected rows by row id after score-based selection. Ensures attention scans compressed K/V in cache order.

## Configuration

### Engine Options (`ds4_engine_options`)

Model shape determines indexer dimensions at load time:
- `attention.indexer.head_count` → `DS4_N_INDEXER_HEAD` (validated)
- `attention.indexer.key_length` → `DS4_N_INDEXER_HEAD_DIM` (validated)
- `attention.indexer.top_k` → `DS4_N_INDEXER_TOP_K` (validated)
- `rope.dimension_count` → `DS4_N_ROT` (validated)

### Per-Layer Weights

| Tensor | Shape | Purpose |
|---|---|---|
| `indexer_proj` | F16/F32, `[DS4_N_EMBD, DS4_N_INDEXER_HEAD]` | Embed → indexer head count projection |
| `indexer_k_norm` | F32, `[DS4_N_INDEXER_HEAD_DIM]` | Indexer key norm (V4) |
| `indexer_k_norm_b` | F32, `[DS4_N_INDEXER_HEAD_DIM]` | Indexer key norm bias (V4) |
| `indexer_compressor_ape` | — | Absolute positional encoding for compressor |
| `indexer_compressor_norm` | F32, `[DS4_N_INDEXER_HEAD_DIM]` | Compressor normalization |

## Lifecycle

```
[Graph Init Allocation] → [Build: Compression → Populate Indexer Cache]
    → [Search: Score → Top-K → Mask] → [Indexed Attention]
    → [Rebuild: Compressor State Refresh → Update Indexer State]
    → (repeat Search→Rebuild on window advance)
```

### Phases

**Init** (ds4.c ~L16977-L17001)
Graph init per ratio-4 layer. Allocates three tensors:
- `layer_index_comp_cache[il]`: managed KV cache tensor, `comp_cap * N_INDEXER_HEAD_DIM * sizeof(float)`.
- `layer_index_state_kv[il]`: compressor state KV, `index_width * index_rows * sizeof(float)`, zero-filled.
- `layer_index_state_score[il]`: compressor state score, `index_width * index_rows`, initialized to `DS4_NEG_INF`.

**Build** (Compression, ds4.c ~L21932-L21989)
Ratio-4 compress phase produces new indexer keys. Prefill path:
1. `ds4_gpu_compressor_prefill_tensor`: APE + norm → compressed indexer keys into `layer_index_comp_cache[il]`.
2. `ds4_gpu_dsv4_indexer_qat_tensor`: Hadamard-128 + FP4 quantize on compressed cache.
3. `metal_graph_refresh_ratio4_compressor_state`: update `layer_index_state_kv[il]` and `layer_index_state_score[il]`.

Decode compression (per-token, non-zero prefix):
1. `ds4_gpu_compressor_update_tensor`: incremental state update → new compressed row.
2. QAT on the new row.
3. `layer_n_index_comp[il]` incremented.

**Search** (ds4.c ~L22046-L22189)
Triggered for every decode step or prefill chunk with `n_comp > DS4_N_INDEXER_TOP_K`:
1. **Score**: Q (projected + QAT) dot-product with all compressed indexer keys.
2. **Top-K**: partial argsort, keep top `DS4_N_INDEXER_TOP_K` row indices.
3. **Mask** (optional, speculative decode): materialize dense -inf/0 mask.
4. **Indexed attention**: gather only selected compressed rows + raw SWA rows.

**Rebuild** (ds4.c ~L27248, L27570)
Compressor state refresh runs when KV window advances (e.g., context window shift or compaction triggers `comp_cap` near-exhaustion). The ratio-4 state is rebuilt from the last 4 tokens using small-batch projection for bit-exact recurrence. Refreshed via `metal_graph_refresh_ratio4_compressor_state`. New indexer keys populate `layer_index_comp_cache[il]` and update `layer_index_state_kv[il]`/`layer_index_state_score[il]`.

### Error transitions

- Allocation failure during init → graph init aborts (tensor NULL checks, `layer_cache_ok`).
- `layer_n_index_comp[il]` exceeds `layer_comp_cap[il]` → compression disabled for that layer until rebuild.
- QAT produces no valid rows after top-K → falls back to non-indexed `ds4_gpu_attention_decode_heads_tensor`.

### Relationship to KV cache lifecycle

Indexer lifecycle mirrors the compressed KV cache. Each ratio-4 compression trigger (window advance, context shift) invalidates the current indexer state; the next rebuild populates fresh indexer keys and resets `layer_n_index_comp[il]`. The indexer score/top-k/mask pipeline only runs when compressed rows exist (`n_comp > 0`) and exceeds the top-K threshold. See [kv-cache-lifecycle.md](kv-cache-lifecycle.md).

## See Also

- [attention-compression.md](attention-compression.md) — compression scheme details (CSA/HCA ratios, E4M3 encoding)
- [kv-cache-lifecycle.md](kv-cache-lifecycle.md) — KV cache lifecycle and compression triggers

## Relationship

- **Depends on**: [kv-cache-lifecycle.md](kv-cache-lifecycle.md) (compressed rows must exist before indexer), compressor (produces indexer keys), [attention-compression.md](attention-compression.md) (ratio-4 layers require indexer), RoPE (query gets tail-only rotation before scoring).
- **Used by**: attention (decode, batch decode, prefill apply mask), Metal graph encoder, TP (score/topk replicated, attention row-split).
- **Alternatives**: full compressed attention (no mask, O(n) per token), sliding-window only (no long context), causal range select ([glm-model-path.md](glm-model-path.md) uses position-based selection instead of scoring).

## Code Pattern

### Decode path (single token, ratio-4 layer, n_comp > top_k)

Source: ds4.c ~L22046-L22189, ds4_gpu.h L412-L500.

```c
// --- Score: Q · compressed indexer keys ---
//   Q after projection:  [1][n_indexer_head * n_indexer_head_dim]
//   index_comp cache:    [n_comp][n_indexer_head_dim]
//   scores output:       [1][n_comp]
ds4_gpu_indexer_score_one_tensor(
    scores,                      // ds4_gpu_tensor* output
    q,                           // ds4_gpu_tensor* query (projected + QAT)
    weights,                     // ds4_gpu_tensor* indexer weights
    layer_index_comp_cache[il],  // ds4_gpu_tensor* compressed key cache
    layer_n_index_comp[il],      // uint32_t n_comp
    DS4_N_INDEXER_HEAD,          // uint32_t n_head (def=64)
    DS4_N_INDEXER_HEAD_DIM,      // uint32_t head_dim (def=128)
    index_scale);                // 1/sqrt(head_dim * n_head)

// --- Top-K: keep highest-scoring rows ---
//   scores input:        [1][n_comp]
//   selected output:     [1][top_k]  (indices)
ds4_gpu_indexer_topk_tensor(
    selected,                    // ds4_gpu_tensor* output (row indices)
    scores,                      // ds4_gpu_tensor* scores from previous step
    layer_n_index_comp[il],      // uint32_t n_comp
    1,                           // uint32_t n_tokens
    DS4_N_INDEXER_TOP_K);        // uint32_t top_k (def=512)

// --- Mask (speculative decode only): -inf for unselected rows ---
//   mask output:         [n_comp][n_tokens]
ds4_gpu_dsv4_topk_mask_tensor(
    mask,                        // ds4_gpu_tensor* output (-inf/0.0f)
    selected,                    // ds4_gpu_tensor* top-k indices
    layer_n_index_comp[il],      // uint32_t n_comp
    n_tokens,                    // uint32_t n_tokens
    DS4_N_INDEXER_TOP_K);        // uint32_t top_k

// --- Indexed attention over raw + selected compressed rows ---
ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
    heads,                       // ds4_gpu_tensor* output
    model_map,                   // model weights
    ...,                         // model size, sinks offset
    q,                           // ds4_gpu_tensor* query
    raw_kv,                      // ds4_gpu_tensor* uncompressed KV
    comp_kv,                     // ds4_gpu_tensor* compressed KV
    ...,                         // comp_kv_f16 flag
    selected,                    // ds4_gpu_tensor* top-k indices
    n_tokens, pos0, n_raw, ...,
    n_comp,                      // uint32_t total compressed rows
    n_selected,                  // uint32_t selected rows (min(top_k, n_comp))
    window, ratio, n_head, ...);
```

### Batch decode / prefill path

Same sequence with `ds4_gpu_indexer_scores_decode_batch_tensor` or `ds4_gpu_indexer_scores_prefill_tensor` instead of the `_score_one_` call. The batch score kernels select dispatch strategy based on batch size: NAX (n_tokens ≥ 16, MPP available), tiled f16, or tiled f32 (quality mode).

### Key constants (ds4.c ~L716-L718)

```c
#define DS4_N_INDEXER_HEAD       (g_ds4_shape.n_indexer_head)        // 64
#define DS4_N_INDEXER_HEAD_DIM   (g_ds4_shape.n_indexer_head_dim)    // 128
#define DS4_N_INDEXER_TOP_K      (g_ds4_shape.n_indexer_top_k)       // 512
```

## Notes

- `ds4_gpu_dsv4_indexer_qat_tensor` applies to both indexer Q (after projection) and indexer compressor KV (after compression). The Hadamard + FP4 transform must match training graph exactly, or top-K selection loses fidelity.
- Indexer weights are separate from the main attention MLA weights. They are projected from the normalized attention input, not from the Q/KV projection path.
- GLM uses different score semantics: position-based causal range select (`ds4_gpu_glm_fill_selected_range_batch_tensor`) as alternative to score-based top-K. The `use_scalar_indexer` path iterates per-token with row views.
- Frontier snapshot buffers (`spec_index_*`) are allocated per ratio-4 layer for speculative decode rollback. Compressor state must be snapshotted alongside compressed cache.
- Metal top-K uses argsort + iterative merge tree — general but slower for large `n_comp`. CUDA has optimized paths for sizes 1024/2048/4096 common in practice; generic fallback handles others.

[← Back to Index](../README.md)
