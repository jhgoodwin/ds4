# Attention Compression (CSA / HCA)

## Definition

Compressed Sparse Attention (CSA, ratio-4 layers) and Heavily Compressed Attention (HCA, ratio-128 layers).
DeepSeek V4 compresses the KV cache to reduce memory at long contexts. After the initial 128-token sliding window, tokens are compressed into fewer rows. Attention attends over both the raw window and compressed history.

## Why It Exists

Total KV cache for 1M tokens across all layers at 32 heads × 128 dim × 2 (K+V) = 32 GB (~0.7 GB/layer for 43-layer Flash). Compression reduces total cache to ~2-4 GB depending on layer ratio. Enables 1M-context on consumer hardware.

## E4M3-Style Encoding

Compressed KV stored through an E4M3-style format (4 exponent bits, 3 mantissa bits). Decompressed to float before attention. After RoPE applied to full row, only the non-RoPE portion (first `n_head_dim - n_rot` elements) is E4M3 encoded. RoPE tail (last `n_rot` elements) kept in f32.

## Where It Appears

| File | Role |
|---|---|
| `ds4.c` | ratio computation, compression table, KV cache management, compressor logic |
| `ds4.h` | compression ratio type definitions |
| `ds4.c` (metal-graph) | GPU compressed attention kernels |

## Variants

| Model | n_swa | CSA (ratio-4) | HCA (ratio-128) | Notes |
|---|---|---|---|---|
| Flash | 128 | yes | yes | Layers 0,1 raw; even ≥2 ratio-4; odd ≥3 ratio-128 |
| Pro | 128 | yes | yes | Layers 0,1 ratio-128 (not raw); even ≥2 ratio-4; odd ≥3 ratio-128 |
| GLM52 | 0 | yes | yes | No sliding window; all layers compressed |

## Compression Scheme

```
raw KV (128 tokens, full precision)
  → compressor (E4M3-style encoding)
  → compressed KV row appended to compressed cache
```

Two compression ratios alternate per layer:

| Variant | Layer | Ratio | Compressed Row Format | Indexer |
|---|---|---|---|---|
| Flash | 0, 1 | none | — | — |
| Flash | even ≥ 2 | 4:1 | E4M3-encoded KV + indexer KV | yes — selects visible compressed rows |
| Flash | odd ≥ 3 | 128:1 | E4M3-encoded KV | no |
| Pro | 0, 1 | 128:1 | E4M3-encoded KV | no |
| Pro | even ≥ 2 | 4:1 | E4M3-encoded KV + indexer KV | yes — selects visible compressed rows |
| Pro | odd ≥ 3 | 128:1 | E4M3-encoded KV | no |

## E4M3-Style Encoding

See [E4M3-Style Encoding](#e4m3-style-encoding) above for format details.

## Indexer (Ratio-4 Layers)

Ratio-4 layers produce an additional indexer row during compression. The indexer learns which compressed rows each token should attend to. During attention, the indexer mask filters compressed rows, reducing the effective attention span.

## Attention Over Compressed Rows

Single concatenated `score[]` array over all `n_raw + n_comp` positions, one softmax over all positions.

Compressed scores computed by:

1. Decompress compressed KV rows
2. Apply indexer mask (ratio-4 layers only)
3. Compute attention scores against Q, appended to raw scores
4. Softmax over single concatenated score array

## Lifecycle

```
token arrives → store raw KV in window
→ if window full, compress oldest raw row
→ append compressed row to compressed cache
→ update indexer (ratio-4 layers)
→ next token
```

## Relationship to Other Concepts

- **Depends on**: [kv-cache-lifecycle.md](kv-cache-lifecycle.md) (raw → compressed flow)
- **Used by**: attention pipeline (reads compressed rows)
- **Alternatives**: MQA, GQA, MLA (no compression, just fewer heads)

## Code Pattern

```c
// From ds4.c:kv_cache_push_raw → compressor_decode_one
// E4M3 round-trip encoding of non-RoPE portion
if (compressor_decode_one(comp, model,
                          layer->attn_compressor_kv,
                          layer->attn_compressor_gate,
                          layer->attn_compressor_ape,
                          layer->attn_compressor_norm,
                          attn_norm,
                          cache->attn_state_kv,
                          cache->attn_state_score,
                          DS4_N_HEAD_DIM,
                          ratio,
                          il,
                          pos)) {
    kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, comp);
}
```

Compressor invoked after each token's KV computed. If ratio is 4, a second call compresses the indexer row. See `ds4.c:L12981` for the full decode path.

[← Back to Index](../README.md)
