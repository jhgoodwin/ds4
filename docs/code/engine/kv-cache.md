# KV Cache, Compressors, CPU Layer Execution

## Files

- `ds4.c`
- `ds4.h`

## Purpose

Maintain raw SWA KV rows, optional compressed KV rows, indexer mask for ratio-4 layers, and reusable decode scratch arena.  CPU path is correctness reference for all backends.

## Cache Structure

Per-layer breakdown of KV cache state:

| Model | Layer | Ratio | Compressed Rows | Extra State |
|---|---|---|---|---|
| Flash | 0, 1 | none | 0 | raw window only |
| Flash | even ≥ 2 | 4 | ctx/4 | compressed KV + indexer KV |
| Flash | odd ≥ 3 | 128 | ctx/128 | compressed KV only |
| Pro | 0, 1 | 128 | ctx/128 | compressed KV only |
| Pro | even ≥ 2 | 4 | ctx/4 | compressed KV + indexer KV |
| Pro | odd ≥ 3 | 128 | ctx/128 | compressed KV only |
| GLM52 | all | 0 | 0 | n_swa=0, no raw window; separate GPU-managed full/compact cache |

## Raw Window

Circular buffer over last 128 tokens at full precision (f16 round-trip: f32→f16→f32).  Each decode step stores the current token's KV projection into the window, overwriting the oldest slot when full.  All layers maintain a raw window.

## Compressed Cache

See [kv-cache-lifecycle.md](../concepts/kv-cache-lifecycle.md) for compressed cache lifecycle and [fp8-kv-quantization.md](../concepts/fp8-kv-quantization.md) for FP8 quantization format.

Compressed KV capacity per layer: `comp_cap = ctx_size / ratio + 2`. The +2 accounts for rounding and the initial unfilled slot before the first compression trigger.

## Indexer

See [indexer-subsystem.md](../concepts/indexer-subsystem.md) for indexer architecture and search algorithm.

## Memory Accounting

Allocated at `cpu_decode_scratch_init()` during `ds4_session_create`, freed at `cpu_decode_scratch_free()` during `ds4_session_free`.  Kept resident for the whole generation to avoid per-step malloc churn.

Field groupings (47 total in `ds4_cpu_decode_scratch`):

- **Core**: `plain`, `cur` (hc_dim), `next` (hc_dim)
- **Attention**: `attn_cur`, `attn_norm`, `attn_residual`, `q`, `qr`, `qr_norm`, `kv_raw`, `kv`, `heads`, `attn_low`, `attn_out`, `after_attn_hc`, `attn_score`
- **Compression**: `comp`, `index_comp`, `comp_kv_cur`, `comp_sc_cur`, `comp_pooled`
- **Indexer**: `index_allowed`, `index_q`, `index_weights`, `index_scores`
- **FFN**: `ffn_cur`, `ffn_norm`, `ffn_moe`, `ffn_shared`, `ffn_out`, `shared_gate`, `shared_up`, `shared_mid`, `routed_mid_all`
- **Q8 quantize (routed/mid)**: `routed_xq`, `routed_midq`, `routed_q8_xq`, `routed_q8_xscale`, `routed_q8_midq`, `routed_q8_midscale`
- **Q8 activation quantize**: `q8_xq`, `q8_xscale`
- **Output**: `hc_flat`, `output_flat`, `output_pre`, `output_weights`, `output_embd`, `output_norm`

No allocations in hot loop.  Arena sized for worst-case profile.

## Invariants

- Raw KV window is a circular buffer (overwrite oldest when full).
- Compressed rows appended monotonically (never overwritten).
- Indexer rebuilt when new compressed rows added at ratio-4 layers.
- Scratch arena allocated at session create, freed at session free (kept resident whole generation).

## See Also

- [attention-compression.md](../concepts/attention-compression.md) — CSA/HCA encoding and compressed score computation

[← Back to Index](../README.md)
