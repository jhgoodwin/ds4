# Attention: Projections, RoPE, Compressed Attention

## Files

- `ds4.c` — attention pipeline (CPU reference)
- `ds4.c` — E4M3-style KV decompression, indexer rotation

## Purpose

Implement the attention half of each transformer layer: HC pre, attention RMSNorm, low-rank Q projection, KV projection with layer-specific RoPE, sink-aware attention over raw SWA window + compressed KV rows, grouped LoRA output projection back to embedding width.

## Pipeline

```
HC pre (4→1 stream mixing)
→ attention RMSNorm
→ Q projection (low-rank Q8_0 → LoRA-Q → RMSNorm → Q8_0 → all heads → head RMSNorm)
→ KV projection (Q8_0 → K and V for all heads)
→ RoPE on Q and KV tail 64 dims
→ attention over raw SWA window + compressed KV rows
→ inverse RoPE on attention output
→ grouped LoRA O projection → back to HC 4-stream
```

**Abbreviations:**
- **SWA** — Sliding Window Attention. Raw KV cache of the last 128 tokens (Flash) or 0 (GLM52). Tokens outside the window are compressed.
- **LoRA-Q** — Low-Rank Q projection. Projects from 4096 → 1536 (rank), RMSNorm, then 1536 → 4096 (all heads). Reduces parameter count vs direct 4096×4096.
- **HC pre** — Hidden State pre-processing. Merges the 4-stream HC state (shared, routed, output, residual) into a single stream before attention.

## Q Projection

Low-rank design: project from 4096 → 1536 (LoRA-Q rank), RMSNorm, project to 32 heads × 128 dim = 4096, then head RMSNorm (per-head normalization).  Reduces parameters vs direct 4096×4096 Q matrix.

## KV Projection

Single projection produces both K and V.  Output split: first half = K, second half = V.  Each head gets 128-dim K and 128-dim V.

## RoPE Per Layer

All layers apply RoPE to the tail (last `n_rot=64` dims) of each head's Q and K. Layer-specific differences:

| Variant | Layer | Ratio | RoPE Notes |
|---|---|---|---|
| Flash | 0, 1 | 0 (raw) | base freq, no YaRN extension |
| Flash | even ≥ 2 | 4 | compressed freq base, YaRN extension |
| Flash | odd ≥ 3 | 128 | compressed freq base, YaRN extension |
| Pro | 0, 1 | 128 | compressed freq base, YaRN extension (not raw) |
| Pro | even ≥ 2 | 4 | compressed freq base, YaRN extension |
| Pro | odd ≥ 3 | 128 | compressed freq base, YaRN extension |

Inverse RoPE applied to attention output before grouped O projection.
DeepSeek V4 graph rotates indexer activations with 64-wide rotary embedding before they select compressed rows.

## Compressed Attention

See [attention-compression.md](../concepts/attention-compression.md) for compressed attention (CSA/HCA) encoding and score computation.

## Output Projection

Grouped LoRA: output from all heads projected back to 4096 via low-rank factorized matrix.  Groups of heads share the up-projection.

## Invariants

- Q and KV projections always Q8_0 quantized.
- RoPE applied in-place on K buffer.
- Attention always uses causal mask (no future tokens).
- Compressed rows only attend to older tokens (strictly before current position).

## See Also

- [kv-cache-lifecycle.md](../concepts/kv-cache-lifecycle.md) — compressed cache lifecycle and compression triggers
- [hc-state.md](../concepts/hc-state.md) — HC state structure and lifecycle

[← Back to Index](../README.md)
