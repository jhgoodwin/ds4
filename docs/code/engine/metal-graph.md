# Metal Release Graph

## Files

- `ds4.c` — Metal graph state, alloc, decode, prefill, diagnostic hooks
- `ds4_metal.m` — Objective-C Metal runtime wrappers
- `metal/*.metal` — GPU compute kernels

## Purpose

Own GPU-side state for Metal backend: one fixed tensor set for decode, another for batched prefill.  Tensor names follow model stages, not generic graph nodes.  Primary production inference path.

## Graph Construction

The `ds4_gpu_graph` struct (ds4.c:14840) owns all GPU-side tensor pointers. Fields are grouped by lifecycle:

### Class P — decode scratch (per-tier replicated)

```
cur_hc_by_tier[DS4_MAX_GPUS]         — current token HC state
flat_hc_by_tier, hc_mix_by_tier      — HC flatten/mix scratch
hc_split_by_tier                     — split HC for multi-tier
hc_pre_by_tier, hc_post_by_tier      — views of hc_split
hc_comb_by_tier                      — combined HC view
attn_cur_by_tier                     — attention input HC
attn_norm_by_tier                    — attention RMS norm input
qr_by_tier                           — Q after RoPE
qr_norm_by_tier                      — Q norm (RMS) buffer
q_by_tier                            — projected Q (prior to RoPE)
kv_raw_by_tier                       — raw K,V projection output
kv_by_tier                           — K,V after RoPE
comp_kv_cur_by_tier                  — compressor KV workspace
comp_sc_cur_by_tier                  — compressor score workspace
attn_comp_stage_by_tier              — compressor stage buffer
indexer_q_by_tier                    — indexer Q for ratio-4 layers
indexer_weights_by_tier              — indexer weights
indexer_scores_by_tier               — indexer scores
comp_mask_by_tier                    — compressor mask
comp_selected_by_tier                — compressor selected rows
heads_by_tier                        — multi-head Q view
attn_low_by_tier                     — low-rank attention output
attn_out_by_tier                     — attention output
after_attn_hc_by_tier                — HC after attention (residual)
ffn_cur_by_tier                      — FFN input HC
ffn_norm_by_tier                     — FFN RMS norm input
shared_gate_by_tier                  — shared expert gate proj
shared_up_by_tier                    — shared expert up proj
shared_mid_by_tier                   — shared expert mid activations
shared_out_by_tier                   — shared expert output
router_logits_by_tier                — router logits
router_probs_by_tier                 — router probabilities
router_selected_by_tier              — selected expert indices
router_weights_by_tier               — selected expert weights
routed_gate_by_tier                  — routed expert gate proj
routed_up_by_tier                    — routed expert up proj
routed_mid_by_tier                   — routed expert mid activations
routed_down_by_tier                  — routed expert down proj
routed_out_by_tier                   — routed expert output
ffn_out_by_tier                      — FFN output
after_ffn_hc_by_tier                 — HC after FFN (residual)
```

### Class H — output head (per-tier, head tier only)

```
output_pre_by_tier, output_weights_by_tier
output_embd_by_tier, output_norm_by_tier
logits_by_tier                       — final logits
```

All share `head_tier` slot; non-head tiers are NULL. Read via `metal_graph_logits()` / `metal_graph_output_*()` accessors.

### Class L — persistent KV state (per-layer)

```
layer_raw_cache[DS4_MAX_LAYER]           — raw SWA KV ring (window)
layer_attn_comp_cache[DS4_MAX_LAYER]     — compressed KV cache
layer_attn_state_kv[DS4_MAX_LAYER]       — compressor KV frontier
layer_attn_state_score[DS4_MAX_LAYER]    — compressor score frontier
layer_index_comp_cache[DS4_MAX_LAYER]    — ratio-4 indexer compressed KV
layer_index_state_kv[DS4_MAX_LAYER]      — indexer KV frontier
layer_index_state_score[DS4_MAX_LAYER]   — indexer score frontier
layer_raw_cache_tp[DS4_MAX_LAYER]        — TP mirror of raw cache
layer_attn_comp_cache_tp[DS4_MAX_LAYER]  — TP mirror of comp cache
```

`layer_n_comp[]` / `layer_n_index_comp[]` store actual compressed-row count per layer.  Layer-indexed arrays (not per-GPU) because KV is partitioned per-layer placement.

### Class B — batched prefill scratch (per-tier replicated)

```
batch_cur_hc_by_tier, batch_next_hc_by_tier   — ping-pong HC
batch_flat_hc_by_tier, batch_hc_mix_by_tier
batch_hc_split_by_tier
batch_attn_cur_by_tier, batch_attn_norm_by_tier
batch_qr_by_tier, batch_qr_norm_by_tier
batch_q_by_tier
batch_kv_raw_by_tier, batch_kv_by_tier
batch_comp_kv_by_tier, batch_comp_sc_by_tier
batch_indexer_q_by_tier, batch_indexer_weights_by_tier
batch_heads_by_tier
batch_attn_low_by_tier, batch_attn_out_by_tier
batch_group_tmp_by_tier, batch_low_tmp_by_tier
batch_after_attn_hc_by_tier
batch_ffn_cur_by_tier, batch_ffn_norm_by_tier
batch_shared_gate_by_tier, batch_shared_up_by_tier
batch_shared_mid_by_tier, batch_shared_out_by_tier
batch_router_logits_by_tier, batch_router_probs_by_tier
batch_router_selected_by_tier, batch_router_weights_by_tier
batch_routed_gate_by_tier, batch_routed_up_by_tier
batch_routed_mid_by_tier, batch_routed_down_by_tier
batch_routed_out_by_tier
batch_ffn_out_by_tier
```

## Alloc Phase

`metal_graph_alloc_raw_cap()` sizes all buffers for a given context size.  Tier-aware: per-layer KV allocations land on each GPU's home tier when placement table non-NULL.

## Decode Phase

See [attention.md](attention.md) for attention pipeline, [hc-transforms.md](hc-transforms.md) for HC transforms, and [moe-ffn.md](moe-ffn.md) for MoE FFN pipeline.

Each step dispatches Metal compute kernels.  Env flags disable fusion for diagnostics: `DS4_METAL_DISABLE_HC_FUSION`, `DS4_METAL_DISABLE_KV_FUSION`, `DS4_METAL_DISABLE_QKV_NORM_FUSION`, etc. (see `metal_graph_use_reference_*()`)  Diagnostic helpers (`max_abs_diff`, `rms_abs_diff`), layer trace dumps (`DS4_METAL_GRAPH_DUMP_PREFIX`), and per-stage tensor comparisons against CPU reference available.

## Prefill Phase

Batched prefill processes multiple tokens simultaneously.  Uses separate prefill graph with larger buffers.  Same kernel logic but matrix-matrix (GEMM) instead of matrix-vector (GEMV).

## Invariants

See [metal.md](../backends/metal.md) for Metal backend invariants and memory model.

## See Also

- [gpu-tensor-primitives.md](../concepts/gpu-tensor-primitives.md) — GPU tensor primitives used in Metal graph kernels

[← Back to Index](../README.md)
