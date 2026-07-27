# GLM Model Path

## Definition

GLM (General Language Model, family `DS4_MODEL_FAMILY_GLM_DSA`) is an alternate model architecture supported alongside DeepSeek Flash/Pro. Uses low-rank KV projections (LoRA), compact KV cache (no raw K/V stored), separate Q/K/V projection tensors with RoPE, a fixed-schedule indexer, and a different attention kernel pipeline without head-wise compression or attn_sinks. Engine detects GLM via GGUF metadata keys (prefix `glm-dsa.*`) and activates GLM-specific config validation, weight binding, cache layout, and GPU kernel dispatch.

## Why It Exists

Single engine binary for both DeepSeek and GLM architectures. GLM models (GLM-4, GLM-52B, GLM-Edge, GLM-130B derivatives) have different tensor layouts from DeepSeek Flash/Pro but share the same HC-state inference loop, MoE router, and KV compression concepts. Reusing the engine avoids a separate inference stack.

## Where It Appears

### Source Files

- `ds4.c` — model config validation, weight binding, GPU graph construction, decode dispatch, MTP spec cycle, KV cache planning, payload serialization, dense cache tracking
- `ds4.h` — engine options fields, public query functions
- `ds4_gpu.h` — GLM-specific GPU primitive declarations

### Detection & Family

| Symbol | File | Role |
|---|---|---|
| `DS4_MODEL_FAMILY_GLM_DSA` | `ds4.c` | Compile-time constant: model family discriminator |
| `DS4_VARIANT_GLM52` | `ds4.c` | Shape variant for GLM-52B |
| `g_ds4_shape = DS4_SHAPE_GLM52` | `ds4.c` | Shape set during `config_validate_glm_dsa_model` |
| `config_validate_glm_dsa_model` | `ds4.c` | Reads GGUF metadata keys `glm-dsa.*` into compile-time constants |
| `ds4_engine_is_glm_dsa` | `ds4.c` | Public query: returns `DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA` |

### Engine Struct Fields (`struct ds4_engine`)

| Symbol | Type | Role |
|---|---|---|
| `glm_mtp` | `bool` | Enable GLM MTP speculative decoding |
| `glm_mtp_timing` | `bool` | Log GLM MTP timing breakdown to stderr |
| `cuda_tensor_parallel` | `bool` | CUDA TP for GLM via `glm_token_prefill` mode |
| `glm_tp_token_prefill` | `bool` | Runtime TP token prefill mode (set from `tp_options.glm_token_prefill`) |

### Session Struct Fields (`struct ds4_session`)

| Symbol | Type | Role |
|---|---|---|
| `glm_graph` | `ds4_glm_gpu_graph` | GLM GPU graph workspace |
| `glm_graph_ready` | `bool` | Flag set after successful graph init |
| `glm_dense_cache_len` | `uint32_t` | Live rows in dense (full) KV cache on indexer layers |
| `glm_mtp_draft` | `int` | GLM MTP: greedy draft token from last step |
| `glm_mtp_have` | `int` | GLM MTP: whether draft buffer populated |
| `glm_spec_inside` | `int` | Reentrancy guard for spec cycle eval |
| `glm_mtp_min_pos` | `uint32_t` | Minimum position tracked for MTP range |
| `glm_mtp_hc` | `float *` | GLM MTP: softmax HC buffer for draft head |
| `glm_mtp_logits0` | `float *` | GLM MTP: logits storage for first speculative position |

### GPU Graph Struct (`ds4_glm_gpu_graph`)

Key fields beyond standard tensor slots:

| Symbol | Type | Role |
|---|---|---|
| `layer_kv_lora_cache[DS4_MAX_LAYER]` | `ds4_gpu_tensor *` | Per-layer compact LoRA latent cache: `[cache_cap × kv_lora_dim]` |
| `layer_k_rope_cache[DS4_MAX_LAYER]` | `ds4_gpu_tensor *` | Per-layer RoPE-K cache: `[cache_cap × rot_dim]` |
| `layer_indexer_key_cache[DS4_MAX_LAYER]` | `ds4_gpu_tensor *` | Per-layer indexer key cache (indexer layers only) |
| `full_kv_cache` | `bool` | True when expanded (full-dim) KV cache allocated instead of compact |
| `mtp_kv_lora_cache` | `ds4_gpu_tensor *` | MTP draft head: private compact LoRA cache |
| `mtp_k_rope_cache` | `ds4_gpu_tensor *` | MTP draft head: private RoPE-K cache |
| `mtp_concat` | `ds4_gpu_tensor *` | MTP: concatenated enorm + hnorm scratch |
| `mtp_selected` | `ds4_gpu_tensor *` | MTP: indexer selected indices for compact cache |
| `compact_cache_cap` | `uint32_t` | Compact cache capacity (positions) |
| `indexer_full_layers` | `uint32_t` | Number of layers with full indexer |

### Engine Options (`ds4_engine_options`)

| Field | Type | Default | Effect |
|---|---|---|---|
| `glm_mtp` | `bool` | false | Enable GLM MTP speculative decode |
| `glm_mtp_timing` | `bool` | false | Log GLM MTP timing breakdown |
| `cuda_tensor_parallel` | `bool` | false | Enable CUDA tensor parallelism for GLM |
| `tp_options.glm_token_prefill` | `bool` | false | TP token prefill mode for GLM |

### GPU Primitives

All declared in `ds4_gpu.h`.

**KV LoRA & Norm**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_kv_lora_rms_norm_tensor` | RMS norm on KV LoRA input |
| `ds4_gpu_glm_k_b_project_tensor` | K head projection from LoRA latents (qk_nope → n_head) |
| `ds4_gpu_glm_k_b_project_typed_tensor` | Same, typed weight variant |
| `ds4_gpu_glm_store_compact_kv_tensor` | Store kv_lora + k_rope into compact cache |
| `ds4_gpu_glm_qkv_norm_store_compact_kv_tensor` | Fused Q RMS norm + KV LoRA RMS norm + compact cache store (decode hot path). Does NOT fuse K-bias/V-bias projections — those are separate kernels (`ds4_gpu_glm_k_b_project_*`, `ds4_gpu_glm_value_project_*`). |

**Indexer**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_store_indexer_k_tensor` | Project hidden → indexer key, norm + RoPE, store |
| `ds4_gpu_glm_indexer_rope_tail_tensor` | Apply RoPE to indexer query |
| `ds4_gpu_glm_indexer_score_one_tensor` | Single-row indexer score (decode) |
| `ds4_gpu_glm_indexer_scores_batch_tensor` | Multi-row indexer scores (prefill) |
| `ds4_gpu_glm_fill_selected_range_tensor` | Fill top-K selected indices (decode) |
| `ds4_gpu_glm_fill_selected_range_batch_tensor` | Fill top-K selected indices (batch) |

**QK Low-Rank**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_qk_lowrank_q8_0_tensor` | Q·k_b low-rank attention bias q8_0 (single). Shape: [n_head, kv_lora] per Q row |
| `ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor` | Same, batch variant |
| `ds4_gpu_glm_qk_lowrank_typed_tensor` | Typed weight variant |
| `ds4_gpu_glm_qk_lowrank_typed_batch_tensor` | Typed batch variant |

**Value Projection**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_value_project_q8_0_batch_heads_tensor` | V head projection q8_0 (batch, per head). Maps kv_lora → value_mla |
| `ds4_gpu_glm_value_project_typed_batch_heads_tensor` | Typed variant |

**Attention (Indexed)**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_attention_indexed_decode_tensor` | Single-row indexed decode: score = qk_low·kv_lora + q_rope·k_rope; weighted sum value |
| `ds4_gpu_glm_attention_indexed_decode_typed_tensor` | Typed weight variant |
| `ds4_gpu_glm_attention_indexed_decode_split_group8_tensor` | Split decode (group8 blocks) for large selected sets |
| `ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor` | Typed split variant |
| `ds4_gpu_glm_attention_indexed_batch_tensor` | Batch indexed attention (prefill) |
| `ds4_gpu_glm_attention_indexed_batch_typed_tensor` | Typed batch variant |
| `ds4_gpu_glm_attention_indexed_batch_lora_tensor` | Batch LoRA attention |
| `ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor` | Causal LoRA batch attention |
| `ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor` | Valid-range LoRA batch attention |

**Full Attention**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_attention_full_tensor` | Full (dense) attention from raw K/V cache |
| `ds4_gpu_glm_attention_flash_tensor` | Flash attention from raw K/V cache |
| `ds4_gpu_glm_attention_flash_staged_tensor` | Staged flash attention |

**KV Cache Build**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_build_kv_cache_tensor` | Expand compact KV → full-dim K/V for flash attention |
| `ds4_gpu_glm_build_kv_cache_flash_tensor` | Flash variant of KV expansion |

**RoPE**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_rope_tail_tensor` | Apply RoPE to Q/K (tail-only on rot_dim) |

**MoE Router & Fused MoE**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_router_select_tensor` | Softmax router: logits → selected experts + weights |
| `ds4_gpu_glm_router_select_batch_tensor` | Batch variant |
| `ds4_gpu_glm_routed_moe_one_tensor` | Fused routed MoE forward (single token): gate_out = gate·x, up_out = up·x, down = down·(gate_out⊙up_out) |
| `ds4_gpu_glm_routed_moe_batch_tensor` | Batch variant |
| `ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor` | Direct scalar q4 batch MoE |

**Streaming Expert Cache**

| Primitive | Purpose |
|---|---|
| `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor` | Begin async expert cache load for selected layer |

### Payload Serialization

| Symbol | Role |
|---|---|
| `payload_write_glm_compact_span` | Write compact KV span to file |
| `payload_read_glm_compact_span` | Read compact KV span from file |
| `payload_write_glm_full_kv_span` | Write full KV span (dense cache) to file |
| `payload_read_glm_full_kv_span` | Read full KV span from file |
| `payload_read_or_skip_glm_full_kv_span` | Conditional read with skip |
| `session_glm_compact_live_rows` | Live rows in compact cache for payload |
| `session_glm_full_live_rows` | Live rows in full KV cache for payload |
| `session_glm_payload_live_tensor_bytes` | Total live tensor bytes for payload |
| `ds4_engine_glm_layer_payload_bytes` | Per-layer payload byte count |

## KV LoRA

GLM uses low-rank KV projections distinct from DeepSeek's shared `attn_kv` tensor. Key design: single-head KV LoRA input projected per-head through separate K-bias and V-bias tensors.

### Weight Tensors

- `attn_kv_a_mqa` — 2D `[embd, head_dim]`. Single-head KV LoRA input projection.
- `attn_kv_a_norm` — 1D `[kv_lora]` (DS4_N_KV_LORA = 512). RMS norm on KV LoRA latent (after `attn_kv_a_mqa`).
- `attn_k_b` — **3D** `[qk_nope, kv_lora, n_head]`. Per-head K projection from LoRA latents. No DeepSeek equivalent.
- `attn_v_b` — **3D** `[kv_lora, value_mla, n_head]`. Per-head V projection from LoRA latents. No DeepSeek equivalent.

Tensor validation (`weights_validate_glm_dsa_layout`) asserts exact shapes for `attn_kv_a_mqa`, `attn_k_b`, and `attn_v_b`.

### Attention Sequence (KV LoRA Portion)

In each decode step, after Q projection and RoPE, the KV LoRA path differs by attention mode:

**Indexed (compact cache) decode path:**

1. `attn_kv_a_mqa` — project hidden `[embd → head_dim]` (single head)
2. `ds4_gpu_glm_qkv_norm_store_compact_kv_tensor` — fused Q RMS norm + KV RMS norm + store:
   - Q RMS norm: `q → q_rank_norm` (via `attn_q_a_norm`)
   - KV LoRA RMS norm: `kv_raw → layer_kv_lora_cache` (via `attn_kv_a_norm`)
   - KV RoPE tail: `kv_raw → layer_k_rope_cache`
3. `ds4_gpu_glm_qk_lowrank_*` — K-bias low-rank projection: `q_nope · attn_k_b` → `[n_head, kv_lora]` per Q row
4. Indexed decode attention (`ds4_gpu_glm_attention_indexed_decode_*`): `score = qk_low · kv_lora + q_rope · k_rope` (from caches), V projection via `attn_v_b`, weighted sum of value

**Full (dense cache) decode path:**

1. `attn_kv_a_mqa` — project hidden `[embd → head_dim]` (single head)
2. `attn_kv_a_norm` — RMS norm
3. `ds4_gpu_glm_k_b_project_typed_tensor` — K head projection: `[qk_nope, kv_lora, n_head]` → k_nope per head
4. `ds4_gpu_glm_value_project_*` — V head projection: `[kv_lora, value_mla, n_head]` → value per head
5. `ds4_gpu_glm_build_kv_cache_tensor` — expand compact latent → full-dim K/V cache
6. Full attention (`ds4_gpu_glm_attention_full_tensor` or flash variant)

No `attn_compressor_*` tensors (no head-wise compression). GLM uses LoRA latent directly.

### Cache Layout

```
layer_kv_lora_cache[il]:  compact LoRA latents    [cache_cap × kv_lora_dim]
layer_k_rope_cache[il]:   RoPE-applied K head     [cache_cap × rot_dim]
```

These two per-layer caches hold the entire KV state. No raw full-dim K/V stored unless `full_kv_cache` is set (flash attention prefill path).

### GPU Primitives

KV LoRA operations in `ds4_gpu.h`: RMS norm (`ds4_gpu_glm_kv_lora_rms_norm_tensor`), K head projection (`ds4_gpu_glm_k_b_project_tensor`, typed variant), store compact KV (`ds4_gpu_glm_store_compact_kv_tensor`), fused Q norm + KV store (`ds4_gpu_glm_qkv_norm_store_compact_kv_tensor`). Value projection batch kernels: `ds4_gpu_glm_value_project_q8_0_batch_heads_tensor`, typed variant.

## Compact KV

GLM stores only compressed (LoRA latent + RoPE-K) state — no raw K/V in cache. Indexer layers add a separate key cache for top-K selection.

### Three Tier Layout

```
layer_kv_lora_cache[il]:       LoRA latents           [cache_cap × kv_lora_dim]
layer_k_rope_cache[il]:        RoPE-K head            [cache_cap × rot_dim]
layer_indexer_key_cache[il]:   indexer key vectors    [cache_cap × indexer_head_dim] (indexer layers only)
```

Row width computed by `engine_glm_per_layer_kv_bytes_planner`:

```
row_width = DS4_N_KV_LORA + DS4_N_ROT
if engine_glm_layer_uses_full_indexer(il):
    row_width += DS4_N_INDEXER_HEAD_DIM
```

No `layer_key_cache` / `layer_value_cache` allocated at graph level unless `full_kv_cache` is set (flash attention prefill path).

### Indexer Schedule

`engine_glm_layer_uses_full_indexer` determines which layers get an indexer:

```c
il < DS4_N_LEADING_DENSE                   // all leading dense layers
|| (il >= 6 && (il - 6) % 4 == 0)          // every 4th MoE layer starting at 6
```

Indexer components per layer: `indexer.attn_q_b`, `indexer.attn_k`, `indexer.k_norm` (+ bias), `indexer.proj`.

### Dense Cache

On full-indexer layers, `glm_dense_cache_len` tracks live rows in the expanded (full-dim) KV cache. Managed by `ds4_session_glm_reset_dense_cache`, `ds4_session_glm_cap_dense_cache`, `ds4_session_glm_note_dense_cache`.

### GPGPU Kernels

Compact KV expanded to full-dim via `ds4_gpu_glm_build_kv_cache_tensor` (flash variant: `ds4_gpu_glm_build_kv_cache_flash_tensor`). Full attention from raw cache: `ds4_gpu_glm_attention_full_tensor`. Flash attention: `ds4_gpu_glm_attention_flash_tensor`, `ds4_gpu_glm_attention_flash_staged_tensor`.

## Streaming Prefill

GLM streaming layer logic distinct from DeepSeek. Set of guards and capability checks activate on decode layers:

| Symbol | Role |
|---|---|
| `glm_stream_resident_decode_layer_supported` | Checks if layer supports resident decode (no expert swap) |
| `glm_stream_resident_decode_layer_enabled` | Checks if layer is within resident decode range |
| `glm_stream_expert_cache_addr_layout_supported` | Checks addressing layout for expert cache |
| `glm_stream_decode_experts_are_streamed` | Whether a layer's experts must be streamed at decode |
| `glm_stream_selected_expert_cache_supported` | Selected expert cache support |
| `glm_graph_stream_layer_expert_cache_supported` | Per-layer streaming cache support |
| `glm_graph_stream_prefill_full_layer_enabled` | Full-layer prefill streaming |
| `ds4_backend_supports_glm_streaming_full_layers` | Backend capability check |
| `ds4_engine_glm_streaming_memory_guard` | Streaming memory guard |
| `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor` | GPU primitive: async expert load |

Environment variables prefixed `DS4_ROCM_GLM_*` control ROCm-specific streaming features.

## GLM MTP (Next-N Block)

GLM uses its own MTP variant via the next-n prediction block (last `DS4_N_NEXTN_PREDICT` layers).

Enabled when: backend != CPU, `DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA`, `DS4_N_NEXTN_PREDICT != 0`, `e->glm_mtp` option set. Returns 2 draft tokens when active.

### Draft Generation

`glm_graph_mtp_step` operates on next-n layer `il = DS4_N_LAYER - DS4_N_NEXTN_PREDICT`:
1. Requires `nextn_eh_proj`, `nextn_enorm`, `nextn_hnorm`, `nextn_shared_head_norm` weights
2. Embeds `next_token` via `ds4_gpu_embed_token_quant_tensor`
3. Concatenates `enorm(embed(next)) + hnorm(h[pos])` → `eh_proj` → layer input
4. Runs forward through MoE + attention (reads compact caches at `[min_pos..pos]`)
5. Applies `nextn_shared_head_norm` → logits → argmax → single greedy draft token
6. Stores draft in `s->glm_mtp_draft`
7. Uses private compact caches (`mtp_kv_lora_cache`, `mtp_k_rope_cache`) separate from main caches

### Spec Cycle

`ds4_session_glm_spec_cycle`:
1. First entry (no draft): normal eval, then `glm_graph_mtp_step` to seed draft
2. Subsequent entry (have draft): verify `draft == argmax(s->logits)` from previous eval
3. If draft matches: both tokens accepted as NTB (next-token-batch) pair
4. If draft mismatches: accept token-0 only, replay token-1 via normal eval
5. After commit: seed next draft via `glm_graph_mtp_step`
6. Both TP ranks run mirrored spec cycle — derive identical drafts from identical logits

Verify path: `glm_graph_verify_rows` or `glm_graph_forward_indexed_tokens` batch-forward both tokens to verify acceptance.

## Relationship

- **Depends on**: [kv-cache-lifecycle.md](kv-cache-lifecycle.md), attention indexed decode, [indexer-subsystem.md](indexer-subsystem.md), [moe-routing.md](moe-routing.md), GPU graph allocator.
- **Used by**: engine graph (GLM branch: `glm_graph_forward_token`, `glm_graph_forward_tokens`, `glm_graph_prefill_range`), Metal backend (`ds4_gpu_glm_*` primitives), CUDA backend (GLM kernels with `glm_token_prefill` mode for TP).
- **Alternatives**: DeepSeek Flash/Pro native path (different tensor layout, same HC-state inference loop).
- **Incompatible with**: [dspark.md](dspark.md) (GLM has its own MTP), [mtp.md](mtp.md) (GLM uses nextn block). GLM MTP is compatible with TP.

[← Back to Index](../README.md)
