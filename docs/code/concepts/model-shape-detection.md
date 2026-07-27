# Model Shape Detection & Support Kind

## Definition

Model shape detection identifies the loaded GGUF model's architecture variant by examining GGUF metadata and tensor names. The result is a `ds4_support_kind` enum value that controls weight binding, cache sizing, and kernel dispatch throughout the engine. Shape detection is orthogonal to support kind detection — shape selects compile-time constants, support kind selects decode path.

## Why It Exists

Engine supports two model families — DeepSeek Flash/Pro (`DS4_MODEL_FAMILY_DEEPSEEK4`) and GLM (`DS4_MODEL_FAMILY_GLM_DSA`) — with different tensor layouts, compression ratios, and speculative decode mechanisms. Shape detection selects the right `ds4_shape` profile without user specification. Support kind detects whether the loaded draft model enables speculative decode and which variant.

## Where It Appears

**Files:**
- `ds4.c` — `ds4_support_kind` enum, `support_model_detect`, engine struct fields, engine open dispatch, eval speculative path, TP interaction.
- `ds4.h` — `ds4_engine_options::dspark`, `ds4_engine_options::dspark_strict`, `ds4_engine_options::dspark_confidence_threshold`, `ds4_engine_options::dspark_confidence_threshold_set`.

**Types:**

| Symbol | File | Role |
|---|---|---|
| `ds4_support_kind` | `ds4.c` | Enum: `DS4_SUPPORT_NONE`, `DS4_SUPPORT_MTP_LEGACY`, `DS4_SUPPORT_DSPARK` |
| `ds4_shape` | `ds4.c` | Struct: shape constants loaded at engine init |

See [dspark.md](dspark.md) for DSpark detection and configuration.

## Support Kind Impact

| Kind | Weight Binding | KV / Cache | Eval Path |
|---|---|---|---|
| `DS4_SUPPORT_NONE` | standard DeepSeek weights | raw + compressed | standard autoregressive decode |
| `DS4_SUPPORT_MTP_LEGACY` | + `mtp.0.*` tensors via `mtp_weights_bind` | same as NONE | speculative decode via `ds4_session_eval_speculative_argmax` |
| `DS4_SUPPORT_DSPARK` | + per-stage dspark tensors via `dspark_weights_bind_optional` | + dspark ring KV | multi-stage speculative decode + TP suspend |

## Detection Algorithm

See [dspark.md](dspark.md) for DSpark detection, metadata scanning, and decision logic.

## Model Profiles

Constants defined as `static const ds4_shape` structs.

| Property | Flash | Pro | GLM52 |
|---|---|---|---|
| Layers | 43 (`n_layer`) | 61 (`n_layer`) | 79 (`n_layer`) |
| Hidden dim | 4096 (`n_embd`) | 7168 (`n_embd`) | 6144 (`n_embd`) |
| Attention heads | 64 (`n_head`) | 128 (`n_head`) | 64 (`n_head`) |
| KV heads | 1 (`n_head_kv`) | 1 (`n_head_kv`) | 1 (`n_head_kv`) |
| Routed experts | 256 (`n_expert`) | 384 (`n_expert`) | 256 (`n_expert`) |
| Used experts | 6 (`n_expert_used`) | 6 (`n_expert_used`) | 8 (`n_expert_used`) |
| KV compression | LoRA Q=1024, O=1024 | LoRA Q=1536, O=1024 | LoRA Q=2048, KV LoRA=512 |
| MTP support | legacy MTP | legacy MTP | GLM nextn (1 stage) |

Active shape selected at engine open via `g_ds4_shape`, populated from GGUF metadata. Compile-time macros (`DS4_N_LAYER`, `DS4_N_EMBD`, `DS4_N_HEAD`, etc.) alias to `g_ds4_shape` fields.

**GLM detection** (`ds4_engine_is_glm_dsa`): compile-time check on `DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA`. Orthogonal to `support_kind` — both GLM and non-GLM models can have any support kind (though DSPARK only valid under DeepSeek V4).

## Relationship

### TP Interaction

- **Legacy MTP**: disabled when TP active (`opt->tp.role != DS4_TP_NONE`). Model closed, `support_kind` set to `DS4_SUPPORT_NONE`. Check in `support_model_detect` path at engine open. Verify path in `ds4_session_eval_speculative_argmax` gated on `!tp.active`.
- **DSPARK**: handles TP via `ds4_gpu_tp_suspend_expert_sharding` — suspends expert sharding (`tp_world=0`) during DSpark stage eval block, restores after each DSpark draft block. `dspark_exec_tier` selects which GPU runs DSpark stages. Cross-GPU tensor copy from `metal_graph_prefill_tokens` to `g->dspark_draft_tokens`.
- **GLM MTP**: compatible with TP — both ranks run mirrored `ds4_session_glm_spec_cycle`.

### Dependencies

- **Depends on**: [gguf-format.md](gguf-format.md) (`model_find_tensor`, `model_get_u32_any`, `model_get_u32_array_any`), `ds4_tensor_mtp_stage` for stage number extraction.
- **Used by**: `ds4_engine_open` (selects weight binding path), `ds4_session_eval_speculative_argmax` (selects verify loop variant), `ds4_session_prepare_legacy_mtp_draft` (legacy draft eval), `metal_graph_dspark_ring_maintain` (DSpark KV ring).
- **Alternatives**: user-specified architecture flag (fragile, violates "detect don't specify").

### Notes

- `support_kind` logged at engine open: `ds4: MTP support model loaded: ...` for legacy, `ds4: DSpark support model detected: ...` for DSPARK, `"ds4: ignore under tensor parallelism"` for TP rejection.
- DSpark metadata scanning and tensor presence checks documented in [dspark.md](dspark.md).

[← Back to Index](../README.md)
