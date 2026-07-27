# Weight Binding

## Files

- `ds4.c` — tensor name binding, shape validation, layer struct population

## Purpose

Convert GGUF tensor directory into DS4-specific pointer table.  After binding, code addresses weights by semantic fields (`layer->attn_q_a`, `layer->ffn_gate_exps`) instead of string lookup.

## Binding Process

For each expected tensor name pattern, linear scan of tensor directory with memcmp (not hash).  Validate type, dimensionality, and shape dimensions match active profile.  Store pointer in layer struct.  Fail on missing tensor or dim mismatch.  Extra tensors in GGUF pass silently (only expected tensors validated; unexpected tensors are ignored).

### Lookup: linear scan with memcmp

`model_find_tensor(m, name)` iterates `m->tensors[]` comparing `memcmp(name.ptr, name, len)`.  No hash table, no sorting assumption.  Linear scan is fast because the tensor directory is small (hundreds, not millions).

### Validation: dimension and type checks (not stride validation)

Every expected tensor is validated against the active model profile.  Checks cover:
- Tensor type (e.g. F16, Q8_0, Q4_0)
- Dimensionality (2D vs 3D)
- Shape dimensions (hidden size, number of heads, expert count, etc.)

Stride validation is not performed — only dimension sizes are compared.

### Extra tensors: pass silently

Only expected tensors are validated.  Any tensor in the GGUF file that is not part of the expected binding list is silently ignored.  There is no "fail on unexpected extra tensor" check.

## Binding Entry Points

### `weights_bind`

```c
static void weights_bind(
    ds4_weights     *w,
    const ds4_model *m,
    bool             load_slice,
    uint32_t         load_layer_start,
    uint32_t         load_layer_end,
    bool             require_output);
```

Main binding function.  Iterates layers `[load_layer_start, load_layer_end]`, binding each via `weights_bind_layer`.  When `load_slice` is true:

- `token_embd.weight` is bound only if `load_layer_start == 0` (first layer in slice owns the embedding).  Otherwise it is looked up via `model_find_tensor` (optional — may be NULL).
- `require_output` controls whether the output projection is required.
- GLM nextn/MTP blocks are bound only when loading the full range (start=0, end=executable_layers-1).

After binding all layers, `weights_validate_layout` checks invariants.

### `weights_bind_glm_dsa_layer`

```c
static void weights_bind_glm_dsa_layer(
    ds4_layer_weights *l,
    const ds4_model   *m,
    uint32_t           il);
```

GLM DSA model-family layer binding.  Binds the DSA-specific attention and FFN tensor set:

**Attention tensors (all required):**
- `attn_norm`, `attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_a_norm`, `attn_k_b`, `attn_v_b`, `attn_output`
- `indexer_attn_q_b`, `indexer_attn_k`, `indexer_k_norm`, `indexer_k_norm_b`, `indexer_proj`

**FFN tensors:**
- `ffn_norm` (required)
- Leading dense layers (`il < DS4_N_LEADING_DENSE`): `ffn_gate`, `ffn_up`, `ffn_down`
- MoE layers otherwise: `ffn_gate_inp`, `ffn_exp_probs_b`, `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`

**Nextn tensors (conditional, when `DS4_N_NEXTN_PREDICT != 0` and `il + DS4_N_NEXTN_PREDICT >= DS4_N_LAYER`):**
- `nextn_eh_proj`, `nextn_enorm`, `nextn_hnorm`, `nextn_shared_head_norm`

All bindings use `required_tensorf` — missing tensors exit.

### `weights_bind_layer`

```c
static void weights_bind_layer(
    ds4_layer_weights *l,
    const ds4_model   *m,
    uint32_t           il);
```

Dispatches to `weights_bind_glm_dsa_layer` when `DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA`.  Otherwise binds the standard DeepSeek V4 layer tensor set (hc_attn, MLA attention, compressors, indexers, hc_ffn, routed+shared MoE FFN).

### `mtp_weights_bind`

```c
static void mtp_weights_bind(
    ds4_mtp_weights *w,
    const ds4_model *m);
```

MTP (Multi-Token Prediction) draft head weight binding.  Binds tensors under the `mtp.0.` prefix:

**Head tensors:** `hc_head_base`, `hc_head_fn`, `hc_head_scale`, `e_proj`, `h_proj`, `enorm`, `hnorm`, `norm`

**Transformer block (full layer equivalent):** `hc_attn_fn`, `hc_attn_scale`, `hc_attn_base`, `attn_norm`, `attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv`, `attn_kv_a_norm`, `attn_sinks`, `attn_output_a`, `attn_output_b`, `hc_ffn_fn`, `hc_ffn_scale`, `hc_ffn_base`, `ffn_norm`, `ffn_gate_inp`, `ffn_exp_probs_b`, `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`

All bindings use `required_tensor` — missing tensors exit.  After binding, calls `mtp_weights_validate_layout`.

### `dspark_weights_bind_optional`

```c
static void dspark_weights_bind_optional(
    ds4_dspark_weights       *dw,
    const ds4_model          *m,
    const ds4_dspark_summary *summary);
```

DSpark speculative decoding weight binding.  Optional — if the GGUF file lacks DSpark tensors, the function returns without error.

Uses `tensor_by_mtp_stage_suffix` (wraps `model_find_tensor`) internally via `dspark_bind_tensor`, which counts present vs missing tensors per stage.  Binds:

**Per stage (0..n_stages-1):** Full transformer block (same layout as MTP block above: hc_attn, MLA attention, hc_ffn, MoE FFN).
**Stage 0 only:** `main_proj`, `main_norm`
**Final stage:** `norm`, `hc_head_base`, `hc_head_fn`, `hc_head_scale`, `markov_head.markov_w1`, `markov_head.markov_w2`, `confidence_head.proj`

After binding, calls `dspark_weights_validate_layout`.

## `required_tensorf` vs `tensor_by_namef`

### `tensor_by_namef`

```c
static ds4_tensor *tensor_by_namef(
    const ds4_model *m,
    const char      *fmt,
    uint32_t         layer);
```

Formats a tensor name using `snprintf(fmt, layer)` and calls `model_find_tensor`.  Returns `NULL` if the tensor is not found.  Used for optional tensors — caller must check return value.

### `required_tensorf`

```c
static ds4_tensor *required_tensorf(
    const ds4_model *m,
    const char      *fmt,
    uint32_t         layer);
```

Same name formatting as `tensor_by_namef`, but calls `required_tensor` which prints an error and calls `exit(1)` if the tensor is missing.  Used for all mandatory weight bindings.

### Non-layered variants

- `required_tensor(m, name)` — non-layered version (no `%u` format).  Used for global tensors like `token_embd.weight` and MTP/DSpark head tensors.
- `model_find_tensor(m, name)` — raw lookup returning NULL on miss.  Used directly when the caller wants to distinguish "absent" from "error" without formatting.

## Tensor Validation

- Every tensor type and dimension checked against active profile.
- Routed expert tensors validated as 3D.
- Shared expert tensors validated as 2D.
- Compressor tensors validated per-layer ratio metadata.

## Shape Assertions

See [model-shapes.md](model-shapes.md) for expected tensor shapes and dimension constants.

## Invariants

- After binding, all weight pointers are non-NULL for required tensors.
- Optional tensors (GLM-specific, directional steering) checked separately.
- Binding order matches layer iteration order for cache-friendly access.

[← Back to Index](../README.md)
