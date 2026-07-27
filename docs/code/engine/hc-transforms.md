# Hyper-Connection Transforms

## Engine-Level Reference

Describes HC state transform functions (pre, post, output head) as implemented in ds4.c and GPU kernels. For conceptual overview, state structure, and lifecycle, see [hc-state.md](../concepts/hc-state.md).

## Files

- `ds4.c` — CPU decode path, sinkhorn, weighted sum, output head
- `ds4_gpu.h` — GPU kernel declarations for HC transforms
- `metal/*.metal` — Metal GPU kernels

## Dependencies

- **HC state**: 4 parallel residual streams (see hc-state.md)
- **Weight tensors**: `output_hc_fn`, `output_hc_scale`, `output_hc_base`, `output_norm` (loaded from GGUF)
- **RMSNorm**: `rms_norm_no_weight`, `rms_norm_weight`
- **Matmul**: `matvec_f16`, `matvec_q8_0`
- **Sinkhorn**: alternating row/column normalization (CPU only)

## Key Types

| Type | Role |
|---|---|
| `float hc_state[DS4_N_HC][DS4_N_EMBD]` | 4 parallel streams of embedding-dim vectors |
| `float pre_weights[DS4_N_HC]` | Sigmoid-gated weights for HC pre weighted sum |
| `float post_gates[DS4_N_HC]` | Sigmoid-gated weights for HC post stream gating |
| `float combine_matrix[DS4_N_HC * DS4_N_HC]` | Sinkhorn-normalized doubly-stochastic 4×4 mix matrix |

## Constants

| Symbol | Value (Flash/Pro) | Definition |
|---|---|---|
| `DS4_N_HC` | 4 (both) | Number of HC streams |
| `DS4_N_EMBD` | 4096 / 7168 | Embedding dimension per stream |
| `DS4_N_HC_SINKHORN_ITER` | 20 (both) | Sinkhorn normalization iterations |
| `DS4_HC_EPS` | model-specific | Epsilon for combine matrix stability |

All macros defined in `ds4.c:699-731`. Model shapes in `ds4.c:540-680`.

## API Surface

### HC Pre (4→1)

Reduces 4 HC streams into 1 embedding vector consumed by attention/FFN.

| Function | Line | Signature | Role |
|---|---|---|---|
| `hc_pre_from_state_one_scratch` | 9690 | `(float* out, const float* residual_hc, ds4_model*, ds4_hparams*, float* scratch, int n_vocab)` | Full HC pre pipeline using caller-provided scratch buffer |
| `hc_pre_from_state_one` | 9723 | `(float* out, const float* residual_hc, ds4_session*, ds4_model*)` | Allocates scratch, calls scratch variant |
| `hc_split_sinkhorn_one` | 9592 | `(float* pre_weights, float* post_gates, float* combine_matrix, const float* mix, int n_hc)` | Decodes mix → sigmoid weights, gates, doubly-stochastic combine matrix |
| `hc_weighted_sum_one` | 9673 | `(float* out, const float* res_hc, const float* w, int n_embd, int n_hc)` | Weighted sum of n_hc streams into one |

**Call graph:**

```
hc_pre_from_state_one
  └─ hc_pre_from_state_one_scratch
       ├─ rms_norm_no_weight(flat, residual_hc, hc_dim, DS4_RMS_EPS)
       ├─ matvec_f16(mix, model, output_hc_fn, flat)     → mix[24]
       ├─ hc_split_sinkhorn_one(pre_weights, post_gates, combine_matrix, mix, n_hc)
       │    ├─ sigmoid(pre_weights[i])                    mix[0..3]
       │    ├─ post_gates[i] = sigmoid(mix[4+i]) * 2.0    mix[4..7]
       │    └─ sinkhorn(combine_matrix, &mix[8], n_hc)     mix[8..23] → 4×4 double-stochastic
       └─ hc_weighted_sum_one(out, residual_hc, pre_weights, n_embd, n_hc)
```

**Learnable weights:** `output_hc_fn` tensor, shape `(DS4_N_EMBD * DS4_N_HC) × DS4_N_HC`, type F16.

**mix[24] layout**: `[0..3]` pre_weights (sigmoid), `[4..7]` post_gates (sigmoid, scaled [0,2]), `[8..23]` combine matrix logits (sinkhorn → row/col sums = 1).

**Sinkhorn** (`hc_split_sinkhorn_one:9592`): alternating row softmax + column normalization on `float c[16*16]`. Fixed 20 iterations. No early exit.

### HC Post (1→4)

Expands 1 sublayer output vector back into 4 HC streams.

| Function | Line | Signature | Role |
|---|---|---|---|
| `hc_post_one` | 9772 | `(float* out_hc, const float* block_out, const float* residual_hc, const float* post_gates, const float* combine_matrix, int n_embd, int n_hc)` | Single-token: gate + cross-stream residual combine |
| `hc_post_batch` | 9812 | `(float* out_hc, const float* block_out, const float* residual_hc, const float* post_gates, const float* combine_matrix, int n_tokens, int n_embd, int n_hc)` | Batched HC post via thread workers |
| `hc_post_batch_worker` | 9794 | Thread worker for batch variant | |

**Call graph:**

```
hc_post_one / hc_post_batch
  ├─ for each dst stream:
  │    block_out[d] × post_gates[dst]             ← gated sublayer output
  │    + Σ_src combine[dst][src] × residual_hc[src][d]  ← cross-stream residual
  └─ out_hc[dst][d] = gated + residual_mix
```

**Combine matrix access:** `comb[dst + src * n_hc]` — 1 scalar per (dst, src) pair. Small 4×4 = 16 scalars, same weights across all embedding positions.

### Output Head (HC → Logits)

Collapses final 4-stream HC state into logits.

| Function | Line | Role |
|---|---|---|
| `output_hc_head_one` | 13879 | CPU: HC collapse + sigmoid gates + weighted sum |
| `output_logits_one` | 13910 | CPU: HC head + RMSNorm + Q8_0 vocab projection |
| `output_logits_one_decode_scratch` | 14730 | CPU: allocation-free logits head for decode |
| `metal_graph_encode_output_head` | 23942 | Metal GPU: fused output head graph node |
| `ds4_gpu_output_hc_weights_tensor` | ds4_gpu.h:2463 | GPU kernel: sigmoid gates from pre, scale, base |
| `ds4_gpu_hc_weighted_sum_tensor` | ds4_gpu.h:2389 | GPU kernel: weighted sum of all 4 streams |

**Call graph:**

```
output_logits_one / output_logits_one_decode_scratch
  ├─ output_hc_head_one
  │    ├─ rms_norm_no_weight(flat, inp_hc, hc_dim, DS4_RMS_EPS)
  │    ├─ matvec_f16(pre, model, output_hc_fn, flat)
  │    ├─ w[i] = sigmoid(pre[i] * output_hc_scale[0] + output_hc_base[i]) + eps
  │    └─ hc_weighted_sum_one(embd, inp_hc, w, DS4_N_EMBD, DS4_N_HC)    ← ALL 4 streams
  ├─ rms_norm_weight(norm, embd, output_norm, DS4_N_EMBD, DS4_RMS_EPS)
  └─ matvec_q8_0(logits, model, output, norm)
```

**Learnable components:**

| Tensor | Shape | Type | Role |
|---|---|---|---|
| `output_hc_fn` | (DS4_N_EMBD * DS4_N_HC) × DS4_N_HC | F16 | Maps normalized flat HC → stream weights |
| `output_hc_scale` | [1] | F32 | Scales pre-activation before sigmoid |
| `output_hc_base` | [DS4_N_HC] | F32 | Bias per stream before sigmoid |
| `output_norm` | [DS4_N_EMBD] | F32 | RMSNorm weight after HC collapse |
| `output` | DS4_N_EMBD × DS4_N_VOCAB | Q8_0 | Vocab projection matrix |

## Data Flow

```
token embedding
  → hc_from_plain_embedding:9764 (replicate 4× into hc[4][embd])
  → for each layer:
       hc_pre_from_state_one:9723 (4→1, preserves residual_hc)
         → sublayer (attention / FFN)
         → hc_post_one:9772 (1→4, gate + cross-stream residual mix)
  → output_logits_one:13910 (HC collapse → RMSNorm → Q8_0 vocab proj)
       └─ output_hc_head_one:13879 (all-stream weighted sum, NOT stream 0 only)
```

## Invariants

- Every layer executes exactly: HC pre → sublayer → HC post
- HC pre writes `residual_hc` (the 4 input streams) before reducing — HC post reads these for cross-stream residual
- HC post reads `post_gates` and `combine_matrix` produced by HC pre in the same layer
- Output head collapses ALL 4 streams via learned per-stream gates, NOT stream 0 only
- Combine matrix is Sinkhorn-normalized to doubly-stochastic (rows + columns both sum to 1)
- First layer: `hc_from_plain_embedding` replicates token embedding into all 4 streams (line 9764)
- Sinkhorn runs fixed 20 iterations regardless of convergence — no early exit

### Metal/CUDA

- Metal fuses entire output head into `metal_graph_encode_output_head:23942` — single GPU graph node avoids per-kernel launch overhead
- GPU kernels `ds4_gpu_output_hc_weights_tensor` and `ds4_gpu_hc_weighted_sum_tensor` operate on tensor-shaped buffers, not flat CPU arrays
- Sinkhorn is CPU-only — GPU reads combine matrix produced by CPU prior to graph encoding
- HC pre/post on GPU follows same call graph as CPU but uses tensor-level primitives (`ds4_gpu_matmul_q8_0_tensor`, etc.)
- Weighted sum on GPU is fused kernel to avoid intermediate buffer allocation for per-stream gated values

## Configuration

- `DS4_N_HC_SINKHORN_ITER`: compile-time constant (default 20)
- Model shape selects `DS4_N_EMBD` (4096 Flash / 7168 Pro)

## Notes

- Initial implementation used stream-0-only for output head. Switched to all-stream sigmoid-weighted sum for better logit quality.
- Function signatures and line ranges here are engine-level reference. For state structure, lifecycle, and design rationale, see [hc-state.md](../concepts/hc-state.md).

## See Also

- [gpu-tensor-primitives.md](../concepts/gpu-tensor-primitives.md) — GPU tensor-level primitives for HC pre/post/output head

[← Back to Index](../README.md)
