# Hyper-Connection State (4-Stream)

## Definition

Token state maintained as 4 parallel streams of `DS4_N_EMBD`-dim vectors. Before each sublayer, a learned projection reduces 4→1 stream. After the sublayer, another projection expands 1→4 streams. Each stream can carry different semantic content.

`DS4_N_EMBD` = 4096 (Flash) or 7168 (Pro). See model shape definitions in ds4.c:540-680.

## Why It Exists

Standard transformers use a single residual stream. DeepSeek V4 uses 4 streams to separate content, position, task-specific info, etc., without interference. The model learns per-layer how to mix them.

## Where It Appears

| File | Role |
|---|---|
| `ds4.c` | HC decode, sinkhorn, combine (lines 9592-9820) |
| `ds4.c` | HC pre in attention/FFN pipelines (lines 9690-9761) |
| `ds4.c` | HC post in sublayer pipelines (lines 9772-9830) |
| `ds4.c` | Output head with sigmoid-weighted sum (lines 13879-13908, 14730-14755) |
| `ds4_gpu.h` | GPU kernel declarations for HC transforms |

## Variants

| Variant | When Used | Key Difference |
|---|---|---|
| `n_hc = 4` (default) | DeepSeek V4 Flash, Pro | Full HC with 4 streams, 4 pre-weights, 4 post-gates, 4×4 sinkhorn combine matrix. `DS4_N_HC_SINKHORN_ITER = 20` |
| `n_hc = 0` | GLM52 (P0 fix, ds4.c:636) | HC bypassed entirely. All loops iterate zero times (`i < n_hc` → 0 iterations). Sinkhorn iterations also zero. Token processed as single stream. Model still allocates no HC tensors |

Per-model differences:

| Model | `n_hc` | `n_embd` | `n_hc_sinkhorn_iter` | `hc_eps` |
|---|---|---|---|---|
| DeepSeek V4 Flash (ds4.c:540) | 4 | 4096 | 20 | `DS4_DEFAULT_HC_EPS` |
| DeepSeek V4 Pro (ds4.c:595) | 4 | 7168 | 20 | `DS4_DEFAULT_HC_EPS` |
| GLM52 (ds4.c:633) | 0 | 6144 | 0 | 0.0f |

## State Structure

```
hc[4][DS4_N_EMBD] — 4 streams × embedding dim

hc[0] — stream A
hc[1] — stream B
hc[2] — stream C
hc[3] — stream D
```

## HC Pre (4→1)

See `hc_pre_from_state_one_scratch` (ds4.c:9690).

```
input: hc[4][DS4_N_EMBD]
  ↓ 1. RMSNorm (no weight) on flat HC state
  ↓ 2. Learned projection: normalized flat HC → mix[24] via output_hc_fn
  ↓ 3. Decode mix → pre weights (sigmoid), post gates (sigmoid), combine matrix (4×4 sinkhorn)
  ↓ 4. Weighted sum of 4 streams using pre weights
output: 1 × DS4_N_EMBD (fed to attention or FFN)
```

## HC Post (1→4)

See `hc_post_one` (ds4.c:9772).

```
input: 1 × DS4_N_EMBD (sublayer output)
         residual hc[4][DS4_N_EMBD] (preserved before HC pre)
  ↓ per stream: block_out × post_gate[dst] + Σ_src combine[dst][src] × residual[src]
output: 4 × DS4_N_EMBD
```

The combine matrix is a small `4 × 4 = 16` scalar matrix — it mixes the 4 residual streams per dimension using the same weights across all embedding positions. Normalized via Sinkhorn iterations (alternating row/column normalization, default 20 iters).

## Initial State

Token embedding replicated 4 ways to form initial HC state. All streams start identical; divergence learned through layers. See `hc_from_plain_embedding` (ds4.c:9764).

## Final State

Last layer's HC post feeds into output head:

```
hc[4][DS4_N_EMBD]
  ↓ 1. RMSNorm (no weight) on flat HC
  ↓ 2. Learned projection → per-stream sigmoid gates
  ↓ 3. Weighted sum of ALL 4 streams (NOT single stream — learned per-stream weights)
  ↓ 4. RMSNorm (with weight) → vocab projection → logits
```

See `output_hc_head_one` (ds4.c:13879) and `output_logits_one` (ds4.c:13910).

## Lifecycle

```
embedding → replicate 4× → HC state
  → for each layer:
      HC pre (4→1) → sublayer → HC post (1→4) → residual add
  → HC post → RMSNorm → head → logits
```

## Relationship to Other Concepts

- **Depends on**: RMSNorm, sinkhorn normalization
- **Used by**: attention pipeline, MoE FFN pipeline
- **Alternatives**: single residual stream (standard transformer), dual-stream (some MoE architectures)

## Engine-Level Reference

For function signatures, line ranges, tensor shapes, and GPU kernel details, see `docs/code/engine/hc-transforms.md`.

## Code Pattern

```c
// HC state initialization — replicate token embedding 4 ways (ds4.c:9764)
static void hc_from_plain_embedding(float *out_hc, const float *x,
                                    uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t h = 0; h < n_hc; h++) {
        memcpy(out_hc + (uint64_t)h * n_embd, x, (size_t)n_embd * sizeof(x[0]));
    }
}

// HC pre (4→1) — RMSNorm → learned projection → sinkhorn split → weighted sum (ds4.c:9690)
// Pre-weights: 4 scalars (not [4096])      — ds4.c:9710
// Post-gates:  4 scalars (not [4096])      — ds4.c:9714
// Combine:     4×4 scalars (not [4096][4096]) — ds4.c:9717
static void hc_pre_from_state_one_scratch(...) {
    rms_norm_no_weight(flat, residual_hc, hc_dim, DS4_RMS_EPS);  // ds4.c:9707
    matvec_f16(mix, model, fn, flat);           // learned projection → 24-dim mix
    hc_split_sinkhorn_one(split, mix, scale, base, n_hc, n_iter, 1e-6f);
    hc_weighted_sum_one(out, residual_hc, split, DS4_N_EMBD, n_hc);
    memcpy(post, split + n_hc, n_hc * sizeof(post[0]));
    memcpy(comb, split + 2 * n_hc, n_hc * n_hc * sizeof(comb[0]));
}

// HC post (1→4) — block_out × gate[dst] + Σ comb[dst][src] × residual[src] (ds4.c:9772)
static void hc_post_one(float *out_hc, const float *block_out,
                        const float *residual_hc, const float *post,
                        const float *comb, uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = block_out[d] * post[dst];
            for (uint32_t src = 0; src < n_hc; src++) {
                acc += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * n_embd + d];
            }
            out_hc[(uint64_t)dst * n_embd + d] = acc;
        }
    }
}

// Output head — ALL 4 streams contribute via learned sigmoid gates (ds4.c:13879)
// NOT "only one stream used" — per-stream learned weights via sigmoid(scale[i] + base[i])
static void output_hc_head_one(...) {
    rms_norm_no_weight(flat, inp_hc, hc_dim, DS4_RMS_EPS);
    matvec_f16(pre, model, weights->output_hc_fn, flat);
    for (uint32_t i = 0; i < n_hc; i++) {
        w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + DS4_HC_EPS;
    }
    hc_weighted_sum_one(out, inp_hc, w, DS4_N_EMBD, n_hc);
}

// n_hc = 0 variant (GLM52, ds4.c:636): all loops iterate zero times, no-op
// Verified: no conditional branches, just loop bounds drive bypass
```

[← Back to Index](../README.md)
