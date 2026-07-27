# Mixture-of-Experts FFN

## Files

- `ds4.c` — MoE pipeline (CPU reference)
- `ds4.c` — Q2_K/Q4_K batch matvec workers

## Purpose

FFN half of each layer: HC pre, RMSNorm, shared expert, routed expert selection, IQ2_XXS gate/up projections, SwiGLU activation, Q2_K down projection, merge, HC post step.

## FFN Pipeline

```
HC pre (1→4 stream expansion via Sinkhorn)
→ RMSNorm
→ router: softplus(√logits)→probs, bias for selection, unbiased probs for weighting, top-k
→ for each selected expert (routed MoE):
    IQ2_XXS gate/up projections → clamp(gate), clamp(up)
    → SiLU(gate)*up*router_weight → Q2_K down projection → accumulate
→ shared expert gate/up (dense F16) → SwiGLU → shared expert down (dense F16)
→ routed + shared → directional steering → HC post (4→1 stream collapse)
```

Both routed MoE and shared expert receive the same RMSNorm-normalized input.

## Router

- Input: RMSNorm-normalized activation
- Computes raw logits via dense F16 matvec: `logits[i] = dot(weight[i], x)`
- Converts to probabilities: `probs[i] = sqrt(softplus(logits[i]))`
  - `softplus(x)` = `log(1 + exp(x))` with stable clamping at ±20
  - `sqrt` scales for numerical stability before top-k selection
- Expert count: 256 (Flash), 384 (Pro), 256 (GLM52)
- Top-k selection: k=6 (Flash, Pro), k=8 (GLM52)
- **Selection** uses biased scores: `probs[i] + bias[i]` (learned per-expert bias via `ffn_exp_probs_b`)
- **Weighting** uses unbiased probabilities: selected experts weighted by `probs[i]`, renormalized to sum=1, then scaled by `expert_weight_scale` (1.5 Flash, 2.5 Pro)
- Router weights are dense F16

## Expert Evaluation

Per-expert matrices:

| Matrix | Format | Dims |
|---|---|---|
| gate/up | IQ2_XXS | 256 × hidden_in |
| down | Q2_K | hidden_out × 256 |
| Q4_K variant | Q4_K | same dims |

See [moe-routing.md](../concepts/moe-routing.md) for expert quantization format details.

SwiGLU activation per expert:

```
swiglu(gate, up) = silu(gate) * up
```

Where `silu(x) = x * sigmoid(x)`.

**Clamp**: Applied to both gate and up after matvec, before SwiGLU.
- Gate: positive-only clamp (if `gate > clamp`, set to `clamp`)
- Up: bidirectional clamp (if `up > clamp`, set to `clamp`; if `up < -clamp`, set to `-clamp`)
- Clamp disabled when value < `1e-6`

## Shared Expert

See [moe-routing.md](../concepts/moe-routing.md) for shared expert structure details.

## Output Merge

Combines shared + routed expert output, applies directional steering, then HC post projection (1 stream → 4 streams, Sinkhorn normalize for collapse).

## Invariants

- See [moe-routing.md](../concepts/moe-routing.md) for expert weight 3D tensor indexing details.
- Down projection accumulates directly into output buffer (no temp copy for sum)
- Q2_K down uses batch workers when multiple experts selected
- Expert locality profiling hooks inserted after router, before weight fetch

## See Also

- [gguf-format.md](../concepts/gguf-format.md) — quant format specifications for expert weights
- [gpu-expert-streaming-cache.md](../concepts/gpu-expert-streaming-cache.md) — expert weight streaming and cache management on GPU

[← Back to Index](../README.md)
