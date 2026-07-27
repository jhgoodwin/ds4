# MoE Routing

## Definition

Each token activates only a subset of expert networks.  A learned router computes logits over all experts, selects the top-k, and routes the token's activation through those experts' gate/up/down matrices.

## Why It Exists

MoE increases model capacity (total parameters) without proportional compute.  Flash: 284B total, 13B active per token.  Pro: 1.6T total, 49B active.  Router selects 6 out of 256 (2.34%) for Flash, 6 out of 384 (1.56%) for Pro.

## Where It Appears

| File | Role |
|---|---|
| `ds4.c` | router logit computation, top-k selection |
| `ds4.c` | expert probability calculation (softplus, sqrt) |

See [moe-ffn.md](../engine/moe-ffn.md) for expert FFN pipeline and shared expert details.

## Router Architecture

```
input: FFN-normalized activation
→ dense F16 projection → expert logits
→ top-k selection (k=6 Flash, k=6 Pro, k=8 GLM52)
→ softmax over selected → expert weights
```

## Top-K Selection

Sort expert logits, pick top-k values.  Only selected experts' weights fetched from memory.  Non-selected experts contribute zero.

### Hash-Based Routing (Early Layers)

First `n_hash_layer` layers (3 for Flash/Pro, 0 for GLM52) bypass learned top-k and use a precomputed token-id-to-expert lookup table (`ffn_gate_tid2eid.weight`).  Router scores are still computed via sqrt(softplus(logit)) and normalized, but selection is deterministic by token id.  Reduces training collapse at early layers where representation is weak.

## Variants

| Model | n_expert | n_expert_used | top-k | hash layers | routing notes |
|---|---|---|---|---|---|
| Flash | 256 | 6 | 6 | 3 | first 3 layers use token-id hash routing; indexer top-k=512 |
| Pro | 384 | 6 | 6 | 3 | first 3 layers use token-id hash routing; indexer top-k=1024 |
| GLM52 | 256 | 8 | 8 | 0 | all layers use learned top-k; indexer top-k=2048 |

### Density

Fraction of experts activated per token:

| Model | activation fraction |
|---|---|
| Flash | 6 / 256 = 2.34% |
| Pro | 6 / 384 = 1.56% |
| GLM52 | 8 / 256 = 3.12% |

### Routing Logic Per Model

- **Flash / Pro**: learned router with softplus → sqrt → top-k selection; first `n_hash_layer=3` layers switch to token-id hash routing for stability.
- **GLM52**: same learned router (softplus → sqrt → top-k) but all 79 layers use learned routing (`n_hash_layer=0`); no hash-based early layers.

Router decision logic (score + selection) is documented here; the expert FFN pipeline that consumes routing decisions is in [moe-ffn.md](../engine/moe-ffn.md).

## Relationship to Other Concepts

- **Depends on**: quantized kernels (IQ2_XXS, Q2_K)
- **Used by**: [moe-ffn.md](../engine/moe-ffn.md) (FFN pipeline consumes routing decisions), imatrix collection
- **Alternatives**: dense FFN (no routing, all parameters active)

[← Back to Index](../README.md)
