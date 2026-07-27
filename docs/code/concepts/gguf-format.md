# GGUF Format (DS4 Subset)

## Definition

GGUF is the model file format for storing quantized weights.  DS4 reads a subset: F16, F32, Q8_0, Q8_K, Q2_K, Q4_K, IQ2_XXS, Q5_K, Q6_K.

## Why It Exists

GGUF packs quantized tensors and metadata (architecture, layer count, tokenizer) into a single self-describing file.  Self-description removes guesswork: the file declares its own shape and type so the loader adapts without external config.

## Where It Appears

| File | Role |
|---|---|
| `ds4.c` | GGUF parser |
| `ds4.c` | quantized kernel implementations |
| `gguf-tools/` | quantization, imatrix, inspection |

## Layout

### File Structure

```
magic (0x46554747 "GGUF") → version → tensor count
→ metadata KV count → metadata KV pairs (string key + typed value)
→ tensor directory (name + ndim + dims + type + offset)
→ tensor data (at offsets)

`type` maps to a quantization format (see table below) and determines the per-element byte size via `tensor_nbytes()`. Without `type`, the parser cannot compute tensor memory footprint or select the correct dequant kernel.
```

### Metadata Keys Used

| Key | Type | Purpose |
|---|---|---|
| `general.architecture` | string | model architecture |
| `deepseek4.block_count` | u32 | layer count |
| `deepseek4.embedding_length` | u32 | embedding dimension |
| `deepseek4.expert_count` | u32 | total experts |
| `deepseek4.expert_used_count` | u32 | top-k experts |
| `deepseek4.attention.head_count` | u32 | attention heads |
| `deepseek4.attention.key_length` | u32 | head dimension |
| `deepseek4.attention.compress_ratios` | u32 | compression ratios per layer |
| `deepseek4.attention.compress_rope_freq_base` | f32 | compressed RoPE freq base |
| `tokenizer.ggml.*` | various | tokenizer data |

### DS4-Specific Convention

DeepSeek V4 stores MoE expert weights as 3D tensors: `[n_expert, out_dim, in_dim]`.  Standard GGUF only defines 1D and 2D; DS4 infers the 3D layout from tensor name patterns.

## Quant Formats Used

| Format | Block Size | Bytes/Block | Bits/Weight |
|---|---|---|---|
| F16 | 1 | 2 | 16 |
| F32 | 1 | 4 | 32 |
| Q8_0 | 32 | 34 | 8.5 |
| Q8_K | 256 | 292 | 9.125 |
| Q2_K | 256 | 84 | 2.625 |
| Q4_K | 256 | 144 | 4.5 |
| IQ2_XXS | 256 | 66 | 2.0625 |
| Q5_K | 256 | 176 | 5.5 |
| Q6_K | 256 | 210 | 6.5625 |

## Relationship

- **Depends on**: [model-shape-detection.md](model-shape-detection.md) (validates metadata against profile)
- **Used by**: GGUF loading (parser), quantized kernels (runtime format handling)

[← Back to Index](../README.md)
