# Quantized Tensor Kernels (CPU Reference)

## Files

- `ds4.c` — scalar conversion, Q8_0/Q2_K/IQ2_XXS dot products, quantize routines

## Purpose

CPU reference math for quantized tensor formats present in DeepSeek V4 Flash GGUF.  Used by C backend (reference/debug) and Metal diagnostics.  Not used in production Metal path (GPU has own kernels).

## Quant Formats

See [gguf-format.md](../concepts/gguf-format.md) for quant format specifications (block sizes, bytes/block).

## Kernel Implementations

### Conversions

- `f16_to_f32()` / `f32_to_f16()` — half-precision conversion (NEON or software)
- `quantize_q8_0_activation()` — float → Q8_0 block quantization
- `ds4_quantize_row_q8_K()` — float → Q8_K block quantization

### Dot Products

- `dot_q2_16()` — Q2_K × Q8_0 partial dot (16-element)
- `dot_iq2_pair_16()` — IQ2_XXS × Q8_0 partial dot
- `matvec_q8_0()` — single-token Q8_0 matrix-vector
- `matvec_q2_k_expert()` — single-expert Q2_K down projection (line 8118)
- `matvec_q4_k_experts_mid_prequant()` — Q4_K gate/up matvec with mid prequant (line 8844)
- `matvec_q4_k_experts_accum_prequant()` — Q4_K down projection with accum prequant (line 8916)

### Quantize-Activation Pipeline

Decode scratch owns temporary activation quantization so generation does not allocate in hot loop.  Two paths exist:

**Q8_0 path** — attention Q projections (`quantize_q8_0_activation` at line 7051):
```
input float activation → quantize_q8_0_activation() → Q8_0 block (int8 xq + fp32 scale)
→ matvec_q8_0_rows_prequant() / matvec_q8_0_decode_scratch()
→ dot product with Q8_0 weight → accumulate float result
```
Call sites: `layer->attn_q_a`, `layer->attn_q_b`, `layer->attn_kv`, `layer->attn_output_b`, `layer->ffn_gate_shexp`, `layer->ffn_up_shexp`, `layer->ffn_down_shexp`, `weights->output`.

**Q8_K path** — expert FFN mid/accum (`ds4_quantize_row_q8_K` at line 3306):
```
input float activation → ds4_quantize_row_q8_K() → Q8_K block
→ matvec_q2_k_expert() / matvec_q4_k_experts_mid_prequant() / matvec_q4_k_experts_accum_prequant()
→ dot product with Q2_K/Q4_K weight → accumulate float result
```
Call sites: `layer->ffn_gate_exps`, `layer->ffn_up_exps`, `layer->ffn_down_exps`.

## Performance Characteristics

See [cpu.md](../backends/cpu.md) for SIMD availability (ARM NEON on aarch64, scalar on x86-64).  Throughput adequate for single-sequence debug and reference validation only.  Hot path (Metal GPU) runs equivalent operations on-device with parallel reduction.

Q8_K activation blocks are ephemeral: allocated in decode scratch, valid only within one decode step.  Avoids per-step allocator pressure at cost of repeated quantize work each step.

IQ2_XXS dot product uses pre-computed signed grid tables initialized once via `pthread_once`.  Table construction cost amortized over model lifetime.

## Invariants

- All quantized formats are little-endian.
- Block sizes are fixed per format and assumed by all consumers.
- Quantize functions assume contiguous float input; no stride or batch support.

## See Also

- [cuda.md](../backends/cuda.md) — CUDA backend quantized kernel implementations
- [metal.md](../backends/metal.md) — Metal GPU quantized kernel implementations
- [rocm.md](../backends/rocm.md) — ROCm backend quantized kernel implementations

[← Back to Index](../README.md)
