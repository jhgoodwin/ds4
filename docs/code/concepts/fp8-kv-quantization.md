# FP8 KV Cache Quantization

## Definition

DS4 applies FP8 quantization to the compressed KV cache — E4M3 format (4 exponent, 3 mantissa, bias 7). The quantize is an in-place f32 round-trip: values are quantized to E4M3 then immediately dequantized back to f32 at write time. This adds quantization noise matching FP8 precision but **does not reduce storage** — the tensor remains f32 (or f16 if `DS4_GPU_ATTN_COMP_CACHE_F16` is set). Attention kernels read compressed rows as plain `float*` — no on-read decompression.

Storage reduction comes from KV compression (dimension reduction via pooling), not from FP8 quantization.

Only the non-RoPE portion (`head_dim - n_rot` elements) is quantized. The RoPE tail stays f32, matching DeepSeek V4 reference: only the nope (non-positional) subspace is quantized.

Indexer keys use a different scheme: 128-wide Hadamard transform followed by E2M1 FP4 round-trip via `ds4_gpu_dsv4_indexer_qat_tensor`. Indexer keys remain f16 for score computation — FP4 is quantization-aware training (QAT) for the indexer projection, not a storage optimization.

## Why It Exists

KV cache dominates memory at long contexts (1M+ tokens). FP8 quantization improves attention output quality for a given memory budget by eliminating noise that a full f32 write would preserve, at negligible accuracy loss. The E4M3 round-trip is lossy but error is small relative to the attention softmax distribution.

Combined with KV compression ratios (4x, 128x) — which are the actual storage savings — total KV memory for 1M context fits consumer GPU VRAM. E4M3 quantize runs once per compressed row at write time with no decode-time cost.

Note: FP8 quantize is a write-time noise-injection technique, not a storage format. Storage reduction comes from the compressor (pools KV pairs into lower-dimension rows), not from FP8 quantization.

## E4M3 Format

```
Bit:  7   6 5 4 3    2 1 0
     Sign Exponent   Mantissa
```

- Sign: 1 bit
- Exponent: 4 bits (bias 7)
- Mantissa: 3 bits (explicit leading bit when exponent > 0)
- Range: ±448 (max representable)
- Min normalized: 2⁻⁷ ≈ 0.0078
- Zero: sign=0, exp=0, mant=0
- Subnormal: exp=0 → mant * 2⁻⁹ (step 0.001953125)
- Normal: exp>0 → (1 + mant × 0.125) × 2^(exp − 7)

### Value Table

```c
// dsv4_e4m3fn_value_cpu
static const float exp_scale[16] = {
    0.0f, 0.015625f, 0.03125f, 0.0625f,
    0.125f, 0.25f, 0.5f, 1.0f,
    2.0f, 4.0f, 8.0f, 16.0f,
    32.0f, 64.0f, 128.0f, 256.0f,
};

// i = (exp << 3) | mant, 7-bit index
int exp = (i >> 3) & 0x0f;
int mant = i & 0x07;
return exp == 0
    ? (float)mant * 0.001953125f     // subnormal
    : (1.0f + (float)mant * 0.125f) * exp_scale[exp];  // normal
```

127 representable values (index 0..126, sparse via 7-bit encoding). Index 127 (exp=15, mant=7) is NaN in standard E4M3 but unused here.

## Where It Appears

| File | Symbol | Role |
|---|---|---|
| `ds4_gpu.h` | `ds4_gpu_dsv4_fp8_kv_quantize_tensor` | GPU primitive: in-place E4M3 round-trip on nope portion. Takes `n_tok`, `head_dim`, `n_rot`. |
| `ds4_gpu.h` | `ds4_gpu_dsv4_indexer_qat_tensor` | GPU primitive: Hadamard + E2M1 FP4 round-trip on 128-wide indexer keys. |
| `ds4_gpu.h` | `ds4_gpu_compressor_prefill_tensor` | Compressor: pools raw KV, applies RMS norm, RoPE, optional quantize. Gated by `quantize_fp8` bool. |
| `ds4_gpu.h` | `ds4_gpu_compressor_prefill_ratio4_replay_tensor` | Same as prefill for ratio-4 replay path. |
| `ds4_gpu.h` | `ds4_gpu_compressor_update_tensor` | Compressor: pools single token, applies RMS norm, RoPE on emit. Caller applies quantize separately on emitted row. |
| `ds4.c` | `dsv4_e4m3fn_value_cpu` | E4M3 value table: `exp_scale[16]`, `(exp, mant) → f32`. 127 representable values. |
| `ds4.c` | `dsv4_e4m3fn_dequant_cpu` | Nearest-neighbor binary search over 127 E4M3 values. Round-ties-to-even. |
| `ds4.c` | `dsv4_fp8_kv_quantize_row_inplace_cpu` | Per-block amax → scale = 2^ceil(log2(amax/448)) → clamp to [-448, 448] → dequant → restore. CPU reference. |
| `ds4.c` | `dsv4_indexer_qat_row_inplace_cpu` | Hadamard 128 → E2M1 FP4 round-trip. CPU reference. |
| `ds4.c` | `dsv4_indexer_qat_rows_inplace_cpu` | Batched row QAT. |
| `ds4.c` | Graph decode/update/DSpark caller paths | Graph code invokes quantize on emitted compressed rows; DSpark raw store applies RMS norm, RoPE, then quantize. |
| `ds4_cuda.cu` | `fp8_kv_quantize_kernel` | CUDA kernel: 64 threads, per-block amax reduction, 64-wide groups. |
| `ds4_cuda.cu` | `fp8_kv_quantize_row` | Device function: shared-memory amax reduction → scale → clamp → dequant → multiply back. |
| `ds4_cuda.cu` | `fp8_kv_quantize_store_rows_kernel` | Combined quantize + f16 store to raw decode cache. |
| `ds4_cuda.cu` | `indexer_hadamard_fp4_kernel` | 128-wide Hadamard → FP4 E2M1 round-trip. 128 threads, butterfly network, per-32-lane amax. |
| `ds4_cuda.cu` | `dsv4_e4m3fn_dequant_dev` | Device binary search over 127 values. Same logic as CPU. |
| `ds4_cuda.cu` | `dsv4_e4m3fn_value_dev` | Device E4M3 value: `exp2f((float)exp - 7.0f) * (1.0f + mant * 0.125f)`. |
| `ds4_metal.m` | `g_dsv4_fp8_kv_quantize_pipeline` | Metal pipeline state for `kernel_dsv4_fp8_kv_quantize_f32`. |
| `ds4_metal.m` | `g_dsv4_indexer_qat_pipeline` | Metal pipeline state for `kernel_dsv4_indexer_hadamard_fp4_f32`. |
| `rocm/ds4_rocm_fp8_kv_launch.cuh` | `ds4_gpu_dsv4_fp8_kv_quantize_tensor` | ROCm launch: `dim3(n_tok, groups)` grid, 64 threads. |

## Quantization Process

### Per-Row (non-RoPE portion only)

```
raw KV row (f32, head_dim elements)
  → split at n_rot boundary
  → nope portion: [0, head_dim - n_rot)
  → for each 64-wide block:
      1. amax = max(|x[off+i]|) over block
      2. clamp to 1e-4 if amax < 1e-4
      3. scale = 2^ceil(log2(amax / 448))
      4. for each element:
           a. v = x[i] / scale
           b. clamp v to [-448, 448]
           c. binary-search nearest E4M3 value
           d. x[i] = nearest_value * scale
```

Scale is a power of 2 so multiply/divide is exact — no floating-point precision loss from scaling.

### GPU Kernel (CUDA)

`fp8_kv_quantize_kernel`: 64 threads per block (two warps; CUDA warp = 32 threads). Shared-memory scratch buffer holds 64 floats for reduction tree.

1. Cooperative load block into shared memory.
2. Parallel warp-tree reduction: stride 32 → 16 → … → 1, each step `scratch[tid] = fmaxf(scratch[tid], scratch[tid+stride])`.
3. `scratch[0]` = block amax.
4. `scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1e-4f) / 448.0f)))`.
5. Each thread: `dsv4_e4m3fn_dequant_dev(clamp(v / scale, -448, 448)) * scale`.

Per-block amax uses 64-element groups independent of head_dim. Head_dim=256 → 4 groups, each with separate scale. Matches CPU reference: `for (off = 0; off < n_nope; off += 64)`.

### Prefill Path

`ds4_gpu_compressor_prefill_tensor`: pool raw KV pairs → RMS norm → RoPE → `ds4_gpu_dsv4_fp8_kv_quantize_tensor` (gated by `quantize_fp8` param). Same pattern for `ds4_gpu_compressor_prefill_ratio4_replay_tensor`.

### Decode Update Path

`ds4_gpu_compressor_update_tensor`: store single-token KV → on emit, pool → RMS norm → RoPE. Caller then applies `ds4_gpu_dsv4_fp8_kv_quantize_tensor` explicitly on the emitted row. Quantize is outside the compressor function — differs from prefill where it's internal and gated.

### DSpark Path

dspark_store → RMS norm → RoPE → `ds4_gpu_dsv4_fp8_kv_quantize_tensor` → `ds4_gpu_store_raw_kv_batch_tensor`.

### ROCm Variation

ROCm splits nope region into `groups = ceil(n_nope / 64)` independent grids: `dim3(n_tok, groups)`. Each grid processes one 64-wide block. Same kernel body.

## Dequantization

Binary search over the 127-member E4M3 value set with round-ties-to-even:

```c
// dsv4_e4m3fn_dequant_dev — device path
float sign = x < 0.0f ? -1.0f : 1.0f;
float ax = fminf(fabsf(x), 448.0f);
int lo = 0, hi = 126;
while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
    else hi = mid - 1;
}
int best = lo;
if (best < 126) {
    float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
    float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
    if (nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0)) best++;
}
return sign * dsv4_e4m3fn_value_dev(best);
```

CPU and device share the same logic. The binary search index maps through the 127-value set produced by `dsv4_e4m3fn_value`.

## Relationship

- **Depends on**: [kv-cache-lifecycle.md](kv-cache-lifecycle.md) (raw rows exist before quantization), compressor prefill/update (produce compressed rows), RMS norm + RoPE (applied before quantize).
- **Used by**: compressed KV cache (read by attention decode), raw KV store (decode path), DSpark raw cache.
- **Alternatives**: f16 KV cache (2 bytes/elem, bit-exact, no quantize), f32 KV cache (4 bytes/elem, full precision), FP8 without round-trip (DS4 stores f32 through round-trip for attention compatibility).
- **Distinct from**: [indexer-subsystem.md](indexer-subsystem.md) (E2M1 FP4 with Hadamard, different codec and purpose), model weight quantization (E4M3 weight storage in GGUF, separate pipeline).

## Code Pattern

### Function Signature

```c
// ds4_gpu.h — E4M3 round-trip on non-RoPE (nope) portion of each KV row
// x: f32 tensor [n_tok, head_dim], quantized in-place
// n_rot: RoPE dimension count — last n_rot elements stay f32
int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);
```

### Prefill Path — Inside Compressor

```c
// ds4_cuda.cu:ds4_gpu_compressor_prefill_tensor
// Pool → RMS norm → RoPE → quantize (gated by quantize_fp8)
if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot))
    return 0;
```

### Decode DSpark Store — Quantize Then Store Raw

```c
// rocm/ds4_rocm_attention_launch.cuh:ds4_gpu_kv_fp8_store_raw_tensor
// Quantize a single raw KV row, then store to circular raw cache
return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) &&
       ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, head_dim);
```

### CUDA Kernel Launch

```c
// ds4_cuda.cu:ds4_gpu_dsv4_fp8_kv_quantize_tensor — 64 threads (two warps)
fp8_kv_quantize_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
```

[← Back to Index](../README.md)
