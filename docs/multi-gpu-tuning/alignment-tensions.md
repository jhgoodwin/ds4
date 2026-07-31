# Alignment Tensions

## Purpose

Document CUDA alignment requirements, quantization format alignment constraints, and the tension between compact representation and aligned access. Measure cost of misalignment where applicable.

## CUDA Alignment Requirements

| Level | Alignment | Rationale |
|---|---|---|
| Cache line (L1/L2) | 128 bytes | Cache line size on Blackwell |
| Warp | 32 elements (128B for F32) | Warp-level memory access |
| Vectorized (vec4) | 16 bytes (4×F32) | Native vector load/store |
| Vectorized (vec2) | 8 bytes (2×F32) | Half-vector load/store |
| cudaMemcpyPeerAsync | No measurable penalty | [measured: pcie_bw_bench — alignment test] |

### Measured: Cross-Device Transfer Alignment

Tested cudaMemcpyPeerAsync GPU0→GPU1 at 1MB with offsets from 0 to 1024B:

| Offset | Bandwidth (GB/s) | Latency (µs) |
|---|---|---|
| 0 (aligned) | 27.63 | 38.0 |
| 1B | 27.44 | 38.2 |
| 32B | 27.52 | 38.1 |
| 128B | 27.56 | 38.1 |
| 1024B | 27.62 | 38.0 |

[measured: pcie_bw_bench alignment test, 980 trials per offset]

**Finding**: CUDA peer DMA shows no measurable alignment sensitivity. The driver handles misalignment internally, likely via staging copies through aligned internal buffers. No fallback to cudaMemcpy (device-side) observed.

## Quantization Format Alignment

### Q8_0 (block size 32)

| Property | Value |
|---|---|
| Block size | 32 elements |
| Block format | 1×F16 scale + 32×Q8 values |
| Alignment requirement | 32-element boundaries for efficient dequant |
| Padding cost | Wasted dimensions at block boundaries |

### Q4_K (block size 32)

| Property | Value |
|---|---|
| Block size | 32 elements (2 half-blocks × 16 elements) |
| Block format | 2×F16 scales + 2×F16 mins + 32×Q4 values |
| Alignment requirement | 32-element boundaries |
| Padding cost | Higher due to per-half-block metadata |

### Alignment Tension

The tension: **compact representation wants minimum bytes per weight** (Q4_K is 4.5 bits/weight including block overhead), but **aligned access wants dimensions to be multiples of block size** (32 for Q8_0, 32 for Q4_K).

| Dimension | Model Value | Block-Aligned? | Padding Waste |
|---|---|---|---|
| n_embd (Flash) | 4096 | 4096/32 = 128 ✓ | 0% |
| n_embd (Pro) | 7168 | 7168/32 = 224 ✓ | 0% |
| n_vocab | 129280 | 129280/32 = 4040 ✓ | 0% |
| n_expert_used | 6 | Not applicable | 0% (routed) |

[derived: model shapes are naturally block-aligned for all quantization formats used]

**Finding**: No alignment tension for this model at current quantization. All dimensions are multiples of Q8_0/Q4_K block size (32). Padding waste is 0%.

### Hidden State Alignment

Hidden state size: 7168 (Pro) or 4096 (Flash) F16 elements = 14KB or 8KB. Both are multiples of 128B cache line:
- 7168 × 2B = 14336B / 128 = 112 cache lines ✓
- 4096 × 2B = 8192B / 128 = 64 cache lines ✓

### KV Cache Alignment

KV cache rows are:
- Raw KV: 1 head × 512 dim × 2 (K+V) × 2 bytes (F16) = 2048B per token per layer
- 2048 / 128 = 16 cache lines ✓

- Compressed KV: depends on compression scheme, but typically 512 dim × 2 bytes = 1024B per row
- 1024 / 128 = 8 cache lines ✓

## Shared Memory Bank Conflicts

For attention kernels using shared memory for tile storage:

| Access Pattern | Bank Conflict Risk |
|---|---|
| QK dot product tile (head_dim=512) | 512/32 = 16 banks → conflict-free with proper stride |
| Softmax reduction | Column-wise access → no conflict |
| Weighted sum accumulation | Row-wise access → no conflict with warp-aligned rows |

[hypothesis: Blackwell shared memory has 32 banks × 4 bytes]

Given head_dim=512 and 32 shared memory banks, the access stride for QK tiles can be arranged to be conflict-free [hypothesis: Blackwell shared memory layout. No bank conflict experiments run — requires NVCC PTX inspection or Nsight Compute shared memory metrics].

## Non-Coalesced Global Memory Access

### HC Transform Kernels

The HC pre/post/output kernels access 4 parallel streams of n_embd dimensions:

```
for each stream s in [0..3]:
  for each dimension d in [0..n_embd):
    out[s][d] = f(in[s][d])
```

If stored as `hc_state[4][n_embd]`, access pattern is:
- Sequential within stream: coalesced ✓
- Cross-stream: separate vectors, each coalesced ✓

If stored as interleaved `hc_state[n_embd][4]`:
- Access by stream requires stride-4 access
- Non-coalesced: only 1/4 of cache line used
- Bandwidth waste: ~75%

[hypothesis: hc_state layout affects effective bandwidth]

**Recommendation**: Verify that HC state is stored as `streams[4][n_embd]` (not interleaved) to maintain coalesced access. This is the expected layout from hc-transforms.md.

## PCIe Transfer Alignment

From the PCIe characterization:

| Alignment Condition | Effect |
|---|---|
| 128B-aligned buffers | No observable difference |
| Non-aligned buffers | No observable difference |
| cudaMemcpyPeerAsync fallback | Not triggered by misalignment |

The CUDA driver handles misaligned cudaMemcpyPeerAsync by staging through aligned internal buffers. Cost of this staging could not be isolated from measurement noise (CV ~1-3%).

## Tension Summary

| Tension Pair | Severity | Current Status |
|---|---|---|
| Q4_K block padding vs compactness | None | Dimensions are block-aligned |
| Q8_0 block size vs dimension granularity | None | 32 is a common divisor |
| Shared memory banks vs head_dim | Low | Can be arranged conflict-free |
| HC state layout vs coalesced access | Low | Likely already correct layout |
| PCIe alignment vs fallback | None | No degradation measured |
| Activation size vs cache line | None | Always cache-line-aligned |

**No alignment tensions are active for the current model and quantization on this system.** The model dimensions, quantization block sizes, and access patterns are all naturally aligned to their respective requirements.

## If Alignment Tensions Arise (Future Models)

For future models where dimensions may not be block-aligned:

1. **Padding cost**: For a dimension that's N elements with block size B, padding to next multiple of B wastes `(B - N%B) × bytes_per_element` per row. At scale (vocab = 130K, embedding = 7K, 43 layers), even 1% padding adds ~1.5 GB of wasted data.

2. **Mitigation**: Use variable-length quant blocks at boundaries, or pad during model conversion and accept the storage overhead.

3. **Detection**: Compare effective bandwidth (bytes transferred) vs theoretical bandwidth (bytes of actual data). If effective < 90% of theoretical, padding waste is likely.

## See Also

- [experiments/alignment-cost/](experiments/alignment-cost/) — alignment micro-benchmarks
- [pcie-characterization.md](pcie-characterization.md) — PCIe alignment measurements
- [roofline-analysis.md](roofline-analysis.md) — effective bandwidth analysis
- [cuda.md](../code/backends/cuda.md) — CUDA kernel categories
