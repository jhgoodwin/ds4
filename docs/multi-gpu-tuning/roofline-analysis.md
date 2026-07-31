# Roofline Analysis

## Purpose

Compute mathematical throughput ceilings for each operation in the 2-GPU pipeline-parallel inference system, then compare measured throughput against those ceilings. Identifies which bottleneck dominates and quantifies each gap.

## System Ceilings

### Hardware Parameters

| Parameter | GPU 0 | GPU 1 | Source |
|---|---|---|---|
| SMs | 188 | 188 | [measured: cudaDeviceProp] |
| SM clock (max) | 3090 MHz | 3090 MHz | [datasheet: nvidia-smi] |
| CUDA cores/SM | 128 (Blackwell) | 128 | [datasheet: NVIDIA Blackwell arch] |
| Peak FMA TFLOPS | 188×128×3.09×2 = 148.7 TFLOPS | same | [derived: SMs × cores × clock × 2] |
| HBM BW (read) | 1502 GB/s | 1508 GB/s | [measured: compute_peak_bench] |
| HBM BW (copy) | 1290 GB/s | 1295 GB/s | [measured: compute_peak_bench] |
| PCIe unidirectional | 28.4 GB/s | 28.2 GB/s | [measured: pcie_bw_bench_v2] |
| PCIe bidirectional | 51.8 GB/s aggregate | | [measured: pcie_bw_bench_v2] |
| VRAM | 97.2 GiB | 97.2 GiB | [measured: nvidia-smi] |

### Theoretical Peak Computations

**Compute ridge point** (transition from compute-bound to memory-bound):

```
ridge_point = peak_compute / peak_memory = 148.7 TFLOPS / 1500 GB/s = 99 FLOPs/byte
```

[derived: roofline model formula]

This means: any kernel with arithmetic intensity < 99 FLOPs/byte is memory-bound on this GPU. Most transformer kernels (matmul, attention) have intensity < 10 FLOPs/byte — they are firmly in the memory-bound regime.

### Composite Pipeline Ceilings

For 2-GPU pipeline-parallel decode (DeepSeek V4 Flash, F16 hidden state):

**Per-token HBM read** derived from model architecture (Q4_0 weights, 6 active + 1 shared expert per layer):
- Per expert FFN (gate+up+down): 3 × 4096 × 2048 = 25.2M params × 4.5 bits/param = 14.2 MB [derived: model shape × Q4_0 block format]
- Active FFN per layer: 7 experts × 14.2 MB = 99 MB [derived]
- Attention (MLA absorbed, estimated): ~15 MB [derived]
- Per layer total: ~114 MB [derived]
- 43 layers + output head: 43 × 114 MB + ~300 MB = ~5.2 GB per token [derived: model shape × weight format × active expert count]

| Resource | Per-Token Requirement | Peak Capacity | Utilization at 68 t/s |
|---|---|---|---|
| HBM read (GPU0, 24 layers) | ~2.7 GB | 1500 GB/s | 68 × 2.7 / 1500 = 12.2% |
| HBM read (GPU1, 19 layers + head) | ~2.2 GB | 1500 GB/s | 68 × 2.2 / 1500 = 10.0% |
| PCIe xfer | 8 KB (activation F16, 1 hop) | 28 GB/s | 68 × 8KB / 28GB/s = 0.002% |
| FMA compute | ~200 GFLOPs (43 layers, FFN+attn) | 148.7 TFLOPS | <1% |

[derived: model shape × throughput / peak]

## Measured Throughput

### Pipeline Decode Baseline

| Metric | Value | Condition |
|---|---|---|
| Decode steady-state | 68.5 t/s | ctx=1024, 2-GPU, batch=1, gen=32 tokens |
| Decode overall | 68.2 t/s | ctx=1024, 2-GPU, batch=1, gen=32 tokens |
| First token latency | 14.5 ms | ctx=1024, 2-GPU |
| Prefill | 1758 t/s | ctx-prefill 512 tokens, 2-GPU |

[measured: ds4-bench pipeline_bench_baseline.csv, single run per frontier]

### Compute Operations

| Operation | Achieved TFLOPS | Arithmetic Intensity (FLOPs/byte) | Bottleneck |
|---|---|---|---|
| FMA (memory-bound test) | 0.40 TFLOPS | 0.26 | HBM bandwidth |
| HBM read-only | — | ~0.25 | HBM bandwidth |
| Decode attention (estimated) | ~1-5 TFLOPS | ~3-10 | Memory bandwidth |
| MoE FFN (estimated) | ~0.5-2 TFLOPS | ~1-3 | Memory bandwidth |

[measured: compute_peak_bench for FMA/HBM; decode estimates from model shape]

### Pipeline Component Breakdown

For each decode step at 68 t/s (14.7ms per token), estimated breakdown:

| Component | Time (µs) | % of Step | Bound By |
|---|---|---|---|
| 24 layers on GPU0 | ~7000 | 48% | HBM bandwidth |
| Activation xfer GPU0→GPU1 | ~1.0 | 0.01% | PCIe latency |
| 19 layers on GPU1 | ~5500 | 37% | HBM bandwidth |
| Output head | ~500 | 3.4% | HBM bandwidth |
| Synchronization/barriers | ~1200 | 8% | CUDA driver + event overhead |
| Kernel launch overhead | ~500 | 3.4% | Driver + graph overhead |
| Pipeline bubble (idle while GPU1 waits for GPU0) | ~1000 | 6.8% | Load imbalance (24 vs 19 layers) |
| **Total** | **~14700** | **100%** | |

[derived: component estimates scaled from measured total decode time]

## Gap Analysis

### Gap 1: Decode Throughput vs HBM Roofline

**Observation**: 68 t/s vs theoretical HBM-limited maximum.

**Theoretical maximum (single-GPU)**: if the entire model fit on one GPU and were perfectly HBM-bound:
```
max_tokens = HBM_bw / bytes_per_token = 1500 GB/s / 5.2 GB ≈ 288 t/s
```

**Per-GPU pipeline-parallel ceiling**: each GPU loads only its assigned layers from HBM:
```
GPU0 ceiling = 1500 GB/s / 2.7 GB ≈ 555 t/s
GPU1 ceiling = 1500 GB/s / 2.2 GB ≈ 681 t/s
```
Bottleneck is GPU0 at 555 t/s.

[derived: HBM roofline formula using per-device bytes per token from Composite Pipeline Ceilings table]

Measured: 68 t/s, which is **23.6% of the HBM roofline** (accounting for kernel launch, sync, and quantization overhead).

**Attribution**:
1. **Kernel launch overhead**: ~500µs per step = ~3.4% of decode time [hypothesis: 43 layers × multiple kernels each]
2. **Synchronization barriers**: cross-device sync, cudaDeviceSynchronize [hypothesis: implicit in pipeline]
3. **Quantization dequant overhead**: Q4_0 dequant adds ~1.43× to effective memory traffic [measured: O1.2-q4k-dequant-overhead, research-log.md] — **not** the dominant gap
4. **Pipeline load imbalance**: GPU0 24 layers, GPU1 19 layers + output head [measured: ds4 multi-GPU layout]
5. **Sequential layer execution**: no overlap between layers within a device [by architecture]

### Gap 2: FMA Throughput vs Theoretical Peak

**Observation**: 0.40 TFLOPS vs 148.7 TFLOPS theoretical (0.27%).

**Attribution**: The test kernel reads all operands from global memory. At HBM bandwidth of 1500 GB/s and 4 bytes per read (float), the maximum FMA throughput is:
```
max_FMA_HBM_limited = 1500 GB/s / 4B per operand / 2 operands = 187.5 GFLOPs ≈ 0.19 TFLOPS
```

But measured is 0.40 TFLOPS, which is ~2× higher. This indicates the kernel achieves ~2× the naive bound because of some cache reuse. It's still firmly memory-bound.

**True compute peak requires register-resident operands** — not measured here but known to be >100 TFLOPS for well-optimized kernels [datasheet: NVIDIA Blackwell].

### Gap 3: PCIe Bandwidth Utilization

**Observation**: At 68 t/s, PCIe moves ~0.5 MB/s (8KB × 68), using 0.002% of available bandwidth.

**Attribution**: No gap — PCIe is massively over-provisioned for activation transfers. This is by design: PCIe Gen 5 x8 can support >1000× the current throughput.

### Gap 4: Pipeline Bubble

**Observation**: Pipeline bubble from cross-device handoff estimated at ~1.5µs per hop, plus ~500µs from load imbalance.

**Bubble fraction**: 500µs / 14700µs = **3.4%** of decode time.

[derived: load imbalance from 24 vs 19 layers]

This is the only pipeline-specific overhead. All other gaps (kernel launch, synchronization) would exist in single-GPU mode too.

## Roofline Summary

```
Throughput (tokens/sec)
    ^
    |  Overhead ceiling: ~450 t/s (launch + sync + bubble bound)
    |  |
    |  |  HBM ceiling: ~288 t/s (architectural bytes/token)
    |  |  |
    |  |  |  Composite ceiling: ~288 t/s
    |  |  |  |
    |  |  |  |  Measured: 68 t/s
    |  |  |  |  |
    |  |  |  |  |--> Quantization dequant overhead
    |  |  |  |  |--> Memory system inefficiency
    |  |  |  |  |--> Kernel launch overhead: ~500µs
    |  |  |  |  |--> Synchronization: ~1200µs
    |  |  |  |  |--> Pipeline bubble: ~500µs
    +--+--+--+--+-------------------> Arithmetic intensity
```

## Gap 5: Overhead Ceiling vs Compute Roofline

Two distinct ceilings govern throughput: the **compute roofline** (HBM bandwidth) and the **overhead ceiling** (kernel launch + sync + pipeline bubble).

### Overhead Ceiling

Kernel launch overhead (~500µs), synchronization (~1200µs), and pipeline bubble (~500µs) sum to ~2200µs of non-compute time per decode step. This sets an **overhead ceiling**:

```
max_tokens_overhead_limited = 1 / 2200µs ≈ 450 t/s
```

[derived: sum of non-compute overheads]

### Compute Roofline

The HBM bandwidth limits compute throughput to:

```
max_tokens_compute_limited = 1500 GB/s / 5.2 GB ≈ 288 t/s
```

[derived: HBM roofline formula]

### Composite Ceiling

The system is simultaneously limited by both. The lower of the two ceilings governs:

```
effective_ceiling = min(288 t/s, 450 t/s) = 288 t/s
```

[derived: composite roofline]

Measured throughput: 68 t/s, which is **23.6% of the HBM roofline** and **15.1% of the overhead ceiling**. The residual gap after accounting for both ceilings is:

| Factor | Time (µs) | % of Step |
|---|---|---|
| Kernel launch overhead | ~500 | 3.4% |
| Synchronization | ~1200 | 8.2% |
| Pipeline bubble (load imbalance) | ~500 | 3.4% |
| Quantization dequant overhead | ~1300 | 8.8% |
| F16 BW inefficiency (vs F32 roofline) | ~1200 | 8.2% |
| Memory system inefficiency | ~2000-4000 | 14-27% |
| Graph/workspace overhead | ~500-1000 | 3-7% |
| Attention kernel inefficiency | ~2000-4000 | 14-27% |
| **Compute (ideal HBM-bound)** | **~3470** | **24%** |
| **Total** | **~14700** | **100%** |

**CORRECTION (2026-07-28)**: Dequant overhead originally estimated at 3000-5000 µs, but O1.2 [measured: research-log.md] measured dequant overhead factor at 1.43× (639 GB/s Q4_K vs 912 GB/s F16). Revised estimate: ~1300 µs derived from (2.7 GB / 639 GB/s − 2.7 GB / 912 GB/s). Updated table adds separate line for F16 BW inefficiency (F16 achieves 912 GB/s vs 1500 GB/s roofline, costing ~1200 µs).

[derived: component estimates scaled from measured total decode time]

### Residual Gap

After accounting for HBM roofline (288 t/s theoretical), kernel launch + sync (450 t/s ceiling), the measured 68 t/s is ~3-4× below the composite ceiling.

**Dequant overhead is NOT the dominant gap** — O1.2 [measured: research-log.md] measured Q4_K dequant overhead factor at 1.43× (639 GB/s vs 912 GB/s F16), contributing only ~1300 µs (~8.8%) of the 14.7 ms step. The residual gap is dominated by:

1. **Memory system inefficiency**: TLB misses, non-coalesced access [hypothesis: unknown, to be measured]
2. **Attention kernel inefficiency**: decode attention may not fully utilize HBM bandwidth [hypothesis: to be measured via O1.4 Nsight Compute]
3. **cuBLAS workspace overhead**: 64 MiB reserve plus per-call overhead [hypothesis: small]
4. **Graph construction overhead**: per-step graph rebuild cost [hypothesis: unknown, not measured]

This residual gap requires Nsight Compute profiling (O1.4) to attribute. The roofline-analysis assumption from Phase 1 — that dequant dominates — is FALSIFIED by O1.2 measurement.

## See Also

- [pcie-characterization.md](pcie-characterization.md) — PCIe bandwidth ceiling
- [instrumentation-guide.md](instrumentation-guide.md) — how to measure each gap
- [tuning-guide.md](tuning-guide.md) — decision framework
- [experiments/compute-peak/](experiments/compute-peak/) — micro-benchmark source
- [data/compute_peak_results.txt](data/compute_peak_results.txt) — raw measurements
