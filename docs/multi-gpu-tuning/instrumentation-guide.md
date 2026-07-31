# Instrumentation Guide

## Purpose

How to measure each component of the multi-GPU inference pipeline with minimal disruption. Documents the recommended instrumentation approach for each subsystem.

## Design Principles

1. **Lightweight by default**: Built-in stats structs + env-var gated timing. Always available, minimal overhead.
2. **Progressive detail**: Start with coarse counters, drill down with cudaEvent pairs, then full Nsight profiles.
3. **Minimal disruption**: Avoid implicit synchronization from instrumentation. Prefer non-blocking timers.
4. **Source-tag all data**: Every measurement must carry `[measured: <method>]` tag per GROUND-RULES §1.2.

## Measurement Hierarchy

```
Level 0: Built-in stats (env vars, always on) 
  → Level 1: cudaEvent pairs (bounded sync overhead)
    → Level 2: Nsight Systems (full timeline, root required)
      → Level 3: Nsight Compute (per-kernel metrics, high overhead)
```

## Per-Component Measurement

### 1. Cross-Device Transfer (PCIe Bandwidth/Latency)

| Method | Overhead | Precision | Disruption |
|---|---|---|---|
| cudaEvent pair around `xdev` call | ~1µs [hypothesis] | ±0.1µs [hypothesis] | Implicit sync drains pipeline |
| DS4_DIST_DECODE_PROFILE | Stderr write per step [datasheet] | ±10µs [hypothesis] | write() syscall overhead |
| Nsight Systems GPU trace | <5% [datasheet: NVIDIA Nsight docs] | ±0.01µs [datasheet] | Requires root |

**Recommended**: Use `DS4_DIST_DECODE_PROFILE` for routine monitoring. Use custom cudaEvent pairs for detailed characterization (as in pcie_bw_bench.cu).

**Code pattern**:
```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, stream);
cudaMemcpyPeerAsync(dst, dev_dst, src, dev_src, bytes, stream);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

### 2. Compute Kernel Throughput (TFLOPS)

| Method | Overhead | Precision | Notes |
|---|---|---|---|
| cudaEvent pair per kernel | <1µs/kernel [hypothesis] | ±0.1µs [hypothesis] | Adds event records to stream |
| Nsight Compute | 5-20% [datasheet: NVIDIA Nsight Compute] | Cycle-accurate [datasheet] | Full kernel profile |
| Built-in timer in benchmark | None (standalone) | ±1µs [datasheet: clock_gettime resolution] | compute_peak_bench.cu approach |

**Recommended**: Standalone micro-benchmarks (compute_peak_bench.cu) for ceiling measurement. cudaEvent pairs for production kernel profiling. Nsight Compute for optimization targeting.

### 3. Pipeline Bubble (Idle GPU Time)

| Method | Overhead | Precision | Notes |
|---|---|---|---|
| cudaEvent on both devices | ~2µs [hypothesis] | ±0.5µs [hypothesis] | Record events on both streams |
| DS4_DIST_DECODE_PROFILE | Stderr write [datasheet] | ±10µs [hypothesis] | Per-hop timing |
| Nsight Systems timeline | <5% [datasheet: NVIDIA Nsight docs] | ±0.01µs [datasheet] | Visual idle detection |

**Measurement technique**:
```c
// On GPU0
cudaEventRecord(event_gpu0_done, stream0);
// After GPU0 work

// On GPU1
cudaEventRecord(event_gpu1_start, stream1);
// Before GPU1 work

cudaEventSynchronize(event_gpu0_done);
float idle_ms;
cudaEventElapsedTime(&idle_ms, event_gpu0_done, event_gpu1_start);
```

### 4. Kernel Launch Overhead

| Method | Overhead | Precision | Notes |
|---|---|---|---|
| CPU timer around cudaLaunchKernel | None [datasheet] | ±0.5µs [datasheet: clock_gettime] | Just measure host-side launch |
| CUDA driver API trace (cupti) | 1-5% [datasheet: CUDA profiling docs] | ±0.1µs [hypothesis] | Per-call tracing |

**Recommendation**: Use `clock_gettime` before and after kernel launch on host. The GPU execution time is excluded; only the launch submission is measured.

### 5. Synchronization Overhead

| Method | Notes |
|---|---|
| cudaEventSynchronize timing | Direct measure of sync wait |
| cudaStreamSynchronize timing | Coarser, waits for all pending work |
| cudaDeviceSynchronize timing | Most expensive, global barrier |

**Recommendation**: Prefer `cudaEventSynchronize` for fine-grained sync timing. Use `cudaStreamSynchronize` for per-stream measurement.

### 6. Cache Miss Penalty

| Method | Notes |
|---|---|
| Built-in cache stats (hits, misses, evictions) | Metal only, ds4_metal.m counters |
| Per-step timing with/without cache clear | A/B comparison |
| Nsight memory trace | Full memory access pattern |

**Current status**: CUDA backend has no per-step cache hit/miss counters. Only Metal backend tracks these (g_stream_expert_cache_layer_hits/misses/evictions).

### 7. Memory Bandwidth (HBM)

**Standalone**: compute_peak_bench.cu read/copy kernels measure HBM bandwidth directly.

**In pipeline**: Use Nsight Compute's `l1tex__throughput` and `dram__bytes_read` metrics for per-kernel HBM utilization.

### 8. Pipeline Load Balance

**Method**: Compare kernel execution time on each GPU using cudaEvent pairs around each device's work.

```c
cudaSetDevice(0);
cudaEventRecord(gpu0_start, stream0);
// ... GPU0 work
cudaEventRecord(gpu0_end, stream0);

cudaSetDevice(1);
cudaEventRecord(gpu1_start, stream1);
// ... GPU1 work
cudaEventRecord(gpu1_end, stream1);

// Compare gpu0_end - gpu0_start vs gpu1_end - gpu1_start
```

## Fused Kernel Analysis (Analytical — Not Measured)

### Trade-offs

Fused kernels merge multiple sequential operations (e.g., attention QK dot + softmax + weighted sum) into a single kernel, reducing launch overhead and intermediate memory traffic. Key trade-offs per PRD §4.4:

| Factor | Benefit | Cost |
|---|---|---|
| Launch overhead | Eliminates N-1 launches per fusion | Register pressure increases with operations per thread |
| Intermediate memory | Keeps results in registers/SMEM vs global memory | Larger tile requires more shared memory, reducing occupancy |
| Memory traffic | Fewer global memory reads/writes | May increase L1/L2 pressure from larger working set |

[derived: PRD §4.4, CUDA optimization principles]

### Detection Method

To measure whether fusion helps on this system:
1. Compare kernel time pre-fusion vs post-fusion using cudaEvent pairs
2. Check occupancy changes via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
3. If occupancy drops >20%, fusion may hurt despite launch reduction [hypothesis]

### Cross-Device Fusion

Fusion across device boundaries (e.g., fused HC pre on GPU0 + attention on GPU1) is not possible — each GPU executes its own kernel stream. The PCIe transfer at layer boundaries is a hard synchronization point [by architecture: multi-gpu-pipeline.md].

### Recommended Approach

Start with built-in fused kernels (attention decode, HC weighted sum, output head). Profile kernel launch count before attempting custom fusions — if launch overhead is <3% of step time, fusion has low return. See roofline-analysis.md Gap 1 for current launch overhead estimate.

## Environment Variable Reference

| Variable | What It Enables | Overhead |
|---|---|---|
| `DS4_DIST_DECODE_PROFILE` | Per-hop telemetry (distributed) | Low (stderr write) |
| `DS4_DSPARK_SPEC_LOG` | Per-round spec decisions | Low (stderr write) |
| `DS4_DSPARK_PROBE` | Draft pipeline profiling | Medium (full draft compute) |
| `DS4_METAL_DECODE_STAGE_PROFILE` | Per-stage timing (Metal) | Low |
| `DS4_CUDA_EXACT_SCORE_SPLIT_DECODE` | Forces specific attention path | Changes behavior |

## Analysis Template

For each measurement, record:

```
### Measurement: [name]
**Instrument**: [method used]
**Result**: [value ± spread, n_trials]
**Steady-state**: [warm/cold]
**Confounds**: [known issues]
**Tag**: [measured: <ref>]
```

## See Also

- [PRD.md §4.9](PRD.md) — detailed instrumentation discussion
- [GROUND-RULES.md](GROUND-RULES.md) — measurement methodology
- [experiments/pcie-bw/](experiments/pcie-bw/) — PCIe instrumentation example
- [experiments/compute-peak/](experiments/compute-peak/) — compute measurement example
