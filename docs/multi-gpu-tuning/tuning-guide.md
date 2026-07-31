# Tuning Guide

## Purpose

Decision framework for tuning the multi-GPU pipeline-parallel inference system. Given workload, model, and hardware configuration, which knobs to turn and what improvement to expect.

## Tuning Philosophy

1. **Measure first**: Every tuning decision must be informed by measurement, not intuition.
2. **One knob at a time**: Change one parameter, measure impact, then change next.
3. **Roofline-aware**: Don't optimize past the dominant ceiling.
4. **Cost-benefit**: Only optimize if expected improvement > measurement noise (3% for this system).

## System Parameters

### Fixed (Hardware)

| Parameter | Value | Source |
|---|---|---|
| GPU count | 2 | [measured: nvidia-smi] |
| GPU model | RTX PRO 6000 Blackwell Max-Q | [measured: nvidia-smi] |
| VRAM per GPU | 97 GiB | [measured: nvidia-smi] |
| GPU-GPU interconnect | PCIe Gen 5 x8 (28 GB/s) | [measured: pcie_bw_bench_v2] |
| HBM bandwidth | 1500 GB/s per GPU | [measured: compute_peak_bench] |
| Power limit | 300W default (250-325W range) | [measured: nvidia-smi] |

### Configurable (ds4)

| Knob | Default | Range | Effect |
|---|---|---|---|
| `--gpu-vram` | auto | manual budgets | Layer distribution |
| `--ctx` | 32768 | 128-1048576 | KV cache size |
| `--prefill-chunk` | 4096 | 512-8192 | Prefill granularity |
| `--power` | 100 | 1-100 | GPU duty cycle |
| `--gpu-devices` | auto | device indices | Which GPUs to use |
| `--cuda-tensor-parallel` | off | on/off | Expert sharding |

## Decision Trees

### Tree 1: Decode Throughput Optimization

```
Is decode throughput < HBM roofline (1000 t/s)?
  ├── Yes (almost certainly):
  │   Is the bottleneck kernel launch overhead?
  │   ├── Yes (decode time dominated by >500 launch events):
  │   │   → Fuse kernels (reduce launch count)
  │   │   → Use CUDA graphs (amortize launch cost)
  │   │   → Expected: 20-50% improvement [hypothesis]
  │   │
  │   └── No (compute/memory bound):
  │       → Optimize quantization format (Q4_K → Q8_0 for critical layers)
  │       → Increase batch size if possible
  │       → Expected: 10-30% improvement [hypothesis]
  │
  └── No (near roofline — unlikely on this hardware):
      → Profile with Nsight to find remaining bottleneck
      → Report finding
```

### Tree 2: Pipeline Balance

```
Compare GPU0 vs GPU1 execution time per step.

If GPU0 time > GPU1 time by >10%:
  ├── GPU0 has more layers (24 vs 19 in current layout) [measured: ds4 multi-GPU layout]
  │   → Adjust --gpu-vram budgets to move layer(s) to GPU1
  │   → Move output head to GPU0 if it's on GPU1
  │   → Expected: 5-15% throughput improvement for balanced load
  │
  └── Other imbalance source:
      → Profile per-layer timing to identify slow layer
      → Check for memory contention or power throttling
```

### Tree 3: DSpark Speculative Decode

```
Is DSpark support GGUF loaded?
  ├── Yes:
  │   Is acceptance rate >70%?
  │   ├── Yes: Enable DSpark decode
  │   │   → --dspark (or engine option)
  │   │   → Expected: 1.5-2.5× decode speedup [hypothesis]
  │   │
  │   └── No: Profile draft chain latency
  │       → If draft chain > 3ms, optimize stage blocks
  │       → If confidence threshold too high, lower (default 0.9)
  │       → Expected: variable
  │
  └── No (DSpark not available):
      → Legacy MTP may be available (--mtp)
      → Without speculation: ~68 t/s baseline [measured]
```

### Tree 4: Memory Pressure

```
Is (model weights + KV cache + overhead) < available VRAM?
  ├── Yes (current state):
  │   → Full GPU residency achieved
  │   → No SSD streaming needed
  │   → Optimal configuration
  │
  └── No (future models or larger context):
      → Enable SSD streaming (--ssd-streaming)
      → Set --ssd-streaming-cache-experts to fit hot experts
      → Reduce context (--ctx) to free KV cache space
      → Use --ssd-streaming-cold for cold-start scenario
      → Expected throughput drop: 20-50% [hypothesis]
```

### Tree 5: Power/Thermal

```
Is GPU clock throttling during sustained inference?
  ├── Yes (clock drops below 3.0 GHz after 30-60s):
  │   → Reduce --power to stay below throttle threshold
  │   → Improve VM cooling / airflow
  │   → Expected: 10-20% throughput loss from throttling [hypothesis]
  │
  └── No:
      → Power is sufficient (300W cap per GPU)
      → No action needed
```

## Measured Baseline

| Metric | Value | Condition |
|---|---|---|
| Decode throughput | 68.5 t/s | ctx=1024, 2-GPU, batch=1, F32 logits |
| Prefill throughput | 1758 t/s | ctx-prefill=512 tokens, 2-GPU |
| First token latency | 14.5 ms | ctx=1024 |
| PCIe transfer latency | 1.0 µs [derived: 8KB activation, interpolated from bench] | F16 activation (8KB Flash) |

[measured: pipeline_bench_baseline.csv, pcie_bw_bench_v2]

## Recommended Configuration

For current hardware and workload:

```
--gpu-devices 0,1          # Use both GPUs
--gpu-vram auto             # Auto-detect budgets (90 GB each)
--ctx <desired>             # Based on workload (32768 default)
--prefill-chunk 4096        # Default, adequate
--cuda                      # CUDA backend
--power 100                 # Full power (monitor for throttling)
# Optional:
# --dspark                  # If DSpark support GGUF loaded
```

## Expected Impact Matrix

| Tuning Action | Expected Improvement | Risk | Complexity |
|---|---|---|---|
| Fuse decode kernels | 20-50% on decode | Kernel may exceed register limit | High |
| CUDA graphs for decode | 15-30% on decode | Graph rebuild overhead | Medium |
| Optimize GPU0/GPU1 balance | 5-15% | May cause OOM if budget wrong | Low |
| DSpark speculative decode | 50-150% | Depends on acceptance rate | Medium |
| Lower dspark_confidence_threshold | 10-30% more accepted | May reduce quality | Low |
| Increase batch size | 20-40% per token | Higher latency per batch | Medium |
| Power limit reduction (to 80%) | 0-5% loss | Prevents throttling | Low |
| Q8_0 for attention layers | 10-20% | Increases VRAM usage | Low |

## See Also

- [roofline-analysis.md](roofline-analysis.md) — throughput ceilings
- [pcie-characterization.md](pcie-characterization.md) — PCIe limits
- [speculative-decode-multi-gpu.md](speculative-decode-multi-gpu.md) — DSpark analysis
- [instrumentation-guide.md](instrumentation-guide.md) — how to measure
- [scalability-analysis.md](scalability-analysis.md) — scaling to N GPUs
- [PRD.md §6.3](PRD.md) — decision framework reference
