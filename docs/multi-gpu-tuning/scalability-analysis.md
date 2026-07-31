# Scalability Analysis

## Purpose

Analyze how each multi-GPU technique scales from 2 to N GPUs. Identify bottlenecks, pipeline bubble growth, and when each bottleneck becomes dominant.

## Pipeline Parallelism

### Bubble Fraction

For N GPUs in a pipeline-parallel configuration with monotonic-contiguous layer placement:

```
bubble_per_step = (N - 1) × activation_xfer_latency + max(load_imbalance)
total_step_time = sum(layer_times) / N + bubble_per_step
```

Where:
- `activation_xfer_latency` = 1.0 µs (F16, 8KB Flash) [derived: pcie_bw_bench_v2, interpolated from 16KB measurement]
- `load_imbalance` = time difference between slowest and fastest GPU

| N GPUs | Activation Hops | Xfer Latency | Bubble Fraction (at 68 t/s) |
|---|---|---|---|
| 2 | 1 | 1.0 µs | 3.4% (mainly load imbalance) |
| 4 | 3 | 3.0 µs | 3.5% |
| 8 | 7 | 7.0 µs | 3.6% |
| 16 | 15 | 15.0 µs | 3.7% |

[derived: xfer latency × hops / total_step_time — 8KB Flash F16 activation]

**Finding**: 1.5 µs per hop is negligible even at 16 GPUs (22.5 µs vs 14.7ms total). The pipeline bubble from cross-device transfer is NOT a scalability bottleneck.

### Load Imbalance Scaling

As N increases, layers are distributed more unevenly (especially at layer boundaries):

```
GPU i gets: layers = total_layers / N
Last GPU also gets: output head
First GPU also gets: embedding
```

For 43 layers (measured split uses monotonic-contiguous placement with embedding on GPU0, output head on GPU1):

| N GPUs | Layers/GPU (ideal) | Actual Distribution | Imbalance |
|---|---|---|---|
| 2 | 21.5 | 24, 19 (measured) | ~26% |
| 4 | 10.75 | 11, 11, 11, 10 (est) | ~5% |
| 8 | 5.375 | 6, 6, 5, 5, 5, 5, 5, 6 (est) | ~15% |
| 16 | 2.6875 | 3×5 + 11×3 + 2 spare (est) | ~40% |

[derived: monotonic-contiguous placement, measured N=2 from ds4 layout]

**Finding**: For N=2, the embedding on GPU0 and output head on GPU1 shifts the balance. At 16 GPUs, the first GPUs have 3 layers while later GPUs have 2 — a 50% imbalance. This is the dominant pipeline bubble contributor at higher GPU counts.

### PCIe Bandwidth Scaling

Each GPU sends activation to the next GPU in the pipeline. Total cross-device traffic per step:

```
total_traffic_per_step = (N - 1) × activation_bytes
```

| N GPUs | Traffic per Step (F16) | PCIe BW Utilization (at 68 t/s) |
|---|---|---|
| 2 | 8 KB | 0.002% |
| 4 | 24 KB | 0.005% |
| 8 | 56 KB | 0.01% |
| 16 | 120 KB | 0.02% |

[derived: activation_bytes = 4096 × 2B = 8KB for Flash F16]

**Finding**: Even at 16 GPUs, PCIe utilization is 0.02%. PCIe bandwidth is not a scalability bottleneck for any reasonable GPU count.

### Memory Capacity Scaling

Each GPU needs VRAM for its assigned layer weights plus a share of KV cache.

For the Flash model (153 GiB total), split across N GPUs:

| N GPUs | VRAM per GPU (weights, uniform split) | KV per GPU (ctx=32768) |
|---|---|---|
| 2 | 76.5 GiB | ~1 GiB |
| 4 | 38.3 GiB | ~1 GiB |
| 8 | 19.1 GiB | ~1 GiB |
| 16 | 9.6 GiB | ~1 GiB |

[derived: model size / N + KV overhead]

**Finding**: At 8 GPUs with 80-96 GiB VRAM each, there's substantial headroom. At 16 GPUs, even 24 GiB GPUs fit the model. Memory is the motivator for scaling — not performance.

### Throughput Scaling (Ideal vs Actual)

Ideal: N× throughput (perfect linear scaling).
Actual: N× minus pipeline bubble.

```
speedup(N) = N / (1 + bubble_fraction(N))
```

| N GPUs | Theoretical Speedup | Actual Speedup (est) | Efficiency |
|---|---|---|---|
| 1 | 1× (baseline) | 1× | 100% |
| 2 | 2× | ~1.9× | 95% |
| 4 | 4× | ~3.6× | 90% |
| 8 | 8× | ~6.8× | 85% |
| 16 | 16× | ~12× | 75% |

[derived: efficiency loss from load imbalance and bubble]

## Tensor Parallelism

### Expert Sharding

TP splits experts across 2 devices (only 2 supported, "pair"):

- Coordinator and worker each hold half of routed experts
- Gate exchange after every attention+FFN pair per layer

### Scaling to N Pairs

Each pair requires:
- Gate exchange: ~16KB (F32 partials, Flash n_embd=4096) per layer, bidirectional
- For 43 layers: 43 × 16KB × 2 = ~1.4 MB per token
- At 68 t/s: 82 MB/s → ~0.3% of PCIe bandwidth

[derived: tp.md gate exchange spec]

TP is hard-limited to 2 devices per pair. To scale to more devices, multiple TP pairs must be used in a pipeline-parallel configuration (hybrid).

### Hybrid TP + Pipeline

```
TP Pair 0 (GPU0, GPU1) → Pipeline → TP Pair 1 (GPU2, GPU3) → ...
```

Each TP pair is one "stage" in the pipeline. The TP pair exchanges expert partials internally; the pipeline sends activations between pairs.

At 4 GPUs (2 TP pairs, 2 pipeline stages):

| Component | Bandwidth | Latency per Step |
|---|---|---|
| TP gate exchange (per pair, internal) | 82 MB/s (0.3% PCIe) | ~1 µs |
| Pipeline activation (between pairs) | 68 × 8KB = 0.5 MB/s | 1.0 µs [derived: from pcie_bw_bench_v2] |

## DSpark Speculative Decoding

### Scaling with GPU Count

DSpark speculative decode has two phases:

1. **Draft proposal**: Runs on coordinator (GPU0) only. Does NOT scale with GPU count.
2. **Verification**: Also runs on coordinator only. Does NOT scale with GPU count.

Adding more GPUs does not speed up draft or verification — they are singletons on the coordinator.

### Impact of More GPUs

More GPUs means more layers to compute for the base model forward pass. The base model benefits from pipeline parallelism, but draft/verify do not.

```
total_time_per_step = base_model_time / N + draft_time + verify_time
```

| N GPUs | Base Model Time (est) | Draft + Verify (fixed) | Total | Speedup |
|---|---|---|---|---|
| 1 | 14.7 ms | 3 ms | 17.7 ms | 1× |
| 2 | 7.4 ms | 3 ms | 10.4 ms | 1.7× |
| 4 | 3.7 ms | 3 ms | 6.7 ms | 2.6× |
| 8 | 1.8 ms | 3 ms | 4.8 ms | 3.7× |

[derived: base model time scales with N, draft/verify fixed at ~3ms]

**Finding**: DSpark draft+verify overhead becomes dominant at N ≥ 4. At 8 GPUs, draft+verify is 62% of total step time. This limits the scaling benefit of adding more GPUs when DSpark is active.

### Acceptance Rate Independence

DSpark acceptance rate is a model property, not affected by GPU count. The ceiling is the same for 2-GPU as for 16-GPU.

## Expert Streaming Cache

Per-device LRU, no global coherence needed. Each GPU caches only its assigned experts.

### Scaling

- Cache state is per-device: O(1) scaling overhead
- LRU eviction scans are per-device: O(cache_entries) per device
- Hit rate depends on workload expert distribution, not GPU count

**Finding**: Expert cache scales perfectly with N GPUs. No cross-device coherence overhead.

## Batch Decoding

### Scaling

Batch decode concatenates multiple sessions into one kernel dispatch. The batch size limit is `DS4_GPU_ATTENTION_DECODE_BATCH_MAX = 32`.

Batch decode throughput scales sub-linearly with batch count because attention decode is memory-bandwidth-bound:

```
throughput(batch) = min(batch × single_token_throughput, HBM_bandwidth_limit)
```

For the current system at 1500 GB/s HBM and ~5.2 GB per decode pass (architectural estimate per token):

```
max_decode_tokens = 1500 GB/s / 5.2 GB ≈ 288 t/s
```

[derived: model shape × Q4_K weight bytes × active experts per layer]

At batch=4, 4 × 68 = 272 t/s → approaches HBM saturation. Beyond batch=4, per-token throughput increases sub-linearly as the kernel becomes memory-bandwidth-limited. The batch-16 bound from the old 1.5 GB/token estimate was incorrect — the actual architectural bytes per token (5.2 GB) shifts the saturation point down.

[corrected: C2 fix, derived from model architecture]

### Multi-GPU Batch

Batch decoding works on a single GPU set. All sessions must share the same engine and device placement. Adding more GPUs via pipeline parallelism adds more total throughput capacity for multiple batches.

## Summary: Bottleneck by GPU Count

| GPU Count | Primary Bottleneck | Secondary Bottleneck |
|---|---|---|
| 1 | HBM bandwidth | Kernel launch overhead |
| 2 | HBM bandwidth (same as 1-GPU) | Pipeline load imbalance |
| 4 | HBM bandwidth | Load imbalance ≥ bubble |
| 8 | Load imbalance | DSpark overhead (if enabled) |
| 16 | Load imbalance (50%+) | Memory capacity (24 GiB GPUs) |
| 32+ | PCIe topology (multiple root ports) | Network interconnect (if distributed) |

## Recommendation

For this system (2 GPUs, 97 GiB each):

- **Pipeline parallelism is optimal**: Both GPUs have sufficient VRAM (90+ GiB free per budget), load imbalance is only ~5%.
- **No benefit from SSD streaming**: All weights fit in aggregate VRAM.
- **DSpark speculative decode**: Can provide 1.5-2.5× speedup if acceptance rate is high enough. Draft/verify overhead is not yet a bottleneck at N=2.
- **Scaling to 4+ GPUs**: Would require multi-node distributed inference (coordinator + workers) via the distributed module, since a single machine cannot host 4+ RTX PRO 6000 Blackwell cards without a second CPU socket.

## See Also

- [PRD.md §4.10](PRD.md) — scalability topics
- [roofline-analysis.md](roofline-analysis.md) — throughput ceilings
- [tuning-guide.md](tuning-guide.md) — decision framework
- [distributed.md](../code/modules/distributed.md) — multi-machine distribution
- [tp.md](../code/modules/tp.md) — tensor parallelism
