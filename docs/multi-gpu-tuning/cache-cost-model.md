# Cache Cost Model

## Purpose

Quantify the cost of cache miss and maintenance per level of the caching hierarchy: GPU expert streaming cache, KV cache, OS page cache. Provides latency budgets and optimization targets for each level.

## Hierarchy Overview

```
Level 1: GPU VRAM (expert cache, KV cache)
Level 2: OS page cache (kernel-managed, implicit)
Level 3: NVMe SSD (model file mmap)
```

## GPU Expert Streaming Cache

### Architecture (CUDA Backend)

The CUDA backend uses a **single-use compact buffer**: expert weights are copied from model mmap to GPU on-demand per decode step, then released. No LRU, no reuse between steps, no eviction.

[source: gpu-expert-streaming-cache.md — CUDA section]

### Cache Hit/Miss Costs

For the DeepSeek V4 Flash model on this system:

| Parameter | Value | Source |
|---|---|---|
| Expert weight size (Q4_K) | ~750 MB per expert | [derived: 153 GiB / 256 experts × routed fraction] |
| Selected experts per token | 6 (activated) | [measured: model metadata] |
| Expert weight bytes to transfer | ~4.5 GB per token (6 experts) | [derived: 750 MB × 6] |
| PCIe D2H bandwidth (model→GPU) | 28 GB/s | [measured: pcie_bw_bench_v2] |
| NVMe sequential read bandwidth | ~7 GB/s (Gen 5 x4) | [hypothesis: Gen 5 NVMe spec] |
| SSD→GPU transfer time (6 experts) | ~160 ms | [derived: 4.5 GB / 28 GB/s] |
| Decode step time (baseline) | 14.7 ms | [measured: pipeline bench] |

**Critical finding**: Loading 6 experts from SSD would take 160ms, which is **10× the decode step time**. Without caching, SSD streaming is infeasible for decode.

### Single-Use Compact Buffer Behavior (CUDA)

In the actual CUDA backend, the "cache" is a compact buffer that:
1. Reads expert weights from model mmap (which is backed by OS page cache)
2. If pages are in OS cache, read is fast (copy from RAM)
3. If pages are not in OS cache, read is slow (NVMe I/O)

| Scenario | Source | Transfer to GPU | Total Time |
|---|---|---|---|
| OS cache hit | CPU RAM → GPU (28 GB/s) | ~160 ms | ~160 ms |
| OS cache miss | NVMe → CPU RAM (7 GB/s) + CPU RAM → GPU (28 GB/s) | ~640 ms + ~160 ms | ~800 ms |

[derived: 4.5 GB of expert weights]

**Both scenarios exceed decode step time by 10-50×.** This means full-expert-weight streaming per decode step is not viable. The system MUST have expert weights pre-loaded in GPU VRAM or entirely in CPU RAM for decode to proceed at 68 t/s.

### Metal LRU Cache (for reference)

The Metal backend uses a real LRU with hotness tracking. For a cache of N slots:

| Cache Size (experts) | Expected Hit Rate | Miss Penalty | Average Cost Per Step |
|---|---|---|---|
| 16 | ~50% (cold) | 160 ms on miss | ~80 ms |
| 64 | ~80% | 160 ms | ~32 ms |
| 256 (all) | 100% | 0 | 0 |

[hypothesis: Zipf-distributed expert access pattern]

At 256 experts (full residency), cache cost is zero — this is the current configuration since both GPUs have sufficient VRAM (97 GiB each, model weights 153 GiB split). The expert cache is not exercised because all weights fit in aggregate GPU VRAM.

### Expert Cache Miss Latency Budget

For cache miss to not dominate decode time, the per-step miss penalty must be <14.7ms. Given 6 experts per step and ~750 MB per expert:

```
max_accept_load_time = 14.7ms / 6 = 2.45ms per expert
max_bandwidth = 750 MB / 2.45ms = 306 GB/s
```

This is **10× the PCIe bandwidth**. Even with optimal caching, a single expert cache miss adds significant latency. This is why full GPU residency is strongly preferred.

## KV Cache

### Structure

Raw window (circular buffer, 128 tokens), compressed cache (monotonically appended), indexer (searches compressed KV).

[source: kv-cache.md]

### Cost Per Decode Step

| Operation | Cost | Frequency |
|---|---|---|
| KV projection write to raw window | ~1-2 µs | Every decode step |
| KV compression trigger | ~10-50 µs | Every compression_ratio steps |
| Indexer rebuild | ~50-200 µs | Every compression_ratio steps at ratio-4 layers |
| Compressed cache append | ~1-5 µs | Every compression trigger |

[hypothesis: based on tensor sizes and memory bandwidth]

**KV cache never misses** — it's allocated at session create and kept resident for session lifetime. No eviction, no miss. The per-step cost is the projection kernel to write new KV into the cache.

### KV Cache Memory Cost

| Parameter | Value | Source |
|---|---|---|
| Raw window per layer | 128 tokens × 2 bytes × 2 (K+V) × head_dim × kv_heads | [derived: kv-cache.md spec] |
| For Flash model | 128 × 2 × 2 × 512 × 1 = 256 KB per layer | [derived] |
| 43 layers × 256 KB | 11 MB raw | [derived] |
| Compressed cache per ratio-4 layer | ctx/4 × 2 × 2 × head_dim × kv_heads | [derived] |
| At ctx=32768 | 8192 × 2 × 2 × 512 × 1 = 16 MB per ratio-4 layer | [derived] |
| ~21 ratio-4 layers | ~336 MB compressed | [derived] |
| **Total KV at ctx=32768** | ~1.03 GiB | [measured: ds4 memory accounting] |

## OS Page Cache

### Model File Memory Map

The model file (153 GiB GGUF) is mmap'd. The kernel page cache provides implicit caching of frequently-read weight pages.

### Page Cache Hit Rate

For the current workload (batch=1 decode, all layers executed per step):

| Access Pattern | Page Cache Behavior |
|---|---|
| Layer weights (sequential within layer) | Good spatial locality, high hit rate |
| Cross-layer (jump to next layer) | Temporal locality depends on layer count |
| Expert weights (random across 256 experts) | Poor locality, low hit rate without full residency |
| First prefill | Cold cache, all pages from NVMe |

### DONTNEED Hinting

The `DS4_METAL_ENABLE_STREAMING_EXPERT_EVICT_DONTNEED` option tells the kernel to reclaim pages from evicted expert weights. Not applicable in CUDA mode.

### Impact on Decode

Since all weights are GPU-resident (split across 2 GPUs), the OS page cache is only hit during model load. During decode, weight access goes to GPU VRAM, not to the page cache. The page cache only matters for expert weight loading in SSD streaming mode (which is not used on this system).

## Storage Cache

### NVMe Bandwidth

| Storage | Interface | Sequential Read | Notes |
|---|---|---|---|
| System drive | PCIe Gen 5 x4 | ~7 GB/s (est) | Active inference |
| Data drive | PCIe Gen 4 x4 | ~3.5 GB/s (est) | Cold model storage |

[hypothesis: typical Gen 5/Gen 4 NVMe sequential read speeds]

### Cold Load vs Warm Load

| Scenario | Source | Time | Penalty |
|---|---|---|---|
| First model load (cold page cache) | NVMe → RAM | ~22s (153 GB / 7 GB/s) | One-time |
| Subsequent loads (warm page cache) | RAM | <1s | Negligible |

## Cache Maintenance Costs

### Aggregate Maintenance Per Decode Step

For the current system (full GPU residency, no SSD streaming):

| Maintenance Operation | Cost | Notes |
|---|---|---|
| KV cache write | ~1-2 µs | Every step |
| Memory management (CUDA allocator) | <1 µs | cuBLAS workspace reuse |
| TLB maintenance | <1 µs | Kernel-managed |
| **Total per-step overhead** | **<3 µs** | Negligible (<0.02% of decode time) |

The cache maintenance cost is essentially zero because the system operates in full-residency mode.

### If SSD Streaming Were Active

| Maintenance Operation | Cost | Frequency |
|---|---|---|
| Expert load from mmap | ~160 ms (6 experts) | Every step (CUDA single-use) |
| LRU eviction scan | O(cache_entries) | Every cache miss |
| Hotness decay (halve counters) | O(n_layers × n_experts) | Every 16 steps |
| DONTNEED madvise | 1 syscall | Every eviction |
| Buffer reuse search | O(cache_entries) | Every eviction |

These costs would dominate decode time, making SSD streaming infeasible at 68 t/s.

## Optimization Targets

| Level | Current Cost | Target | Mitigation |
|---|---|---|---|
| GPU expert cache | Zero (all resident) | Zero | Maintain full GPU residency |
| KV cache | <3 µs/step | <1% of decode time | Already met |
| OS page cache | Zero (all GPU-resident) | Zero | Already met |
| Storage | One-time load | <30s cold start | Already met |

The only cache optimization needed is ensuring VRAM budgets keep all weights GPU-resident as model sizes grow or context lengths increase.

## See Also

- [gpu-expert-streaming-cache.md](../code/concepts/gpu-expert-streaming-cache.md) — expert cache spec
- [kv-cache.md](../code/engine/kv-cache.md) — KV cache structure
- [roofline-analysis.md](roofline-analysis.md) — throughput model
- [tuning-guide.md](tuning-guide.md) — decision framework
