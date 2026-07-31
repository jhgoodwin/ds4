# Multi-GPU Tuning — Research PRD

## 1. Purpose

Characterize the gap between **mathematical throughput limits** and **measured throughput** on a multi-GPU pipeline-parallel inference system. Identify root causes of throughput loss, quantify each contributor, and produce a decision framework for tuning.

Primary case study: dual RTX PRO 6000 Blackwell 96GB Max-Q (PCIe Gen 5 x8 per card, split from x16) running inside Proxmox VE with direct GPU passthrough, 96GB OS RAM, PCIe Gen 5 x4 storage, PCIe Gen 4 x4 storage.

**All experiments governed by [GROUND-RULES.md](GROUND-RULES.md)** — experimental methodology, hypothesis structure, abandonment criteria, learning rate optimization, and assumption hygiene rules apply to every test in this project.

## 2. System Under Test (SUT)

### 2.1 Hardware

| Component | Detail |
|---|---|
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell 96GB Max-Q |
| GPU 1 | NVIDIA RTX PRO 6000 Blackwell 96GB Max-Q |
| GPU interconnect | PCIe Gen 5 x8 per card (x16 CPU lanes split across 2 slots) |
| Peer access | GPU peer-to-peer DMA via PCIe (not NVLink) |
| System RAM | 96 GB OS RAM |
| Storage 0 | PCIe Gen 5 x4 NVMe |
| Storage 1 | PCIe Gen 4 x4 NVMe |
| Virtualization | Proxmox VE with direct GPU passthrough (IOMMU groups, VFIO) |
| CPU | (TBD — socket count, NUMA topology, PCIe root port layout) |
| PCIe topology | (TBD — which root complex, switch topology, peer-to-peer routing) |

### 2.2 Software Stack

| Layer | Detail |
|---|---|
| OS | Proxmox VE (Debian-based, custom kernel with VFIO/IOMMU) |
| GPU driver | NVIDIA driver (open kernel module or proprietary), CUDA 12.x |
| Inference engine | ds4 (DwarfStar v4) — CUDA backend |
| Workload | DeepSeek Flash/Pro model, quantized (Q8_0, Q4_0), variable batch sizes |

### 2.3 Constraints

- No NVLink — PCIe Gen 5 x8 is the only GPU-GPU data path.
- Peer access may be direct (PCIe P2P) or require host bounce depending on root port topology.
- Virtualization adds IOMMU translation overhead for DMA — quantify vs bare metal.
- Two storage tiers (Gen 5 vs Gen 4) enable split: hot cache on Gen 5, cold weights on Gen 4.
- Max-Q variant has power/temperature constraints that may throttle under sustained load.

## 3. Research Objectives

### O1 — Establish Mathematical Upper Bound

For every operation in the inference pipeline, compute the **roofline limit**:

- **Compute-bound**: FLOPs per token / peak FMA throughput of 2× RTX PRO 6000 Blackwell
- **Memory-bound (device)**: bytes per token / HBM bandwidth of both GPUs
- **Memory-bound (cross-device)**: bytes transferred per token / PCIe Gen 5 x8 bandwidth (unidirectional + bidirectional)
- **Memory-bound (storage)**: bytes read per token / NVMe sequential read bandwidth
- **Latency-bound**: kernel launch overhead, synchronization barriers, pipeline bubble

Derive the **composite ceiling** for: prefill (tokens/sec), decode (tokens/sec), speculative decode (tokens/sec with acceptance rate).

### O2 — Measure Actual Throughput Per Component

Instrument each subsystem independently:

- **Compute kernels**: FLOPs achieved vs peak for matmul, attention, MoE FFN, norm, HC transforms
- **Cross-device transfer**: achieved PCIe bandwidth per transfer direction, latency per transfer
- **Storage read**: achieved NVMe bandwidth for model weight loads, cache miss penalties
- **Synchronization**: time spent in cudaDeviceSynchronize, stream wait events, cross-device barriers
- **Kernel launch**: submission overhead per kernel, per graph node
- **Pipeline bubble**: idle GPU time due to inter-device dependency chains

### O3 — Decompose the Gap

For each measured component, attribute the gap between measured and mathematical limit to specific causes:

| Cause Category | Examples |
|---|---|
| Memory alignment | Misaligned access reduces effective bandwidth; padding tension between alignment and compact representation |
| Kernel fusion | Fused kernels reduce launch overhead but may increase register pressure or reduce occupancy |
| Quantization overhead | Dequantize cycles on-the-fly, format conversion (F32↔F16↔Q8_0) |
| PCIe protocol overhead | TLP/DLLP framing, ACK/NACK, flow control credit round-trips |
| IOMMU translation | Virtualization adds DMA translation latency, may force bounce buffering |
| Cache maintenance | LRU eviction decisions, hotness decay, slab allocator fragmentation |
| Power throttling | Max-Q power cap reduces clock under sustained load |
| NUMA effects | Cross-socket PCIe topology increases inter-device latency |
| Tensor parallelism | Expert sharding overhead, partial reduction, coordinator serialization |

### O4 — Map Speculative Decoding Overlap

For DSpark-style speculative decoding across multiple GPUs:

- Where does draft computation run relative to base model forward pass?
- How does multi-GPU placement interact with stage-target-layer capture?
- What is the cost of TP expert-sharding suspend/resume during draft chain?
- Can draft computation on one GPU overlap with verification on another?
- What is the acceptance-rate ceiling given the mathematical limits of Markov-chain draft quality?

### O5 — Produce Tuning Guidelines

For each finding, produce:

1. **Detection method** — how to measure/observe the issue on the target system
2. **Mitigation** — configuration change, code change, or workload change that reduces the gap
3. **Trade-off** — what degrades when the mitigation is applied
4. **Expected impact** — estimated throughput recovery (as % of gap closed)

## 4. Topics of Investigation

### 4.1 Multi-GPU Pipeline Placement

**Ref**: multi-gpu-pipeline.md, layer-packing-engine.md

- Monotonic-contiguous greedy first-fit: how close to optimal bin-packing?
- Entry footprint estimation accuracy (tensor weights + KV per layer)
- Budget pre-subtraction: safety margin, cuBLAS workspace, per-tier graph scratch
- KV cache distribution: raw window on each device vs compressed cache locality
- Impact of imbalance: GPU0 saturated at 95% while GPU1 at 40%

### 4.2 Cross-Device Communication (XDEV)

**Ref**: multi-gpu-pipeline.md (Communication section)

- PCIe Gen 5 x8 bidirectional bandwidth: 32 GB/s theoretical, ~28 GB/s achievable
- Peer DMA vs host bounce: threshold where bounce is cheaper (small transfers, high latency)
- Event-based synchronization overhead: cudaEventRecord + cudaStreamWaitEvent round-trip
- Grouped transfers (xdev3) vs individual: breakeven transfer size
- Ordered transfers (xdev_ordered) vs unordered: when ordering adds measurable cost
- Pipeline prefill: overlapping send with compute on source device
- Impact of PCIe topology on peer access (root port vs switch vs NUMA)

### 4.3 DSpark / Speculative Decoding on Multi-GPU

**Ref**: dspark.md

- Stage chain eval requires TP expert sharding suspend — what is the idle-GPU cost on the non-coordinator device?
- Capture-based HC recording: per-layer overhead on the device owning the target layer
- Support model KV ring: where does it live (which device)? Cross-device access cost?
- Confidence threshold gating: CPU-in-the-loop vs GPU-only path
- Markov argmax: GPU fused path (requires Q8_0 w1/w2, rank multiple of 32) vs CPU fallback
- Scheduler adaptive skip: how does pipeline bubble from skipped cycles interact with multi-GPU?

#### Key Questions

1. Can stage 0 eval run on device 0 while stage chain runs on device 1 (overlap)?
2. Does the capture mechanism record HC on the correct device, or does it require cross-device readback?
3. What is the critical path through draft proposal → verify → accept/reject when layers span 2 GPUs?
4. Can verification (base model forward pass) overlap with next draft proposal in a steady state?

### 4.4 Fused Kernels

**Ref**: cuda.md, hc-transforms.md

- Which kernel fusions exist in the CUDA backend (attention decode, HC weighted sum, output head)?
- Register pressure vs launch overhead trade-off for each fused kernel
- Fusion across device boundary: is cross-device fusion possible (e.g., fused HC pre on device 0 + attention on device 1)?
- Effect of fusion on occupancy and achievable FLOPs
- Fusion interaction with tensor parallelism (partial results must be exchanged before fused kernel can complete)

### 4.5 Aligned Datatypes & Alignment Tensions

- CUDA alignment requirements: 128B cache line, 32B warp, 16B vec4, 8B vec2
- Tensor quantization formats: Q8_0 (block size 32), Q4_0 (block size 32), F16, F32
- Alignment tension: Q8_0 wants block size 32 for efficient dequant → forces padded dimensions → wastes memory bandwidth
- Cross-device transfer alignment: cudaMemcpyPeerAsync alignment requirements, bounce buffer alignment
- Shared memory bank conflicts from misaligned access patterns in attention kernels
- Tension between compact representation (minimum bytes per weight) and aligned access (minimum wasted bandwidth)

#### Specific Measurements

| Alignment Scenario | Cost to Measure |
|---|---|
| Q8_0 block padding at dimension boundaries | Extra bytes transferred / effective bandwidth loss |
| Misaligned cross-device copy (non-128B offset) | Fallback to cudaMemcpy (device-side) vs cudaMemcpyPeerAsync |
| Shared memory bank conflicts in decode attention | Cycles wasted per warp |
| Non-coalesced global memory access in HC pre/post | Achieved vs peak bandwidth for HC transform kernels |

### 4.6 Caching Hierarchy

**Ref**: gpu-expert-streaming-cache.md, kv-cache.md

#### GPU Expert Streaming Cache (Metal LRU / CUDA compact buffer)

- Hit rate vs cache size for various workloads (chat, code, analysis)
- Miss penalty: SSD read latency + PCIe transfer + GPU memcpy
- LRU eviction overhead: victim selection scan, buffer release, potential DONTNEED hint
- Hotness decay: cost of halving all counters every 16 tokens
- Buffer reuse (take_reusable): allocation churn reduction vs complexity cost
- CUDA single-use compact buffer: always re-reads from mmap — cost per decode step

#### KV Cache

- Raw window (circular buffer, last 128 tokens): write per step, no eviction cost
- Compressed cache (appended monotonically): write per compression trigger, no eviction
- Indexer: rebuilt when new compressed rows added at ratio-4 layers — cost per rebuild
- Cache residency: KV stays resident for session lifetime — no eviction, no miss

#### Storage Cache (OS page cache / NVMe)

- Model file mmap: kernel page cache provides implicit caching of frequently-read weight pages
- Expert streaming reads: sequential within expert, random across experts — page cache hit rate
- DONTNEED hints: tell kernel to reclaim pages — reduces pressure but causes re-read on next access

#### Cross-Device Cache Coherency

- Each GPU caches only its assigned tensors (selective cache via ds4_gpu_device_cache_tensors)
- No cross-device cache coherence — each device has independent view of model weights
- Host memory map registered once (ds4_gpu_register_model_map_no_copy) — no per-device copy
- Strict lookup (ds4_gpu_lookup_cache_strict): no fallback — miss means CPU read

### 4.7 Cost of Cache Miss

#### Expert Cache Miss

| Stage | Operation | Latency `[hypothesis]` |
|---|---|---|
| 1 | Router selects expert not in cache | — |
| 2 | SSD read: pread from model file (sequential, ~expert_size bytes) | ~10-50 µs `[hypothesis: Gen 5 NVMe QD1 latency range from vendor specs]` |
| 3 | PCIe transfer: host pinned → GPU device | ~5-20 µs `[hypothesis: depends on expert size, derived from PCIe Gen 5 x4 bandwidth]` |
| 4 | GPU memcpy: temporary → cache slot | ~1-5 µs `[hypothesis: GPU memcpy bandwidth ~500 GB/s, expert ~1-10 MB]` |
| 5 | Cache prune (per-layer + global) | ~0.1-1 µs `[hypothesis: O(n) scan over resident entries, n ≤ budget]` |
| Total | Miss penalty per expert | ~16-76 µs `[hypothesis: sum of above, to be measured]` |

If 2-4 experts miss per decode step, miss penalty adds 32-304 µs `[hypothesis: derived from per-expert penalty × miss count]` to a decode step that would otherwise be ~5-10 ms `[hypothesis: typical decode step time from prior bench runs]`. At 8 experts miss (cold start), penalty reaches 128-608 µs — significant fraction of decode time.

#### KV Cache Miss (not applicable — always resident)

KV cache allocated at session create and kept resident. No eviction, no miss. However, compressed cache rebuild (indexer) on new row addition is a per-trigger cost.

#### Storage Cache Miss (OS page cache)

If model weights evicted from OS page cache (memory pressure), next read goes to NVMe directly. Page cache miss adds ~10-50 µs per page. For a large model (100+ GB), thrashing page cache can add seconds to prefill.

### 4.8 Cost of Cache Maintenance

| Maintenance Operation | Frequency | Cost |
|---|---|---|
| LRU eviction victim scan (per-layer) | Every seed/load that exceeds layer cap | O(n_experts_in_layer) scan |
| LRU eviction victim scan (global) | Every seed/load that exceeds total budget | O(total_entries) scan |
| Hotness decay (halve all counters) | Every 16 decode tokens | O(total_layers × total_experts) memset/bit-shift |
| Hotness increment | Every router decision | O(1) atomic |
| Buffer reuse search (take_reusable) | Every eviction | O(total_entries) byte-size comparison |
| DONTNEED hint | Every eviction (when enabled) | 1 syscall (madvise) |
| Indexer rebuild | Every compressed row addition at ratio-4 layer | O(ctx/4 × n_heads) compute |
| KV compressed cache append | Every compression trigger | O(n_embd) projection |

The aggregate maintenance cost should be measured as fraction of total decode time. Optimization threshold determined per GROUND-RULES.md §3.2 (model refinement limit).

### 4.9 Instrumentation & Observability

#### Existing Mechanisms

| Mechanism | Scope | Overhead |
|---|---|---|
| `DS4_DSPARK_SPEC_LOG` | Per-round speculative decisions | Stderr write per decode step — measurable I/O latency |
| `DS4_DSPARK_PROBE` | Draft pipeline profiling (no consumption) | Full draft compute + stats tracking — same as production |
| `DS4_DIST_DECODE_PROFILE` | Per-hop telemetry (distributed mode) | Timing collection + emit per decode step |
| `DS4_METAL_DECODE_STAGE_PROFILE` | Per-stage breakdown (Metal) | Timing collection per graph stage |
| `DS4_CUDA_EXACT_SCORE_SPLIT_DECODE` | Attention score split profiling | Forces specific code path — may change performance |
| `ds4_dspark_spec_stats` | Per-session timing breakdown | Struct updates in hot path — atomic or non-atomic counters |
| `clock_gettime(CLOCK_MONOTONIC)` calls | Explicit timing in bench.c | syscall overhead per measurement point |
| GPU timers (cudaEvent) | GPU-side timing | cudaEventRecord overhead, synchronization for readback |

#### Disruption Analysis

| Instrumentation | Disruption |
|---|---|
| Stderr logging per decode step | write() syscall, may block on pipe buffer full |
| cudaEvent readback | Implicit sync — drains GPU pipeline, destroys overlap |
| Per-kernel timing | Requires separate cudaEvent pairs per kernel — adds launch overhead |
| Atomic counter updates | Cache line bouncing between SMs, contention |
| Memory tracing (valgrind/cuda-memcheck) | 10-100x slowdown — only for validation, not profiling |
| Nsight Systems / Nsight Compute | Hardware performance counters — low overhead but requires root/GPU access |
| Proxmox host monitoring | Host-level perf counters — no GPU visibility |

#### Recommended Approach

1. **Lightweight**: Built-in stats structs + env-var gated timing — minimal overhead, always available
2. **Medium**: cudaEvent pairs around critical sections (cross-device transfers, draft chain, verify) — requires sync but bounded overhead
3. **Heavy**: Nsight Systems timeline — captures full kernel execution view, necessary for overlap analysis
4. **Validation**: cuda-memcheck for correctness runs (not performance) — ensures alignment and access patterns are valid

### 4.10 Scalability of Ideas

| Technique | Scales to N GPUs | Bottleneck `[hypothesis]` |
|---|---|---|
| Pipeline parallelism (monotonic-contiguous) | Yes, up to n_layers | PCIe bandwidth (each device sends to next), pipeline bubble proportional to depth |
| Tensor parallelism (expert sharding) | 2 (pair) | Coordinator serializes logit combine, network latency for gate exchange |
| DSpark speculative decoding | Yes (per-device draft chain) | TP expert sharding suspend on coordinator, capture mechanism must span all devices |
| Expert streaming cache | Yes (per-device independent) | LRU state is per-device — no global coherence needed |
| Batch decoding (session batch) | Yes (all sessions on same device set) | Attention decode row kernel scales to 32 rows, then memory bandwidth bound |
| Mixed prefill+decode | Yes (single device) | Prefill chunk size bounded by VRAM, decode latency increases with prefill overlap |

#### Key Scalability Questions

1. What is the pipeline bubble fraction for 2 GPUs vs 4 GPUs vs 8 GPUs?
   - Bubble = sum of all cross-device transfer latencies per step / step time
   - Increases with depth because each device waits for previous device's output

2. At what GPU count does PCIe bandwidth become the bottleneck for pipeline parallelism?
   - Each device sends full activation vector to next device
   - With N devices, total cross-device traffic per step = (N-1) × activation_bytes
   - At N=2: 1 transfer. At N=4: 3 transfers. At N=8: 7 transfers.
   - Activation size for DeepSeek Pro (7168 hidden dim, F32 = 28KB per token, F16 = 14KB)

3. Does DSpark draft quality degrade with deeper pipeline (more devices)?
   - Stage target layers are model-level, not device-level — capture quality depends on placement
   - If target layers span multiple devices, capture requires cross-device hidden state transfer



## 5. Deliverables

### 5.1 Documents

| Document | Content |
|---|---|
| `PRD.md` | This document |
| `roofline-analysis.md` | Roofline model for each operation, with measured vs theoretical ceilings |
| `pcie-characterization.md` | PCIe Gen 5 x8 bidirectional bandwidth, latency, peer access topology |
| `speculative-decode-multi-gpu.md` | DSpark overlap analysis for 2-GPU pipeline |
| `cache-cost-model.md` | Cost of cache miss and maintenance per level (expert cache, KV cache, page cache) |
| `alignment-tensions.md` | Alignment requirements per datatype, measured cost of misalignment |
| `instrumentation-guide.md` | How to measure each component with minimal disruption |
| `tuning-guide.md` | Decision framework: given workload X and hardware Y, which knobs to turn |
| `scalability-analysis.md` | How each technique scales to N GPUs, where bottlenecks appear |

### 5.2 Experimental Artifacts

| Artifact | Location |
|---|---|
| CUDA micro-benchmarks | `experiments/pcie-bw/`, `experiments/compute-peak/`, `experiments/alignment-cost/` |
| ds4 instrumented runs | `experiments/dspark-profile/`, `experiments/cache-miss/` |
| Raw data | `data/*.csv` — per-experiment measurements |
| Analysis notebooks | `analysis/` — Python/R scripts for model fitting and visualization |

### 5.3 Decision Framework

A flowchart or table mapping:

```
If workload is [chat / code / analysis / batch]
  and model is [Flash / Pro / GLM]
  and GPU count is [1 / 2 / 4 / 8]
  and bottleneck is [compute / memory / PCIe / storage / synchronization]
Then recommended tuning is [X]
  with expected improvement [Y%]
  at cost [Z]
```

## 6. Constraints & Risks

### 6.1 Constraints

- Proxmox VE passthrough: IOMMU groups must allow both GPUs to be passed to the same VM.
- Max-Q power limits: sustained inference may trigger throttling after ~30-60 seconds.
- No NVLink: PCIe Gen 5 x8 is the only GPU-GPU interconnect.
- 96 GB system RAM limits host-side buffers for cross-device bounce transfers.
- Two storage tiers: Gen 5 x4 (system) and Gen 4 x4 (data) — benchmark both.

### 6.2 Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| IOMMU prevents peer DMA | Medium | High — forces host bounce, halves PCIe efficiency | Test with DS4_FORCE_HOST_BOUNCE=1 to characterize |
| Power throttling during sustained bench | High | Medium — measured throughput varies with temperature | Monitor clock freq, short bench runs, power limit control |
| Proxmox CPU scheduler interference | Medium | Low — adds noise to timing | Pin vCPUs, isolate cores, use real-time priority |
| Storage bandwidth shared with host OS | High | Medium — Gen 5 x4 may be contended | Use dedicated NVMe for model storage, measure baseline |
| Insufficient system RAM for large models | Low | High — cannot load model | Use quantized (Q4_0) or smaller model variant |


## 7. References

- [docs/code/concepts/multi-gpu-pipeline.md](../code/concepts/multi-gpu-pipeline.md) — Multi-GPU pipeline architecture
- [docs/code/concepts/dspark.md](../code/concepts/dspark.md) — DSpark speculative decoding
- [docs/code/concepts/layer-packing-engine.md](../code/concepts/layer-packing-engine.md) — Layer placement algorithm
- [docs/code/concepts/gpu-expert-streaming-cache.md](../code/concepts/gpu-expert-streaming-cache.md) — Expert caching hierarchy
- [docs/code/concepts/session-batch-decode.md](../code/concepts/session-batch-decode.md) — Batch decoding architecture
- [docs/code/backends/cuda.md](../code/backends/cuda.md) — CUDA backend details
- [docs/code/engine/hc-transforms.md](../code/engine/hc-transforms.md) — HC transform kernels
- [docs/code/engine/kv-cache.md](../code/engine/kv-cache.md) — KV cache structure
- [docs/code/modules/tp.md](../code/modules/tp.md) — Tensor parallelism
- [docs/code/modules/distributed.md](../code/modules/distributed.md) — Distributed inference
- [docs/code/modules/bench.md](../code/modules/bench.md) — Benchmark harness
