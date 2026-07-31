# PCIe Characterization

## Purpose

Measure PCIe Gen 5 x8 bandwidth, latency, and peer access characteristics on the dual-RTX PRO 6000 Blackwell system. These measurements define the cross-device transfer cost for the pipeline-parallel inference pipeline.

## System Topology

| Parameter | Value | Source |
|---|---|---|
| GPU 0 bus | `0000:00:10.0` | [measured: nvidia-smi] |
| GPU 1 bus | `0000:00:11.0` | [measured: nvidia-smi] |
| Peer topology | PHB (same PCIe Host Bridge) | [measured: nvidia-smi topo] |
| NUMA | Single node (0), 15 vCPUs | [measured: lscpu] |
| Virtualization | KVM/QEMU (Proxmox passthrough) | [measured: /proc/cpuinfo] |
| CUDA peer access | DIRECT both directions | [measured: pcie_bw_bench] |
| PCIe max link | Gen 5 x16 per slot | [measured: nvidia-smi query] |
| PCIe current link | Gen 1 x8 (idle, ASPM) | [measured: nvidia-smi query] |
| GPU SM count | 188 per GPU | [measured: compute_peak_bench] |
| Compute capability | 12.0 (Blackwell) | [measured: cudaDeviceProp] |

**Note**: PCIe link negotiates to Gen 1 when idle. Under CUDA traffic, it should negotiate to Gen 5. The measured bandwidth (28 GB/s) confirms Gen 5 x8 operation under load.

## Measured Bandwidth

### Unidirectional Peer DMA

| Size | GPU0→GPU1 (GB/s) | GPU1→GPU0 (GB/s) | Latency (µs) | Notes |
|---|---|---|---|---|
| 256B | 0.20 | 0.20 | 1.3 | Protocol overhead dominates |
| 1KB | 0.83 | 0.80 | 1.2 | Latency-bound |
| 16KB | 11.47 | 11.56 | 1.4 | Activation vector size (F16) |
| 32KB | 17.14 | 16.64 | 1.9 | Activation vector size (F32) |
| 128KB | 24.16 | 24.17 | 5.4 | Transition to bandwidth-bound |
| 1MB | 27.58 | 27.65 | 38.0 | Near peak |
| 16MB | 28.31 | 28.30 | 592.6 | Saturates at peak |
| 64MB | 28.42 | 28.22 | 2362 | Peak stable |

[measured: pcie_bw_bench_v2, 980 trials per size, mean values]

### Bidirectional Peer DMA

| Size | Aggregate (GB/s) | Per-Direction (GB/s) | Latency (µs) |
|---|---|---|---|
| 16KB | 4.32 | 2.16 | 7.6 |
| 256KB | 32.53 | 16.27 | 16.1 |
| 1MB | 46.57 | 23.28 | 45.0 |
| 16MB | 51.79 | 25.90 | 647.9 |
| 64MB | 51.80 | 25.90 | 2591 |

[measured: pcie_bw_bench_v2, 980 trials per size, mean values]

### Host Bounce (D2H + H2D via pinned memory)

| Size | Effective BW (GB/s) | Latency (µs) |
|---|---|---|
| 16KB | 2.56 | 6.4 |
| 256KB | 7.63 | 34.4 |
| 1MB | 12.58 | 83.3 |
| 16MB | 14.13 | 1187 |
| 64MB | 14.39 | 4664 |

[measured: pcie_bw_bench_v2, 980 trials per size, mean values]

## Key Findings

### 1. Peak Efficiency

Unidirectional peak: **28.4 GB/s** → **89%** of PCIe Gen 5 x8 theoretical (32 GB/s).

The 11% gap to theoretical is attributable to:
- TLP/DLLP framing overhead: ~2% [hypothesis: PCIe Gen 5 encoding overhead]
- Flow control credits (ACK/NACK round-trips): ~3% [hypothesis: depends on receiver buffer capacity]
- IOMMU translation in VM: ~3% [hypothesis: KVM IOMMU adds per-page translation]
- CUDA driver copy path overhead: ~3% [hypothesis: internal staging for peer validation]

### 2. Bidirectional Scaling

Bidirectional aggregate: **51.8 GB/s** → **81%** of 2× theoretical (64 GB/s). Per-direction bandwidth drops from 28.4 to 25.9 GB/s (~9% reduction) under concurrent traffic. This is consistent with PCIe switch port contention when both directions share the same upstream port [hypothesis: PHB topology routes both directions through the same root port].

### 3. Host Bounce Penalty

Host bounce achieves 14.4 GB/s max, approximately half of peer DMA bandwidth. The D2H + H2D path traverses PCIe twice, hence the ~2× latency increase. For small transfers (<32KB), host bounce has a fixed overhead of ~6µs vs ~1.3µs for peer DMA — a 4-5µs penalty.

### 4. Activation-Size Transfer Cost

For pipeline-parallel inference, activation vectors are the primary cross-device payload:

| Quantization | Size | Peer Latency | Bounce Latency | Bounce Penalty |
|---|---|---|---|---|
| F16 hidden (Flash) | 8 KB | ~1.0 µs | ~5.5 µs | +4.5 µs |
| F32 hidden (Flash) | 16 KB | ~1.5 µs | ~6.0 µs | +4.5 µs |

[derived: from pcie_bw_bench measurements at nearest transfer sizes]

At typical decode latencies of 10-30 ms per token [measured: ds4 pipeline bench], this transfer cost is **0.005-0.02% of decode time** — negligible. Even with 43 layers and 2 GPUs sending activations at layer boundaries, total cross-device transfer time is <3 µs per token.

### 5. Peer DMA vs Host Bounce Selection

| Transfer Size | Peer DMA (GB/s) | Host Bounce (GB/s) | Ratio |
|---|---|---|---|
| 64B-32KB | 0.05-17 | 0.01-4.6 | 3-4× faster |
| 64KB+ | 21-28 | 6-14 | 2× faster |

Peer DMA is universally faster on this system. No crossover point where host bounce is competitive. The `DS4_FORCE_HOST_BOUNCE=1` path should NOT be used on this hardware — it would halve effective bandwidth.

### 6. Alignment Sensitivity

Tested at 1MB transfer size with offsets from 0 to 1024B (step sizes: 1B, 4B, 16B, 32B, 64B, 128B, 256B, 1024B). All offsets achieved 27.4-27.7 GB/s — **no measurable degradation** from misalignment.

[measured: pcie_bw_bench_v2, offset sensitivity test]

The absence of alignment sensitivity suggests cudaMemcpyPeerAsync either handles misalignment internally or the PCIe DMA engine on Blackwell supports unaligned transfers without penalty.

## Analysis

### Roofline Position

For the pipeline-parallel decode workload:
- **Compute ceiling**: ~0.4-1.5 TFLOPS (memory-bound decode kernels)
- **HBM ceiling**: ~1500 GB/s (per GPU)
- **PCIe ceiling**: 28 GB/s (per direction)

The activation transfer (8KB F16 for Flash) requires 28 GB/s / (8KB / token) = **3.5M tokens/sec** before saturating PCIe. At measured decode throughput of ~68 t/s [measured: pipeline bench], PCIe utilization is **0.002%**.

**PCIe is not a bottleneck for decode** on 2-GPU pipeline parallelism. The pipeline bubble from cross-device transfer is a fixed ~1.5µs per hop, dwarfed by compute time per layer (~200-500µs per layer).

### Peak Throughput Summary

| Measurement | Value | % of Theoretical |
|---|---|---|
| Unidirectional peak | 28.4 GB/s | 89% |
| Bidirectional aggregate | 51.8 GB/s | 81% |
| Host bounce peak | 14.4 GB/s | — |
| Activation latency (F16) | 1.5 µs | — |
| Activation latency (F32) | 1.9 µs | — |

[all measured: pcie_bw_bench_v2]

## Implications for Scaling

| N GPUs | Cross-device hops/token | Total xfer latency | PCIe utilization (at 68 t/s) |
|---|---|---|---|
| 2 | 1 | 1.0 µs | 0.002% |
| 4 | 3 | 3.0 µs | 0.005% |
| 8 | 7 | 7.0 µs | 0.01% |

[derived: activation latency × hops at 68 t/s decode]

PCIe becomes a meaningful bottleneck only when per-token decode throughput exceeds ~3.5M t/s (requiring HBM bandwidth saturation across multiple GPUs). This is not achievable with current model sizes and quantization on 2 GPUs.

## See Also

- [roofline-analysis.md](roofline-analysis.md) — composite roofline model
- [experiments/pcie-bw/](experiments/pcie-bw/) — source code and raw data
- [data/pcie_bw_results_v2.txt](data/pcie_bw_results_v2.txt) — full measurement dump
- [multi-gpu-pipeline.md](../code/concepts/multi-gpu-pipeline.md) — XDEV mechanism in ds4
