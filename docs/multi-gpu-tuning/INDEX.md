# Multi-GPU Tuning Research — Index

## Documents

| Document | Status | Description |
|---|---|---|
| [GROUND-RULES.md](GROUND-RULES.md) | Final | Experimental methodology, hypothesis structure, abandonment criteria, learning rate optimization, assumption hygiene. All experiments inherit these rules. |
| [PRD.md](PRD.md) | Final | Project requirements, objectives, topics, deliverables. All experiments governed by GROUND-RULES.md. |
| [research-log.md](research-log.md) | Final | Chronological log of experiments following GROUND-RULES.md §8 format. |
| [PRD-2.md](PRD-2.md) | Phase 2 — partial falsification | Code generation throughput optimization. H1 (dequant dominant) FALSIFIED by O1.2. H3 (acceptance rate) FALSIFIED by 5% measured rate. Inline correction notes applied. |
| [PRD-3.md](PRD-3.md) | Phase 3 — active investigation | Speculative decode structural validation. Tests MTP/DSpark assumptions before throughput optimization. Governed by P0.0 support_kind detection. Experimental priority per learning rate §4. |

## Deliverables

| Document | Status | Dependencies |
|---|---|---|
| [roofline-analysis.md](roofline-analysis.md) | Complete | PCIe characterization, compute peak measurement |
| [pcie-characterization.md](pcie-characterization.md) | Complete | PCIe bandwidth + latency experiments |
| [speculative-decode-multi-gpu.md](speculative-decode-multi-gpu.md) | Complete | DSpark analysis on 2-GPU |
| [cache-cost-model.md](cache-cost-model.md) | Complete | Cache miss/maintenance cost analysis |
| [alignment-tensions.md](alignment-tensions.md) | Complete | Alignment micro-benchmarks |
| [instrumentation-guide.md](instrumentation-guide.md) | Complete | Derived from PRD §4.9 |
| [tuning-guide.md](tuning-guide.md) | Complete | All prior deliverables |
| [scalability-analysis.md](scalability-analysis.md) | Complete | Tuning guide + analytical extrapolation |

## Experimental Artifacts

| Path | Status | Contents |
|---|---|---|
| [experiments/pcie-bw/](experiments/pcie-bw/) | Complete | `pcie_bw_bench.cu` — PCIe bandwidth micro-benchmark (CUDA), compiled binary |
| [experiments/compute-peak/](experiments/compute-peak/) | Complete | `compute_peak_bench.cu` — HBM BW + FMA micro-benchmark, compiled binary |
| [experiments/alignment-cost/](experiments/alignment-cost/) | Empty | Alignment cost measured via PCIe bench (no separate benchmark needed) |
| [experiments/dspark-profile/](experiments/dspark-profile/) | Empty | DSpark not profiled on GPU — requires DSpark support model loaded |
| [experiments/cache-miss/](experiments/cache-miss/) | Empty | Cache miss analyzed analytically (full GPU residency achieved) |
| [data/](data/) | Complete | Raw measurement files: `pcie_bw_results*.txt`, `compute_peak_results.txt`, `pipeline_bench_baseline.csv` |
| [analysis/](analysis/) | Complete | `analyze_results.py` — Python analysis script |

## Reference Documents (Codebase)

- [Multi-GPU Pipeline](../code/concepts/multi-gpu-pipeline.md) — pipeline architecture, XDEV, peer access
- [DSpark](../code/concepts/dspark.md) — speculative decoding architecture
- [Layer Packing Engine](../code/concepts/layer-packing-engine.md) — placement algorithm
- [GPU Expert Streaming Cache](../code/concepts/gpu-expert-streaming-cache.md) — expert cache hierarchy
- [Session Batch Decode](../code/concepts/session-batch-decode.md) — batch decoding
- [CUDA Backend](../code/backends/cuda.md) — kernel categories, memory model
- [HC Transforms](../code/engine/hc-transforms.md) — HC pre/post/output head kernels
- [KV Cache](../code/engine/kv-cache.md) — cache structure, memory accounting
- [Tensor Parallelism](../code/modules/tp.md) — TP transport, gate exchange
- [Distributed Inference](../code/modules/distributed.md) — pipeline-parallel multi-machine
- [Benchmark](../code/modules/bench.md) — frontier walk, measurement methodology

## Key Findings Summary

| Finding | Value | Source |
|---|---|---|
| PCIe unidirectional peak | 28.4 GB/s (89% of theoretical) | [measured: pcie_bw_bench_v2] |
| PCIe bidirectional peak | 51.8 GB/s aggregate | [measured: pcie_bw_bench_v2] |
| Host bounce peak | 14.4 GB/s (~50% of peer DMA) | [measured: pcie_bw_bench_v2] |
| Activation xfer latency (16KB) | 1.5 µs | [measured: pcie_bw_bench_v2] |
| HBM read bandwidth | 1502-1508 GB/s per GPU | [measured: compute_peak_bench] |
| HBM copy bandwidth | 1290-1295 GB/s per GPU | [measured: compute_peak_bench] |
| Decode throughput (2-GPU) | 68.5 t/s steady-state | [measured: pipeline bench] |
| Prefill throughput (2-GPU) | 1758 t/s | [measured: pipeline bench] |
| PCIe utilization at 68 t/s | 0.003% | [derived: bandwidth / throughput] |
| Load imbalance | ~26% (24 vs 19 layers) | [measured: ds4 multi-GPU layout] |
| Dequant overhead factor | 1.43× (not 2-4×) | [measured: O1.2 q4-dequant-overhead, research-log.md] |
| DSpark acceptance rate (greeting prompt) | 5.00% (3/60 proposed) | [measured: DSpark-acceptance-rate, research-log.md] |
| MTP spec decode default gating | Disabled (mtp_draft_tokens=1). Requires --mtp-draft-tokens N with N>=2 | [derived: ds4_engine_options, ds4.c guard check] |
| DSpark/MTP mutual exclusivity | DSpark checked first, MTP fallback. Combined GGUF → always DSpark | [measured: support_model_detect, ds4.c line 2787] |
| DSpark capture init cost | ~2800 ms seed-from-prefill, then ~11.4 ms/step draft chain | [measured: research-log.md DSpark stats] |
| Alignment sensitivity | None detected | [measured: pcie_bw_bench_v2] |
| DSpark draft/verify bottleneck | Coordinator-only, doesn't scale with N | [derived: dspark.md spec] |
