# Multi-GPU Tuning Research — Index

## Documents

| Document | Status | Description |
|---|---|---|
| [GROUND-RULES.md](GROUND-RULES.md) | Draft | Experimental methodology, hypothesis structure, abandonment criteria, learning rate optimization, assumption hygiene. All experiments inherit these rules. |
| [PRD.md](PRD.md) | Draft | Project requirements, objectives, topics, deliverables. All experiments governed by GROUND-RULES.md. |
| [research-log.md](research-log.md) | Active | Chronological log of experiments following GROUND-RULES.md §8 format. |

## Planned Deliverables

| Document | Status | Dependencies |
|---|---|---|
| `roofline-analysis.md` | Not started | PCIe characterization, compute peak measurement |
| `pcie-characterization.md` | Not started | PCIe bandwidth + latency experiments |
| `speculative-decode-multi-gpu.md` | Not started | DSpark profile on 2-GPU |
| `cache-cost-model.md` | Not started | Cache miss/maintenance cost experiments |
| `alignment-tensions.md` | Not started | Alignment micro-benchmarks |
| `instrumentation-guide.md` | Not started | Derived from PRD §4.9 |
| `tuning-guide.md` | Not started | All prior deliverables |
| `scalability-analysis.md` | Not started | Tuning guide + analytical extrapolation |

## Experimental Artifacts

| Path | Status |
|---|---|
| `experiments/pcie-bw/` | Not started |
| `experiments/compute-peak/` | Not started |
| `experiments/alignment-cost/` | Not started |
| `experiments/dspark-profile/` | Not started |
| `experiments/cache-miss/` | Not started |
| `data/` | Not started |
| `analysis/` | Not started |

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
