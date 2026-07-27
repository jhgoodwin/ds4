# CUDA Backend

## Files

- `ds4_cuda.cu` — CUDA kernel implementations and runtime wrappers
- `ds4_gpu.h` — shared GPU type definitions, tensor ops
- `ds4_gpu_args.c` — GPU argument parsing and config
- `ds4_gpu_args.h` — GPU config types
- `ds4_gpu_mgpu.h` — multi-GPU plumbing types and API declarations

## Purpose

NVIDIA GPU backend via CUDA.  Targets DGX Spark, multi-L40S, and consumer NVIDIA GPUs.  Support for single-GPU and multi-GPU (NVLink/PCIe pipeline parallelism).

## Kernel Categories

- **Quantized matmul** — `ds4_gpu_matmul_q8_0_*` variants for Q8_0 quantized weights
- **Attention decode** — `ds4_gpu_attention_decode_*` fused attention for autoregressive decode
- **Cross-device transfer** — P2P copies over NVLink or PCIe, fallback to host staging

## Tensor Ops

See [gpu-tensor-primitives.md](../concepts/gpu-tensor-primitives.md) for GPU tensor operation primitives.

## Memory Model

GPU memory allocated via `cudaMalloc` (CUDA runtime API, not cuMemAlloc).
See [multi-gpu-pipeline.md](../concepts/multi-gpu-pipeline.md) for multi-GPU memory model and device assignment.

## Invariants

- CUDA kernels compiled for sm_75+ (Turing) minimum; sm_80+ (Ampere) required for 16-pair MoE tile kernels.
- Multi-GPU prefers peer access between all pairs; falls back to pinned-host bounce when peer unavailable or `DS4_FORCE_HOST_BOUNCE=1` is set.
- Tensor handles reference GPU memory, not host memory.
- All GPU operations asynchronous; synchronization explicit.

## See Also

- [Quantization Kernels](../engine/quant-kernels.md)
- [KV Cache](../engine/kv-cache.md)
- [Attention](../engine/attention.md)
- [MoE FFN](../engine/moe-ffn.md)
- [GPU Tensor Primitives](../concepts/gpu-tensor-primitives.md)
- [Multi-GPU Pipeline](../concepts/multi-gpu-pipeline.md)

[← Back to Index](../README.md)
