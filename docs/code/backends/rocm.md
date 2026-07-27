# ROCm Backend

## Files

- `ds4_rocm.cu` — ROCm kernel implementations (HIP)
- `ds4_rocm.h` — ROCm-specific declarations
- `ds4_rocm_compat.cu` — CUDA→HIP compatibility wrappers
- `ds4_rocm_unavailable.cu` — stubs for unsupported CUDA features
- `rocm/*.cuh` — 23 kernel implementation files (all GPU kernels live here, not in `ds4_rocm.cu`)

## Purpose

AMD GPU backend via ROCm/HIP. Targets Strix Halo (Radeon 8060S, gfx1151) and similar AMD GPUs.

## ROCm Specifics

HIP port of the CUDA backend. Uses hipify-style translation: CUDA API calls mapped to HIP equivalents via `ds4_rocm_compat.cu`. Unsupported features get no-op stubs in `ds4_rocm_unavailable.cu`.

Dual-compilation design: `ds4_rocm.cu` compiles for both CUDA and ROCm via `#ifdef __HIP_PLATFORM_AMD__`. The AMD branch includes `<hipblaslt/hipblaslt.h>`, sets HIP defines; the CUDA branch includes `<cuda_runtime.h>`, `<mma.h>`, `<cublas_v2.h>`. All kernel bodies (`.cuh` files) are `#include`d inside the AMD branch, keeping the CUDA path minimal.

GPU kernel implementations live in `rocm/*.cuh` files, each corresponding to a CUDA backend kernel file. These are included from `ds4_rocm.cu` under the AMD preprocessor guard. This avoids file-level duplication — the `.cuh` is the single source for both build targets.

rocWMMA required for warp-level matrix operations. Header tree must be installed separately — Ubuntu packages miss internal headers. See `STRIXHALO.md` for setup.

## Strix Halo

- 128 GB unified memory (CPU+GPU)
- Radeon 8060S integrated GPU
- gfx1151 architecture (RDNA 3.5)
- ROCm 7.1+ recommended

## Kernel Differences

| Aspect | CUDA | ROCm |
|---|---|---|
| Compiler | nvcc | hipcc |
| Math library | cuBLAS | hipBLAS/hipBLASLt |
| WMMA | nvcuda::wmma | rocWMMA |
| Peer access | CUDA P2P | Single-GPU (no P2P) |
| Tensor cores | CUDA Tensor Cores | Matrix Cores (gfx1151+) |

## Invariants

- ROCm backend compiled via `make rocm` (separate Makefile target).
- CUDA-only features have no-op stubs.
- Kernel launch parameters tuned for gfx1151 wavefront size (64 vs CUDA's 32).

## See Also

- [Quantization Kernels](../engine/quant-kernels.md)
- [GPU Tensor Primitives](../concepts/gpu-tensor-primitives.md)

[← Back to Index](../README.md)
