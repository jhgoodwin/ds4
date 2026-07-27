# CPU Backend

## Files

- `ds4.c` — CPU reference sections: all quantized kernels, layer runner, attention, FFN
- `ds4_gpu_args.c` — GPU CLI argument parser (compiled as `ds4_gpu_args_cpu.o` with `-DDS4_NO_GPU` in `make cpu`)
- `ds4.c` — GPU stubs under `#ifdef DS4_NO_GPU` (lines 15335+; main block at 15335-15417 holds ~40 inline stubs, plus smaller blocks at 4595, 6142, 48368). Stubs in `ds4.c`, **not** `ds4_gpu.h`. Returns 0, 1, or -1 depending on function.

### Macro: `DS4_NO_GPU`

Define at compile time (`-DDS4_NO_GPU`) to build the CPU-only reference backend. Guards all GPU-specific code: stubs in `ds4.c`, `make cpu` target, GPU argument parser recompilation. Greppable entry point for understanding the CPU/GPU split.

## Purpose

CPU-only reference/debug backend.  Not for production inference.  Used for:

- Correctness verification against GPU results
- Debugging quantized math
- Running on machines without GPU
- Small-scale testing

## Quantized Kernels

All quantized kernel implementations live in ds4.c.  Math is single-threaded or thread-pool parallel (row-parallel kernels split output rows across workers).  No x86-64 SIMD intrinsics beyond compiler auto-vectorization.  ARM NEON intrinsics (~211 call sites in `ds4.c`) are compiled when `__ARM_NEON` is defined.  No SSD streaming or distributed inference support.

Thread pool created at engine init.  Workers sleep on condition variable, wake when work arrives.

## Scalar Fallback

No x86 SIMD intrinsics or hand-tuned assembly on x86-64; ARM NEON (~211 call sites) exists when `__ARM_NEON` is defined.  All kernel loops on x86 compile as scalar operations, relying entirely on the compiler for auto-vectorization.  ARM NEON intrinsics (~211 call sites) are used when `__ARM_NEON` is defined.  Adequate for correctness checking; unsuitable for performance measurement.

## Invariants

- CPU backend compiled via `make cpu` (separate target).
- GPU tensor functions return 0, 1, or -1 depending on function (no-op stubs in `ds4.c` under `#ifdef DS4_NO_GPU`).
- Metal/CUDA-specific code excluded at compile time.
- Used only for reference comparison, not performance measurement.

## See Also

- [Quantization Kernels](../engine/quant-kernels.md)
- [GGUF Format](../concepts/gguf-format.md)

[← Back to Index](../README.md)
