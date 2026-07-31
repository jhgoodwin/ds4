# Alignment Cost — No Separate Benchmark Needed

## Status: Not Run (Separate Benchmark)

Alignment cost was measured as a sub-test of the PCIe bandwidth benchmark (`experiments/pcie-bw/pcie_bw_bench.cu`), not as a standalone benchmark. The PCIe bench includes an offset sensitivity test that covers all alignment scenarios.

## Abandonment Rationale

Per GROUND-RULES.md §3 (Abandonment Criteria):

- **Learning rate** [per §4.1]: Creating a standalone alignment-cost benchmark would duplicate the alignment test already in `pcie_bw_bench.cu`. The PCIe bench measures:
  - cudaMemcpyPeerAsync with offsets 0-1024B (step sizes 1B, 4B, 16B, 32B, 64B, 128B, 256B, 1024B)
  - Bandwidth vs offset — detects alignment sensitivity
  - Result: No measurable degradation at any offset

- **Plateau rule**: The alignment result is negative (no penalty detected). Further independent measurement would produce the same result.

- Additional alignment tensions (Q4_K block padding, shared memory bank conflicts) are analyzed analytically in alignment-tensions.md. These are architectural properties not amenable to micro-benchmarking without model-weight-level access.

## What Was Measured

See `data/pcie_bw_results_v2.txt` for alignment test data. Summary in alignment-tensions.md.

## If a Future Model Shows Alignment Sensitivity

Create a standalone CUDA benchmark that:
1. Allocates misaligned device buffers via pointer arithmetic
2. Measures effective bandwidth for copy kernels at each offset
3. Detects fallback thresholds where CUDA switches from optimized to generic copy path
