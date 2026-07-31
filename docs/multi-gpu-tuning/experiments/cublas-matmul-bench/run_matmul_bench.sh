#!/usr/bin/env bash
# O1.3 — cuBLAS matmul Q4_0 vs F16 TFLOPS comparison
#
# Builds and runs the matmul benchmark at decode shapes:
#   4096×2048 (FFN gate/up)
#   4096×4096 (QKV projection)
#   2048×2048 (FFN down)
#
# Usage: bash run_matmul_bench.sh
#
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="$EXPERIMENT_DIR/matmul_bench"

echo "=== O1.3 — cuBLAS Matmul Q4_0 vs F16 TFLOPS ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

# Build
echo "--- Building matmul_bench ---"
nvcc -O3 -arch=sm_120 -o "$BENCH" "$EXPERIMENT_DIR/matmul_bench.cu" \
    -lcuda -lcudart -lcublas 2>&1
echo "Build complete"
echo ""

# System info
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
echo ""

# Run
echo "--- Running benchmark ---"
"$BENCH" 2>&1 | tee "$EXPERIMENT_DIR/matmul_bench_results.txt"

echo ""
echo "=== Complete ==="
echo "Results saved to: $EXPERIMENT_DIR/matmul_bench_results.txt"
