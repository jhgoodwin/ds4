#!/usr/bin/env bash
# P2 — Fused Decode Kernel Prototype Benchmark
#
# Builds and runs the fused vs unfused decode kernel comparison.
#
# Measures:
#   - Per-layer time (fused vs unfused)
#   - Projected savings for 24-layer GPU0 deployment
#   - Launch overhead vs register pressure trade-off
#
# Usage: bash run_fused_bench.sh
#
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH="$EXPERIMENT_DIR/fused_decode"

echo "=== P2 — Fused Decode Kernel Prototype Benchmark ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

# Build
echo "--- Building fused_decode ---"
nvcc -O3 -arch=sm_120 -o "$BENCH" "$EXPERIMENT_DIR/fused_decode.cu" \
    -lcuda -lcudart 2>&1
echo "Build complete"
echo ""

# Run
echo "--- Running benchmark ---"
"$BENCH" 2>&1 | tee "$EXPERIMENT_DIR/fused_results.txt"

echo ""
echo "=== Complete ==="
echo "Results saved to: $EXPERIMENT_DIR/fused_results.txt"
