#!/usr/bin/env bash
# O1.4 — Nsight Compute HBM Utilization Profiling
#
# Profiles one decode step to measure:
# - Achieved DRAM throughput
# - L2 hit rate
# - Warp stall reasons
# - Compute utilization
#
# Usage:
#   bash run_nsight.sh                # Full profiling (saves .ncu-rep)
#   bash run_nsight.sh --light        # Lightweight profiling (faster)
#   bash run_nsight.sh --timing-only  # Skip ncu, run timing benchmark only
#
# System: Dual RTX PRO 6000 Blackwell 96GB Max-Q
#
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$EXPERIMENT_DIR"
BENCH="$BUILD_DIR/decode_step_bench"

# Compile if needed
if [[ ! -f "$BENCH" ]]; then
    echo "=== Building decode_step_bench ==="
    nvcc -O3 -arch=sm_120 -o "$BENCH" "$EXPERIMENT_DIR/decode_step_bench.cu" \
        -lcuda -lcudart -lcublas
    echo "Build complete"
    echo ""
fi

MODE="${1:-full}"

# Handle --timing-only: skip ncu entirely, just run standalone timing
if [[ "$MODE" == "--timing-only" || "$MODE" == "timing-only" ]]; then
    echo "=== Standalone timing run (--timing-only) ==="
    echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo ""
    "$BENCH" 2>&1 | tee "$EXPERIMENT_DIR/standalone_timing.txt"
    echo ""
    echo "=== Timing complete ==="
    exit 0
fi

# Ensure ncu is available (not needed for --timing-only)
if ! command -v ncu &>/dev/null; then
    echo "ERROR: ncu (Nsight Compute CLI) not found in PATH"
    echo "Install: cuda-nsight-compute-12-9 package"
    echo "Or use --timing-only to run without ncu"
    exit 1
fi

NCU_VERSION=$(ncu --version 2>&1 | head -1)
echo "=== O1.4 — Nsight Compute HBM Utilization ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "NCU: $NCU_VERSION"
echo ""

case "$MODE" in
    --light|-l|light)
        echo "--- Lightweight profiling ---"
        echo "Sections: MemoryWorkloadAnalysis, SchedulerStats, WarpStateStats"
        echo ""
        ncu --section MemoryWorkloadAnalysis \
            --section SchedulerStats \
            --section WarpStateStats \
            --page details \
            --target-processes all \
            --launch-skip 0 \
            --launch-count 1 \
            --set full:0 \
            -o "$EXPERIMENT_DIR/decode_step_light" \
            "$BENCH" 2>&1 | tee "$EXPERIMENT_DIR/ncu_light_output.txt"
        echo ""
        echo "Profiling output saved to: ${EXPERIMENT_DIR}/decode_step_light.ncu-rep"
        ;;
    *)
        echo "--- Full profiling ---"
        echo "All sections, full details"
        echo ""
        ncu --set full \
            --target-processes all \
            --launch-skip 0 \
            --launch-count 1 \
            -o "$EXPERIMENT_DIR/decode_step_full" \
            "$BENCH" 2>&1 | tee "$EXPERIMENT_DIR/ncu_full_output.txt"
        echo ""
        echo "Profiling output saved to: ${EXPERIMENT_DIR}/decode_step_full.ncu-rep"
        ;;
esac

echo ""
echo "=== Key Metrics to Inspect ==="
echo ""
echo "Open the .ncu-rep file in Nsight Compute GUI, or use ncu --import:"
echo "  ncu --import ${EXPERIMENT_DIR}/decode_step_${MODE}.ncu-rep"
echo ""
echo "Target hypotheses (PRD-2 §O1.4):"
echo "  H1: Achieved DRAM throughput = 300-600 GB/s (20-40% of 1500 GB/s peak)"
echo "  H2: Dominant warp stall = 'long scoreboard' (waiting on memory dequant dependency)"
echo "  H3: Compute utilization < 30% during Q4_K matmul kernels"
echo "  H4: L2 hit rate < 50% (weights exceed L2 capacity)"
echo ""
echo "Contrast if:"
echo "  Achieved DRAM throughput > 1000 GB/s → dequant is NOT the main limiter"
echo "  Compute utilization > 60% → decode is not memory-bound"
echo "  Warp stall 'no instruction' dominant → instruction fetch pressure"
echo ""

# Run the benchmark standalone too for timing reference
echo "=== Standalone timing run ==="
"$BENCH" 2>&1 | tee "$EXPERIMENT_DIR/standalone_timing.txt"

echo ""
echo "=== Profiling complete ==="
