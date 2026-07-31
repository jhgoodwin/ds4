#!/usr/bin/env bash
# O2.1 — Draft Model Design Space Analysis
#
# Runs analytical model to estimate achievable draft t/s for
# 100M-500M param draft models on one GPU.
#
# Usage: bash run_draft_bench.sh
#
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== O2.1 — Draft Model Design Space Analysis ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""
echo "Method: Analytical model calibrated against measured HBM BW"
echo "  HBM read peak: 1502 GB/s [measured: compute_peak_bench]"
echo "  F16 BW util:   ~61% [measured: q4_dequant_bench]"
echo ""

# Run analysis script
python3 "$EXPERIMENT_DIR/analyze_draft_throughput.py" 2>&1 | \
    tee "$EXPERIMENT_DIR/draft_bench_results.txt"

echo ""
echo "=== Complete ==="
echo "Results saved to: $EXPERIMENT_DIR/draft_bench_results.txt"
echo "JSON results:    $EXPERIMENT_DIR/draft_model_analysis.json"
