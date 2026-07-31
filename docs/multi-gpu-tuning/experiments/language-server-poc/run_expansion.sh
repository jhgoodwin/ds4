#!/usr/bin/env bash
# Language Server Integration POC — Template Expansion
#
# Runs deterministic template expansion on speed-bench prompts.
# Measures free tokens generated per prompt.
#
# Usage: bash run_expansion.sh
#
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Language Server Integration POC ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

# Run on all prompts
python3 "$EXPERIMENT_DIR/template_expander.py" --all 2>&1 | \
    tee "$EXPERIMENT_DIR/expansion_output.txt"

echo ""
echo "=== Complete ==="
echo "Results saved to: $EXPERIMENT_DIR/expansion_output.txt"
echo "JSON:            $EXPERIMENT_DIR/template_expansion_results.json"
