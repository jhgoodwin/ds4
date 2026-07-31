#!/usr/bin/env bash
# DSpark Acceptance Rate Benchmark — Rerun with n>=1024
#
# Previous run at n=128 failed: capture needs ~2.8s (~183 tokens at 65 t/s)
# to initialize hidden-state capture (prop_cache=2868ms from research-log.md).
# This run uses n=1024 to ensure capture completes.
#
# Usage: bash run_dspark_acceptance_v2.sh
#
# System: Dual RTX PRO 6000 Blackwell 96GB Max-Q
#
set -euo pipefail

DS4_DIR="/opt/ds4"
MODEL="$DS4_DIR/ds4flash.gguf"
DSPARK_GGUF="/opt/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf"
PROMPT_DIR="$DS4_DIR/speed-bench/prompts"
RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RESULTS_DIR"

GPU_FLAGS="--cuda --gpu-devices 0,1"

echo "=== DSpark Acceptance Rate Benchmark v2 (n=1024) ==="
echo "Model: $MODEL"
echo "DSpark: $DSPARK_GGUF"
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

cd "$DS4_DIR"

if [[ ! -f "$DSPARK_GGUF" ]]; then
    echo "ERROR: DSpark GGUF not found at $DSPARK_GGUF"
    exit 1
fi

# Warmup — short run to prime GPU state
echo "--- Warmup (ctx=4096, n=10, hello) ---"
DS4_DSPARK_STATS=1 DS4_DSPARK_SPEC_LOG=1 timeout 120 \
./ds4 --dspark --mtp "$DSPARK_GGUF" $GPU_FLAGS -m "$MODEL" -c 4096 -p "hello" -n 10 \
    > "$RESULTS_DIR/warmup_v2.log" 2>&1
echo "Warmup complete"
echo ""

# Run on each speed-bench prompt with n=1024
for pf in "$PROMPT_DIR"/*.txt; do
    pname=$(basename "$pf" .txt)
    echo "--- Benchmarking: $pname (n=1024) ---"
    echo "  Expected duration: ~15s at ~65 t/s baseline, may be slower with DSpark"
    
    # n=1024 ensures capture has time to complete (~2.8s)
    # --temp 0 for greedy (reproducible)
    DS4_DSPARK_STATS=1 DS4_DSPARK_SPEC_LOG=1 timeout 300 \
    ./ds4 --dspark --mtp "$DSPARK_GGUF" $GPU_FLAGS -m "$MODEL" -c 4096 \
        --prompt-file "$pf" -n 1024 --temp 0 --seed 42 \
        > "$RESULTS_DIR/${pname}_dspark_n1024.log" 2>&1
    
    echo "  Run complete"
    echo ""
done

# Parse results
echo "=== Results Summary ==="
echo ""

for lf in "$RESULTS_DIR"/*_dspark_n1024.log; do
    pname=$(basename "$lf" .log | sed 's/_dspark_n1024//')
    echo "--- $pname ---"
    
    prefill=$(grep -oP 'prefill:\s*\K[\d.]+' "$lf" | tail -1)
    gen=$(grep -oP 'generation:\s*\K[\d.]+' "$lf" | tail -1)
    echo "  Prefill: ${prefill:-N/A} t/s"
    echo "  Generation: ${gen:-N/A} t/s"
    
    # Parse DSpark stats
    dspark_line=$(grep "DSpark stats" "$lf" | tail -1)
    if [[ -n "$dspark_line" ]]; then
        echo "  DSpark stats: $dspark_line"
        
        cycles=$(echo "$dspark_line" | grep -oP 'cycles=\K\d+')
        proposed=$(echo "$dspark_line" | grep -oP 'proposed=\K\d+')
        accepted=$(echo "$dspark_line" | grep -oP 'accepted_draft=\K\d+')
        accept_rate=$(echo "$dspark_line" | grep -oP 'accept_rate=\K[\d.]+')
        avg_accept=$(echo "$dspark_line" | grep -oP 'avg_accept=\K[\d.]+')
        
        echo "  Cycles: ${cycles:-N/A}"
        echo "  Proposed: ${proposed:-N/A}"
        echo "  Accepted: ${accepted:-N/A}"
        echo "  Accept rate: ${accept_rate:-N/A}%"
        echo "  Avg accept/cycle: ${avg_accept:-N/A}"
        
        # Timing breakdown
        propose=$(echo "$dspark_line" | grep -oP 'propose=\K[\d.]+')
        prop_cache=$(echo "$dspark_line" | grep -oP 'prop_cache=\K[\d.]+')
        prop_chain=$(echo "$dspark_line" | grep -oP 'prop_chain=\K[\d.]+')
        verify=$(echo "$dspark_line" | grep -oP 'verify=\K[\d.]+')
        target=$(echo "$dspark_line" | grep -oP 'target=\K[\d.]+')
        
        echo "  Timing (ms): propose=${propose:-N/A}, prop_cache=${prop_cache:-N/A}, prop_chain=${prop_chain:-N/A}, verify=${verify:-N/A}, target=${target:-N/A}"
        
        # Net savings
        net_saved=$(echo "$dspark_line" | grep -oP 'net_saved=\K[\d.]+')
        echo "  Net saved: ${net_saved:-N/A} ms"
    else
        echo "  No DSpark stats found (capture may not have completed)"
        grep -c "DSpark" "$lf" || true
    fi
    
    echo ""
done

# Generate JSON results
python3 "$RESULTS_DIR/analyze_dspark_acceptance_v2.py" --results-dir "$RESULTS_DIR"

echo ""
echo "=== Benchmark complete ==="
