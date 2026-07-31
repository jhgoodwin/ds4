#!/usr/bin/env bash
# DSpark Acceptance Rate Benchmark
# Measures: per-cycle acceptance rate, avg accepted per cycle, timing breakdown
# on each speed-bench prompt.
#
# Usage: bash run_dspark_acceptance.sh
#
set -euo pipefail

DS4_DIR="/opt/ds4"
MODEL="$DS4_DIR/ds4flash.gguf"
DSPARK_GGUF="/opt/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf"
PROMPT_DIR="$DS4_DIR/speed-bench/prompts"
RESULTS_DIR="/opt/ds4/docs/multi-gpu-tuning/experiments/dspark-acceptance"
mkdir -p "$RESULTS_DIR"

GPU_FLAGS="--cuda --gpu-devices 0,1"

echo "=== DSpark Acceptance Rate Benchmark ==="
echo "Model: $MODEL"
echo "DSpark: $DSPARK_GGUF"
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

cd "$DS4_DIR"

# Check DSpark GGUF exists
if [[ ! -f "$DSPARK_GGUF" ]]; then
    echo "ERROR: DSpark GGUF not found at $DSPARK_GGUF"
    exit 1
fi

# Warmup
echo "--- Warmup (ctx=4096, n=10, hello) ---"
DS4_DSPARK_STATS=1 DS4_DSPARK_SPEC_LOG=1 \
./ds4 --dspark --mtp "$DSPARK_GGUF" $GPU_FLAGS -m "$MODEL" -c 4096 -p "hello" -n 10 \
    > "$RESULTS_DIR/warmup.log" 2>&1
echo "Warmup complete"
echo ""

# Run on each prompt
for pf in "$PROMPT_DIR"/*.txt; do
    pname=$(basename "$pf" .txt)
    echo "--- Benchmarking: $pname ---"
    
    # Run with gen=128, ctx=4096
    DS4_DSPARK_STATS=1 DS4_DSPARK_SPEC_LOG=1 \
    ./ds4 --dspark --mtp "$DSPARK_GGUF" $GPU_FLAGS -m "$MODEL" -c 4096 \
        --prompt-file "$pf" -n 128 \
        > "$RESULTS_DIR/${pname}_dspark.log" 2>&1
    
    # Parse main stats from stderr (redirected to stdout)
    prefill=$(grep -oP 'prefill:\s*\K[\d.]+' "$RESULTS_DIR/${pname}_dspark.log" | tail -1)
    gen=$(grep -oP 'generation:\s*\K[\d.]+' "$RESULTS_DIR/${pname}_dspark.log" | tail -1)
    
    echo "  → prefill: ${prefill:-N/A} t/s, gen: ${gen:-N/A} t/s"
    
    # Parse DSpark stats
    dspark_line=$(grep "DSpark stats" "$RESULTS_DIR/${pname}_dspark.log" | tail -1)
    if [[ -n "$dspark_line" ]]; then
        echo "  → DSpark: $dspark_line"
    else
        echo "  → No DSpark stats found (may not have been enabled)"
    fi
    
    echo ""
done

# Parse and tabulate results
echo "=== Summary ==="
echo ""

echo "| Prompt | Gen (t/s) | Cycles | Drafts Proposed | Drafts Accepted | Accept Rate | Avg Accept/Cycle | Saved (ms) | Extra (ms) | Net Saved (ms) |"
echo "|--------|-----------|--------|-----------------|----------------|-------------|-----------------|------------|------------|----------------|"

for pf in "$PROMPT_DIR"/*.txt; do
    pname=$(basename "$pf" .txt)
    log="$RESULTS_DIR/${pname}_dspark.log"
    
    gen=$(grep -oP 'generation:\s*\K[\d.]+' "$log" | tail -1)
    
    dspark_line=$(grep "DSpark stats" "$log" | tail -1)
    if [[ -z "$dspark_line" ]]; then
        echo "| ${pname} | ${gen:-N/A} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        continue
    fi
    
    cycles=$(echo "$dspark_line" | grep -oP 'cycles=\K\d+')
    proposed=$(echo "$dspark_line" | grep -oP 'proposed=\K\d+')
    accepted=$(echo "$dspark_line" | grep -oP 'accepted_draft=\K\d+')
    accept_rate=$(echo "$dspark_line" | grep -oP 'accept_rate=\K[\d.]+')
    avg_accept=$(echo "$dspark_line" | grep -oP 'avg_accept=\K[\d.]+')
    saved=$(echo "$dspark_line" | grep -oP 'saved=\K[\d.]+')
    net_saved=$(echo "$dspark_line" | grep -oP 'net_saved=\K[\d.]+')
    extra=$(echo "$dspark_line" | grep -oP 'spec_total=\K[\d.]+')
    
    echo "| ${pname} | ${gen:-N/A} | ${cycles:-N/A} | ${proposed:-N/A} | ${accepted:-N/A} | ${accept_rate:-N/A}% | ${avg_accept:-N/A} | ${saved:-N/A} | ${extra:-N/A} | ${net_saved:-N/A} |"
done

echo ""
echo "=== Complete ==="
