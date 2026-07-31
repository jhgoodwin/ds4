#!/usr/bin/env bash
# O1.1 — Decode throughput benchmark
# Measures steady-state decode t/s on each speed-bench prompt
# Runs: ds4 with gen=64, ctx=1024, batch=1 at current quantization
set -euo pipefail

DS4_DIR="/opt/ds4"
MODEL="$DS4_DIR/ds4flash.gguf"
PROMPT_DIR="$DS4_DIR/speed-bench/prompts"
RESULTS_DIR="$DS4_DIR/docs/multi-gpu-tuning/experiments/decode-comparison"
mkdir -p "$RESULTS_DIR"

GPU_FLAGS="--cuda --gpu-devices 0,1"

echo "=== O1.1 — Decode Throughput Benchmark ==="
echo "Model: $MODEL"
echo "GPU flags: $GPU_FLAGS"
echo "Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo ""

# Warmup
echo "--- Warmup (ctx=4096, n=10) ---"
cd "$DS4_DIR"
./ds4 $GPU_FLAGS -m "$MODEL" -c 4096 -p "hello" -n 10 > "$RESULTS_DIR/warmup.log" 2>&1
echo "Warmup complete"
echo ""

# Run on each prompt
for pf in "$PROMPT_DIR"/*.txt; do
    pname=$(basename "$pf" .txt)
    echo "--- Benchmarking: $pname ---"
    
    # Run with gen=64, ctx=1024
    ./ds4 $GPU_FLAGS -m "$MODEL" -c 1024 --prompt-file "$pf" -n 64 \
        > "$RESULTS_DIR/${pname}_gen64_ctx1024.log" 2>&1
    
    # Parse results
    prefill=$(grep -oP 'prefill:\s*\K[\d.]+' "$RESULTS_DIR/${pname}_gen64_ctx1024.log" | tail -1)
    gen=$(grep -oP 'generation:\s*\K[\d.]+' "$RESULTS_DIR/${pname}_gen64_ctx1024.log" | tail -1)
    echo "  → prefill: ${prefill:-N/A} t/s, gen: ${gen:-N/A} t/s"
    
    # Also run with gen=128 for comparison with existing baseline
    ./ds4 $GPU_FLAGS -m "$MODEL" -c 1024 --prompt-file "$pf" -n 128 \
        > "$RESULTS_DIR/${pname}_gen128_ctx1024.log" 2>&1
    
    prefill128=$(grep -oP 'prefill:\s*\K[\d.]+' "$RESULTS_DIR/${pname}_gen128_ctx1024.log" | tail -1)
    gen128=$(grep -oP 'generation:\s*\K[\d.]+' "$RESULTS_DIR/${pname}_gen128_ctx1024.log" | tail -1)
    echo "  → (n=128) prefill: ${prefill128:-N/A} t/s, gen: ${gen128:-N/A} t/s"
    echo ""
done

echo "=== Benchmark complete ==="
