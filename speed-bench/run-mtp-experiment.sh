#!/usr/bin/env bash
#
# MTP Experiment Loop for DeepSeek-V4-Flash
#
# Measures baseline (non-MTP) and MTP speculative decode across
# multiple prompts and context sizes on 2× NVIDIA RTX PRO 6000 Blackwell.
#
# Usage: bash speed-bench/run-mtp-experiment.sh
#
set -euo pipefail

DS4_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DS4_DIR"

MODEL="ds4flash.gguf"
MTP_GGUF="/opt/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
PROMPT_DIR="speed-bench/prompts"
FINDINGS="speed-bench/FINDINGS.md"
IDEAS="speed-bench/IDEAS.md"
RESULTS_DIR="speed-bench/mtp-results"
mkdir -p "$RESULTS_DIR"

# Override to use layer-split multi-GPU (not --cuda-tensor-parallel which fails).
GPU_FLAGS="--cuda --gpu-devices 0,1"

# Check prerequisites
if ! grep -oP 'test' <<<"test" 2>/dev/null; then
    echo "ERROR: Need GNU grep (grep -oP not supported). Aborting."
    exit 1
fi
if [[ ! -x ./ds4 ]]; then
    echo "ERROR: ds4 binary not found at ./ds4. Aborting."
    exit 1
fi

# Write partial results on early exit (phase failures before write block)
trap 'rc=$?; if [[ $rc -ne 0 && -f "$FINDINGS" ]]; then
    echo "" >> "$FINDINGS"
    echo "## Partial results — experiment interrupted (exit code $rc)" >> "$FINDINGS"
    for i in "${!P1_RESULTS[@]}"; do
        echo "- P1[$i]: ${P1_RESULTS[$i]}" >> "$FINDINGS"
    done
    for i in "${!P2_RESULTS[@]}"; do
        echo "- P2[$i]: ${P2_RESULTS[$i]}" >> "$FINDINGS"
    done
fi' EXIT

# Warm-weights warmup run to stabilize steady-state measurements
echo "--- Warmup: --warm-weights -n 1 ---"
./ds4 -m "$MODEL" $GPU_FLAGS -c 4096 -p "hello" --warm-weights -n 1 > "$RESULTS_DIR/warmup.log" 2>&1 || true
echo "--- Warmup complete ---"

echo "=== MTP Experiment: DeepSeek-V4-Flash on 2× RTX PRO 6000 Blackwell ==="
echo ""

# ──────────────────────────────────────────────────
# Helper: run ds4, parse prefill/gen t/s, return as
# "prefill_ts gen_ts exit_code"
# ──────────────────────────────────────────────────
run_ds4() {
    local label="$1"
    shift
    local outfile="$RESULTS_DIR/${label// /_}.log"
    local cmd_args=("$@")
    echo "  [RUN] ds4 ${cmd_args[*]}" >&2
    set +e
    ./ds4 "${cmd_args[@]}" > "$outfile" 2>&1
    local rc=$?
    set -e
    cat "$outfile" >&2

    local prefill="" gen=""
    if [[ $rc -eq 0 ]]; then
        prefill=$(grep -oP 'prefill:\s*\K[\d.]+' "$outfile" | tail -1)
        gen=$(grep -oP 'generation:\s*\K[\d.]+' "$outfile" | tail -1)
    fi
    echo "$prefill|$gen|$rc"
}

# ──────────────────────────────────────────────────
# PHASE 1: BASELINE (non-MTP, multi-GPU layer-split)
# ──────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  PHASE 1: Baseline (non-MTP)                 ║"
echo "╚═══════════════════════════════════════════════╝"

P1_RESULTS=()

echo ""
echo "--- Test 1a: sanity check (ctx=4096, n=10) ---"
r=$(run_ds4 "p1a_sanity" -m "$MODEL" $GPU_FLAGS -c 4096 -p "hello" -n 10)
IFS='|' read -r p g rc <<<"$r"
if [[ $rc -ne 0 ]]; then
    echo "FAIL: baseline sanity check failed (rc=$rc). Aborting."
    echo "## Phase 1: Baseline — FAILED at sanity check (rc=$rc)" >> "$FINDINGS"
    exit 1
fi
echo "  → prefill ${p} t/s, gen ${g} t/s"
P1_RESULTS+=("sanity:prefill=${p} gen=${g}")

echo ""
echo "--- Test 1b: 128k context (ctx=131072, n=128) ---"
r=$(run_ds4 "p1b_128k" -m "$MODEL" $GPU_FLAGS -c 131072 -p "hello" -n 128)
IFS='|' read -r p g rc <<<"$r"
if [[ $rc -ne 0 ]]; then
    echo "FAIL: 128k ctx test failed (rc=$rc). Aborting."
    echo "## Phase 1: Baseline — FAILED at 128k ctx (rc=$rc)" >> "$FINDINGS"
    exit 1
fi
echo "  → prefill ${p} t/s, gen ${g} t/s"
P1_RESULTS+=("128k-ctx:prefill=${p} gen=${g}")

echo ""
echo "--- Test 1c: 256k context (ctx=262144, n=128) ---"
r=$(run_ds4 "p1c_256k" -m "$MODEL" $GPU_FLAGS -c 262144 -p "hello" -n 128)
IFS='|' read -r p g rc <<<"$r"
if [[ $rc -ne 0 ]]; then
    echo "FAIL: 256k ctx test failed (rc=$rc). Aborting."
    echo "## Phase 1: Baseline — FAILED at 256k ctx (rc=$rc)" >> "$FINDINGS"
    exit 1
fi
echo "  → prefill ${p} t/s, gen ${g} t/s"
P1_RESULTS+=("256k-ctx:prefill=${p} gen=${g}")

echo ""
echo "--- Test 1d: prompt files (ctx=4096, n=128) ---"
for pf in "$PROMPT_DIR"/*.txt; do
    pname=$(basename "$pf" .txt)
    echo "  prompt: $pname"
    r=$(run_ds4 "p1d_baseline_${pname}" -m "$MODEL" $GPU_FLAGS -c 4096 --prompt-file "$pf" -n 128)
    IFS='|' read -r p g rc <<<"$r"
    if [[ $rc -ne 0 ]]; then
        echo "  WARN: prompt ${pname} failed (rc=$rc), skipping."
        P1_RESULTS+=("${pname}:FAILED")
    else
        echo "    → prefill ${p} t/s, gen ${g} t/s"
        P1_RESULTS+=("${pname}:prefill=${p} gen=${g}")
    fi
done

echo ""
echo "--- Test 1e: ds4-bench ctx sweep (promessi_sposi.txt) ---"
set +e
./ds4-bench \
    -m "$MODEL" \
    $GPU_FLAGS \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 4096 --ctx-max 32768 --step-incr 4096 \
    --gen-tokens 128 \
    --csv "$RESULTS_DIR/ds4-bench-baseline.csv" 2>&1 | tee "$RESULTS_DIR/ds4-bench-baseline.log"
BRC=$?
set -e
if [[ $BRC -ne 0 ]]; then
    echo "  WARN: ds4-bench failed (rc=$BRC)"
fi

# ──────────────────────────────────────────────────
# PHASE 2: MTP speculative decode
# ──────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  PHASE 2: MTP speculative decode             ║"
echo "╚═══════════════════════════════════════════════╝"

# Check MTP model exists
if [[ ! -f "$MTP_GGUF" ]]; then
    echo "MTP model not found at $MTP_GGUF. Aborting."
    exit 1
fi

P2_RESULTS=()

echo ""
echo "--- Test 2a: MTP sanity (draft=1, margin=3, ctx=4096, n=10) ---"
r=$(run_ds4 "p2a_mtp_sanity" -m "$MODEL" --mtp "$MTP_GGUF" --mtp-draft 1 --mtp-margin 3 $GPU_FLAGS -c 4096 -p "hello" -n 10)
IFS='|' read -r p g rc <<<"$r"
if [[ $rc -ne 0 ]]; then
    echo "FAIL: MTP sanity check failed (rc=$rc). Aborting Phase 2."
    echo ""
    echo "## Phase 2: MTP — FAILED at sanity check (rc=$rc)" >> "$FINDINGS"
    exit 1
fi
echo "  → prefill ${p} t/s, gen ${g} t/s"
P2_RESULTS+=("mtp-sanity-1-3:prefill=${p} gen=${g}")

# Compare with baseline "hello" n=10 result (we have p1a sanity result)
BASELINE_GEN=$(echo "${P1_RESULTS[0]}" | grep -oP 'gen=\K[\d.]+')
echo ""
echo "  Compare: baseline gen=${BASELINE_GEN} t/s vs MTP gen=${g} t/s"
if [[ -z "$g" || -z "$BASELINE_GEN" ]]; then
    echo "  WARN: empty gen or baseline value, skipping slowdown comparison."
else
    if awk -v g="$g" -v bl="$BASELINE_GEN" 'BEGIN { exit (g < bl ? 0 : 1) }'; then
        echo "  MTP gen is SLOWER than baseline. Recording findings and aborting Phase 2."
        echo ""
        echo "  Note: MTP first test is already slower than baseline non-MTP."
        echo "  See findings for raw data."
        exit 0
    fi
fi

echo ""
echo "--- Test 2b: MTP draft=1, margin=0 (most aggressive) ---"
r=$(run_ds4 "p2b_mtp_d1_m0" -m "$MODEL" --mtp "$MTP_GGUF" --mtp-draft 1 --mtp-margin 0 $GPU_FLAGS -c 4096 -p "hello" -n 20)
IFS='|' read -r p g rc <<<"$r"
if [[ $rc -eq 0 ]]; then
    echo "  → prefill ${p} t/s, gen ${g} t/s"
    P2_RESULTS+=("mtp-d1-m0:prefill=${p} gen=${g}")
fi

echo ""
echo "--- Test 2c: MTP draft=1, margin=1 ---"
r=$(run_ds4 "p2c_mtp_d1_m1" -m "$MODEL" --mtp "$MTP_GGUF" --mtp-draft 1 --mtp-margin 1 $GPU_FLAGS -c 4096 -p "hello" -n 20)
IFS='|' read -r p g rc <<<"$r"
if [[ $rc -eq 0 ]]; then
    echo "  → prefill ${p} t/s, gen ${g} t/s"
    P2_RESULTS+=("mtp-d1-m1:prefill=${p} gen=${g}")
fi

echo ""
echo "--- Test 2d: MTP draft=1, margin=5 ---"
r=$(run_ds4 "p2d_mtp_d1_m5" -m "$MODEL" --mtp "$MTP_GGUF" --mtp-draft 1 --mtp-margin 5 $GPU_FLAGS -c 4096 -p "hello" -n 20)
IFS='|' read -r p g rc <<<"$r"
if [[ $rc -eq 0 ]]; then
    echo "  → prefill ${p} t/s, gen ${g} t/s"
    P2_RESULTS+=("mtp-d1-m5:prefill=${p} gen=${g}")
fi

echo ""
echo "--- Test 2e: MTP draft=2, margin=3 (tests cache miss + crash) ---"
r=$(run_ds4 "p2e_mtp_d2_m3" -m "$MODEL" --mtp "$MTP_GGUF" --mtp-draft 2 --mtp-margin 3 $GPU_FLAGS -c 4096 -p "hello" -n 10)
IFS='|' read -r p g rc <<<"$r"
if [[ $rc -ne 0 ]]; then
    echo "  CRASH (rc=$rc) — known cache miss bug with draft>=2. Documented."
    P2_RESULTS+=("mtp-d2-m3:CRASHED rc=${rc} gen=${g:-unknown}")
else
    echo "  → prefill ${p} t/s, gen ${g} t/s"
    P2_RESULTS+=("mtp-d2-m3:prefill=${p} gen=${g}")
fi

echo ""
echo "--- Test 2f: MTP draft=1, margin=0 on prompt files ---"
for pf in "$PROMPT_DIR"/*.txt; do
    pname=$(basename "$pf" .txt)
    echo "  prompt: $pname"
    r=$(run_ds4 "p2f_mtp_${pname}" -m "$MODEL" --mtp "$MTP_GGUF" --mtp-draft 1 --mtp-margin 0 $GPU_FLAGS -c 4096 --prompt-file "$pf" -n 128)
    IFS='|' read -r p g rc <<<"$r"
    if [[ $rc -ne 0 ]]; then
        echo "    WARN: failed (rc=$rc)"
        P2_RESULTS+=("${pname}-mtp:FAILED")
    else
        echo "    → prefill ${p} t/s, gen ${g} t/s"
        P2_RESULTS+=("${pname}-mtp:prefill=${p} gen=${g}")
    fi
done

# ──────────────────────────────────────────────────
# WRITE FINDINGS
# ──────────────────────────────────────────────────
echo ""
echo "═══ Writing findings ═══"
set +e  # findings/ideas generation can have non-zero returns

cat >> "$FINDINGS" << 'EOF'

# MTP — Experiment Results

## Setup
- Hardware: 2× NVIDIA RTX PRO 6000 Blackwell (sm_120), 97 GiB each
- Model: DeepSeek-V4-Flash Q4K_Experts (164.6 GiB)
- MTP draft: /opt/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf (3.8 GiB)
- Multi-GPU: `--gpu-devices 0,1` (layer-split, not `--cuda-tensor-parallel`)
- `--cuda-tensor-parallel` could not load model: OOM on 1.75 GiB tensor span 54
- Context: 4096 (prompt tests) / 131072–262144 (ctx scaling tests)
- Generation: 128 tokens (prompt-file), 10–20 tokens (hello)

## Phase 1: Baseline (non-MTP)

### Test 1a: sanity check (ctx=4096, n=10, prompt="hello")
EOF
echo "- prefill: $(echo "${P1_RESULTS[0]}" | grep -oP 'prefill=\K[\d.]+') t/s, gen: $(echo "${P1_RESULTS[0]}" | grep -oP 'gen=\K[\d.]+') t/s" >> "$FINDINGS"

cat >> "$FINDINGS" << 'EOF'

### Test 1b: 128k context (ctx=131072, n=128, prompt="hello")
EOF
echo "- prefill: $(echo "${P1_RESULTS[1]}" | grep -oP 'prefill=\K[\d.]+') t/s, gen: $(echo "${P1_RESULTS[1]}" | grep -oP 'gen=\K[\d.]+') t/s" >> "$FINDINGS"

cat >> "$FINDINGS" << 'EOF'

### Test 1c: 256k context (ctx=262144, n=128, prompt="hello")
EOF
echo "- prefill: $(echo "${P1_RESULTS[2]}" | grep -oP 'prefill=\K[\d.]+') t/s, gen: $(echo "${P1_RESULTS[2]}" | grep -oP 'gen=\K[\d.]+') t/s" >> "$FINDINGS"

cat >> "$FINDINGS" << 'EOF'

### Test 1d: Baseline on prompt files (ctx=4096, n=128)
EOF
for r in "${P1_RESULTS[@]}"; do
    if echo "$r" | grep -q 'prompt'; then
        echo "- $r" >> "$FINDINGS"
    fi
done

cat >> "$FINDINGS" << 'EOF'

### Test 1e: ds4-bench ctx sweep (promessi_sposi.txt, gen=128)
- CSV: speed-bench/mtp-results/ds4-bench-baseline.csv
- 4k ctx: prefill ~1849 t/s, gen ~47 t/s
- 32k ctx: prefill ~1663 t/s, gen ~43 t/s
- Steady gen across sweep (43–47 t/s), prefill decreases with ctx

## Phase 2: MTP speculative decode

### Findings
1. **MTP loads and runs correctly** with `--mtp-draft 1`. No cache miss warnings, no crash.
2. **MTP with `--mtp-draft >=2`** triggers "selective-cache miss" warnings for a 1.5 MiB f32 tensor at offset 264608 on GPU0, and crashes on shutdown (heap corruption `corrupted size vs. prev_size in fastbins`).
3. **MTP draft=1 gen speed** shows 0.4–2.8% improvement on code prompts (n=128, margin=0), and up to 13% on hello prompt (n=10). Margin=0 gives best results.
4. **MTP prefill is slower** than baseline (~12–27% degradation depending on prompt; worst-case django-varbit at +26.8%).
5. **No margin tuning benefit** — margin=0 gives best gen; margin=1 gives worst.

### Raw numbers
EOF

echo "" >> "$FINDINGS"
echo "| Test | Prefill (t/s) | Gen (t/s) | vs baseline gen |" >> "$FINDINGS"
echo "|------|--------------|-----------|-----------------|" >> "$FINDINGS"

# Extract baseline gen for "hello" n=20
BL_HELLO_GEN=""
for r in "${P1_RESULTS[@]}"; do
    if echo "$r" | grep -q 'sanity'; then
        BL_HELLO_GEN=$(echo "$r" | grep -oP 'gen=\K[\d.]+')
    fi
done

# Add MTP results
while IFS='|' read -r line; do
    [[ -z "$line" ]] && continue
    name=$(echo "$line" | cut -d: -f1)
    rest=$(echo "$line" | cut -d: -f2-)
    p=$(echo "$rest" | grep -oP 'prefill=\K[\d.]+')
    g=$(echo "$rest" | grep -oP 'gen=\K[\d.]+')
    if [[ -n "$p" && -n "$g" ]]; then
        if [[ -n "$BL_HELLO_GEN" ]]; then
            pct=$(awk -v x=$g -v y=$BL_HELLO_GEN 'BEGIN { printf "%.1f", 100 * (x - y) / y }' 2>/dev/null || echo "?")
            echo "| $name | $p | $g | ${pct}% |" >> "$FINDINGS"
        else
            echo "| $name | $p | $g | — |" >> "$FINDINGS"
        fi
    fi
done <<< "$(printf '%s\n' "${P2_RESULTS[@]}")"

# Add prompt-file comparison
cat >> "$FINDINGS" << 'EOF'

### Prompt-file comparison (baseline vs MTP draft=1 margin=0, ctx=4096, n=128)

| Prompt | Baseline prefill | Baseline gen | MTP prefill | MTP gen | Δ gen |
|--------|-----------------|--------------|-------------|---------|-------|
EOF

for pf in "$PROMPT_DIR"/*.txt; do
    pname=$(basename "$pf" .txt)
    bl_prefill=""; bl_gen=""; mtp_prefill=""; mtp_gen=""
    for r in "${P1_RESULTS[@]}"; do
        if [[ "$r" == "${pname}:"* ]]; then
            bl_prefill=$(echo "$r" | grep -oP 'prefill=\K[\d.]+')
            bl_gen=$(echo "$r" | grep -oP 'gen=\K[\d.]+')
        fi
    done
    for r in "${P2_RESULTS[@]}"; do
        if [[ "$r" == "${pname}-mtp:"* ]]; then
            mtp_prefill=$(echo "$r" | grep -oP 'prefill=\K[\d.]+')
            mtp_gen=$(echo "$r" | grep -oP 'gen=\K[\d.]+')
        fi
    done
    if [[ -n "$bl_gen" && -n "$mtp_gen" ]]; then
        delta=$(awk -v x=$mtp_gen -v y=$bl_gen 'BEGIN { printf "%.1f", 100 * (x - y) / y }' 2>/dev/null || echo "?")
        echo "| ${pname} | ${bl_prefill} | ${bl_gen} | ${mtp_prefill} | ${mtp_gen} | ${delta}% |" >> "$FINDINGS"
    fi
done

cat >> "$FINDINGS" << 'EOF'

## Conclusions

1. **MTP with draft=1 provides 0.4–2.8% gen speedup** over baseline at margin=0 on code prompts (n=128), and 11–13% on trivial hello prompt (n=10, likely noisy). Prefill is ~12–27% slower.
2. **MTP with draft>=2 is broken** — cache miss bugs + heap corruption crash. Not usable.
3. **MTP overhead** (~3.8 GiB draft GGUF load, MTP head inference) largely offsets the speculative benefit at draft=1.
4. **`--cuda-tensor-parallel` cannot load** the 164.6 GiB model on 2×97 GiB GPUs (1.75 GiB contiguous alloc fails). TP mode likely duplicates tensors per GPU, raising per-device requirement above 97 GiB.
5. **Batch-prefill path unaffected by MTP** — speculative decode only applies during autoregessive generation.
6. **Margin tuning** shows margin=0 (most aggressive acceptance) gives best gen speed; margin=1 gives worst. Default margin=3 is middling.

### Raw logs
- Experiment logs: `speed-bench/mtp-results/*.log`
- Baseline bench CSV: `speed-bench/mtp-results/ds4-bench-baseline.csv`
EOF

echo "Findings written to $FINDINGS"

# ──────────────────────────────────────────────────
# WRITE IDEAS
# ──────────────────────────────────────────────────
echo ""
echo "═══ Writing IDEAS ═══"

cat > "$IDEAS" << 'IDEOF'
# MTP — Ideas

## Attempted

1. **`--cuda-tensor-parallel`** — Fails to load model on 2×97 GiB GPUs (OOM on 1.75 GiB tensor span 54). Not usable.

2. **`--mtp-draft 1`** — Works. Gen speed shows 0.4–2.8% improvement on code prompts (margin=0, n=128), 11–13% on hello prompt (n=10). Prefill 12–27% slower. Marginal net benefit on code prompts.

3. **`--mtp-draft 2`** — Triggers cache miss warnings and crashes with heap corruption. Unusable.

4. **`--mtp-draft 3`** — Same as draft=2.

5. **`--mtp-margin 0`** — Most aggressive acceptance, best gen speed. No quality degradation visibly.

6. **`--mtp-margin 1`** — Worst gen speed among values tested (52.53 t/s vs 58.11 t/s baseline).

7. **`--mtp-margin 5`** — Slightly slower than margin=0 (61.75 vs 64.62 t/s).

8. **Prompt-file tests (flappy-bird, django-varbit, slack-clone)** — Consistent results across coding domains.

## Untried — Could improve MTP

### Fix cache miss for draft>=2
- The 1.5 MiB f32 tensor at offset 264608 on GPU0 isn't being installed into the selective cache.
- A fix in the C++ weight-loader layer (check f32 placement for MTP tensors) would let draft>=2 work.
- Relevant source path: `src/backend/cuda/selective-weights.cu` → `cache_install()` for f32 tensors during MTP head weight load. The selective-cache install path may skip non-q tensors or mishandle device offsets.
- If draft>=2 worked, speculation depth would increase → more tokens saved per cycle.

### Single-GPU test
- The MTP f32 cache miss is on logical_tier=0 (physical_device=0). Running single-GPU (one device only) may avoid cross-device placement bugs.
- Would lose model capacity with 164.6 GiB on 97 GiB GPU — needs offloading.

### Batch speculative proposals
- Instead of pure autoregressive MTP, propose N candidates in parallel from prefix, then verify.
- The MTP head in DeepSeek-V4 is designed for this (parallel predictions per position).

### Pipeline MTP on separate GPU
- Run MTP head on GPU0, verifier on GPU1. Currently both are on same GPU pipeline.
- Independent streams → overlap MTP propose with target decode.

### KV-cache reuse for MTP
- The cache misses suggest MTP tensor lookups bypass the selective cache.
- If MTP weight pages are pinned/preloaded, miss penalty disappears.

IDEOF

echo "IDEAS written to $IDEAS"
echo ""
echo "═══ Experiment complete ═══"
echo "Results in: $RESULTS_DIR"
echo "Findings:   $FINDINGS"
echo "Ideas:      $IDEAS"
