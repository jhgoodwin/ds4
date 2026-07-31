# PRD-2: Code Generation Throughput Optimization

## Purpose

Scope research into achieving 5000+ t/s on dual RTX PRO 6000 Blackwell 96GB Max-Q for code generation workloads. Address the 8.2× gap between measured 68.5 t/s decode and the 555 t/s HBM roofline ceiling [derived: roofline-analysis.md §Gap Analysis], then project further past the roofline via speculative decoding, template expansion, and CPU/GPU hybrid techniques. Identify which hypotheses are testable. No conclusions asserted from untested strategies.

## System Under Test

Same hardware as PRD.md §2. No changes.

Reference: [PRD.md §2](PRD.md) for full SUT table.

Key parameters for this analysis:

| Parameter | Value | Source |
|---|---|---|
| GPU0 layers | 0-23 + embedding (82.6 GB) | [measured: speculative-decode-multi-gpu.md §2-GPU Placement Context] |
| GPU1 layers | 24-42 + output head (71.5 GB) | [measured: speculative-decode-multi-gpu.md §2-GPU Placement Context] |
| Decode steady-state | 68.5 t/s | [measured: roofline-analysis.md §Pipeline Decode Baseline] |
| Per-step decode time | 14.7 ms | [derived: 1/68.5 t/s] |
| GPU0 compute per step | ~7.0 ms (48%) | [derived: roofline-analysis.md §Pipeline Component Breakdown] |
| GPU1 compute per step | ~5.5 ms (37%) | [derived: roofline-analysis.md §Pipeline Component Breakdown] |
| HBM roofline (GPU0) | ~555 t/s | [derived: roofline-analysis.md §Gap Analysis, 1500 GB/s / 2.7 GB] |
| Overhead ceiling | ~450 t/s | [derived: roofline-analysis.md §Gap 5, 1/2200µs] |
| DSpark draft chain | 1-3 ms (est) | [hypothesis: speculative-decode-multi-gpu.md §Key Finding] |
| Weight format | Q4_0 block (32 elements, 2×F16 scales) | [measured: PRD.md §4.5] |
| Model total weights | 153 GiB (Q4_0) | [derived: PRD.md §6.1 + GPU VRAM usage] |

## Research Objectives

### O1 — Validate Quantization Dequant Overhead

The roofline analysis attributes the dominant gap to Q4_0 dequant overhead [hypothesis: roofline-analysis.md §Residual Gap]. Q4_0 stores 32 4-bit values per block, with 2 F16 scale factors + 1 F16 min — that's 16 bytes payload + 6 bytes metadata = 22 bytes for 32 values, or 5.5 bits/value. Dequant on read: for each block, load metadata (2×F16 = 4 bytes + 2 bytes min), then for each value: multiply-add scale.

**Hypothesis**: Q4_0 dequant adds 2-4× effective memory traffic because:

1. Block format overhead: 22 bytes to represent 32 values = 5.5 bits/value vs 4 bits naive. Metadata is 6 bytes/block = 27% overhead [derived: Q4_0 block format].
2. Scale load per block: reading 2 F16 scales per 32 values triggers 4× F16 loads (2 scales × 2 bytes each) on every block read — these are not coalesced with value loads [hypothesis].
3. Dequant ALU: for each of 32 values, F16 multiply + F16 add = ~2 FLOPs per value. At 5.2 GB/token weight read, that's ~1.3B values × 2 FLOPs = ~2.6 GFLOPs dequant compute per token — negligible vs 148.7 TFLOPS peak (<0.002%) [derived: model parameter count × 2 FLOPs / value].
4. **Implicit HBM read amplification**: Because dequant is done on-the-fly, the read pattern is: read block metadata → compute → read value bytes → dequant. This is a read-modify pattern where every weight byte must be read AND processed. The dequant pipeline stalls on the extra ALU ops, reducing effective bandwidth utilization [hypothesis].

**Experiment O1.1 — Q4_0 vs F16 decode throughput comparison**

| Element | Detail |
|---|---|
| Hypothesis | If dequant is the dominant gap, F16 decode throughput will be closer to roofline than Q4_0. Q4_0 throughput fraction of F16 throughput ≈ Q4_0_effective_bytes / F16_bytes. |
| Test | Run ds4 decode at Q4_0, Q8_0 (block size 32, 8-bit + scale), and F16 on same model (same architecture, different weight formats). Measure t/s decoded, gen=64, ctx=1024, batch=1. |
| Predicted | F16 throughput ≥ Q8_0 throughput × (9/4.5) (bytes ratio) but less due to HBM bandwidth being the same regardless of format. Actually: F16 uses 16 bits/weight = 2× bytes of Q8_0 = 4× bytes of Q4_0. But HBM bandwidth is fixed. So F16 should be SLOWER in t/s because it reads more bytes per token. Corrected prediction: F16 t/s ≈ Q4_0 t/s × (effective_hbm_utilization_F16 / effective_hbm_utilization_Q4_0). If dequant overhead is the dominant gap, Q4_0 will have lower effective HBM utilization than F16 (because dequant cycles stall HBM reads). So F16 throughput will be HIGHER than Q4_0 throughput, despite reading 4× more weight bytes. |
| Contrast | If F16 t/s ≈ Q4_0 t/s × (bytes_per_token_Q4_0 / bytes_per_token_F16) ≈ Q4_0 t/s × 0.25, then dequant overhead is negligible and the gap is elsewhere. If F16 t/s > Q4_0 t/s, dequant overhead is confirmed dominant. |
| Learning Rate | High. Single experiment rules out or confirms the primary hypothesis from roofline analysis. ~1 hr setup + run. |

**Experiment O1.2 — Q4_0 weight read micro-benchmark**

| Element | Detail |
|---|---|
| Hypothesis | Effective memory bandwidth when reading Q4_0 weights with on-the-fly dequant is lower than raw F16 read bandwidth by a factor of 2-4×. |
| Test | Write CUDA micro-benchmark: allocate a large Q4_0 buffer on GPU (~2 GiB). Kernel reads blocks, dequantizes to F16, accumulates values (to prevent dead-code elimination). Compare elapsed time vs a kernel that reads raw F16 buffer of same logical size. Use Nsight Compute to measure achieved DRAM throughput. |
| Predicted | Q4_0 effective BW = 400-750 GB/s (vs 1500 GB/s F16 read) [hypothesis: 2-4× dequant overhead]. |
| Contrast | If Q4_0 effective BW > 1200 GB/s, dequant overhead is small (<25%). If <400 GB/s, dequant overhead is extreme (3.75×+). |
| Learning Rate | High. Directly measures the hypothesized bottleneck. ~2 hrs write + run + analyze. |

**Experiment O1.3 — cuBLAS matmul Q4_0 vs F16 TFLOPS**

| Element | Detail |
|---|---|
| Hypothesis | Quantized matmul (Q4_0 weights, F16 activations) achieves lower effective TFLOPS than F16 matmul at typical decode shapes (4096×2048, 4096×4096, 2048×2048) due to dequant overhead in the matmul kernel. |
| Test | Use cuBLAS (or CUTLASS) to measure TFLOPS of: (a) F16 matmul with F16 weights, (b) simulated Q4_0 matmul where weights are dequantized on-the-fly. Compare to theoretical peak (148.7 TFLOPS). |
| Predicted | Q4_0 matmul achieves 30-50% of F16 matmul TFLOPS at same shape [hypothesis: dequant overhead reduces effective arithmetic intensity]. F16 matmul itself is memory-bound at these shapes (arithmetic intensity < 99 FLOPs/byte). |
| Contrast | If Q4_0 matmul achieves >80% of F16 matmul TFLOPS, dequant overhead is small for matmul specifically. If <30%, dequant overhead dominates even in matmul. |
| Learning Rate | Medium. ~3 hrs setup (CUTLASS or custom kernel). |

**Experiment O1.4 — Nsight Compute HBM utilization during decode**

| Element | Detail |
|---|---|
| Hypothesis | During decode, achieved DRAM throughput is significantly below HBM peak (1500 GB/s) due to: (a) stalls from dequant data dependency, (b) non-coalesced scale reads, (c) instruction fetch pressure. |
| Test | Profile one decode step with Nsight Compute (--section MemoryWorkloadAnalysis --section SchedulerStats). Measure: achieved DRAM throughput, L2 hit rate, warp stall reasons, compute utilization. |
| Predicted | Achieved DRAM throughput: 300-600 GB/s (20-40% of peak) [hypothesis]. Compare to prediction that dequant-free F16 decode would achieve 800-1200 GB/s (53-80% of peak). |
| Contrast | If achieved DRAM throughput >1000 GB/s (67% of peak) during decode, dequant is NOT the main bandwidth limiter — look elsewhere (kernel launch, sync). |
| Learning Rate | Very high. Direct measurement resolves the open question from roofline-analysis.md. ~4 hrs (Nsight setup, profiling, analysis). |

### O2 — Decompose the 5000 t/s Claim

User intuition (with 100K hours expertise caveat): 5000+ t/s should be achievable for code generation using fused kernels, minimal I/Os, proper pipelining, clever speculative decoding, parallel predict ahead, short circuit evaluation, language server, redirecting cells, and parallel CPU use.

#### O2.1 — HBM Roofline Bound

HBM bandwidth bounds maximum model inference throughput.

For pipeline-parallel (current config):
- GPU0 reads ~2.7 GB/token [derived: roofline-analysis.md]
- Max t/s = 1500 GB/s / 2.7 GB = 555 t/s [derived: roofline-analysis.md]

For tensor-parallel (hypothetical, each GPU reads half the expert weights):
- Each GPU reads ~1.35 GB/token (half of 2.7 GB on GPU0's pipeline partition, but layers are split differently in TP) [hypothesis]
- Approx: each GPU reads ~2.6 GB/token of weights. Total usable BW = 2 × 1500 GB/s = 3000 GB/s. Max t/s = 3000 / 2.6 = 1154 t/s [derived].

**Hypothesis**: To exceed 555 t/s (pipeline-parallel) or 1154 t/s (tensor-parallel), techniques must produce multiple output tokens per model inference step. The fraction of tokens requiring full model inference determines achievable throughput.

#### O2.2 — Analytical Model

Define throughput as:

```
t/s = 1 / (t_model_per_token + t_model_free_tokens_per_token)
```

Where:
- `t_model_per_token` = time the base model spends on one output token (including verification overhead)
- `t_model_free_tokens_per_token` = free tokens generated per model-inferred token

Actually more precisely:

```
total_output_tokens = N_model_tokens + N_free_tokens
total_time = N_model_tokens × t_model_step (spec decode) + T_free_generation

t/s = (N_model_tokens + N_free_tokens) / (N_model_tokens × t_model_step + T_free_generation)
```

Where t_model_step includes the full base model decode + draft + verify for speculative decode.

Let me build the model systematically.

**Baseline**: 68.5 t/s, all tokens from full model.

**Technique 1: Speculative Decoding with Small Draft Model**

Draft model: 100M params, F16 = 200 MB, 8-layer transformer, n_embd=512, n_heads=8.

Draft throughput on one GPU (memory-bandwidth bound):
- Each draft token: read ~200 MB weights from HBM (if memory-bandwidth bound)
- Max draft t/s = 1500 GB/s / 0.2 GB = 7500 t/s [derived: HBM roofline]
- But draft model is smaller — it may be compute-bound. 100M params × 2 FLOPs/param/token = 200 MFLOPs/token. At 148.7 TFLOPS peak: max 148.7 / 0.0002 = 743,500 t/s. Not compute-bound. [derived]
- Actual draft t/s likely bandwidth-limited: ~7500 t/s for a memory-bound draft model. [hypothesis]

Draft chain cost: 8 tokens × (200µs per draft token at 5000 t/s, or more realistically 133µs at 7500 t/s) = 1.07 ms for 8-token draft chain [derived: 1/7500 × 8].

Acceptance rate regime:
- Low acceptance (code): 0.5-0.7 per token, chain length 8 → expected accepted = 1/(1-0.5) = 2 tokens (geometric) [hypothesis]
- High acceptance (code): 0.7-0.9 per token → expected accepted = 1/(1-0.7) = 3.3 tokens [hypothesis]
- Very high (boilerplate): 0.9-0.95 per token → expected accepted = 1/(1-0.9) = 10 tokens [hypothesis]

**CORRECTION (2026-07-27)**: The following assumptions in this section are falsified by source code analysis:
- **GPU0-only verify**: Source (ds4.c line 33895-33915) shows `metal_graph_verify_suffix_tops_impl` iterates ALL 43 layers with `g->placement[il+1]` tier switching — full cross-GPU, not GPU0-only.
- **GPU0-only draft**: Source (ds4.c line 59334-59340) shows draft runs on `dspark_exec_tier` (default GPU1, selected by free VRAM, override via `DS4_DSPARK_EXEC_TIER=0`).
- **t_verify ≈ 7.0ms**: False because verify uses both GPUs. Measured DSpark batch-verify: 8.6ms/call (research-log.md). Full decode verify (single-token path): ~14.7ms.

**Corrected speculative decode timing** (from measured DSpark stats, research-log.md):
- t_draft = 11.4ms (prop_chain=228ms/20cycles, DSpark on default GPU1 exec tier)
- t_verify = 8.6ms (batch-encode across all 43 layers)
- t_overhead = 1.7ms (unchanged)
- **t_spec_step = 21.7ms** → 46.1 steps/s

| Draft Acceptance | Steps/s | Accepted/Step | t/s (corrected) |
|---|---|---|---|
| Low (0.5/token) | 46.1 | 2.0 | 92 |
| Medium (0.7/token) | 46.1 | 3.3 | 152 |
| High (0.9/token) | 46.1 | 10.0 | 461 |
| Perfect (8/8 chain) | 46.1 | 8.0 | 369 |

[corrected: t_step=21.7ms from DSpark measured values. Re-run needed on speed-bench prompts.]

**Technique 2: Template Expansion / Language Server**

Language server deterministically generates boilerplate tokens: imports, class definitions, function signatures, docstrings, closing braces, semicolons, annotations.

For code generation tasks:
- Django varbit: ~30% of output tokens are predictable boilerplate (class definition, imports, test structure) [hypothesis: from reading prompt]
- Slack clone: ~40% boilerplate (HTML boilerplate, channel list rendering, message list rendering, emoji picker UI) [hypothesis]
- Flappy Bird: ~20% boilerplate (Canvas setup, game loop structure, score display) [hypothesis]

If language server produces B tokens instantly per model step, and remaining tokens come from speculative decode:

```
total_tokens_per_model_step = B + accepted_draft_tokens
t/s = (B + accepted) / t_step
```

| Task | B (free tokens) | Accepted | Total | t/s |
|---|---|---|---|---|
**NOTE**: The table below was derived from t_step=11.7ms (falsified GPU0-only verify assumption). Corrected t_step=21.7ms from DSpark measured values. Re-run after O2.X + DSpark acceptance rate benchmark.

| Task | B (free tokens) | Accepted | Total | t/s (requires rerun) |
|---|---|---|---|---|
| All values | — | — | — | [requires rerun with corrected t_step] |

[derived: original t_step=11.7ms invalid — verify is full 43-layer cross-GPU, not GPU0-only (ds4.c line 33895)]

**Hypothesis (corrected)**: 5000+ t/s may be reachable for high-boilerplate code if language server generates ~100+ deterministic tokens per step and acceptance rate >= 0.7, assuming corrected t_step of ~21.7ms. Without >100 free tokens/step, celing is ~2500 t/s. Requires measurement via O2.X and DSpark acceptance rate benchmark.

**Technique 3: Parallel Predict-Ahead**

At branch points (if/else, try/except, switch), evaluate both branches with bounded compute.

For code generation, branch points occur at predictable frequencies:
- Conditionals: every 15-25 tokens in algorithm-heavy code [hypothesis]
- Exception handling: every 30-50 tokens [hypothesis]
- Pattern matching: every 20-40 tokens [hypothesis]

Evaluating both branches costs 2× compute for the branch-related portion. For full model, this doubles weight reads for the layers affected. Estimated 20-50% compute increase per branch point [hypothesis].

Worst case: every step is a branch point → 2× compute → t/s halves.
Best case: branch points are rare → 5-15% compute increase → t/s drops 5-15%.

**Hypothesis**: This technique trades compute for acceptance rate gain. Net effect unknown without measurement — may help if branch ambiguity causes low acceptance rate, but may hurt if branch costs exceed acceptance gains.

**Technique 4: Short Circuit Evaluation**

For expressions, evaluate common subexpressions once. For function calls with known signatures, bypass model for parameter names.

In code generation:
- Variable names: often derived from context (parameter names, field names). A deterministic name resolver could predict them with high accuracy [hypothesis].
- Function arguments: positional arguments to known functions are predictable from API signature [hypothesis].
- Import names: fully deterministic given package name [hypothesis].

If 10-20% of tokens are name-bound and resolvable via language server + AST analysis: those tokens become free [hypothesis].

**Hypothesis**: This is largely subsumed by language server / template expansion (Technique 2). Estimated additional contribution: 5-10% more free tokens [hypothesis].

**Technique 5: CPU Parallel Generation**

Offload boilerplate to CPU while GPU handles semantic tokens. CPU can generate 1000+ t/s for simple patterns.

CPU generation for dead-simple patterns: F-string templates, brace matching, semicolons, indentation. At 5 GHz × 1 token/cycle (trivial) → 5 billion t/s theoretical. Real limit: PCIe transfer to move tokens to GPU context + Python overhead. CPU-side: ~10M t/s for trivial pattern expansion [hypothesis]. GPU-side: must insert into KV cache, requiring ~1µs per token for cache append + attention update [hypothesis].

If 100 tokens generated on CPU per model step: 100 × 1µs = 100µs to integrate into GPU state = negligible vs corrected ~21.7ms step time [derived].

CPU generation adds 50-100 free tokens per model step [hypothesis].

**NOTE**: Original step time 11.7ms used falsified GPU0-only verify assumption. Corrected: ~21.7ms (DSpark measured). The 100µs KV insertion cost remains negligible either way, but percent overhead drops from 0.85% to 0.46%.

**Technique 6: Fused Kernels**

Eliminate kernel launch overhead. Current: ~500µs per step on 43+ kernels [estimated: roofline-analysis.md §Pipeline Component Breakdown].

With fused single-kernel decode: 1 kernel launch instead of 43+. Saves ~480µs [derived: 43 launches × ~12µs each minus one fused launch].

Impact: reduces t_step by ~480µs (absolute, independent of baseline). Percent gain depends on actual t_step: ~2.2% at corrected 21.7ms, ~3.3% at 14.7ms decode. [derived: 480µs / baseline].

Fused kernels alone are incremental — predicted ~2-4% gain [derived: 480µs savings / baseline t_step]. Fused kernels alone cannot reach 5000 t/s — other techniques (speculative decode, template expansion) are required.

#### O2.3 — Cumulative Speedup Model (Hypothetical — All Values from Untested Hypotheses)

| Technique | Step Time | Free Tokens/Step | Accepted/Step | t/s |
|---|---|---|---|---|
| Baseline (no speculation) | 14.7 ms [measured] | 0 | 1.0 | 68 [measured] |
| + Speculative decode (draft 8, acc=0.7) | 11.7 ms [hypothesis] | 0 | 3.3 [hypothesis] | 282 [derived] |
| + Language server (boilerplate) | 11.7 ms [hypothesis] | 50 [hypothesis] | 3.3 [hypothesis] | 4556 [derived] |
| + CPU parallel generation | 11.7 ms [hypothesis] | 100 [hypothesis] | 3.3 [hypothesis] | 8830 [derived] |
| + Fused kernels (-0.5ms step) | 11.2 ms [hypothesis] | 100 [hypothesis] | 3.3 [hypothesis] | 9195 [derived] |

[derived: cumulative model, each row adds technique. Every hypothesis tagged — see §Hypotheses to Test for falsifying experiments.]

**Hypothesis**: If free tokens ≥ 50/step and acceptance rate ≥ 0.7, high-boilerplate code could reach 4500+ t/s [derived]. If free tokens < 20 or acceptance rate < 0.6, throughput remains in speculative decode range (200-500 t/s) [hypothesis].

#### O2.4 — Critical Experiment

**The single highest-leverage experiment**: Measure the fraction of code output tokens that can be deterministically predicted from the prompt + language server, without model inference.

This determines the ceiling for language-server-based acceleration. The experiment will rule out competing models of token predictability (see §Hypotheses to Test).

```
Experiment O2.X: For each speed-bench prompt, have a language server 
(LSP + AST parser) generate the maximum completions deterministically.
Count tokens that match the model's output.

Method:
1. Run ds4 on each prompt, record full output with token IDs
2. Run language server on each prompt with context, generate 
   deterministic completions for each possible token position
3. Count matching tokens
4. Measure: fraction of output tokens predictable from syntax/context alone
```

### O3 — Code-Specific Speculative Decoding Opportunity

#### O3.1 — Token Predictability by Code Type

Using the three speed-bench prompts as workload archetypes:

**Django Varbit** (~50 LoC, ~250-500 tokens)

| Token Category | Estimated Fraction | Examples |
|---|---|---|
| Syntactically forced | 25-30% | `{`, `}`, `;`, `\n`, indentation, `def`, `class`, `import`, `from` |
| Name-bound | 15-20% | `VarbitField`, `VarbitValue`, `BitLengthExpression` (from prompt context) |
| Pattern-repeating | 15-20% | Test methods with identical structure, SQL operand methods (similar signatures) |
| Semantically creative | 30-45% | Business logic for each operand, SQL generation specifics |

[hypothesis: based on reading the prompt]

**Analysis**: ~60-70% of tokens are predictable from syntax + context + patterns. High speculative decode potential.

**Flappy Bird** (~100-200 LoC, ~500-1000 tokens)

| Token Category | Estimated Fraction | Examples |
|---|---|---|
| Syntactically forced | 15-20% | `{`, `}`, `;`, `function`, `var`, `return` |
| Name-bound | 10-15% | `bird`, `pipe`, `score`, `gameOver`, `flap` (from prompt) |
| Pattern-repeating | 10-15% | Collision checks, update loops (similar structure repeated) |
| Semantically creative | 50-65% | Game physics (gravity, velocity), rendering logic, collision math |

[hypothesis: based on reading the prompt]

**Analysis**: ~35-50% predictable. Game physics and interaction logic require creative decisions. Lower speculative decode potential.

**Slack Clone** (~500-1000 LoC, ~2500-5000 tokens)

| Token Category | Estimated Fraction | Examples |
|---|---|---|
| Syntactically forced | 20-25% | `{`, `}`, `;`, `import`, `const`, `function`, JSX tags |
| Name-bound | 15-20% | `Channel`, `Message`, `User`, `EmojiPicker`, `Sidebar` (from prompt) |
| Pattern-repeating | 25-30% | Message rendering for each simulated user, channel list items, emoji grid items |
| Semantically creative | 25-40% | Message animation, emoji picker UX, responsive layout, accessibility |

[hypothesis: based on reading the prompt]

**Analysis**: ~60-75% predictable. UI-heavy code with high structure repetition. Highest speculative decode potential.

#### O3.2 — Draft Model Architecture for Code

**Requirements**:
- Small enough to fit in VRAM alongside 153 GiB model (2×96 GiB GPUs → available headroom: ~12 GiB on GPU0, ~24 GiB on GPU1) [derived: PRD.md §2.3]
- Fast enough: >5000 draft t/s
- Accurate enough: >0.7 acceptance rate per token for code (REQUIREMENT FALSIFIED — measured 5% acceptance. PRD-3 §E1 isolates model quality vs pipeline loss before this requirement can be re-evaluated.)

**Candidate Architecture**:

| Component | Spec | Bytes |
|---|---|---|
| n_embd | 512 | — |
| n_layers | 8 | — |
| n_heads | 8 | — |
| Vocab | 128256 (same as base) | — |
| Embedding | 128256 × 512 = 65.6M params | 131 MB (F16) |
| Per transformer layer | 4 × 512 × 512 ≈ 1.05M params | 2.1 MB (F16) |
| 8 transformer layers | 8.4M params | 16.8 MB (F16) |
| Output head (tied or separate) | 512 × 128256 = 65.6M params | 131 MB (F16) |
| + KV cache (ctx=32768, 8 layers) | 2 × 8 × 128 × 32768 × 2B = 134 MB | 134 MB (F16) |
| **Total** | ~140M params | **~413 MB** (F16) |

[derived: standard small transformer architecture]

**VRAM impact**: 413 MB fits in available headroom on EITHER GPU. Trivial.

**With vocabulary sharing** (tie input embedding with output head): 
- 131 + 131 = 262 MB → 262 MB total for weights + 134 MB KV = 396 MB [derived]
- Saves ~17 MB

**With AST-aware features**:
- Add AST grammar constraint: mask logits to valid next tokens based on code grammar. This is a deterministic post-processing step on the output logits — no extra model parameters [hypothesis].
- Implementation: compile parser grammar to token transition matrix. At each step, mask out syntactically invalid tokens before argmax. Reduces vocabulary from 128256 to ~10-50 valid tokens per position [hypothesis].
- Benefit: increases acceptance rate because draft tokens are guaranteed to be syntactically valid. No extra compute.

**With language server feedback**:
- Before draft generation, query LSP for: available symbols (variable names, function names, types in scope). Use these to bias logits for name tokens [hypothesis].
- Implementation: maintain token→symbol mapping for known symbols. During draft generation, boost logits for name tokens that match available symbols.
- Benefit: increases acceptance rate for name-bound tokens. Estimated +5-10% acceptance rate [hypothesis].

#### O3.3 — Expected Acceptance Rate per Code Type

**FALSIFIED (2026-07-27)**: DSpark-acceptance-rate [measured: research-log.md] on a simple greeting prompt measured accept_rate = 5.00% (3/60). All per-code-type predictions below are based on an unvalidated draft model quality assumption. The 5% rate suggests model quality, not token predictability, is the dominant factor. PRD-3 §E1 isolates model quality vs pipeline loss. Tables preserved as historical hypotheses, with rejection markers.

Based on token predictability analysis (HYPOTHESES — ALL REFUTED by measured 5% acceptance):

| Code Type | Expected Acceptance Rate | Expected Accepted Chain | Status |
|---|---|---|---|
| Django Varbit (boilerplate) | 0.70-0.85/token | 3.3-6.7 | REFUTED — measured 5% on greeting prompt |
| Slack Clone (UI-heavy) | 0.75-0.85/token | 4.0-6.7 | REFUTED — same draft model, same GGUF |
| Flappy Bird (algorithm) | 0.50-0.65/token | 2.0-2.9 | REFUTED — same draft model, same GGUF |
| Average code generation | 0.65-0.78/token | 2.9-4.5 | REFUTED — same draft model, same GGUF |

[hypothesis: based on token predictability estimates from O3.1 — all falsified by measured 5% acceptance rate]

With AST-aware masking (+0.05-0.10) and language server feedback (+0.03-0.05):

| Code Type | Enhanced Acceptance Rate | Enhanced Accepted Chain |
|---|---|---|
| Django Varbit | 0.78-0.93 | 4.5-14.3 |
| Slack Clone | 0.83-0.93 | 5.9-14.3 |
| Flappy Bird | 0.58-0.75 | 2.4-4.0 |

[derived: base + AST + LSP boosts — requires draft model with non-zero base acceptance first]

#### O3.4 — Speedup Curve

Formula from task:

```
t/s = base_model_tps × (chain_length × acceptance_rate) / (1 + draft_overhead_ratio)
```

Where:
- `base_model_tps` = 68.5 (baseline decode without speculation) [measured: roofline-analysis.md]
- `chain_length` = number of draft tokens proposed per step (8 typical)
- `acceptance_rate` = probability each draft token is accepted (not same as per-token acceptance rate — this is the aggregate acceptance rate including early rejection)
- Actually the formula has issues. Let me use the precise form:

```
expected_accepted = sum_{k=0}^{chain-1} P(accept >= k+1)
                 = (1 - p^{chain}) / (1 - p)    if p = per-token acceptance rate [geometric series]

For p=0.7, chain=8: (1-0.7^8)/(1-0.7) = (1-0.058)/0.3 = 3.14 tokens
For p=0.8, chain=8: (1-0.8^8)/(1-0.8) = (1-0.168)/0.2 = 4.16 tokens
For p=0.9, chain=8: (1-0.9^8)/(1-0.9) = (1-0.430)/0.1 = 5.70 tokens
```

[derived: standard speculative decoding acceptance formula]

```
**FALSIFIED** (GPU0-only verify assumption disproven by source code ds4.c line 33895):
```
t_spec_step = 7.0ms + 3.0ms + 1.7ms = 11.7ms  ← used GPU0-only verify, contradicts source
```

The historical wrong table is preserved below for audit. Corrected values use full 43-layer cross-GPU verify:

**Corrected** (verify = full 43-layer cross-GPU, draft on dspark_exec_tier):
```
t_spec_step = t_draft + t_verify + t_overhead
            = 11.4ms + 8.6ms + 1.7ms = 21.7ms  [measured: research-log.md]
t/s = expected_accepted / 21.7ms
```

| Acceptance Rate p | Expected Accepted | t/s (corrected 21.7ms) |
|---|---|---|
| 0.50 | 2.00 | 92 |
| 0.60 | 2.44 | 112 |
| 0.70 | 3.14 | 145 |
| 0.80 | 4.16 | 192 |
| 0.85 | 4.85 | 224 |
| 0.90 | 5.70 | 263 |
| 0.95 | 6.62 | 305 |

[derived: chain=8, t_spec_step=21.7ms from measured DSpark values. Re-run needed on speed-bench prompts.]

**With base model optimization** (fused kernels, balanced pipeline -- NOTE: original table used falsified GPU0-only verify assumption):

**Historical (falsified) prior table**: Verify was assumed GPU0-only (~7.0ms). Source code (ds4.c line 33895-33915) confirms verify iterates ALL 43 layers with cross-device placement. Verify always runs full model regardless of optimization.

**Corrected** -- t_verify cannot drop below full 43-layer cross-GPU cost regardless of base model optimization. At minimum: full pipeline decode time (~14.7ms). Using measured DSpark batch-verify (8.6ms for multi-token), the minimum t_spec_step is t_draft + 8.6ms + 1.7ms, where t_draft depends on draft model size and device.

Corrected table requires running E3 (DSpark throughput comparison on speed-bench prompts) to establish actual t_verify and t_draft for the target workload.

**With base model optimization** (fused kernels, balanced pipeline → t_base_GPU0 = 5.0ms, t_draft = 2.0ms, t_overhead = 1.0ms → t_spec_step = 8.0ms):

| Acceptance Rate p | Expected Accepted | t/s |
|---|---|---|
| 0.70 | 3.14 | 393 |
| 0.80 | 4.16 | 520 |
| 0.85 | 4.85 | 606 |
| 0.90 | 5.70 | 713 |
| 0.95 | 6.62 | 828 |

[derived: optimized parameters]

**With both optimization + 50 free tokens from language server**:

**NOTE**: This table derived from falsified GPU0-only verify assumption (t_spec_step=8.0ms). Corrected t_spec_step with full cross-GPU verify is ~14.7ms minimum. Re-run after O2.X + DSpark acceptance rate benchmark.

| Task | Free Tokens | Spec t/s | Total t/s |
|---|---|---|---|
| All values | — | — | [requires rerun — prior values used GPU0-only verify assumption, falsified by ds4.c line 33895-33915] |

**5000 t/s threshold (corrected)**: Even with 50 free tokens/step and 3.14 spec tokens/step at corrected t_step (~21.7ms DSpark measured): ~2450 t/s. Optimistic t_step (~14.7ms, matching non-spec decode) yields ~3400 t/s. 5000 t/s requires >100 free tokens/step or t_step < 12ms.

### O4 — Practical Bottleneck Analysis

#### O4.1 — VRAM Headroom

| Component | GPU0 | GPU1 | Source |
|---|---|---|---|
| Total VRAM | 97.2 GiB | 97.2 GiB | [measured: roofline-analysis.md] |
| Model weights (Q4_0) | 82.6 GiB | 71.5 GiB | [measured: speculative-decode-multi-gpu.md] |
| KV cache (ctx=32768) | ~1-2 GiB | ~1-2 GiB | [hypothesis: PRD.md §4.6.2] |
| cuBLAS workspace | ~0.5 GiB | ~0.5 GiB | [hypothesis: PRD.md §4.9] |
| Scratch/temp | ~1 GiB | ~1 GiB | [hypothesis] |
| **Used** | ~85 GiB | ~74 GiB | [derived] |
| **Available headroom** | **~12 GiB** | **~23 GiB** | [derived] |

Headroom on GPU0: 12 GiB — sufficient for draft model (~400 MB) + language server cache (~500 MB) + headroom.

Headroom on GPU1: 23 GiB — sufficient for additional KV cache sessions or expanded context.

**Constraint**: If context window increases (ctx > 32768), KV cache grows. At ctx=131072, KV cache ~8-10 GiB on each GPU [hypothesis: 4× ctx growth]. Would consume most of GPU0's headroom.

#### O4.2 — KV Cache Cost for Speculative Decode

Speculative decode requires:
1. **Support model KV ring**: stores KV cache entries for support model layers (draft context). Size depends on: layer count × n_heads × d_head × context_length × format.
   - For 8 draft layers, 8 heads, d_head=64, ctx=256 (draft only needs recent context): 2 × 8 × 8 × 64 × 256 × 2B = 4.2 MB [derived].
   - Negligible.

2. **Draft KV cache**: must store KV for draft chain tokens during verification. If draft chain is 8 tokens, need KV for 8 tokens per layer. 8 × 12.5 KB per layer ≈ 100 KB per layer, 8 layers × 100 KB = 800 KB [derived]. Negligible.

3. **Base model KV cache**: unchanged. Already fits [measured: PRD.md].

**KV cache is not a constraint** for speculative decoding on this system.

#### O4.3 — PCIe

At 68.5 t/s: PCIe utilization = 0.002% [measured: roofline-analysis.md §Gap 3].
At 5000 t/s: PCIe utilization for activation transfers = 0.002% × (5000/68.5) = 0.15% [derived].
Still negligible.

**Hypothesis**: PCIe is not a bottleneck at any throughput achievable on this hardware [derived: PCIe utilization < 1% even at 100× projected throughput].

#### O4.4 — CPU Round-Trip

Current Markov argmax has CPU fallback path [measured: speculative-decode-multi-gpu.md §Architecture Recap].

At 5000 t/s: one token every 200µs. CPU round-trip latency for argmax: <1µs for simple operation (dot product + find max) [hypothesis]. Fraction of step time: 0.5%.

GPU-only path is preferable to avoid kernel launch overhead, but the CPU path cost is negligible at this scale.

**Constraint**: If CPU path requires cudaMemcpy for logits (100KB+), latency = 3-5µs + PCIe transfer for partials. At 5000 t/s, this consumes 1.5-2.5% of step time [derived]. Acceptable.

#### O4.5 — Power Throttling

RTX PRO 6000 Blackwell Max-Q: 300W default power limit [measured: roofline-analysis.md]. Sustained inference at full load may trigger thermal throttling after 30-60s [hypothesis: PRD.md §6.2].

At 5000 t/s with speculative decoding + language server: GPU compute time per output token is much less than peak because most tokens bypass the model. GPU duty cycle could drop from 100% (no speculation) to 20-40% (most tokens generated by CPU/language server, GPU only for semantic tokens) [hypothesis].

**Hypothesis**: Lower GPU duty cycle from template expansion and CPU parallel generation may reduce thermal load, decreasing throttling risk [hypothesis]. To be measured by sustained run in experiment H7.

#### O4.6 — Memory Bandwidth Wall

The fundamental constraint: even with all techniques combined, the GPU must still read ~2.7 GB from HBM for each model inference step. At 1500 GB/s, that's a hard 1.8ms floor for GPU0's forward pass [derived: 2.7 GB / 1500 GB/s].

If acceptance rate is 0.7 (3.14 tokens/step), each model inference step produces 3.14 tokens + free tokens. Best case: 50 free tokens + 3.14 spec tokens = 53.14 tokens per 21.7ms (corrected DSpark measured step time) = 2449 t/s. This assumes the free tokens cost zero HBM bandwidth, which is true for template expansion but NOT for CPU-parallel tokens that need KV cache insertion (costs ~1µs per token = negligible) [hypothesis].

**NOTE**: Original 11.7ms step time and 4541 t/s derived from falsified GPU0-only verify assumption. Source code confirms verify = full 43-layer cross-GPU (ds4.c line 33895). Corrected step time from DSpark measured values: ~21.7ms.

**Hard floor on model inference**: ~555 t/s if running model for every token.
**With speculation**: ~92-305 t/s for pure speculative decode (corrected: chain=8, t_step=21.7ms).
**With speculation + templates**: 750-4800 t/s depending on boilerplate fraction [corrected estimate, requires rerun].

## Proposed Experiments (Prioritized)

| Priority | Experiment | Expected Info Gain | Wall Time (hrs) | Learning Rate |
|---|---|---|---|---|
| **P0** | O1.4 — Nsight Compute HBM utilization during decode | Resolves primary hypothesis from roofline-analysis. Measures achieved DRAM throughput, stall reasons. High contrast: >1000 GB/s = dequant not dominant; <500 GB/s = dequant confirmed. | 4 | Very high. Single experiment rules confirms/refutes the #1 hypothesis. |
| **P0** | O1.1 — Q4_0 vs F16 decode comparison | If F16 > Q4_0 t/s, dequant confirmed as dominant gap. If F16 < Q4_0 t/s, gap is elsewhere. | 1 | Very high. Fastest direct test. |
| **P0** | O2.X — Token predictability analysis on speed-bench prompts | Determines upper bound on template expansion benefit. If >40% tokens deterministic, 5000+ t/s thesis is viable. | 3 | Very high. Gates all template-expansion experiments. |
| **P0** | O1.2 — Q4_0 weight read micro-benchmark **[RUN — see research-log]** | Measures effective BW of Q4_0 read+dequant vs raw F16. | 0 | High. Result: Q4_K=639 GB/s, dequant factor=1.43×. H1 FALSIFIED. |
| **P1** | DSpark acceptance rate benchmark **[PARTIAL — needs rerun on speed-bench]** | Greeting prompt: accept_rate=5.00% (n=60 proposed). Speed-bench: capture didn't complete (n=128 insufficient). | 4 | High. Needs n >= 1024 for capture init (~2.8s). PRD-3 §E1-E3 cover this. |
| **P1** | O1.3 — cuBLAS matmul Q4_0 vs F16 TFLOPS | Is dequant within matmul the problem, or is it in other kernels? | 3 | Medium. Narrower scope than O1.2/O1.4. |
| **P2** | O2.1 — Profiling experiment: draft model design space | Measure achievable draft t/s for 100M-500M param draft models on one GPU. | 6 | Medium. Depends on O2.X results. |
| **P2** | Language server integration POC | Write deterministic template expansion for one prompt (Django varbit). Count free tokens generated. | 8 | Medium. Validates O2.3 model. |
| **P2** | Fused kernel prototype | Implement fused decode for GPU0's 24 layers. Measure launch overhead reduction vs register pressure cost. | 16 | Low. Marginal gain (4% of throughput). |

[derived: learning rate = expected hypotheses ruled out per hour]

## Hypotheses to Test

| # | Hypothesis | Predicted Outcome | Falsifying Experiment |
|---|---|---|---|
| H1 | Q4_0 dequant overhead is the dominant gap between 68 t/s and 555 t/s roofline | F16 decode t/s > Q4_0 decode t/s on same hardware (O1.1). Achieved DRAM throughput < 500 GB/s during Q4_0 decode (O1.4). | If F16 t/s ≤ Q4_0 t/s, dequant is NOT dominant. If achieved DRAM throughput > 1000 GB/s, dequant is small. |
| H2 | 5000+ t/s is achievable for high-boilerplate code generation on this hardware | Language server generates >40% of output tokens deterministically (O2.X). Template-free tokens reach 305 t/s with speculative decode (corrected O3.4: full cross-GPU verify, t_step=21.7ms). | If deterministic tokens < 20% of output, 5000 t/s requires unrealistic acceptance rates (>0.95/token). Even with 50 free tokens/step, corrected t_step yields ~2450 t/s — below 5000 t/s threshold without further architectural changes. |
| H3 | Code has 0.7-0.85 acceptance rate per token for speculative decoding | Measured acceptance rate on speed-bench prompts ≥ 0.7 for boilerplate-heavy code, ≥ 0.5 for algorithm-heavy code. | If measured acceptance rate < 0.5 for boilerplate code, the hypothesis overestimates code predictability. |

**FALSIFICATION (2026-07-27)**: DSpark-acceptance-rate [measured: research-log.md] on a simple greeting prompt measured 5.00% acceptance rate (draft accepted 3 of 60 proposed tokens). PRD-3 §E1 isolates whether this is a model quality issue (likely, at 5.99 GiB support GGUF on 153 GiB base) or pipeline loss. Until E1 is run, H3 is FALSIFIED for all prompt types on this support model.
| H4 | AST-aware grammar masking improves acceptance rate by 0.05-0.10 per token | With grammar constraint, acceptance rate increases by ≥ 0.05. | If increase < 0.03, grammar masking provides marginal benefit (model already good at syntax). |
| H5 | Fusion of decode kernels yields ≤ 5% throughput improvement | Fused kernel reduces t_spec_step by ≤ 500µs (absolute, independent of baseline). Percent gain: ~2-4% depending on actual t_step. | If fusion saves > 1ms, kernel launch overhead was underestimated in roofline analysis. |
| H6 | CPU parallel generation can produce 50+ free tokens per model step | Implementation generates 50+ deterministic tokens from speed-bench prompts without model inference. | If < 20 tokens, CPU parallel generation has limited value for these workloads. |
| H7 | Power throttling is not a bottleneck for speculative decode workloads | GPU clock stays above 3.0 GHz during sustained speculative decode (30 min run). Duty cycle < 60% reduces thermal load. | If clock drops below 2.5 GHz within 60 seconds, power/thermal is a primary constraint. |

## Constraints & Risks

| Constraint | Impact | Mitigation |
|---|---|---|
| Language server integration complexity | Template expansion requires real-time code analysis. LSP round-trip latency may add 1-5ms per query. | Cache common patterns. Batch deterministic generation. Async pipeline. |
| Draft model training requirement | Current ds4 does not include a code-specialized draft model. Would need training/fine-tuning. | Start with existing small LM (e.g., SantaCoder 100M). Fine-tune on speed-bench code outputs. |
| Acceptance rate is model-dependent | The draft model's quality determines acceptance rate. A 100M param draft may have lower acceptance than a larger one. | Sweep draft model sizes (100M-500M params). Measure acceptance rate vs draft speed trade-off. |
| Verification correctness on 2-GPU | DSpark verification runs full 43-layer cross-GPU via `metal_graph_verify_suffix_tops_impl` (ds4.c line 33895) with `g->placement[il+1]` tier switching. Prior claim of GPU0-only [PRD-2] was a misreading of speculative-decode-multi-gpu.md, which explicitly says cross-device. | Verify that batch-encode logits match single-token decode logits at the same position (PRD-3.md D3.8). |
| Template expansion may produce wrong code | Deterministic generation may get context wrong (variable shadowing, scoping issues). | Conservative: only expand templates when grammar certainty >99.9%. Use verification to catch errors. |
| 5000 t/s requires 100+ tokens/step | At 100 free tokens + 3.3 spec tokens, the model must sustain 103.3 tokens per ~21.7ms step (corrected). This means the model is idle >95% of the time. Even with 100 free tokens, corrected throughput is ~4750 t/s — marginal. | Pipeline: generate free tokens on CPU while GPU runs model. Overlap CPU and GPU work. Requires validating that free token generation does not exceed model inference rate. |
| PCIe latency for CPU→GPU token transfer | At 5000 t/s, 103.3 tokens/step × 8KB activation (F16, one token's hidden state) = ~826 KB per step over PCIe. Still < 1% of PCIe bandwidth. | No concern. |
| KV cache insertion latency | At 5000 t/s, 100+ tokens/step need KV cache update. Current KV cache append is O(1) per token [PRD.md §4.6.2]. Insertion cost: ~1µs per token. 100 tokens × 1µs = 100µs = 0.9% of step time. | Acceptable. |

## Open Questions

1. **Does dequant overhead dominate the gap?** RESOLVED. O1.2 measured dequant at 1.43× [research-log.md], not 2-4×. Dequant contributes ~1300 µs of 14.7ms step — significant but not dominant. Optimization priority shifts to sync/launch overhead, attention kernel efficiency, and pipeline imbalance. O1.4 (Nsight Compute) should now target these causes, not dequant.

2. **What is the actual token-by-token predictability of code?** O2.X must be run before any language server work. If code is less predictable than estimated, the 5000 t/s thesis fails.

3. **Can draft on GPU0 overlap with verify on GPU1?** Current design runs draft on dspark_exec_tier (dynamically selected: follows output head placement by default, GPU1 on this system) and verify across both GPUs sequentially [measured: ds4.c line 59334, 33895]. No overlap. Potential: move support weights to GPU0 (DS4_DSPARK_EXEC_TIER=0) so draft runs on GPU0 while GPU1 handles verify. This would save ~11.4ms/step (full draft time) if fully overlapped vs sequential 21.7ms total.

4. **What is the optimal draft model size for code generation?** 100M vs 200M vs 500M params. Trade-off: larger = higher acceptance rate but slower draft speed. Need to find the optimal point where t/s = expected_accepted / (t_draft + t_verify) is maximized.

5. **Can CPU parallel generation be implemented within ds4's current architecture?** ds4 has CPU fallback for Markov argmax [measured: speculative-decode-multi-gpu.md]. Adding CPU-side template expansion requires new code paths for: (a) pattern detection, (b) expansion, (c) KV cache integration. Feasibility unknown.

6. **Is GPU parallelism possible for draft+verify?** If GPU0 does draft while GPU1 does verify, and they alternate: step N: GPU0 computes base model (layers 0-23) → GPU1 finishes (layers 24-42), then GPU0 does draft while GPU1 does verify. Could save ~11.4ms per step (full draft time) if fully overlapped [hypothesis: speculative-decode-multi-gpu.md §Overlap Design Proposal]. Not blocked by TP expert sharding on this 2-GPU pipeline-parallel config (g->tp_world=1, no-op suspend) [measured: speculative-decode-multi-gpu.md §TP Expert Sharding: No Conflict on 2-GPU]. Blocked by: (1) support weights cached on GPU1 by default, (2) verify requires both GPUs, (3) capture HC availability.

7. **Is the unified draft model approach better than separate draft+verify phases?** If draft and verify use the same small model (100M params) for both proposing AND rejecting tokens, the verify cost drops from ~14.7ms (full base model) to ~0.1ms (draft model forward pass). But acceptance rate may drop because the draft model is weaker than the base model at verification. Trade-off to evaluate. Note correction: full base model verify costs ~14.7ms (or 8.6ms batch), not 7.0ms as previously assumed.

8. **What if code generation doesn't need attention to distant context?** Code structure is dominated by local patterns (within 100 tokens). KV cache for draft model only needs ctx=256, not 32768. This means draft model KV cache is 128× smaller than base model KV cache, reducing memory pressure further.

## References

- [PRD.md](PRD.md) — Research PRD for multi-GPU tuning
- [roofline-analysis.md](roofline-analysis.md) — Throughput ceilings and gap analysis
- [speculative-decode-multi-gpu.md](speculative-decode-multi-gpu.md) — DSpark overlap analysis
- [tuning-guide.md](tuning-guide.md) — Tuning decision framework
- [scalability-analysis.md](scalability-analysis.md) — Scaling to N GPUs
- [GROUND-RULES.md](GROUND-RULES.md) — Experimental methodology
- [dspark.md](../code/concepts/dspark.md) — DSpark architecture specification
- [multi-gpu-pipeline.md](../code/concepts/multi-gpu-pipeline.md) — Pipeline architecture
- [tp.md](../code/modules/tp.md) — Tensor parallelism
- Speed-bench prompts: `/opt/ds4/speed-bench/prompts/` — Code generation workloads
