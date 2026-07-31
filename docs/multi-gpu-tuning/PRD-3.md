# PRD-3: Speculative Decode Structural Validation

## Purpose

Validate the structural correctness of speculative decoding on the 2-GPU pipeline-parallel system before optimizing for throughput. The previous research phases (PRD, PRD-2) assumed prior code was functionally correct and focused on quantifying performance gaps. This phase questions that assumption.

**Critical architectural constraint**: DSpark and legacy MTP are mutually exclusive support kinds determined by a single GGUF file loaded via `--mtp`. Detection runs in priority order — DSpark check first, fallback to MTP legacy if DSpark fails, else NONE. A GGUF with both DSpark AND MTP tensors (e.g., combined training checkpoint) is always detected as DSpark; MTP experiments unreachable. P0.0 determines which track is active.

Detection logic executed by `support_model_detect` (ds4.c line 2787):
- stages >= 3 && has_main_proj && has_markov_head && has_confidence_head → DSpark (checked first)
- mtp.0.e_proj.weight && mtp.0.h_proj.weight && mtp.0.hc_head_base.weight → MTP legacy (fallback)
- Neither → NONE

Gates: DSpark and MTP experiments cannot run in the same session. Experiments in §2.1 and 2.2 target DIFFERENT GGUFs.

**Additional gating — MTP draft token count**: `ds4_engine_options::mtp_draft_tokens` defaults to 1 (../code/concepts/mtp.md). Speculative verify loop checks `e->mtp_draft_tokens > 1` — at default, MTP speculative decode is DISABLED. To run M3.4-M3.7, explicitly set `--mtp-draft-tokens N` where N >= 2.

**Premise**: Source code analysis [source: ds4.c lines 59330-59360, 33822-33915, 32218-32278] confirms the multi-GPU plumbing code paths for DSpark and MTP exist and compile:
- DSpark draft chain switches to dspark_exec_tier (dynamically selected: output head's placement tier by default, adjustable for TP, overridable via `DS4_DSPARK_EXEC_TIER`) via `ds4_session_prepare_dspark_draft` [source: ds4.c line 59330]
- DSpark verification iterates all 43 layers with cross-device placement switching via `metal_graph_verify_suffix_tops_impl` [source: ds4.c line 33822]
- MTP CUDA backend exists: `metal_graph_eval_mtp_draft` → `metal_graph_eval_mtp_draft_from_hc` uses `ds4_gpu_*` tensor ops [source: ds4.c lines 32346, 32218]
- DSpark capture records one HC slot per target layer per decode step via `metal_graph_dspark_capture_decode_layer` [source: ds4.c line 25843]

**Note**: Code path existence (syntax-valid) does not imply functional correctness (semantic-valid). The experiments below test whether these paths produce correct results.

The 5% DSpark acceptance rate [measured: research-log.md — Experiment: DSpark-acceptance-rate (re-run with DS4_DSPARK_STATS)] has an existing hypothesis in the source: research-log.md concludes "the draft model has fundamental quality issues on this target model, independent of multi-GPU configuration." The 2868ms prop_cache is cumulative over 20 cycles [derived: DS4_DSPARK_PROP_ADD accumulator at ds4.c line 58960], dominated by a one-time seed-from-prefill (~2800ms [derived: 2868ms total prop_cache - per-cycle estimate]). Per-step draft cost is 11.4ms [measured: DS4_DSPARK_STATS from existing run] (228ms/20). See §D3.1 for derivation. The experiments below test specific hypotheses about remaining unknown causes.

**All experiments governed by [GROUND-RULES.md](GROUND-RULES.md)**.

---

## 1. System Under Test

Same hardware as PRD.md §2. No changes.

Reference: [PRD.md §2](PRD.md) for full SUT table.

---

## 2. Structural Assumptions — Test or Refute

Every assumption below is currently unvalidated. Each must pass a dedicated test before the technique can be considered viable.

### 2.1 MTP Assumptions (Legacy)

**Gate**: All MTP assumptions gated on P0.0 returning `DS4_SUPPORT_MTP_LEGACY`. If `support_kind != MTP_LEGACY`, MTP experiments M3.1-M3.7 cannot run. A DSpark GGUF does NOT enable MTP testing.

| # | Assumption | If False, The Technique | Validation Test |
|---|---|---|---|
| A1 | MTP draft head produces tokens with >0% hit rate on this model + hardware | Entirely non-functional | M3.1 — probe-mode acceptance rate |
| A2 | MTP draft head eval runs on the same device where MTP weight tensors were allocated | Produces garbage logits from cross-device weight access | M3.2 — device allocation vs eval consistency (requires discovering allocation device first — see M3.2 method) |
| A3 | MTP HC capture reads hidden state from the live decode graph without cross-device corruption | Silent data corruption | M3.3 — capture consistency check |
| A4 | MTP verification (verify_suffix_tops) produces deterministic results matching non-speculative decode | Correctness hazard | M3.4 — verify equivalence test |
| A5 | MTP frontier snapshot/restore correctly saves and restores GPU state across both devices | State corruption after partial accept | M3.5 — frontier integrity test |
| A6 | MTP spec cycle's raw SWA cache (mtp_n_raw) operates correctly when layers span 2 devices | KV cache corruption | M3.6 — cache consistency test |
| A7 | MTP draft tokens are consumed from the correct GPU buffer and not stale from previous cycles | Token reuse / hallucination | M3.7 — draft freshness test |

### 2.2 DSpark Assumptions

| # | Assumption | If False, The Technique | Validation Test |
|---|---|---|---|
| B1 | DSpark target-layer HC capture completes and produces valid data on 2-GPU | Draft proposals based on garbage HC | D3.1 — capture validity test |
| B2 | Captured HC values match source HC rows within precision of the weighted-sum reduction | Draft proposals based on reduced representation that discards needed information | D3.2 — capture value consistency |
| B3 | Stage chain eval on GPU1 produces logits from support model weights that agree with the base model's logits on the same prefix | Draft proposals diverge from base model distribution | D3.3 — stage chain logit agreement |
| B4 | Markov argmax (GPU and CPU paths) produces reproducible results across devices | Non-deterministic draft proposals | D3.4 — markov determinism test |
| B5 | Confidence threshold gating correctly stops at the right prefix length | Accepts wrong tokens / rejects good ones | D3.5 — confidence calibration test |
| B6 | DSpark support model KV ring (dspark_raw_cache) maintains consistent state across pipeline steps | KV cache corruption | D3.6 — KV ring integrity test |
| B7 | DSpark scheduler adaptive skip does not accumulate errors across multi-GPU pipeline steps | Silent correctness drift | D3.7 — scheduler alignment test |
| B8 | DSpark verify batch-encode logits match single-token decode logits for the same input | Verify rejects valid drafts or accepts invalid ones | D3.8 — verify logit equivalence |
| B9 | GPU Markov path (ds4_gpu_dspark_markov_argmax_tensor) produces same results as CPU fallback | Silent correctness difference | D3.9 — markov path equivalence |

**Gate**: All DSpark assumptions gated on P0.0 returning `DS4_SUPPORT_DSPARK`. If `support_kind != DSPARK`, DSpark experiments D3.1-D3.9, E1-E3 cannot run.

### 2.3 Multi-GPU Pipeline Assumptions (Shared)

| # | Assumption | If False, The Technique | Validation Test |
|---|---|---|---|
| C1 | Cross-device tensor allocation for speculative caches is correctly sized | Buffer overrun / OOB access | P3.1 — allocation bounds test |
| C2 | Device synchronization (cudaEvent, stream wait) between draft + verify phases is correct | Race condition | P3.2 — sync correctness test |
| C3 | DSpark draft chain on GPU1 (executor tier) produces token proposals that match the base model at a rate exceeding 0% | Zero accepted tokens | E1 — draft logit vs base model logit comparison |
| C4 | Host bounce fallback (DS4_FORCE_HOST_BOUNCE=1) does not corrupt speculative state | Data corruption on non-peer path | P3.4 — host bounce integrity test |
| C5 | The speculation code path does not silently deadlock or hang on 2-GPU | Non-termination | P3.5 — timeout survival test |

---

## 3. Experiments (Prioritized by Learning Rate)

### Phase 0a: Pre-Checks

These must pass before any speculation experiment produces interpretable results.

#### P0.0 — Support Kind Classification

| Element | Detail |
|---|---|
| Hypothesis | The GGUF file loaded via --mtp is detected by `support_model_detect` (ds4.c line 2787) as exactly one of: DSpark (stages >= 3 + main_proj + markov_head + confidence_head), MTP legacy (mtp.0.e_proj.weight + mtp.0.h_proj.weight + mtp.0.hc_head_base.weight), or NONE (no support tensors). File existence alone does not determine testability — the support kind does. |
| Context premises | System with ds4 flash model GGUF and --mtp support. One or more support GGUF files present in /opt/ds4/gguf/. |
| Predicted outcome | Each GGUF file maps to exactly one support_kind. No assumption on which kind. |
| What it informs | If support_kind == DS4_SUPPORT_DSPARK: only DSpark experiments (B1-B9, D3.1-D3.9, E1-E3) are runnable. If support_kind == DS4_SUPPORT_MTP_LEGACY: only MTP experiments (A1-A7, M3.1-M3.7) are runnable. If support_kind == DS4_SUPPORT_NONE: no speculation experiments are runnable — skip to non-speculative optimization. |
| Method | For each GGUF file matching *mtp* or *dspark* in /opt/ds4/gguf/, run: `ds4 --mtp <gguf_path> --help` and parse the support model summary line ("DSpark stages N, block M, markov rank K, targets [...]" for DSpark; "MTP legacy" text for legacy). If ds4 --help does not print support kind, inspect GGUF metadata directly: `python3 -c "import gguf; r=gguf.GGUFReader('<path>'); print(r.keys())"` and check for dspark.* vs mtp.0.* tensor names. Record path, size, and support_kind in research-log. |
| Contrast | DSpark → run DSpark track. MTP legacy → run MTP track. NONE → skip all speculation experiments. |
| Learning rate | Very high. ~5 min. Gates every downstream speculation experiment. |

#### P0.1 — Layer Split as Function of Context Size

| Element | Detail |
|---|---|
| Hypothesis | The `multi-GPU layout:` line reported by ds4 varies with ctx and scratch reservation. The split documented in speculative-decode-multi-gpu.md at ctx=32768 (GPU0: 0-23, GPU1: 24-42) may not hold at all context sizes. |
| Context premises | DS4_FORCE_HOST_BOUNCE unset. Default scratch reservation. |
| Predicted outcome | Layer boundary position as function of ctx. No direction bias. |
| What it informs | The position of the GPU0/GPU1 boundary determines which device handles layers near the split point. A shift of 1 layer changes which GPU handles target layer 23 or 24. All experiments must report the split active during measurement. |
| Method | Run ds4 with --help or short run at ctx=1024, 4096, 8192, 32768. Extract `multi-GPU layout:` line from each. Record split (e.g., "GPU0: layers 0-22 + embedding"). |
| Contrast | If all splits match the ctx=32768 measured split (0-23/24-42 from speculative-decode-multi-gpu.md), experiments can use that reference. If split shifts, experiments must report actual split. Note: the split is a measured value (ctx=32768), not an assumption inherited from PRD-2. PRD-2's falsified claims were about execution placement (verify GPU0-only, draft GPU0-only), not the layer split. |
| Learning rate | Very high. ~5 min. Prevents using wrong layer placement in downstream analysis. |

---

### Phase 0b: DSpark Root Cause Isolation

#### E1 — DSpark Draft Logit vs Base Model Logit

| Element | Detail |
|---|---|
| Hypothesis | The DSpark support model (5.6 GiB GGUF, DeepSeek-V4-Flash-DSpark-support.gguf [measured: ls -lh /opt/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf], cached on GPU1 executor tier [hypothesis: default dspark_exec_tier placement, verify via P0.1 and cudaGetDevice]) and the base model (154 GiB, DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf [measured: ls -lh /opt/ds4/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf], pipeline-parallel across both GPUs) produce different top-1 tokens for the same prompt at the same decode position. The fraction of positions where draft top-1 ≠ base top-1 is a property of the support model quality, not of multi-GPU plumbing. |
| Context premises | P0.0 returns DS4_SUPPORT_DSPARK. DSpark support GGUF loaded via --dspark --mtp. DS4_DSPARK_PROBE=1 (draft runs but not consumed). DS4_DSPARK_STATS=1. Simple prompt, gen=256, ctx=4096, temp=0, seed=42. |
| Predicted outcome | No direction bias. Report: (a) fraction of decode positions where draft[0] top-1 token = base model top-1 token, (b) fraction where draft[0] is in base model top-5, top-10, top-100, (c) mean KL(draft_logits || base_logits) at each position. |
| What it informs | If draft top-1 matches base top-1 at rate ≤ 0.05, the support model produces tokens uncorrelated with the base model. This would explain the 5% acceptance rate as a model quality issue, not a plumbing issue. If match rate > 0.50, the support model has predictive power but the speculative mechanism (capture → stage chain → confidence gate) loses it — root cause is in the pipeline, not the model. |
| Method | Run the same prompt TWICE with identical seed (42) and temp=0: (a) ds4 --dspark --mtp DSPARK_GGUF with DS4_DSPARK_PROBE=1 — log draft model logits (from metal_graph_eval_dspark_final_hidden → hc_head → base_logits path, ds4.c line 59020) and generated tokens; (b) ds4 with no speculation (same model, same prompt) — log base model logits and generated tokens. **Correct alignment**: draft at step N produces logits for position N+1 (from hidden state at N). Base model at step N+1 produces logits for position N+1 (from full model state at N+1). Align draft_logits[N+1] (from speculative run) with base_logits[N+1] (from non-speculative run at step N+1). Report match rates for top-1, top-5, top-10, top-100, and mean KL(draft_logits[N+1] || base_logits[N+1]).

**Causal bound**: match_rate from E1 measures an UPPER BOUND on acceptance rate, not an equivalent. Even with a perfect draft model, draft logits for position N+1 (computed from position N's hidden state) can differ from base model logits for position N+1 (computed from full model state at position N+1 — the base model processes the full prefix including the just-decoded token, while draft only sees hidden state before that token is fully decoded). Capture reduction (weighted sum over DS4_N_HC rows) and confidence threshold add further loss. Therefore: actual_acceptance_rate <= match_rate. If match_rate is low, model quality is definitively the cause. If match_rate is high but acceptance rate remains 5%, the loss is in the capture→propose→verify pipeline, not the model. |
| Contrast | match_rate < 0.10 → support model quality is the dominant factor. match_rate in [0.10, 0.50] → support model has partial signal but verification/confidence mechanism introduces loss. match_rate > 0.50 → support model correctly predicts the base model at rate where speculative decode could provide net gain, but current 5% acceptance implies the draft capture → propose → verify pipeline loses the signal. |
| Learning rate | Very high. ~2 hrs setup + run. Determines whether DSpark failure is model quality or pipeline loss. |

#### E2 — DSpark Per-Step Cost Breakdown

| Element | Detail |
|---|---|
| Hypothesis | The DSpark speculative path adds per-step overhead that exceeds the time saved by accepting draft tokens. Specifically: draft chain time + verify time > baseline decode time per token × accepted_tokens. |
| Context premises | DSpark stats enabled (DS4_DSPARK_STATS=1). Any prompt, gen >= 256. |
| Predicted outcome | No direction bias. Report: mean prop_chain_ms per cycle, mean verify_ms per cycle, mean baseline decode time (from non-speculative run at same config), and net_ms = (draft_ms + verify_ms) - baseline_ms × accepted_tokens. |
| What it informs | If net_ms > 0 even at acceptance rate = 1.0, speculative decode is mathematically incapable of improving throughput regardless of draft quality. |
| Method | Parse DS4_DSPARK_STATS output. Extract prop_chain_ms, verify_ms, baseline decode time per token from non-speculative run at same config. Compute: (a) draft_chain_cost per step = prop_chain / cycles, (b) verify_cost = verify / verify_calls, (c) net_total = (draft_chain_cost + verify_cost) - baseline_s_per_token × accepted_tokens, where baseline_s_per_token is the mean per-token decode time from the non-speculative run. Report net_total and net_per_accepted_token = net_total / accepted_tokens. |
| Contrast | If net_per_token > 0 for all acceptance rates ≤ 1.0, DSpark is mathematically net-negative on this hardware. If net_per_token can be negative at some acceptance rate, DSpark has a path to viability. |
| Learning rate | High. Re-run on speed-bench prompts required. Existing stats are from simple greeting prompt at 8 t/s [measured: DS4_DSPARK_STATS from existing run] (DSpark overhead regime), not representative of code generation. Estimated ~4 hrs including new run. |

### Phase 0c: Signal Detection

Before detailed testing, establish the baseline: does MTP or DSpark produce ANY correct speculative tokens? A 0% acceptance rate under probe mode means the technique is structurally broken — skip to abandonment.

#### M3.1 — MTP Probe-Mode Acceptance Rate

| Element | Detail |
|---|---|
| Hypothesis | MTP draft head (`metal_graph_eval_mtp_draft`, ds4.c line 32346) produces a draft token at each decode step. The probe mode (DS4_MTP_PROBE=1) compares draft token to the base model's actual next token. The match rate (hit_rate = mtp_probe_hit / mtp_probe_total) is unknown. |
| Context premises | P0.0 returns DS4_SUPPORT_MTP_LEGACY. Same SUT. Model: ds4flash.gguf (Q4_0). MTP GGUF loaded via --mtp. DS4_MTP_PROBE=1. DS4_MTP_SPEC_LOG=1. Simple greeting prompt, gen=256, ctx=4096. |
| Predicted outcome | No direction bias. hit_rate measured in [0%, 100%]. Report central estimate with 95% CI. |
| What it informs | hit_rate = 0% → draft head produces tokens that never match the base model at the same position (within measurement noise). hit_rate in (0%, 5%] → draft head has trace predictive power but too low for net throughput gain. hit_rate > 5% → draft head may be viable pending cost analysis. Note: no reference threshold from DSpark. MTP draft head architecture (single stage, no Markov, no confidence gating, no capture reduction loss) is architecturally distinct from DSpark. MTP's hit_rate may be higher (simpler path, no capture loss) or lower (no Markov chain to improve proposals). |
| Method | Run ds4 with MTP GGUF loaded, DS4_MTP_PROBE=1. Parse hit_rate from probe stats. Repeat 3 trials [methodological: standard practice for confidence intervals]. Report mean hit_rate, std, and n_probes across trials. If n_probes < 100 across all trials, report as "insufficient samples". |
| Contrast | hit_rate reported as point estimate with CI. No qualitative classification. |
| Learning rate | High. ~0.2 hrs. Determines whether MTP draft head has any predictive relationship to base model output. |

#### D3.1 — DSpark Capture Validity Test

| Element | Detail |
|---|---|
| Hypothesis | DSpark target-layer HC capture (`metal_graph_dspark_capture_hc`, ds4.c line 25795) produces hidden state data that can be read back and compared to the live HC at the same target layer. |
| Context premises | SUT same. DSpark support GGUF loaded via --dspark --mtp. DS4_DSPARK_PROBE=1. DS4_DSPARK_SPEC_LOG=1. Simple prompt, gen=512, ctx=4096. DS4_DSPARK_SCHEDULER=0 (disable skip). The prop_cache=2868ms measured previously is cumulative over 20 decode cycles [derived: DS4_DSPARK_PROP_ADD accumulator at ds4.c line 58960]. First cycle includes one-time `metal_graph_seed_dspark_initial_cache_from_prefill` (~2800ms for 4096-token prefill [derived: first-cycle overhead, dominates cumulative prop_cache]). Per-step crop-to-prefix is a bookkeeping op (~3µs [hypothesis: bookkeeping op, negligible]). |
| Predicted outcome | No direction bias. Effect size will be measured, reported as Pearson r on first 100 elements of first captured target layer. Confidence interval estimated via bootstrap. Note: first 100 elements of a single target layer is a subset (~0.03% of a 7168-dim HC vector at F32). Rationale: capture corruption (misaligned device pointer, wrong buffer offset) produces errors visible across all elements — a 100-element sample is sufficient to detect systematic corruption. If r >= 0.95 on this sample, extend to full vector for precision measurement. |
| What it informs | If r < 0.95, capture introduces error > measurement noise. Possible causes: (a) `ds4_gpu_hc_weighted_sum_tensor` kernel (line 25804) does not compute the same reduction as the support model expects, (b) the HC tensors differ between capture time and live read due to transformer state ordering. Result sets an upper bound on DSpark draft quality attributable to capture quality alone. |
| Method | Instrument `metal_graph_dspark_capture_decode_layer` (line 25843) to dump first 100 float32 elements of captured+live HC at each target layer after cudaDeviceSynchronize. Compute Pearson r with 95% CI via bootstrap (10k resamples). If source modification is not possible, treat this as a separate experiment measuring acceptance rate, not capture validity. Run with DS4_DSPARK_PROBE=1 and gen=512. Report propose/cycle count from stats. |
| Contrast | If r < 0.5, capture introduces systematic error → DSpark cannot work. If r in [0.5, 0.95), capture has resolvable issues. If r >= 0.95, capture is accurate to within measurement noise. |
| Learning rate | High. ~1 hr setup + run. |

---

### Phase 1: Structural Isolation Tests

If Phase 0 shows signal (non-zero acceptance), isolate each component.

#### M3.2 — MTP Device Correctness Check

| Element | Detail |
|---|---|
| Hypothesis | MTP draft head evaluates on the correct CUDA device (the one owning the draft head's weights) and does not accidentally allocate/execute on the wrong GPU |
| Context premises | MTP probe passes (M3.1 hit_rate > 0%). nvidia-smi process monitoring during MTP eval. |
| Predicted outcome | All MTP-related cudaSetDevice calls target the device where MTP weights are allocated. No cross-device memory access for draft head parameters. [hypothesis: draft code path correctly routes to owner device] |
| What it informs | If MTP draft head runs on wrong GPU, any apparent probe hits are coincidental or from stale data. |
| Method | **Step 1: Discover allocation device.** Instrument `mtp_weights_bind` to check `ds4_gpu_set_current_device` state after all MTP tensors are allocated. MTP has NO tier switch — tensors allocate on the device set by `ds4_gpu_set_current_device` at bind time. **Step 2: Eval device.** Instrument `ds4_session_prepare_legacy_mtp_draft` (ds4.c line 58805) to read `g->active_tier` before/after `metal_graph_eval_mtp_draft`. MTP draft runs on `g->active_tier` (the device processing the last base model layer). **Step 3: Compare.** If allocation device ≠ eval device, A2 is refuted. |
| Contrast | If device mismatches, MTP is structurally broken — draft head runs on wrong GPU. Documentation fix: MTP must be constrained to GPU0 or support cross-device weight access. |
| Learning rate | High. ~1 hr. |

#### M3.3 — MTP HC Capture Consistency Check

| Element | Detail |
|---|---|
| Hypothesis | MTP HC capture (`metal_graph_eval_mtp_draft_from_hc`, ds4.c line 32218) reads hidden state from the live decode graph on the correct device without cross-device corruption. The HC tensor read at capture time and the HC tensor consumed by `ds4_session_prepare_legacy_mtp_draft` (ds4.c line 58805) refer to the same GPU memory allocation. |
| Context premises | M3.2 passes (device correctness confirmed). MTP probe passes (M3.1 hit_rate > 0%). MTP GGUF loaded via --mtp. Layers span both GPUs — target layers 40-42 on GPU1. |
| Predicted outcome | The HC pointer passed to `metal_graph_eval_mtp_draft_from_hc` resolves to a valid device allocation on the same GPU that executed `g->active_tier` (the last base model layer). No cudaMemcpyDeviceToDevice errors during capture. No cudaErrorInvalidValue from cross-device pointer access. [hypothesis: HC tensors are device-local, not cross-device references] |
| What it informs | If HC capture reads from a cross-device pointer, the draft head processes corrupted data (silent read of wrong GPU memory). If HC capture reads from the correct device but after the HC tensor has been modified by a subsequent layer, the draft head sees stale or partially-overwritten HC data. Both cases make MTP non-functional regardless of draft head quality. |
| Method | (1) Instrument `metal_graph_eval_mtp_draft_from_hc` entry to log `cudaGetDevice` and the GPU pointer address of `prev_hc` parameter. (2) Instrument the decode loop at the point where `g->active_tier` transitions between GPU0 and GPU1. (3) Compare: does `prev_hc` point to the device that just executed the last base model layer, or to a stale cross-device allocation? (4) Add cudaPointerGetAttributes check on prev_hc to confirm device residency. If cross-device, log the source device vs expected device. Run gen=256, ctx=4096, simple prompt. |
| Contrast | If prev_hc device matches g->active_tier device at all capture points: A3 confirmed — HC capture reads correct device. If prev_hc device ≠ g->active_tier device for any capture: A3 refuted — MTP captures from wrong GPU, draft head never sees correct HC data. |
| Learning rate | High. ~1.5 hrs (instrumentation + run). |

#### D3.2 — Captured vs Live HC Value Consistency

| Element | Detail |
|---|---|
| Hypothesis | For target layers 40-42, the HC tensor captured by `metal_graph_dspark_capture_hc` (ds4.c line 25795) at decode time and the live HC tensor read from `metal_graph_cur_hc(g)` at the same layer differ by less than the precision of the capture mechanism (`ds4_gpu_hc_weighted_sum_tensor`, a dot-product reduction over DS4_N_HC HC rows). |
| Context premises | D3.1 passes (capture completes, r >= 0.5). `metal_graph_dspark_capture_hc` reads cur_hc from the device executing the current layer (GPU1 for layers 40-42, verified by `metal_graph_set_active_tier_decode` in decode loop at ds4.c line 15422). The capture computes `dst[i] = sum_{j=0}^{DS4_N_HC-1} hc[j][i] * weight[j]` via `ds4_gpu_hc_weighted_sum_tensor`. |
| Predicted outcome | No direction bias. The capture operation reduces DS4_N_HC rows to 1 row via weighted sum. This is a lossy reduction. The test measures the difference between the weighted-sum output and the source HC rows to quantify information loss. |
| What it informs | Quantifies information lost during capture reduction. If difference exceeds 1.0 L2 norm per element, the weighted sum discards information the support model needs for correct draft proposals. |
| Method | Dump the full HC tensor (DS4_N_HC × DS4_N_EMBD float32) and the captured single-row output (DS4_N_EMBD float32) for the same decode step and target layer. Compute: (1) the max-weight row index from the weighted sum (which row contributes most), (2) L2 distance between captured output and each source HC row, (3) L2 distance between captured output and the mean of all HC rows. Report all three. |
| Contrast | If captured output is within 1e-6 L2 of the mean HC row, the weighted sum is essentially averaging (information loss = log(DS4_N_HC) bits). If captured output matches a specific HC row within 1e-6, the weighted sum is selecting (information loss = 0 for that row). If neither, the weighted sum is performing a non-trivial projection. |
| Learning rate | Medium. ~2 hrs (instrumentation + run + analyze). |

#### D3.3 — Stage Chain Logit Agreement

| Element | Detail |
|---|---|
| Hypothesis | The DSpark stage chain eval on GPU1 (executor tier via `ds4_session_prepare_dspark_draft`, ds4.c line 59330) produces logits from the support model weights that agree with the base model's logits on the same prefix, within the precision limits of the support model's smaller architecture. |
| Context premises | D3.1 passes (capture completes). D3.2 passes (capture preserves HC information). DSpark support GGUF loaded, stages >= 3, DS4_DSPARK_PROBE=1. Simple prompt, gen=256, ctx=4096, temp=0, seed=42. |
| Predicted outcome | No direction bias. For each decode step where DSpark produces a draft proposal, compare draft logits (from stage chain final output → hc_head → logits path) against base model logits at the same position from a non-speculative run. Report: (a) Pearson r between logit vectors at each position, (b) top-1 agreement fraction, (c) mean KL(draft_p || base_p). |
| What it informs | If stage chain logits are uncorrelated with base model logits (r < 0.5), the support model computes a fundamentally different distribution than the base model on the same prefix. This would explain the 5% acceptance rate as a model architecture mismatch, not a capture or multi-GPU issue. If r >= 0.9, the stage chain is producing a faithful approximation and the acceptance bottleneck is elsewhere (confidence gating, markov argmax, verify path). |
| Method | Run same prompt twice with identical seed (42) and temp=0: (a) ds4 with DS4_DSPARK_PROBE=1, log draft logits per step from `metal_graph_eval_dspark_final_hidden` → hc_head path (ds4.c line 59020). Capture `g->spec_logits` contents after each draft cycle. (b) ds4 non-speculative, log base model logits per step. Align positions: draft logits at position N+1 vs base logits at position N+1 (from non-speculative). Compute per-position correlation and divergence metrics. |
| Contrast | r < 0.5 → stage chain architecture cannot approximate base model on this task. r in [0.5, 0.9) → stage chain has partial agreement but differences accumulate. r >= 0.9 → stage chain is accurate, acceptance problem is downstream (confidence, markov, verify). |
| Learning rate | High. ~3 hrs. Tests a core architectural assumption of DSpark. |

#### D3.4 — Markov Determinism Test

| Element | Detail |
|---|---|
| Hypothesis | The DSpark Markov argmax operation (CPU path via `dspark_markov_q8_0_argmax_worker`, ds4.c line 31830, or GPU path via `ds4_gpu_dspark_markov_argmax_tensor`, ds4.c line 32083) produces reproducible top-1 token selections across repeated invocations with identical input, on the same device and across devices. |
| Context premises | DSpark stage chain works (D3.3 passes). Markov head weights loaded and validated (markov_rank = 256, Q8_0). DS4_DSPARK_PROBE=1. |
| Predicted outcome | Repeated invocations of the Markov argmax with identical logits and markov_bias inputs produce the same top-1 token ID. No non-determinism from: (a) parallel reduction order in `dspark_markov_q8_0_argmax_worker` (ds4_parallel_for with multiple worker threads reducing via best_val), (b) floating-point accumulation order in GPU reduction kernel, (c) uninitialized memory in the markov bias buffer. |
| What it informs | If Markov argmax is non-deterministic, the same decoded prefix can produce different draft proposals on successive spec cycles. This makes acceptance rate measurements unrepeatable and can cause silent correctness issues if verify accepts a non-reproducible draft. |
| Method | (1) Capture a single step's draft input (logits + markov_bias) from a live run via buffer dump. (2) Replay the Markov argmax 1000 times on the same input: 500 on GPU (ds4_gpu_dspark_markov_argmax_tensor) and 500 on CPU (dspark_markov_q8_0_argmax). (3) For each device path, record the top-1 token ID per invocation. (4) Report: fraction of invocations where top-1 token ID matches the majority outcome, per device. Also report if GPU and CPU majority outcomes differ. |
| Contrast | If 100% of invocations produce identical top-1 token on each device: Markov argmax is deterministic. If < 100% but > 90%: minor floating-point non-determinism from reduction order, acceptable. If < 90%: systematic non-determinism — investigate parallel reduction and uninitialized memory. If GPU majority outcome ≠ CPU majority outcome: device path divergence. |
| Learning rate | Medium. ~2 hrs. |

#### D3.5 — Confidence Calibration Test

| Element | Detail |
|---|---|
| Hypothesis | The DSpark confidence threshold gating (ds4.c line 32062: `sigmoid_stable(confidence_logit) < confidence_threshold`) correctly stops draft generation at a prefix length where the support model's confidence in its own predictions drops below the calibrated threshold. The default threshold (0.9, from ds4.c line 55197) produces a false-positive rate (prefix continued past correct tokens) < 10% and a false-negative rate (prefix truncated before wrong tokens) < 10%. |
| Context premises | DSpark capture and stage chain work (D3.1, D3.2, D3.3 pass -- organizational dependency; experiments can run in parallel since D3.5 does not consume D3.3 output). DS4_DSPARK_PROBE=1. Default confidence_threshold=0.9. |
| Predicted outcome | No direction bias. Measure: (a) fraction of confidence-gated prefixes where the gating point is correct (confidence drops before the support model would predict the wrong token), (b) fraction where gating is too early (confidence drops while support model still predicts correctly), (c) fraction where gating is too late (confidence remains high while support model predicts incorrectly). |
| What it informs | If confidence gating stops too early (high false-negative rate), DSpark produces unnecessarily short drafts, reducing potential speedup. If gating stops too late (high false-positive rate), DSpark proposes wrong tokens that must be rejected by verify, wasting compute. The balance determines whether the confidence mechanism adds value or harm. |
| Method | **Pre-check**: DS4_DSPARK_BLOCK_LOG env var absent from source. Instrument confidence logit computation at ds4.c line 31987 directly: insert dump of `confidence_logit` after sigmoid computation at line 32062, gated on DS4_DSPARK_SPEC_LOG (already exists). (1) Run DS4_DSPARK_PROBE=1 with DS4_DSPARK_SPEC_LOG=1 to dump per-draft-token confidence logits and support model logits. (2) For each draft position, compute: (a) sigmoid(confidence_logit) = p_conf, (b) whether the support model's argmax token matches the base model's token at that position. (3) Construct an ROC curve: vary effective threshold from 0.5 to 1.0, plot true positive rate (correct gating) vs false positive rate (incorrect gating). (4) Report precision and recall at default threshold 0.9, and the EER (equal error rate) threshold. |
| Contrast | If default threshold 0.9 achieves precision >= 0.9 and recall >= 0.8: confidence calibration is adequate. If precision < 0.5 or recall < 0.5: confidence threshold is poorly calibrated — the support model's confidence does not correlate with its accuracy. Either retune threshold or disable confidence gating. |
| Learning rate | High. ~3 hrs (collecting data + ROC analysis). Determines whether confidence gating helps or hurts. |

#### M3.4 — Verify Equivalence Test

**NOTE**: MTP has two verify paths with different cross-device behavior. Default (draft_n == 2, production case) uses `metal_graph_verify_decode2_exact` (ds4.c line 34064). Batch path (draft_n > 2) uses `metal_graph_verify_suffix_tops` (ds4.c line 34023). At default `mtp_draft_tokens=1`, NO verify path runs — must set `--mtp-draft-tokens N` with N >= 2.

| Element | Detail |
|---|---|
| Hypothesis | MTP verification produces the same top token as running the base model decoder non-speculatively on the same draft prefix |
| Context premises | MTP probe acceptance > 0%. `--mtp-draft-tokens 2` (gate: ds4_engine_options::mtp_draft_tokens defaults to 1 — speculative verify loop disabled unless > 1). Default verify path: `metal_graph_verify_decode2_exact` (draft_n=2). To test `metal_graph_verify_suffix_tops`, set `DS4_MTP_BATCH_VERIFY=1`. Verify both paths separately. **Cross-ref D3.8**: D3.8 verify logit equivalence (top-1 match rate between batch-encode verify and single-token decode) applies to MTP batch-verify path when DS4_MTP_BATCH_VERIFY=1. Run D3.8 analysis on MTP verify output if batch-verify is enabled. |
| Predicted outcome | Verify output matches non-speculative decode on same prefix: exact token match for all positions. [hypothesis: verification is a correct implementation of base model forward pass over draft tokens] |
| What it informs | If verification produces different results, speculative decode correctness is compromised — accepted tokens may differ from what the base model would have generated. |
| Method | (1) Run non-speculative with `DS4_MTP_PROBE=1 --mtp-draft-tokens 2`, record token sequence. (2) Run same prompt, same seed with DS4_MTP_PROBE=1 --mtp-draft-tokens 2. DS4_MTP_PROBE=1 logs draft tokens without consuming them. Compare logged draft token sequence against non-speculative token sequence from step (1) at corresponding positions — verify equivalence is confirmed if draft tokens at each position match the non-speculative token at that same position. (3) Compare verify output vs non-speculative tokens at the same positions. Report match rate for each verify path separately (`metal_graph_verify_decode2_exact` and `metal_graph_verify_suffix_tops`). |
| Contrast | If verification matches 100%, verification is correct. If verification matches < 100%, verification has bugs — must be fixed before any speculative decode is used for production. |
| Learning rate | High. ~3 hrs. |

#### D3.8 — Verify Logit Equivalence

| Element | Detail |
|---|---|
| Hypothesis | `metal_graph_verify_suffix_tops_impl` (ds4.c line 33822) iterates all 43 layers via `metal_graph_encode_layer_batch` with `g->placement[il+1]` cross-device tier switching [measured: ds4.c line 29049]. The batch-encode logits produced during verification match single-token decode logits for the same input at the same position. |
| Context premises | DSpark draft chain runs on GPU1 via `ds4_session_prepare_dspark_draft` (ds4.c line 59330). Verification runs full 43-layer model with cross-device placement. Model split: GPU0 layers 0-23, GPU1 layers 24-42 + output head, but split varies with scratch budget [hypothesis: depends on ctx and scratch]. Run at ctx=32768 to match PRD-2 split. |
| Predicted outcome | No direction bias. The test measures: (a) fraction of positions where verify top-1 token ≠ decode top-1 token, (b) KL divergence between verify logits and decode logits at the same position. |
| What it informs | If verify batch-encode path and single-token decode path produce different logits for identical input, the verification accept/reject decision is based on a different probability distribution than the generation distribution. This would cause wrong accept/reject decisions even if the multi-GPU plumbing is correct. |
| Method | Run same prompt twice: (1) normal decode (no speculation, temp=0, seed=42). Record token IDs and logits. (2) MTP or DSpark decode with logit dumping (DS4_MTP_FULL_LOGITS=1 or instrument metal_graph_verify_suffix_tops to dump row_logits). Compare verify logits at each accepted position to decode logits at the same position. Report: top-1 match rate, argmax(logits) match rate, mean KL(verify_p || decode_p). |
| Contrast | If top-1 match rate >= 0.99 across all tested positions, verify and decode produce equivalent distributions. If match rate < 0.95, the verify path computes different logits than the decode path — investigate `metal_graph_encode_layer_batch` vs `metal_graph_encode_decode_layer` differences (different kernel paths, different quantization paths, different tensor layouts). |
| Learning rate | High. ~3 hrs. Directly tests whether the verify code path (batch prefill-style) and decode code path (single-token decode-style) produce the same output for identical input. |

#### D3.9 — Markov Path Equivalence

| Element | Detail |
|---|---|
| Hypothesis | The GPU Markov argmax path (`ds4_gpu_dspark_markov_argmax_tensor`, ds4.c line 32083) and the CPU fallback path (`dspark_markov_q8_0_argmax_worker`, ds4.c line 31830) produce the same top-1 token when given identical logits and markov bias inputs, within floating-point precision differences. |
| Context premises | DSpark stage chain works (D3.3 passes). Markov weights loaded (markov_rank = 256, Q8_0). DS4_DSPARK_PROBE=1. Both paths reachable: GPU path enabled by default when `getenv("DS4_DSPARK_NO_GPU_MARKOV") == NULL` and tensor type is Q8_0 (ds4.c line 32074). CPU path is fallback when GPU path disabled or unavailable. |
| Predicted outcome | GPU and CPU Markov argmax produce the same top-1 token for >= 95% of inputs. Differences, if any, are at floating-point precision boundaries (logits within 1e-6 of each other at the max position). [hypothesis: both paths implement the same Q8_0 dot-product argmax with identical arithmetic] |
| What it informs | If GPU and CPU paths produce different results, the DSpark draft proposal depends on which device path executes. This creates reproducibility issues and silent correctness differences depending on env var settings. If results match, the two implementations are consistent and the Markov argmax can safely run on whichever device is faster. |
| Method | (1) Capture draft logits + markov_bias from a live run at 100 decode steps. (2) Fork: run GPU path with `ds4_gpu_dspark_markov_argmax_tensor` on each captured input. (3) Run CPU path with `dspark_markov_q8_0_argmax` on the same inputs. (4) Compare top-1 token per input across paths. (5) Report: fraction where GPU top-1 = CPU top-1; for mismatches, report average logit difference at the max position. Forcing GPU path: ensure `DS4_DSPARK_NO_GPU_MARKOV` is unset. Forcing CPU path: set `DS4_DSPARK_NO_GPU_MARKOV=1`. |
| Contrast | top-1 match >= 0.95: paths equivalent — no concern. Match in [0.5, 0.95): significant divergence — investigate Q8_0 dot-product implementation differences (parallel reduction order, FMA vs separate multiply-add, accumulator precision). Match < 0.5: paths compute different functions — one has a bug. |
| Learning rate | Medium. ~2 hrs. |

---

### Phase 2: Integration Stress Tests

If Phase 1 isolation tests pass, run end-to-end stress tests.

#### M3.5 — MTP Frontier Integrity Test

| Element | Detail |
|---|---|
| Hypothesis | MTP frontier snapshot saves complete GPU state (KV cache, compressor, indexer) across both devices, and frontier restore returns to exactly the same state after a failed verify |
| Context premises | MTP verify works correctly (M3.4 passes). Single session, single batch. |
| Predicted outcome | After restore: (1) Same next-token output as before snapshot, (2) Same KV cache contents (byte-level comparison), (3) Same compressor state, (4) Same indexer state. [hypothesis: snapshot/restore is correct on 2-GPU] |
| What it informs | After a failed verify or partial accept, the engine must rollback to exact pre-speculation state. If restore is incorrect, subsequent tokens degrade. |
| Method | (1) Snapshot frontier state. (2) Generate spec draft, let verify fail or partially accept. (3) Restore frontier. (4) Compare cudaMemcpy of KV cache and compressor state region against pre-snapshot dump. (5) Run one non-speculative token and compare against pre-snapshot same-position output. |
| Contrast | If any mismatch, frontier restore is incorrect — all subsequent tokens after partial accept are corrupt. |
| Learning rate | High. ~4 hrs. |

#### M3.6 — MTP Cache Consistency Test

| Element | Detail |
|---|---|
| Hypothesis | The MTP raw SWA cache (`g->mtp_raw_cache`, ds4.c lines 15010-15011) — a KV cache for SWA streaming across the MTP draft block — operates correctly when layers span 2 devices. The cache records the raw key-value state produced by the MTP draft block's attention (ds4.c line 32304), and increments `g->mtp_n_raw` (ds4.c line 32335) for each processed token. The cache head/tail pointers and n_raw counter remain consistent across spec cycles with cross-device layer execution. |
| Context premises | MTP GGUF loaded, `--mtp-draft-tokens 2` (enables verify). M3.1 probe passes (hit_rate > 0%). M3.2 device correctness confirmed. Default raw_window sizing from `metal_graph_alloc_kv_cache_tensor` (ds4.c line 17118). |
| Predicted outcome | After each MTP draft cycle: (a) `g->mtp_n_raw` increases by exactly 1 (one SWA step per draft cycle), (b) raw KV cache entries for tokens processed on GPU0 and GPU1 have consistent sequence positions, (c) no cache entry overwrites an unprocessed prior entry, (d) no gap in sequence positions between cache entries. [hypothesis: mtp_n_raw bookkeeping and cache storage are device-independent operations on the cache allocator, not GPU-specific] |
| What it informs | If the raw SWA cache loses entries or accumulates stale ones when layers execute on different GPUs, the MTP draft block's self-attention reads incorrect KV state. This can cause the draft head to produce tokens based on a corrupted context window. The `mtp_n_raw` counter reset behavior (ds4.c line 32581: `g->mtp_n_raw = 0` on cycle boundary) must correctly flush on each new spec cycle. |
| Method | (1) Run MTP speculative decode with `--mtp-draft-tokens 2`, DS4_MTP_SPEC_LOG=1, gen=512. (2) Instrument the MTP draft eval path (ds4.c line 32230: entry check of `mtp_raw_cache` and `mtp_n_raw`) to dump: n_raw before/after draft, raw_cache buffer address, sequence position of newest entry. (3) After each verify cycle, dump n_raw and check against expected value. (4) Run the same test with `DS4_FORCE_HOST_BOUNCE=1` to verify cross-device path does not affect cache state. Report: (a) monotonicity of n_raw increments, (b) absence of gaps between cached entries, (c) consistency of cache state before and after verify. |
| Contrast | If n_raw increments monotonically and cache entries have contiguous sequence positions across all cycles: MTP raw SWA cache is consistent on 2-GPU. If n_raw resets incorrectly, decrements, or cache entries show gaps: SWA cache corruption — MTP draft head has degraded context on 2-GPU. |
| Learning rate | Medium. ~3 hrs. |

#### M3.7 — MTP Draft Freshness Test

| Element | Detail |
|---|---|
| Hypothesis | MTP draft tokens are produced from the correct GPU buffer containing the current decode step's hidden state, not from a stale buffer left over from a previous cycle. The HC state passed to `metal_graph_eval_mtp_draft_from_hc` (ds4.c line 32218) via `ds4_session_prepare_legacy_mtp_draft` (ds4.c line 58805) is the current step's HC, not a cached/reused pointer from a prior cycle. |
| Context premises | MTP cache consistency passes (M3.6). MTP GGUF loaded, `--mtp-draft-tokens 2`. DS4_MTP_SPEC_LOG=1. Simple prompt, gen=512. |
| Predicted outcome | Each call to `ds4_session_prepare_legacy_mtp_draft` receives a unique HC tensor pointer (or a valid cyclic buffer slot) that corresponds to the current decode step's last layer output. No two consecutive calls receive pointers to the same memory address unless the buffer is a properly managed ring with no overwrites between read and write. [hypothesis: the HC tensor passed to the MTP draft function is freshly computed per step, not recycled] |
| What it informs | If MTP draft consumes stale HC from a previous step, the draft tokens are computed from outdated hidden state. This can produce tokens that appear valid (decodable) but are positionally incorrect — they correspond to an earlier prefix position. The verify step would detect this via logit mismatch, but the wasted draft compute degrades performance. |
| Method | (1) Instrument `ds4_session_prepare_legacy_mtp_draft` to log the GPU pointer address of the HC tensor passed to `metal_graph_eval_mtp_draft_from_hc`. (2) Also log the pointer address of `metal_graph_cur_hc(g)` at the same decode step. (3) Compare: do the two pointers match (indicating fresh HC)? (4) Over 200 decode steps, track pointer addresses in a set. Report: (a) fraction of steps where draft HC pointer = cur_hc pointer, (b) number of unique pointer values observed (should be ~raw_window, the ring buffer size), (c) any reuse of a pointer before its data is consumed by draft. |
| Contrast | If draft HC pointer == cur_hc pointer for >= 95% of steps: draft consumes fresh HC. If draft HC pointer != cur_hc pointer for > 5% of steps: investigate whether the mismatch is a stale pointer (same address as N steps ago) or a correctly managed ring slot. If stale pointer found: A7 refuted — MTP runs on recycled state. |
| Learning rate | Medium. ~2 hrs. |

#### D3.6 — DSpark KV Ring Integrity Test

| Element | Detail |
|---|---|
| Hypothesis | The DSpark support model KV ring (dspark_raw_cache) maintains consistent state across pipeline steps, and does not accumulate stale entries or lose capture data across cycle boundaries |
| Context premises | DSpark capture works (D3.1, D3.2). Stage chain eval works (D3.3). Non-trivial generation run (>1000 tokens). |
| Predicted outcome | KV ring entries for target layers are consistent with the HC data that was captured at the corresponding checkpoint positions. No entries point to freed or overwritten memory. Ring head/tail bookkeeping is monotonic across cycles. [hypothesis: ring maintenance is correct on 2-GPU] |
| What it informs | The KV ring is the draft model's context window. If corrupted, draft proposals degrade or hallucinate. |
| Method | Dump dspark_raw_cache buffer pointers and lengths before and after each spec cycle. Verify: (1) ring head advances by exactly the accepted token count, (2) no duplicate entries for same checkpoint position, (3) all buffer pointers point to valid GPU memory (not freed allocations). Use DS4_DSPARK_SPEC_LOG for observable logs. |
| Contrast | Mismatches indicate ring buffer bookkeeping bugs — common source of silent correctness issues. |
| Learning rate | Medium. ~5 hrs. |

#### D3.7 — Scheduler Alignment Test

| Element | Detail |
|---|---|
| Hypothesis | The DSpark adaptive scheduler (`ds4_session_dspark_scheduler_reset`, ds4.c line 47593, and related tuning functions at lines 47538-47592) does not accumulate errors across multiple spec cycles by skipping too many cycles, choosing wrong cycle timing, or misaligning the KV cache between draft and verify phases. The scheduler's skip decision (based on configurable window, skip_cycles, min_avg_ms, etc.) correctly balances compute savings against state divergence. |
| Context premises | DSpark capture and KV ring pass (D3.1, D3.2, D3.6 pass). DSpark scheduler enabled by default (`DS4_DSPARK_SCHEDULER` env var, default enabled if unset or non-zero, ds4.c line 47539). DS4_DSPARK_SPEC_LOG=1. Non-trivial generation (>1000 tokens). |
| Predicted outcome | Over 1000+ decode steps: (a) no two consecutive skip decisions cause a gap > `dspark_raw_cache` ring capacity, (b) after each skip, the next capture produces HC data consistent with the skipped step's state (not stale), (c) scheduler decision counters (skip/noskip/break_even) are monotonic and sum to total spec cycles. [hypothesis: the scheduler's state machine (ds4.c lines 47538-47670) maintains invariant counters and correctly detects break-even conditions] |
| What it informs | If the scheduler skips too aggressively, draft proposals become stale — they're based on old HC data. If the scheduler skips too conservatively, it wastes compute on cycles where DSpark provides no benefit. Either error accumulates across multi-GPU pipeline steps, causing silent correctness drift or performance degradation. |
| Method | (1) Run DSpark decode with DS4_DSPARK_STATS=1 and DS4_DSPARK_SPEC_LOG=1, gen=2048, ctx=4096. (2) Parse scheduler stats from DS4_DSPARK_SPEC_LOG output: each cycle shows skip/no_room/invalid/draft_available flags. (3) For each skip cycle, verify: (a) the KV ring has capacity for new entries, (b) the next non-skip cycle produces valid capture data. (4) Override scheduler with `DS4_DSPARK_SCHEDULER=0` to disable skipping and compare: are total accepted tokens different between scheduler-enabled and scheduler-disabled runs? (5) Run the same test with aggressive settings (`DS4_DSPARK_SCHEDULER_SKIP=10, DS4_DSPARK_SCHEDULER_WINDOW=1`) to stress-test the state machine. Report: (a) fraction of cycles skipped, (b) acceptance rate for skipped-adjacent cycles vs steady-state, (c) any state machine counter anomalies. |
| Contrast | If disabled-scheduler acceptance rate >= enabled-scheduler acceptance rate (within ±2%): scheduler does not degrade draft quality. If disabled-scheduler acceptance rate > enabled-scheduler by > 5%: scheduler is skipping too aggressively, causing stale proposals. If scheduler counters show non-monotonic behavior or mismatch between skip+noskip and total cycles: state machine bug. |
| Learning rate | Medium. ~4 hrs. |

#### E3 — DSpark Throughput Comparison

| Element | Detail |
|---|---|
| Hypothesis | DSpark speculative decode achieves different t/s than non-speculative decode on 2-GPU for code generation prompts. |
| Context premises | Same SUT. DS4_DSPARK_STATS=1. Speed-bench prompts (django-varbit, flappy-bird, slack-clone). gen=512, ctx=4096, temp=0, seed=42. |
| Predicted outcome | No direction bias. Report generation t/s for each prompt under both conditions. Effect size and direction measured, not assumed. |
| What it informs | If DSpark t/s < baseline t/s for all prompts, speculation is net-negative regardless of future optimization. If DSpark t/s > baseline for any prompt, speculation provides net benefit on that workload. |
| Method | Run each prompt twice: (a) ds4 --cuda --gpu-devices 0,1 (non-speculative), (b) ds4 --dspark --mtp DSPARK_GGUF (speculative). All other params identical. Parse generation t/s. Report mean and std across prompts. |
| Contrast | If DSpark t/s < baseline t/s for all 3 prompts: DSpark is net-negative. If DSpark t/s > baseline for ≥1 prompt: DSpark has a viable use case on this hardware. If DSpark t/s ≈ baseline t/s within measurement noise: speculation overhead is balanced by accepted tokens for some workloads. |
| Learning rate | High. ~1 hr (3 prompts × 2 conditions × ~10 min each). Directly measures the practical value of DSpark. |

---

### Phase 3: Multi-GPU Specific Stress Tests

#### P3.1 — Allocation Bounds Test

| Element | Detail |
|---|---|
| Hypothesis | Cross-device tensor allocations for speculative caches (`dspark_raw_cache` per stage, ds4.c line 15901; `mtp_raw_cache`, ds4.c line 17118; `mtp_eproj`, `mtp_hproj_hc`, etc. at ds4.c line 17110) are correctly sized to the model's tensor dimensions × cache capacity. No allocation underflows or overflows relative to the actual tensor operations performed. |
| Context premises | Any speculation configuration (MTP or DSpark). Model GGUF tensor dimensions from `required_tensor` / `dspark_bind_tensor` calls. Cache capacity from `g->raw_cap` at allocation time. |
| Predicted outcome | For each speculative cache tensor: (a) allocated bytes >= tensor dimension × capacity × element size, (b) no cudaMalloc returns smaller allocation than requested, (c) all `metal_graph_store_raw_kv_batch_tensor` writes (ds4.c line 30637 for DSpark, line 32304 for MTP) stay within allocation bounds. [hypothesis: allocation sizes match the `layer_raw_cache` / `dspark_raw_cache` / `mtp_raw_cache` layout × capacity computed by `metal_graph_alloc_kv_cache_tensor`] |
| What it informs | If cache tensors are undersized, write operations silently corrupt adjacent GPU memory. This can cause non-deterministic crashes (cudaErrorIllegalAddress) or, worse, silent data corruption where adjacent tensors (logits, HC buffers, compressor state) are overwritten with cache data. |
| Method | (1) Instrument each `ds4_gpu_tensor_alloc` call for speculative caches to record: requested_size, returned_pointer, and allocated_device. (2) After allocation, inject a bounds check: compute max write offset from tensor dimensions * capacity * element_size, verify it fits within allocated size. (3) Run a short speculation session (gen=64) to exercise all cache write paths. (4) After each write, verify via cudaMemcpy that bytes outside allocation bounds are unchanged from initialization. (5) Test at multiple context sizes (1024, 4096, 32768) to verify capacity scaling. Report: (a) allocation margin (allocated - required) for each cache tensor at each ctx size, (b) any OOB write detected. |
| Contrast | If all allocations have margin >= 0 and no OOB writes detected: sizing is correct. If any allocation has negative margin (required > allocated): buffer overflow risk — must fix allocation formula. If OOB writes detected: immediate correctness bug. |
| Learning rate | High. ~2 hrs. |

#### P3.2 — Sync Correctness Test

| Element | Detail |
|---|---|
| Hypothesis | The cudaEventRecord/cudaStreamWaitEvent synchronization between draft phase and verify phase does not cause race conditions where verify reads stale or incomplete draft data |
| Context premises | All single-GPU speculation tests pass. 2-GPU pipeline with any speculation technique. |
| Predicted outcome | Over 10,000 decode steps with speculation: zero instances of verify reading incomplete draft data (detected: verify output doesn't match expectations). [hypothesis: event synchronization is correct] |
| What it informs | Race conditions between draft and verify can cause silent data corruption that produces plausible-looking but wrong tokens. |
| Method | Run long generation (5000+ tokens) with DS4_DSPARK_SPEC_LOG (or DS4_MTP_SPEC_LOG) and log every verify vs expected comparison. On each verify step, check: were the draft tokens finalized before verify started? Injected race detection: add cudaEventSynchronize before verify and compare with non-synchronized path. If results differ, race exists. |
| Contrast | If non-synchronized path produces different results from synchronized path (over 10k steps), there is a race condition. If identical, synchronization is correct. |
| Learning rate | Medium. ~6 hrs. |

#### P3.4 — Host Bounce Integrity Test

| Element | Detail |
|---|---|
| Hypothesis | The host bounce fallback path (`DS4_FORCE_HOST_BOUNCE=1`, which forces GPU peer transfers through host-pinned memory via cudaMemcpyDeviceToHost + cudaMemcpyHostToDevice) does not corrupt speculative cache state, HC tensor data, or draft proposal buffers compared to the peer-DMA direct path. |
| Context premises | Any speculation configuration. Peer DMA direct is the default path (measured: pcie-bw-characterization in research-log.md, `nvidia-smi topo` confirms direct). Host bounce is forced via `DS4_FORCE_HOST_BOUNCE=1`. |
| Predicted outcome | Running with `DS4_FORCE_HOST_BOUNCE=1` produces identical token sequences (temp=0, seed=42) and identical acceptance rates (within measurement noise) as running without it, for the same prompt. [hypothesis: host bounce correctly implements the same data transfer as peer DMA, just through a different physical path] |
| What it informs | If host bounce corrupts speculative state, the issue is in the bounce path's data handling (mismatched copy sizes, incorrect device synchronization, pinned buffer lifecycle). This affects systems without peer DMA (non-NVIDIA GPUs, virtualized environments, certain PCIe topologies). If host bounce and peer DMA produce identical results, the speculative state is robust to transfer mechanism — ruling out a class of cross-device bugs. |
| Method | (1) Run ds4 with speculation (MTP or DSpark, whichever is applicable), temp=0, seed=42, gen=256, ctx=4096, WITHOUT `DS4_FORCE_HOST_BOUNCE`. Log token sequence and stats. (2) Run IDENTICAL config WITH `DS4_FORCE_HOST_BOUNCE=1`. (3) Compare: (a) token sequences — exact match required, (b) acceptance rate — within ±2% for statistical similarity, (c) per-step timing — host bounce will be slower due to 2× data path overhead (measured: pcie-bw-characterization shows host bounce at 14.4 GB/s [measured: pcie-bw-characterization] vs peer at 28.4 GB/s [measured: pcie-bw-characterization]), but token content must match. (4) Repeat 3 times. Report: token match fraction, acceptance rate comparison, and timing overhead factor. |
| Contrast | If token sequences match 100% and acceptance rates within ±2%: host bounce path is correct — speculative state survives the different transfer mechanism. If token sequences differ: host bounce introduces data corruption — investigate copy size mismatches or synchronization differences in the bounce path. If acceptance rate differs significantly: host bounce changes the draft/verify timing, which may affect race-condition-dependent code paths. |
| Learning rate | High. ~2 hrs (3 paired runs + analysis). Directly tests whether speculative state is robust to different cross-device transfer mechanisms. |

#### P3.5 — Timeout Survival Test

| Element | Detail |
|---|---|
| Hypothesis | The speculation code path does not contain infinite loops, deadlocks, or non-terminating conditions when run on 2-GPU. |
| Context premises | Any passing speculation configuration. |
| Predicted outcome | No direction bias. Fraction of runs that complete 30 min without stall > 60s. |
| What it informs | A deadlock or hang makes the technique unusable regardless of throughput. |
| Method | Run ds4 with MTP or DSpark for 30 minutes. Monitor: (1) stdout token generation progress, (2) stderr for DS4_MTP_SPEC_LOG / DS4_DSPARK_SPEC_LOG output, (3) nvidia-smi GPU utilization (should not drop to 0% during decode). Set 60s watchdog: if no token generated in 60s, kill and classify as hang. Repeat 3 times. |
| Contrast | Report fraction of runs with hangs. |
| Learning rate | Medium. ~2 hrs (3 × 30 min runs + analysis). |

---



## 4. Experiment Priority and Learning Rates

| Priority | Experiment | Hypothesis Tested | Support Kind | Learning Rate | Kill Criteria |
|---|---|---|---|---|---|
| **T0** | P0.0 — Support Kind Classification | GGUF support_kind | Both | 12 hypotheses/hr (2 GGUF files / 5 min per GROUND-RULES §4.1) | NONE → skip all speculation experiments |
| **T0** | P0.1 — Layer Split | Split varies with ctx | Both | 4 hypotheses/hr (4 ctx values / 5 min per GROUND-RULES §4.1) | If split differs from ctx=32768 reference, recalibrate layer-dependent experiments |
| **T0** | E1 — Draft Logit vs Base Logit | Support model accuracy | DSpark only | 8 hypotheses/hr | match_rate < 0.10 → DSpark support model quality cannot produce net gain |
| **T0** | E2 — Per-Step Cost Breakdown | Spec overhead vs savings | DSpark only | 6 hypotheses/hr | net_per_token > 0 for all acceptance rates ≤ 1.0 → DSpark mathematically net-negative |
| **T1** | M3.1 — MTP Probe Acceptance | MTP draft head produces hits > 0% | MTP only | 5 hypotheses/hr | hit_rate = 0% across 3 trials → MTP structurally broken |
| **T1** | D3.1 — DSpark Capture Validity | Captured HC matches live HC | DSpark only | 4 hypotheses/hr | Pearson r < 0.5 → capture corrupts data |
| **T1** | M3.2 — MTP Device Correctness | MTP eval runs on correct GPU | MTP only | 3 hypotheses/hr | allocation_device ≠ eval_device → MTP structurally broken |
| **T1** | M3.3 — MTP HC Capture Consistency | HC capture reads correct device memory | MTP only | 3 hypotheses/hr | prev_hc device ≠ g->active_tier device → MTP reads corrupted HC |
| **T1** | D3.8 — Verify Logit Equivalence | Verify batch path matches decode path | DSpark only | 3 hypotheses/hr | top-1 match rate < 0.95 → verify code path differs from decode |
| **T1** | D3.3 — Stage Chain Logit Agreement | Support model logits match base model | DSpark only | 3 hypotheses/hr | r < 0.5 → stage chain cannot approximate base model |
| **T1** | D3.4 — Markov Determinism | Markov argmax produces reproducible tokens | DSpark only | 2 hypotheses/hr | < 90% invocations produce same top-1 → non-determinism |
| **T1** | D3.5 — Confidence Calibration | Confidence threshold correctly gates drafts | DSpark only | 2 hypotheses/hr | precision < 0.5 or recall < 0.5 at default threshold |
| **T1** | D3.9 — Markov Path Equivalence | GPU and CPU markov paths produce same tokens | DSpark only | 2 hypotheses/hr | top-1 match < 0.95 → device path divergence |
| **T2** | E3 — DSpark Throughput | DSpark t/s vs baseline t/s | DSpark only | 2 hypotheses/hr | DSpark t/s ≤ baseline t/s for all prompts → no use case |
| **T2** | D3.2 — Capture Value Consistency | Captured HC preserves needed info | DSpark only | 2 hypotheses/hr | L2 distance ≥ 1.0 → weighted sum discards too much |
| **T2** | M3.4 — Verify Equivalence | MTP verify matches base model | MTP only | 2 hypotheses/hr | verify mismatches < 0.95 → verification path incorrect |
| **T2** | M3.5 — MTP Frontier Integrity | Frontier save/restore correct on 2-GPU | MTP only | 1 hypothesis/hr | frontier restore state mismatch → frontier save/restore incorrect |
| **T2** | M3.6 — MTP Cache Consistency | MTP raw SWA cache consistent on 2-GPU | MTP only | 1 hypothesis/hr | n_raw non-monotonic or cache entries have gaps |
| **T2** | M3.7 — MTP Draft Freshness | MTP consumes fresh HC each step | MTP only | 1 hypothesis/hr | stale pointer reuse > 5% of steps |
| **T2** | P3.1 — Allocation Bounds | Spec cache allocations correctly sized | Both | 1 hypothesis/hr | any OOB write or negative allocation margin |
| **T2** | P3.2 — Sync Correctness | Device sync correct between draft and verify | Both | ~0.17 hypotheses/hr | non-synchronized path differs from synchronized over 10k steps |
| **T2** | D3.7 — Scheduler Alignment | Adaptive skip doesn't accumulate errors | DSpark only | 1 hypothesis/hr | disabled-scheduler acceptance > enabled by 5%+ |
| **T2** | P3.4 — Host Bounce Integrity | Host bounce doesn't corrupt spec state | Both | 1 hypothesis/hr | token sequence differs between bounce and peer paths |
| **T3** | D3.6 — KV Ring Integrity | KV ring consistent across steps | DSpark only | 1 hypothesis/hr | ring corruption detected |
| **T3** | P3.5 — Timeout Survival | Speculation doesn't hang | Both | 0.5 hypothesis/hr | 2/3 runs hang |

[derived: GROUND-RULES §4.1 — learning rate estimates approximate; update after running]

**Execution rule**: P0.0 runs first regardless of priority order. After P0.0 determines `support_kind`, run only the experiments matching that kind. 

If support_kind == DSPARK: run P0.1 → E1 → E2 → D3.1 → D3.3 → D3.4 → D3.5 → D3.8 → D3.9 → D3.2 → D3.6 → D3.7 → E3 → P3.1 → P3.2 → P3.4 → P3.5.

> **Note**: D3.6 (T3, KV Ring Integrity) runs before D3.7 (T2, Scheduler Alignment) and E3 (T2, Throughput) because D3.6 validates KV ring state — a prerequisite for scheduler and throughput measurements.

If support_kind == MTP_LEGACY: run P0.1 → M3.1 → M3.2 → M3.3 → M3.4 → M3.5 → M3.6 → M3.7 → P3.1 → P3.2 → P3.4 → P3.5.

If NONE: skip all speculation experiments, switch to non-speculative optimization.

> **Note on numbering**: No P3.3. C3 (draft chain acceptance) is tested by E1, not a P3 experiment. P3.x numbers skip 3 to keep alignment with C-assumption index for experiments that do map 1:1. If a P3.3 is added later, reassign numbering to avoid collision with C-index alignment.

**Kill-chain override**: T0 priority experiments (P0.0, P0.1, E1, E2) run first regardless. P0.0 gates the entire track. P0.1 recalibrates layer-dependent experiments (informational, not a kill condition). E1 and E2 eliminate the technique if match_rate < 0.10 or net_per_token > 0 for all acceptance rates. If E1 or E2 eliminates, skip all downstream experiments regardless of priority tier.

---

## 5. Kill Criteria

If any P0 experiment (P0.0, P0.1, E1, E2) produces a result that eliminates the technique's path to viability, document the failure and move on.

### Kill Chain — DSpark Track

```
P0.0 support_kind == DSPARK?
    ↓ (yes)
P0.1 Layer split differs from ctx=32768 reference? → Recalibrate layer-dependent experiments.
    ↓
E1 draft match_rate < 0.10? → DSpark support model uncorrelated with base model. DSpark cannot produce net gain.
    ↓
E2 net_per_token > 0 for acceptance rate ≤ 1.0? → DSpark mathematically net-negative. Skip DSpark experiments.
    ↓
D3.1 r < 0.5? → Capture corrupts data. DSpark cannot work.
    ↓
D3.3 r < 0.5? → Stage chain logits uncorrelated with base model. Support model architecture mismatch.
    ↓
D3.4 < 90% deterministic? → Markov argmax non-deterministic. Fix parallel reduction.
    ↓
D3.5 precision < 0.5 or recall < 0.5? → Confidence threshold poorly calibrated. Retune or disable.
    ↓
D3.8 verify top-1 match < 0.95? → Verify code path diverges from decode. Fix verify before speculation viable.
    ↓
D3.9 top-1 match < 0.95? → GPU and CPU markov paths diverge. Investigate Q8_0 dot-product implementation.
    ↓
D3.2 L2 distance >= 1.0? → Weighted sum discards too much info. Fix capture reduction.
    ↓
D3.6 ring corruption detected? → KV ring bookkeeping bug. Fix ring buffer maintenance.
    ↓
D3.7 disabled-scheduler acceptance > enabled by > 5%? → Scheduler skipping too aggressively. Fix skip policy.
    ↓
All speculation dead? → Non-speculative optimization only.
```

### Kill Chain — MTP Track

```
P0.0 support_kind == MTP_LEGACY?
    ↓ (yes)
P0.1 Layer split differs from ctx=32768 reference? → Recalibrate layer-dependent experiments.
    ↓
M3.1 hit_rate = 0% across 3 trials? → MTP draft head not functionally connected. Skip MTP.
    ↓
M3.2 allocation_device ≠ eval_device? → MTP draft head runs on wrong GPU. Fix device routing.
    ↓
M3.3 prev_hc device ≠ g->active_tier device? → MTP captures HC from wrong device. Fix cross-device routing.
    ↓
M3.4 verify matches < 0.95? → Verification path incorrect. Fix verify before speculation viable.
    ↓
M3.5 frontier restore state mismatch? → Frontier save/restore incorrect. Fix snapshot/restore.
    ↓
M3.6 n_raw non-monotonic or cache entries have gaps? → SWA cache corruption. Fix bookkeeping.
    ↓
M3.7 stale pointer reuse > 5% of steps? → MTP consumes stale HC state. Fix HC pointer lifecycle.
    ↓
All speculation dead? → Non-speculative optimization only.
```

### Abandonment Thresholds (per GROUND-RULES §3)

| Condition | Action |
|---|---|
| Any P0 experiment eliminates all speculation paths | Switch to non-speculative optimization. |
| All P0 experiments pass but all P1-P2 speculation experiments fail | Document failure rate and root cause per §6 taxonomy. Switch to non-speculative. |
| All experiments pass but per-step cost (E2) > per-step savings at all acceptance rates | DSpark mathematically net-negative. Document cost breakdown, switch to non-speculative. |

---

## 6. Root Cause Taxonomy

When an experiment fails, classify the failure:

| Root Cause Category | Detection | Classification |
|---|---|---|
| Device routing mismatch | Verify cudaGetDevice matches weight allocation device via instrumentation | Routing bug — measure fix time |
| Sync race | Compare synchronized vs unsynchronized verify path output over 10k steps | Timing race — measure fix time |
| State save/restore incompleteness | Compare KV cache byte-level dump before snapshot vs after restore on all devices | State bug — measure fix time |
| Tensor sizing error | Check allocation bounds against actual tensor dimensions | Memory bug — measure fix time |
| Code path divergence | Compare batch-encode verify logits vs single-token decode logits for same input | Path divergence — measure fix time |
| Capture reduction loss | Compare captured HC output to source HC rows via L2 distance | Reduction bug — measure fix time |
| Quantization mismatch | Compare draft model forward pass in F16 vs actual loaded format | Config mismatch — measure fix time |
| GGUF metadata error | Validate target layer indices against model layout file | Data error — measure fix time |
| TP interaction | Compare DSpark draft with and without TP enabled | Interaction bug — measure fix time |
| Non-determinism | Compare GPU vs CPU argmax path outputs over 1000 iterations | Algorithm bug — measure fix time |
| Memory leak | Track allocation/free counts for speculative cache tensors | Resource bug — measure fix time |

---

## 7. Deliverables

### 7.1 Documents

| Document | Content |
|---|---|
| `PRD-3.md` | This document |
| `experiments/mtp-probe/` | M3.1-M3.7 experiment code + results (MTP track only). Note: directory does not exist yet — create before running MTP experiments. |
| `experiments/dspark-validation/` | D3.1-D3.9 experiment code + results (DSpark track only). Note: directory does not exist yet — create before running DSpark experiments. |
| `experiments/pipeline-stress/` | P3.1-P3.5 experiment code + results. Note: directory does not exist yet — create before running pipeline experiments. |
| `research-log.md` | Phase 3 entries following GROUND-RULES §8 format |
| `speculation-viability-report.md` | Final report: which speculation techniques can work on 2-GPU, what fixes needed, or which are structurally unsalvageable. Note: file does not exist yet — to be created after experiments complete. |

### 7.2 Decision Matrix

After all P0-P2 experiments, fill based on the support_kind actually tested:

```
+-------------------+---------+---------+----------+
| Technique         | Tested? | Viable? | Fixes    |
+-------------------+---------+---------+----------+
| MTP (legacy)      | Y/N     | Y/N     | [list]   |
| DSpark            | Y/N     | Y/N     | [list]   |
| Pure base model   | always  | always  | N/A      |
+-------------------+---------+---------+----------+

If tested technique dead → Non-speculative optimization targets from experiment results (see research-log.md).
Only one support_kind is tested per GGUF file — the other technique has "N" in Tested? column.
```

---

## 8. Constraints & Risks

| Constraint | Impact | Mitigation |
|---|---|---|
| Time budget for Phase 3 | Cannot exhaustively test every assumption | Use GROUND-RULES §4.1 learning rate to prioritize. If P0 fails, skip remaining. |
| DSpark source instrumentation | Some tests require adding debug code to ds4 source | Prefer env-var-gated logging. Minimize source changes. P3.1 (Allocation Bounds Test) explicitly requires instrumenting `ds4_gpu_tensor_alloc` — env-var gating is insufficient for this experiment. |
| Support kind exclusivity | DSpark and MTP legacy are mutually exclusive per GGUF. One `--mtp` path → one `support_kind`. | P0.0 determines which track runs. Only one technique's experiments are runnable per GGUF file. |
| DSpark multi-GPU interaction | Source code [measured: ds4.c lines 54758-54785, 59330, 33822] shows draft on dspark_exec_tier (dynamically selected: follows output head placement, adjustable for TP, overridable via DS4_DSPARK_EXEC_TIER). Default on this system: GPU1 (output head placed on GPU1). Verify iterates all 43 layers with placement[] tier switching (ds4.c line 33822). PRD-2 §O2.2 'draft_chain_time (GPU0-only)' is falsified (ds4.c line 59330). PRD-2 §O2.2 'verification on GPU0 only' is falsified (ds4.c line 33822). | Each experiment tests falsifiable hypotheses, not doc claims. |
| 5% acceptance rate may have multiple causes | Candidates: support model quality, capture reduction loss, confidence threshold calibration | E1 isolates model quality from pipeline loss. D3.1 isolates capture quality. |
| prop_cache=2868ms is cumulative over 20 cycles [derived: DS4_DSPARK_PROP_ADD at ds4.c line 58960] | First cycle includes one-time `metal_graph_seed_dspark_initial_cache_from_prefill` (~2800ms for 4096-token prefill). Not per-step. | Per-step cost is prop_chain (228ms [measured: DS4_DSPARK_STATS from existing run] / 20 = 11.4ms/step avg [measured: DS4_DSPARK_STATS from existing run]). Compare to verify (103ms [measured: DS4_DSPARK_STATS from existing run] / ~12 calls = 8.6ms/call [measured: DS4_DSPARK_STATS from existing run]). |
| Dequant overhead hypothesis falsified | O1.2 [measured: research-log.md] measured dequant overhead factor = 1.43× at 639 GB/s Q4K effective BW. Not the dominant gap (falsified: PRD-2 — Hypothesis H1: Q4_0 dequant overhead is the dominant gap between 68 t/s and 555 t/s roofline). | Non-speculative optimization should target sync/launch overhead, pipeline imbalance, attention kernel tuning — not dequant formats. |
| Proxmox kernel may introduce timer irregularities | Hang detection may produce false positives | Use wall clock + monotonic counter checks. |

---

## 9. References

- [PRD.md](PRD.md) — Research objectives, system config, topics
- [PRD-2.md](PRD-2.md) — Code generation throughput optimization (prior assumptions now questioned)
- [GROUND-RULES.md](GROUND-RULES.md) — Experimental methodology (all tests inherit)
- [research-log.md](research-log.md) — Phase 1-2 experiment results (establishes 5% DSpark acceptance, ~2.8s capture time)
- [dspark.md](../code/concepts/dspark.md) — DSpark architecture specification
- [mtp.md](../code/concepts/mtp.md) — MTP architecture specification
- [multi-gpu-pipeline.md](../code/concepts/multi-gpu-pipeline.md) — Pipeline architecture
- [tp.md](../code/modules/tp.md) — Tensor parallelism
- [roofline-analysis.md](roofline-analysis.md) — Throughput ceilings (establishes non-speculative optimization baseline)
- [speculative-decode-multi-gpu.md](speculative-decode-multi-gpu.md) — DSpark overlap analysis on 2-GPU (speculative-decode-multi-gpu.md correctly states draft on GPU1, verify cross-device. PRD-2 §O2.2 hypothesis of GPU0-only execution is falsified by ds4.c line 59330.)
- ds4.c (CUDA backend) — Source of verified architectural facts:
  - `support_model_detect`: lines 2787-2820 (mutually exclusive DSpark vs MTP legacy detection)
  - Draft chain tier switch: line 59330 (DSpark switches to `dspark_exec_tier`, MTP has NO tier switch)
  - Verify full-layer loop: lines 33822-33915
  - MTP CUDA eval: lines 32346, 32218 (runs on `g->active_tier`, no tier switch)
  - DSpark capture: `metal_graph_dspark_capture_decode_layer` line 25843 → `metal_graph_dspark_capture_hc` line 25795
  - prop_cache accumulator: line 58960
