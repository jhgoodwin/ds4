# DSpark (DeepSeek Speculative Decoding)

## Definition

DSpark is DeepSeek V4's multi-stage speculative decoding framework. Unlike legacy MTP (single draft head at stage 0), DSpark interleaves lightweight transformer blocks at specific target layers through the model stack. Each stage produces a draft token via Markov chain conditioned on previous draft tokens. Stages execute alongside the base model forward pass, then verify and accept/reject via standard speculative decoding.

Architecture:

```
Input → Stage 0 (main_proj + main_norm) → Stage blocks (attn + FFN)
→ Final head (norm → hc_head → base_logits)
→ For each draft position:
    Markov: w1[prev] → features = [hidden | markov_state]
    confidence = sigmoid(features · confidence_proj)
    token = argmax(base_logits + w2[markov_state])
→ Accept prefix where confidence ≥ threshold
```

## Why It Exists

DSpark hides draft computation inside the base model's forward pass. Shallow stage blocks run on idle compute while main transformer occupies the GPU. Marginal GPU cost from extra projections — speculative decoding effectively free on throughput-bound systems. Multi-stage design produces higher-quality drafts than single-head MTP.

## Where It Appears

### Types

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `ds4_dspark_summary` | GGUF metadata summary: stages, block_size, markov_rank, noise_token_id, target_layers[], has_* booleans |
| `ds4.c` | `ds4_dspark_stage_weights` | Per-stage tensors: main_proj, main_norm (stage 0), norm, hc_head_*/markov_w1/w2/confidence_proj (final stage), ds4_layer_weights block (every stage) |
| `ds4.c` | `ds4_dspark_weights` | Aggregate: n_stages, block_size, markov_rank, noise_token_id, target_layers[], present/missing/invalid/metadata_errors counters, stage[DS4_DSPARK_MAX_STAGES] |
| `ds4.c` | `ds4_dspark_layout_kind` | Enum: F32 / PLAIN / DENSE / ROUTED |
| `ds4.c` | `ds4_dspark_spec_stats` | Per-session timing: cycles, proposed/accepted tokens, per-stage breakdown (propose_stage0_ms, propose_chain_ms, propose_conf0_ms, propose_markov_ms, propose_confidence_ms, verify_ms, snapshot_ms, replay_ms, etc.), draft/accepted length histograms |
| `ds4.c` | `ds4_session::dspark_stats` | Embedded `ds4_dspark_spec_stats` in session |
| `ds4.c` | `ds4_session::dspark_draft_tokens[]` | Draft token buffer (block_size max) |
| `ds4.c` | `ds4_session::dspark_draft_len` | Current draft proposal length |
| `ds4.c` | `ds4_session::dspark_draft_valid` | Draft buffer valid flag |
| `ds4.c` | `ds4_session::dspark_sched_*` | Scheduler state: cycles, accepted, skip, lifetime, extra_ms, last_propose_ms, last_confidence0 |
| `ds4.c` | `DS4_DSPARK_MAX_TARGET_LAYERS` | 8 |
| `ds4.c` | `DS4_DSPARK_MAX_STAGES` | 8 |
| `ds4.c` | `DS4_DSPARK_MAX_BLOCK_SIZE` | 16 |
| `ds4.h` | `ds4_engine_options::dspark` | Enable DSpark runtime decode (bool) |
| `ds4.h` | `ds4_engine_options::dspark_strict` | Reject speculative tokens on verify failure (bool) |
| `ds4.h` | `ds4_engine_options::dspark_confidence_threshold` | Confidence floor (float, default 0.9) |
| `ds4.h` | `ds4_engine_options::dspark_confidence_threshold_set` | Whether threshold explicitly set (bool) |
| `ds4.h` | `ds4_support_kind` | `DS4_SUPPORT_DSPARK` value (enum) |

### Detection

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `model_dspark_summary` | Scans GGUF metadata keys (3 aliases each: deepseek4.dspark.*, deepseek4_*, dspark.*) for block_size, markov_rank, noise_token_id, target_layer_ids. Scans tensor names for main_proj, main_norm, markov_head, confidence_head, hc_head_*, norm.weight |
| `ds4.c` | `support_model_detect` | Returns DS4_SUPPORT_DSPARK when stages >= 3 && has_main_proj && has_markov_head && has_confidence_head |
| `ds4.c` | `model_print_dspark_summary` | Prints stage count, block size, markov rank, target layers at engine init |

### Weight Binding & Validation

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `dspark_weights_bind_optional` | Iterates stages, binds per-stage tensors via dspark_bind_block; stage 0 gets main_proj/main_norm; final stage gets norm, hc_head_*, markov_w1/w2, confidence_proj |
| `ds4.c` | `dspark_bind_tensor` | Resolves `mtp.<stage>.<suffix>` via tensor_by_mtp_stage_suffix; tracks present/missing |
| `ds4.c` | `dspark_bind_block` | Binds all 24 ds4_layer_weights tensors (hc_attn_*, attn_*, ffn_*, hc_ffn_*) |
| `ds4.c` | `dspark_weights_validate_metadata` | Checks block_size > 0 && <= MAX_BLOCK_SIZE, markov_rank > 0, noise_token_id < vocab, target layers strictly increasing && < N_LAYER |
| `ds4.c` | `dspark_weights_validate_block_layout` | Validates each tensor's type/ndim/dim against expected layout kind (PLAIN, F32, DENSE, ROUTED) |
| `ds4.c` | `dspark_weights_validate_layout` | Calls validate_metadata + validate_block_layout for each stage + final head tensors |

### GPU Graph (Metal)

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `metal_graph_eval_dspark_stage0` | Stage 0: main_proj + main_norm projection from base HC |
| `ds4.c` | `metal_graph_eval_dspark_stage0_batch` | Batched stage 0 for multi-row prefill |
| `ds4.c` | `metal_graph_prepare_dspark_stage0_setup_block` | Fused stage 0 + setup for runtime path |
| `ds4.c` | `metal_graph_prepare_dspark_setup_block` | Prepare block: embed token, norm, project, fill cache |
| `ds4.c` | `metal_graph_eval_dspark_stage_block` | Single stage block: attention + FFN over support cache |
| `ds4.c` | `metal_graph_eval_dspark_stage_chain` | Full chain: setup + all stages + ring maintain. Suspends expert sharding via ds4_gpu_tp_suspend_expert_sharding |
| `ds4.c` | `metal_graph_eval_dspark_base_logits` | Final stage → norm → hc_head → base logits (no Markov) |
| `ds4.c` | `metal_graph_eval_dspark_final_hidden` | Final stage → norm → hc_head → hidden for confidence0 eval |
| `ds4.c` | `metal_graph_eval_dspark_base_logits_from_hidden` | Recompute logits from cached final hidden |
| `ds4.c` | `metal_graph_seed_dspark_initial_cache_from_prefill` | Seed stage target cache from prefill rows (replicate HC across block rows) |
| `ds4.c` | `metal_graph_seed_dspark_stage_target_cache` | Seed per-stage target cache for draft positions |
| `ds4.c` | `metal_graph_dspark_ring_maintain` | Maintain support-model KV ring buffer during scheduler skip cycles |
| `ds4.c` | `metal_graph_dspark_capture_begin` | Enable capture mode: record HC per layer to support cache |
| `ds4.c` | `metal_graph_dspark_capture_decode_layer` | Capture HC during decode layer |
| `ds4.c` | `metal_graph_dspark_capture_prefill_layer` | Capture HC during prefill layer |

### Capture & Invalidation

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `ds4_session_dspark_capture_current` | Check whether current checkpoint position matches capture state |
| `ds4.c` | `ds4_session_dspark_capture_batch_current` | Check whether current batch region matches capture |
| `ds4.c` | `ds4_session_dspark_capture_invalidate` | Clear draft valid + draft len + GPU capture state |
| `ds4.c` | `ds4_session_dspark_capture_note_checkpoint` | Advance capture checkpoint after successful spec accept |
| `ds4.c` | `metal_graph_dspark_capture_invalidate` | GPU-level: clear capture mask, reset validity |
| `ds4.c` | `metal_graph_dspark_cache_reset` | Reset support KV ring |

### Confidence Threshold

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `dspark_eval_confidence0_runtime` | Pre-check: compute confidence_logit = matvec([hidden|markov_state], confidence_proj) for first draft position. Returns raw logit (not sigmoid) |
| `ds4.c` | `dspark_apply_markov_confidence_lazy_runtime` | Per-token loop: for each draft position, read hidden row, compute markov_state via w1[prev], build feature vector = [hidden|markov_state], confidence_logit = matvec(features, confidence_proj), stop loop when sigmoid(confidence_logit) < threshold. For accepted tokens, apply Markov bias (w2[markov_state]) to base logits, argmax for token |
| `ds4.c` | `dspark_confident_prefix_len` | Given array of confidence logits, return prefix length where all sigmoid(logit) >= threshold |
| `ds4.c` | `dspark_markov_probe_ready` | Check markov_w1/w2 tensors bound |
| `ds4.c` | `dspark_confidence_probe_ready` | Check confidence_proj tensor bound |

### Markov Argmax

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `dspark_argmax_f32` | CPU argmax over float32 array |
| `ds4.c` | `dspark_markov_q8_0_argmax` | Fused CPU: apply Q8_0 Markov bias + argmax in one pass |
| `ds4.c` | `ds4_gpu_dspark_markov_argmax_tensor` | GPU-side: apply Markov bias to logits via matvec(w2[state]), then argmax. Reads model map/size, w1/w2 offsets, prev_token, vocab, markov_rank. Returns packed key containing token index |
| `ds4_gpu.h` | `ds4_gpu_dspark_markov_argmax_tensor` | Declaration |
| `ds4.c` | `dspark_markov_bias_disabled` | Check `DS4_DSPARK_DISABLE_MARKOV_BIAS` env |
| `ds4.c` | `dspark_disable_fused_cpu_markov_argmax` | Check `DS4_DSPARK_DISABLE_FUSED_CPU_MARKOV` env |
| `ds4.c` | `dspark_disable_reuse_confidence0_markov` | Check `DS4_DSPARK_DISABLE_REUSE_CONFIDENCE0_MARKOV` env |

### Draft Proposal Entry

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `ds4_session_prepare_dspark_draft_impl` | Full draft pipeline: capture check → stage0 eval → setup block → stage chain eval → final hidden → confidence0 eval → base logits → markov/confidence loop → fill dspark_draft_tokens[]. Timed sub-stages tracked in dspark_stats |
| `ds4.c` | `ds4_session_prepare_dspark_draft` | Thin wrapper around _impl; also calls ring_maintain |
| `ds4.c` | `ds4_session_eval_dspark_speculative_argmax` | Core speculative decode loop: check draft valid → verify first token → snapshot frontier → push checkpoint → metal_graph_verify_suffix_tops → accept/replay. Includes TP leader commit protocol |

### Scheduler

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `ds4_dspark_scheduler_enabled` | Check `DS4_DSPARK_SCHEDULER` env (default enabled) |
| `ds4.c` | `ds4_dspark_scheduler_window` | `DS4_DSPARK_SCHEDULER_WINDOW` (default 4) |
| `ds4.c` | `ds4_dspark_scheduler_skip_cycles` | `DS4_DSPARK_SCHEDULER_SKIP` (default 2) |
| `ds4.c` | `ds4_dspark_scheduler_slow_skip_cycles` | `DS4_DSPARK_SCHEDULER_SLOW_SKIP` (default 4) |
| `ds4.c` | `ds4_dspark_scheduler_no_draft_skip_cycles` | `DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP` (default 3) |
| `ds4.c` | `ds4_dspark_scheduler_short_accept_no_draft_skip_cycles` | `DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP` (default 4) |
| `ds4.c` | `ds4_dspark_scheduler_cold_low_confidence_skip_cycles` | `DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP` (default 7) |
| `ds4.c` | `ds4_dspark_scheduler_cold_low_confidence_threshold` | `DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI` (default 0.5) |
| `ds4.c` | `ds4_dspark_scheduler_tail_min_tokens` | `DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS` (default 10) |
| `ds4.c` | `ds4_session_dspark_scheduler_reset` | Zero scheduler state |
| `ds4.c` | `ds4_session_dspark_scheduler_should_skip` | Decrement skip counter; return true if should skip this cycle |
| `ds4.c` | `ds4_session_dspark_scheduler_note` | Update scheduler stats after spec cycle; compute next skip based on accept rate, no-draft count, confidence0 |

### GGUF Tools

| File | Symbol | Role |
|---|---|---|
| `gguf-tools/deepseek4-quantize.c` | `dspark_support_options` | block_size, markov_rank, noise_token_id, target_layers[], layer_count |
| `gguf-tools/deepseek4-quantize.c` | `dspark_support_plan` | Tensor plan for writing DSPARK support GGUF |
| `gguf-tools/deepseek4-quantize.c` | `build_dspark_support_plan` | Build tensor plan from hf_dir index |
| `gguf-tools/deepseek4-quantize.c` | `generate_dspark_tensor` | Convert HF safetensors → GGUF tensor |
| `gguf-tools/deepseek4-quantize.c` | `write_dspark_support_gguf` | Write DSPARK support GGUF file |
| `gguf-tools/deepseek4-quantize.c` | `print_dspark_manifest` | Print HF→GGUF name mapping manifest |
| `gguf-tools/deepseek4-quantize.c` | `map_dspark_hf_name` | Map HF name (mtp.<stage>.<hf>) to GGUF name; actions: emit, consume_scale, pack_expert, consume_expert_scale, unknown_dspark, skip_non_dspark |
| `gguf-tools/deepseek4-quantize.c` | `parse_dspark_hf_expert` | Parse expert tensor name → stage, expert#, part (w1/w2/w3), is_scale |
| `gguf-tools/deepseek4-quantize.c` | `parse_dspark_target_layers_arg` | Parse comma-separated target layers from CLI arg |

## Variants

| Variant | When Used | Key Difference |
|---|---|---|
| `DS4_DSPARK_LAYOUT_F32` | debug/validation | All tensors float32 |
| `DS4_DSPARK_LAYOUT_PLAIN` | standard | F16 or F32, no expert split |
| `DS4_DSPARK_LAYOUT_DENSE` | dense-model optimized | F16, F32, or Q8_0 |
| `DS4_DSPARK_LAYOUT_ROUTED` | MoE model | Routed expert quant, 3D tensors |

Compared to legacy MTP: DSpark uses multiple stages (≥3), per-stage block weights, Markov chain conditioning, confidence threshold gating, target-layer injection, and capture-based HC recording. Legacy MTP: single stage 0, no Markov, no confidence, no capture.

## Lifecycle

### Detection & Binding

```
engine_open → model_open(mtp_path) → support_model_detect()
  → model_dspark_summary(): read GGUF metadata (3 alias forms)
  → stages >= 3 && main_proj && markov_head && confidence_head?
    → DS4_SUPPORT_DSPARK → dspark_weights_bind_optional()
      → dspark_weights_validate_layout()
  → else fallback to legacy MTP or NONE
```

### Capture Activation

```
decode_layer / prefill_layer:
  if dspark_capture_enabled:
    metal_graph_dspark_capture_begin()         // init capture mask
    metal_graph_dspark_capture_decode_layer()          // record HC per target layer
    → ds4_session_dspark_capture_note_checkpoint() // on checkpoint commit
```

### Draft Proposal (per decode step)

```
ds4_session_prepare_dspark_draft_impl():
  1. Scheduler skip check → ring_maintain, return false
  2. ds4_session_dspark_capture_current() / _batch_current()
  3. stage0: metal_graph_eval_dspark_stage0()
  4. Setup block: metal_graph_prepare_dspark_setup_block()
  5. Stage chain: metal_graph_eval_dspark_stage_chain()
     → suspends TP expert sharding (coordinator-only)
     → per stage: attention + FFN over support KV ring
  6. Final hidden: metal_graph_eval_dspark_final_hidden()
  7. Pre-check confidence0: dspark_eval_confidence0_runtime()
     → sigmoid(logit) >= threshold?
  8. Base logits: metal_graph_eval_dspark_base_logits()
  9. Per-token markov + confidence:
     dspark_apply_markov_confidence_lazy_runtime()
     → for draft in 0..block_size:
         markov_state = w1[prev_token]
         feature = [hidden | markov_state]
         confidence_logit = matvec(feature, confidence_proj)
         if sigmoid < threshold: break
         argmax(logits + w2[markov_state]) → draft token
  10. Fill dspark_draft_tokens[], set dspark_draft_valid=true
```

### Speculative Verify

```
ds4_session_eval_dspark_speculative_argmax():
  1. Validate draft: valid flag, len > 0, tokens in-vocab
  2. Check room: remaining budget, context capacity
  3. Verify first token: sample_argmax(s->logits) == drafts[0]?
     → miss: return (no admission)
  4. Snapshot frontier → push drafts to checkpoint
  5. metal_graph_verify_suffix_tops() → row_tops per draft
  6. commit_drafts = longest prefix matching row_tops
  7. Full accept:
     → read spec logits → copy to s->logits
     → commit checkpoint → capture_note_checkpoint()
     → return accepted count
  8. Partial accept:
     → spec_frontier_restore() → rollback KV/compressor
     → metal_graph_eval_token_raw_swa() for each committed draft
     → commit checkpoint → capture_note_checkpoint()
  9. Reject:
     → restore frontier → return (no admission)
```

### Scheduler State Machine

```
scheduler_note() after each spec cycle:
  if no_draft && no_draft_skip != 0:
    if lifetime_accepted == 0 && long_accept not seen:
      → cold boot: use no_draft_skip
    if lifetime_accepted > 0 && long_accept not seen:
      → short_accept_no_draft_skip (higher)
    if lifetime_accepted == 0 && confidence0 <= cold_threshold(0.5):
      → cold_low_confidence_skip (7)
  → set dspark_sched_skip for future cycles
```

### Probe Mode

```
DS4_DSPARK_PROBE=1:
  → Same draft pipeline but sets fake_argmax_enabled=true or probe mode
  → Does not use draft for speculative decode
  → Tracks stats only
```

## Configuration

### Engine Options (`ds4_engine_options`)

| Field | Type | Default | Effect |
|---|---|---|---|
| `dspark` | bool | false | Enable DSpark runtime speculative decode |
| `dspark_strict` | bool | false | Reject speculative tokens that fail verification |
| `dspark_confidence_threshold` | float | 0.9 | Stage confidence floor; sigmoid(logit) >= threshold to accept draft position |
| `dspark_confidence_threshold_set` | bool | false | Whether --dspark-confidence-threshold was provided |

### Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `DS4_DSPARK_SCHEDULER` | "1" | Disable scheduler when "0" |
| `DS4_DSPARK_SCHEDULER_WINDOW` | 4 | Rolling window for accept-rate tracking |
| `DS4_DSPARK_SCHEDULER_SKIP` | 2 | Cycles to skip after no-draft event |
| `DS4_DSPARK_SCHEDULER_SLOW_SKIP` | 4 | Skip cycles when decode is slow |
| `DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI` | 1500 | Min avg ms per accept for slow detection |
| `DS4_DSPARK_SCHEDULER_MAX_MS_PER_ACCEPT_MILLI` | 28000 | Max ms per accept threshold |
| `DS4_DSPARK_SCHEDULER_MAX_EXTRA_SAVED_RATIO_MILLI` | 1000 | Extra/saved ratio for break-even |
| `DS4_DSPARK_SCHEDULER_BREAK_EVEN_WINDOW` | 0 | Override break-even window |
| `DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP` | 3 | Skip cycles after no-draft |
| `DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP` | 4 | Higher skip for short-accept-only models |
| `DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP` | 7 | Max skip when cold and low confidence |
| `DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI` | 500 | Cold low-confidence threshold (500ms) |
| `DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS` | 10 | Min remaining tokens to trigger tail skip |
| `DS4_DSPARK_SPEC_LOG` | - | Log per-round decisions to stderr |
| `DS4_DSPARK_PROBE` | - | Probe mode: no draft consumption, stats only |
| `DS4_DSPARK_FAKE_ARGMAX_PROPOSAL` | - | Force fake argmax (skip draft head, use random) |
| `DS4_DSPARK_DISABLE_MARKOV_BIAS` | - | Skip Markov bias addition |
| `DS4_DSPARK_DISABLE_FUSED_CPU_MARKOV` | - | Skip fused Q8_0 Markov argmax, use separate matvec |
| `DS4_DSPARK_DISABLE_REUSE_CONFIDENCE0_MARKOV` | - | Always recompute Markov state for first draft |
| `DS4_DSPARK_NO_GPU_MARKOV` | - | Force CPU Markov path even on CUDA |
| `DS4_DSPARK_DRAFT_LIMIT` | - | Override max draft tokens (within block_size) |

### GGUF Metadata Keys (3 alias forms each)

| Concept | Keys |
|---|---|
| block_size | `deepseek4.dspark.block_size`, `deepseek4.dspark_block_size`, `dspark.block_size` |
| markov_rank | `deepseek4.dspark.markov_rank`, `deepseek4.dspark_markov_rank`, `dspark.markov_rank` |
| noise_token_id | `deepseek4.dspark.noise_token_id`, `deepseek4.dspark_noise_token_id`, `dspark.noise_token_id` |
| target_layer_ids | `deepseek4.dspark.target_layer_ids`, `deepseek4.dspark_target_layer_ids`, `dspark.target_layer_ids` |

## Code Pattern

### Stage 0 + Stage Chain (GPU)

```c
// metal_graph_eval_dspark_stage_chain: coordinator-only DSpark stage eval
const uint32_t saved_tp_world = g->tp_world;
const uint32_t saved_tp_batch_rows = g->tp_batch_rows;
g->tp_world = 0;
g->tp_batch_rows = 0;
ds4_gpu_tp_suspend_expert_sharding(1);  // disarm TP for draft

for (uint32_t stage = 0; ok && stage < dw->n_stages; stage++) {
    ok = metal_graph_eval_dspark_stage_block(g, dspark_model, dw, stage,
                                             pos, support_len, raw_start,
                                             stage + 1 < dw->n_stages, true);
}

ds4_gpu_tp_suspend_expert_sharding(0);  // restore TP
g->tp_world = saved_tp_world;
g->tp_batch_rows = saved_tp_batch_rows;
```

### Confidence Threshold + Markov Argmax (CPU fallback)

```c
// dspark_apply_markov_confidence_lazy_runtime: per-draft-position loop
for (uint32_t draft = 0; ok && draft < dw->block_size; draft++) {
    // Compute confidence
    ok = dspark_dense_row_to_f32(markov_state, model, final->markov_w1, prev_token);
    ok = ds4_gpu_tensor_read(ffn_norm, draft * hidden_bytes, features, hidden_bytes);
    matvec_any(&confidence_logit, model, final->confidence_proj, features);

    if (sigmoid_stable(confidence_logit) < confidence_threshold) break;

    // Apply Markov bias, argmax
    matvec_any(markov_bias, model, final->markov_w2, markov_state);
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) logits[i] += markov_bias[i];
    token = dspark_argmax_f32(logits, DS4_N_VOCAB);

    proposal[draft] = token;
    prev_token = token;
}
```

### Speculative Verify Entry

```c
// ds4_session_eval_dspark_speculative_argmax
if (!s->dspark_draft_valid || s->dspark_draft_len == 0) {
    ds4_session_dspark_scheduler_note(s, 0, true, extra_ms);
    return n_accept;
}
// Verify first token matches base model
if (sample_argmax(s->logits, DS4_N_VOCAB) != drafts[0]) {
    ds4_session_dspark_scheduler_note(s, 0, false, extra_ms);
    return n_accept;
}
// Push drafts, verify, commit or rollback
spec_frontier_snapshot(&frontier, s);
ok = metal_graph_verify_suffix_tops(g, &e->model, &e->weights,
                                    &s->checkpoint, start, draft_n,
                                    false, true, row_tops, NULL, NULL);
// Accept prefix where row_tops[i-1] == drafts[i]
```

## Relationship to Other Concepts

- **Depends on**: engine API, Metal graph (capture, stage eval, verify), [hc-state.md](hc-state.md) (target-layer capture from live graph), [kv-cache-lifecycle.md](kv-cache-lifecycle.md) (support model KV ring in `dspark_raw_cache[]`), frontier snapshot/restore.
- **Used by**: CLI, server (speculative decode via `ds4_session_eval_dspark_speculative_argmax`).
- **Alternatives**: [mtp.md](mtp.md) (single stage, no Markov, no confidence gating, no capture), standard autoregressive (no speculation).
- **Incompatible with**: CPU backend (GPU required for capture/chain eval only — detection, binding, validation, scheduling, verification, and CPU fallback paths all run on CPU). [glm-model-path.md](glm-model-path.md): not compatible (separate draft path).
- **Partial TP support**: coordinator runs draft stages with expert sharding suspended; workers idle during draft but resume normal sharding for verification.
- **GGUF tools**: `gguf-tools/deepseek4-quantize.c` support plan builder for generating DSPARK support GGUF from HF safetensors.

## Notes

- TP expert sharding suspended via `ds4_gpu_tp_suspend_expert_sharding()` during `metal_graph_eval_dspark_stage_chain`. Saves `g->tp_world`/`g->tp_batch_rows`, sets to 0, calls suspend(1), restores after. The coordinator alone runs draft stages.
- Confidence threshold flow: pre-check `dspark_eval_confidence0_runtime` computes sigmoid(confidence0) ≥ threshold before entering per-token loop. Scheduler uses `dspark_last_confidence0` for cold-low-confidence detection (0.5 threshold, skip 7 cycles if zero lifetime accepts).
- Markov argmax has three paths: GPU fused (CUDA: `ds4_gpu_dspark_markov_argmax_tensor`), CPU fused Q8_0 (`dspark_markov_q8_0_argmax`), CPU fallback (`matvec_any` then `dspark_argmax_f32`). Metal falls through to CPU path.
- GPU Markov path requires Q8_0 w1/w2, rank multiple of 32, model map accessible. Returns packed 64-bit key; lower 32 bits masked for token index.
- Support model KV ring (`dspark_raw_cache[DS4_DSPARK_MAX_STAGES]`) is circular buffer sized to `raw_cap`. Captured from base model HC at target layers during normal decode. Invalidated on checkpoint rollback or capture position mismatch.
- Capture mode (`dspark_capture_enabled`) records per-layer HC output during `metal_graph_decode_layer` and `metal_graph_prefill_layer`. Activated only when DSpark draft is needed.
- `ds4_engine_mtp_draft_tokens` returns DSpark block_size from metadata when engine has DSpark loaded and `dspark` option enabled.
- Fake argmax proposal (`DS4_DSPARK_FAKE_ARGMAX_PROPOSAL`) useful for profiling without draft head overhead.
- GGUF metadata uses 3 alias forms for backward compatibility. All resolve to same value via `model_get_u32_any` / `model_get_u32_array_any`.
- Noise token ID from metadata reserved as draft fill token at unsupported positions (not yet implemented — future use).
- Scheduler adaptive skip: starts conservative, increases skip on no-draft events, decreases on successful accepts. Window tracking via `dspark_sched_cycles`/`dspark_sched_accepted`.

[← Back to Index](../README.md)
