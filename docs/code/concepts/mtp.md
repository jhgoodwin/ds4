# Multi-Token Prediction (MTP)

## Definition

MTP augments the base model with a lightweight draft head that predicts multiple future tokens per decode step. Draft verified against full model via speculative decoding: accept verified tokens for free, discard mismatches and replay from first wrong token. Each accepted token advances decode without a full model eval.

Legacy MTP targets DeepSeek Flash/Pro with a single-stage draft head. See [dspark.md](dspark.md) for DSpark multi-stage speculative decoding. See [glm-model-path.md](glm-model-path.md) for GLM MTP architecture.

## Why It Exists

Speculative decoding trades small draft model compute for fewer full-price forward passes. On systems where draft runs at near-zero incremental cost (same GPU, same memory bus), throughput improves 1.5-3x on decode-bound workloads. MTP draft head shares model layers — no separate model load, marginal GPU cost from extra projections.

## MTP Architecture

### Data Flow

```
                        ┌──────────────────────────────────────────────┐
                        │            Base Model (Target)                │
                        │                                              │
Input Embedding ───────▶│  Trunk: Layers 0..N-1                       │
                        │    → HC (hidden concat) output               │
                        │    → output head → base_logits (verified)    │
                        │                                              │
                        └─────────────┬────────────────────────────────┘
                                      │
                                      │ HC output (from last base layer)
                                      │
                                      ▼
                    ┌─────────────────────────────────────────────────────┐
                    │           Draft Head (MTP Stage 0)                 │
                    │                                                     │
                    │  ┌─────────────┐   ┌──────────────────────────┐   │
                    │  │ Embed       │   │ HC Projection            │   │
                    │  │ token ──▶   │   │ HC ──▶ hnorm ──▶ h_proj  │   │
                    │  │ e_proj      │   │                          │   │
                    │  └──────┬──────┘   └──────────┬───────────────┘   │
                    │         │                     │                    │
                    │         └─────────┬───────────┘                    │
                    │                   ▼                                │
                    │          ┌────────────────┐                       │
                    │          │  e_proj + h_proj│                       │
                    │          │  → sum = input  │                       │
                    │          └───────┬────────┘                       │
                    │                  ▼                                 │
                    │    ┌─────────────────────────────┐                │
                    │    │  Own Transformer Block      │                │
                    │    │  (attention + FFN / MoE)    │                │
                    │    │  with HC head (fn/base/scale)│               │
                    │    └──────────┬──────────────────┘                │
                    │               ▼                                   │
                    │         ┌──────────────┐                         │
                    │         │  output_norm  │                         │
                    │         │  + base output│                         │
                    │         │  projection   │                         │
                    │         └──────┬───────┘                         │
                    │                ▼                                  │
                    │          draft_logits ─── argmax ──▶ draft_token  │
                    └─────────────────────────────────────────────────────┘
```

#### Recursive Drafting (Legacy MTP)

For >1 draft tokens, draft head re-runs from its own HC output, alternating between two GPU buffers:

```
Draft round 0:
  base_HC ──▶ mtp_embed(token₀) + mtp_hproj(base_HC) ──▶ draft_block ──▶ mtp_state_hc ──▶ logits₀ ──▶ draft₀

Draft round 1:
  mtp_state_hc ──▶ mtp_embed(draft₀) + mtp_hproj(mtp_state_hc) ──▶ draft_block ──▶ mtp_next_hc ──▶ logits₁ ──▶ draft₁

Draft round 2:
  mtp_next_hc ──▶ mtp_embed(draft₁) + mtp_hproj(mtp_next_hc) ──▶ draft_block ──▶ mtp_state_hc ──▶ logits₂ ──▶ draft₂

... (ping-pong between mtp_state_hc / mtp_next_hc)
```

#### Verification

```
  drafts[0..N-1]  (proposed by MTP draft head)
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │           Verification (base model)                   │
  │                                                      │
  │  N=2 exact (tiny batch path):                        │
  │    metal_graph_verify_decode2_exact()                 │
  │    → row0_top == drafts[1]? accept both              │
  │    → else accept prefix-1 (drafts[0])                │
  │                                                      │
  │  N>2 microbatch (production path):                   │
  │    metal_graph_verify_suffix_tops()                   │
  │    → commit_drafts = verify row_tops vs drafts        │
  │    → full accept, partial accept, or fallback         │
  │                                                      │
  │  Margin threshold (N=2, non-strict):                 │
  │    top0 - top1 < threshold? → skip, raw eval prefix-1│
  └──────────────────────────────────────────────────────┘
       │
       ▼
  Accepted tokens appended to checkpoint.
  DS4_MTP_KEEP_ACCEPTED(n) truncates mtp_n_raw cache.
```

### Variant Detection

Two support kinds determined at engine open:

| Variant | Detection | Key Difference |
|---|---|---|
| `DS4_SUPPORT_MTP_LEGACY` | `mtp.0.e_proj.weight` + `mtp.0.h_proj.weight` + `mtp.0.hc_head_base.weight` present | Single stage-0, shared `ds4_layer_weights` block, HC head projections |
| `DS4_SUPPORT_NONE` | No draft tensors or metadata | Standard autoregressive decode |

See [dspark.md](dspark.md) for DSpark variant detection. See [glm-model-path.md](glm-model-path.md) for GLM MTP variant details.

### Detection Phase

```
engine_open → load mtp_model (mtp_path) → support_model_detect()
  → mtp.0.* tensors found?     → DS4_SUPPORT_MTP_LEGACY → mtp_weights_bind() → mtp_ready = true
  → neither?                   → DS4_SUPPORT_NONE → mtp_ready = false
```

If TP active (`tp.active`), legacy MTP disabled at detection — model closed, support_kind set to NONE.

### Configuration

Engine options (`ds4_engine_options`):

| Field | Type | Default | Effect |
|---|---|---|---|
| `mtp_path` | `const char *` | NULL | Path to separate draft GGUF file. NULL disables MTP. |
| `mtp_draft_tokens` | `int` | 1 | Draft tokens per round. Clamped [1, 16]. |
| `mtp_margin` | `float` | 3.0 | Confidence margin threshold. Higher = fewer accepts, safer. |

Environment variables:

| Variable | Effect |
|---|---|
| `DS4_MTP_STRICT` | Force exact decode verifier (disable batch path). Also set by `--quality`. |
| `DS4_MTP_MIN_MARGIN` | Override `mtp_margin` threshold at runtime. Parsed as float. |
| `DS4_MTP_TIMING` | Log per-round timing breakdown (draft/verify/prefix/replay ms). |
| `DS4_MTP_CONF_LOG` | Log confidence margin per round: top0, top1, margin. |
| `DS4_MTP_FULL_LOGITS` | Read full logits from draft head GPU tensor (not just argmax). |
| `DS4_MTP_SPEC_LOG` | Log speculative decode decisions (miss, fallback, etc.). |
| `DS4_MTP_PROBE` | Enable probe mode: tracks draft hit rate without affecting decode. |
| `DS4_MTP_BATCH_VERIFY` | When unset with `DS4_MTP_STRICT`, select N=2 exact decode verifier. Set to force batch verifier. |
| `DS4_MTP_CAPTURE_PREFIX1` | Use prefix-1 capture (no snapshot/replay) for partial accepts at N=2. |
| `DS4_MTP_EXACT_REPLAY` | Debug: restore frontier and exact-replay accepted prefix after verify. |
| `DS4_MTP_FORCE_SNAPSHOT` | Force frontier snapshot regardless of draft_n. |

### Tensor Names (all under `mtp.0.*`)

| Tensor Name | Weight Struct Field | Purpose |
|---|---|---|
| `mtp.0.hc_head_base.weight` | `hc_head_base` | HC → draft head base projection |
| `mtp.0.hc_head_fn.weight` | `hc_head_fn` | HC → draft head fn projection |
| `mtp.0.hc_head_scale.weight` | `hc_head_scale` | HC → draft head scaling |
| `mtp.0.e_proj.weight` | `e_proj` | Embedding projection into draft |
| `mtp.0.h_proj.weight` | `h_proj` | Hidden-to-vocab projection |
| `mtp.0.enorm.weight` | `enorm` | Embedding norm |
| `mtp.0.hnorm.weight` | `hnorm` | Hidden norm |
| `mtp.0.norm.weight` | `norm` | Draft head input norm |
| `mtp.0.hc_attn_fn.weight` | `block.hc_attn_fn` | Attention HC fn projection |
| `mtp.0.hc_attn_scale.weight` | `block.hc_attn_scale` | Attention HC scale |
| `mtp.0.hc_attn_base.weight` | `block.hc_attn_base` | Attention HC base |
| `mtp.0.attn_norm.weight` | `block.attn_norm` | Attention input norm |
| `mtp.0.attn_q_a.weight` | `block.attn_q_a` | Q low-rank A |
| `mtp.0.attn_q_a_norm.weight` | `block.attn_q_a_norm` | Q low-rank A norm |
| `mtp.0.attn_q_b.weight` | `block.attn_q_b` | Q low-rank B |
| `mtp.0.attn_kv.weight` | `block.attn_kv` | KV projection |
| `mtp.0.attn_kv_a_norm.weight` | `block.attn_kv_a_norm` | KV low-rank A norm |
| `mtp.0.attn_sinks.weight` | `block.attn_sinks` | Attention sinks |
| `mtp.0.attn_output_a.weight` | `block.attn_output_a` | Output low-rank A |
| `mtp.0.attn_output_b.weight` | `block.attn_output_b` | Output low-rank B |
| `mtp.0.hc_ffn_fn.weight` | `block.hc_ffn_fn` | FFN HC fn projection |
| `mtp.0.hc_ffn_scale.weight` | `block.hc_ffn_scale` | FFN HC scale |
| `mtp.0.hc_ffn_base.weight` | `block.hc_ffn_base` | FFN HC base |
| `mtp.0.ffn_norm.weight` | `block.ffn_norm` | FFN input norm |
| `mtp.0.ffn_gate_inp.weight` | `block.ffn_gate_inp` | MoE gate input |
| `mtp.0.exp_probs_b.bias` | `block.ffn_exp_probs_b` | Expert probability bias |
| `mtp.0.ffn_gate_exps.weight` | `block.ffn_gate_exps` | Gate expert weights |
| `mtp.0.ffn_up_exps.weight` | `block.ffn_up_exps` | Up expert weights |
| `mtp.0.ffn_down_exps.weight` | `block.ffn_down_exps` | Down expert weights |
| `mtp.0.ffn_gate_shexp.weight` | `block.ffn_gate_shexp` | Shared expert gate |
| `mtp.0.ffn_up_shexp.weight` | `block.ffn_up_shexp` | Shared expert up |
| `mtp.0.ffn_down_shexp.weight` | `block.ffn_down_shexp` | Shared expert down |

`tensor_by_mtp_stage_suffix` resolves any stage and suffix via `"mtp.%u.%s"` pattern. Legacy uses stage 0 only.

## Where It Appears

| File | Symbol | Role |
|---|---|---|
| `ds4.c` | `ds4_mtp_weights` | Weight struct: e_proj, h_proj, enorm, hnorm, norm, hc_head_base/fn/scale, plus full `ds4_layer_weights block` |
| `ds4.h` | `ds4_engine_options::mtp_path` | Path to separate draft GGUF file (optional; legacy drafts embed tensors in main model) |
| `ds4.h` | `ds4_engine_options::mtp_draft_tokens` | Draft tokens per speculative round (clamped 1-16) |
| `ds4.h` | `ds4_engine_options::mtp_margin` | Confidence margin threshold (default 3.0) |
| `ds4.c` | `ds4_engine::mtp_model` | Separate GGUF model handle for draft weights |
| `ds4.c` | `ds4_engine::mtp_weights` | Bound `ds4_mtp_weights` struct |
| `ds4.c` | `ds4_engine::support_kind` | `DS4_SUPPORT_MTP_LEGACY` or `DS4_SUPPORT_NONE` |
| `ds4.c` | `ds4_engine::mtp_ready` | Flag set after successful weight bind |
| `ds4.c` | `ds4_engine::mtp_draft_tokens` | Runtime draft count (clamped from options) |
| `ds4.c` | `ds4_engine::mtp_margin` | Runtime margin threshold |
| `ds4.c` | `ds4_session::mtp_logits` | Float buffer for draft head logits (GPU readback) |
| `ds4.c` | `ds4_session::mtp_draft_token` | Single draft token from legacy MTP head |
| `ds4.c` | `ds4_session::mtp_draft_valid` | Whether `mtp_draft_token` is usable |
| `ds4.c` | `ds4_session::mtp_probe_total` | Probe mode: total opportunities observed |
| `ds4.c` | `ds4_session::mtp_probe_hit` | Probe mode: draft hit count |
| `ds4.c` | `ds4_support_kind` | Enum: `DS4_SUPPORT_NONE`, `DS4_SUPPORT_MTP_LEGACY` |
| `ds4.c` | `support_model_detect` | Scans GGUF tensors; returns `support_kind` based on tensor/metadata presence |
| `ds4.c` | `ds4_tensor_mtp_stage` | Checks tensor name prefix `"mtp."`; extracts stage number |
| `ds4.c` | `tensor_by_mtp_stage_suffix` | Resolves `"mtp.%u.%s"` pattern for any stage/suffix |
| `ds4.c` | `mtp_weights_bind` | Binds all `mtp.0.*` tensors into `ds4_mtp_weights`; asserts required tensors |
| `ds4.c` | `ds4_engine_has_mtp` | Returns true when backend != CPU, not distributed, `mtp_ready` |
| `ds4.c` | `ds4_engine_mtp_draft_tokens` | Returns draft count (legacy MTP: mtp_draft_tokens) |
| `ds4.c` | `ds4_session_prepare_legacy_mtp_draft` | Runs `metal_graph_eval_mtp_draft`; sets `mtp_draft_token` and `mtp_draft_valid` |
| `ds4.c` | `ds4_session_note_legacy_mtp_probe` | Records hit/miss on `mtp_probe_total`/`mtp_probe_hit`; clears `mtp_draft_valid` |
| `ds4.c` | `ds4_session_eval_speculative_argmax` | Core speculative decode loop: drafts → verifies → accept/replay |
| `ds4.c` | `ds4_session_tp_spec_cycle` | TP worker verify: snapshot frontier, run `metal_graph_verify_suffix_tops`, receive commit frame, rollback/replay |

→ See [dspark.md](dspark.md) for DSpark symbols (ds4_dspark_weights, ds4_dspark_spec_stats, metal_graph_eval_dspark_*).
→ See [glm-model-path.md](glm-model-path.md) for GLM MTP symbols (glm_mtp_*, ds4_session_glm_spec_cycle).

## Draft Heads

### Legacy Draft Head Eval

Draft head runs inside model graph — zero extra CPU overhead, marginal GPU cost from extra projections. Draft reads HC from live GPU buffers without host round-trip.

```c
// ds4_session_prepare_legacy_mtp_draft
if (!e->mtp_ready || !s->mtp_logits || (e->mtp_draft_tokens <= 1 && !mtp_probe_log))
    return false;

int mtp_top = -1;
if (metal_graph_eval_mtp_draft(g, &e->model, &e->weights, &e->mtp_model,
                                &e->mtp_weights, token, pos,
                                getenv("DS4_MTP_FULL_LOGITS") ? s->mtp_logits : NULL,
                                &mtp_top)) {
    s->mtp_draft_token = (mtp_top >= 0) ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
    s->mtp_draft_valid = true;
}
```

### Recursive Drafting

```c
// ds4_session_eval_speculative_argmax — core loop
for (; draft_n < draft_cap; draft_n++) {
    ds4_gpu_tensor *prev_hc = (draft_n & 1) ? g->mtp_state_hc : g->mtp_next_hc;
    ds4_gpu_tensor *out_hc  = (draft_n & 1) ? g->mtp_next_hc : g->mtp_state_hc;
    if (!metal_graph_eval_mtp_draft_from_hc(g, &e->model, &e->weights,
            &e->mtp_model, &e->mtp_weights, prev_hc, out_hc,
            drafts[draft_n-1], pos + draft_n - 1,
            need_logits ? s->mtp_logits : NULL, &mtp_top))
        return n_accept;
    drafts[draft_n] = (mtp_top >= 0) ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
}
```

### GLM Spec Cycle

See [glm-model-path.md](glm-model-path.md) for GLM MTP speculative cycle details.

### Probe Mode

```
ds4_session_prepare_legacy_mtp_draft() → set mtp_draft_token + mtp_draft_valid
  → ds4_session_note_legacy_mtp_probe() on next token eval
  → mtp_probe_total++ ; if match: mtp_probe_hit++
  → Always clears mtp_draft_valid
```

Probe mode runs draft head but never acts on it. Tracks hit rate stats for offline analysis. Clears `mtp_draft_valid` after noting probe to prevent stale draft from affecting real decode.

## Verification

### Speculative Decode Loop (Legacy MTP)

```
ds4_session_eval_speculative_argmax():
  1. Accept first_token via normal eval → base logits produced
  2. Check e->mtp_ready && s->mtp_draft_valid && e->mtp_draft_tokens > 1
  3. Compute draft_cap = min(mtp_draft_tokens, remaining, context_room)
  4. Verify first draft: sample_argmax(s->logits) == drafts[0]?
     → NO: return (no speculative work)
     → YES: continue
  5. For draft_n = 1..draft_cap:
     - Alternate mtp_state_hc / mtp_next_hc buffers
     - metal_graph_eval_mtp_draft_from_hc() → next token logits
     - Sample argmax → drafts[draft_n]
     - Stop on EOS
  6. Apply mtp_margin threshold (non-strict, draft_n == 2):
     - logits_top2() → compute margin
     - margin < threshold? → accept prefix-1 via raw eval, return
  7. Verify suffix:
     a. N=2 exact: metal_graph_verify_decode2_exact() — tiny batch path
        → row0_top == drafts[1]? accept both; else accept prefix-1
     b. Microbatch: metal_graph_verify_suffix_tops() — production path
        → commit_drafts = verify row_tops vs drafts
        → full accept: read spec logits, copy to s->logits
        → partial accept (capture_prefix1): spec_frontier_commit_prefix1
        → partial accept (snapshot): spec_frontier_restore + raw eval
        → fallback: replay committed prefix exactly
  8. DS4_MTP_KEEP_ACCEPTED(n) → truncate mtp_n_raw cache
  9. Return accepted count
```

Margin threshold checked only for N=2 non-strict path. Margin computed as `top0 - top1` from draft head softmax. Margin skip path falls back to single-token raw eval. `DS4_MTP_KEEP_ACCEPTED` macro truncates `g->mtp_n_raw` to hide uncommitted speculative rows — counter-based invalidation sufficient because next attempt overwrites future slots.

### TP Worker Verify

```
ds4_session_tp_spec_cycle():
  → spec_frontier_snapshot() → save frontier
  → Push drafts to checkpoint
  → metal_graph_verify_suffix_tops() → KV/compressor/indexer side effects
  → ds4_tp_recv_verify_commit() → receive leader's decision
  → full_accept? → discard frontier, mark checkpoint valid
  → partial/reject? → spec_frontier_restore() → rollback
  → Run lockstep replay: metal_graph_eval_token_raw_swa() for each accepted token
```

`ds4_session_eval_speculative_argmax` also serves as TP worker verify entry point (`ds4_session_tp_spec_cycle`). Legacy MTP verify path gated on `!tp.active`.

Fallback verifier (sequential exact decode) exists for robustness if microbatch verifier fails. Deliberately slow — should not trigger during normal operation. Logged via `DS4_MTP_SPEC_LOG`.

## Relationship

- **Depends on**: engine API (same session/eval primitives), metal graph (draft + verify kernels), [hc-state.md](hc-state.md) (draft reads HC from live graph), [kv-cache-lifecycle.md](kv-cache-lifecycle.md) (MTP raw SWA cache in `mtp_n_raw`).
- **Used by**: CLI, server (speculative decode path via `ds4_session_eval_speculative_argmax`).
- **Alternatives**: standard autoregressive (no speedup), DSpark (see [dspark.md](dspark.md)), split-kv speculation (different draft source).
- **Incompatible with**: legacy MTP disabled when tensor parallelism active.

[← Back to Index](../README.md)
