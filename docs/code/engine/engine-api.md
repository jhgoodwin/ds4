# Engine API & Sessions

## Files

- `ds4.c` — ds4_engine_open/close, ds4_session_create/sync/eval
- `ds4.h` — public API declarations

## Purpose

Public entry points: acquire instance lock, open GGUF with backend-appropriate mmap policy, expose inference operations to CLI and server.

## Key Types

| Type | Role |
|---|---|
| `ds4_engine` | loaded model handle (one per process) |
| `ds4_session` | mutable inference timeline (one per concurrent request) |
| `ds4_tokens` | growable token array |
| `ds4_session_snapshot` | binary checkpoint for save/restore |
| `ds4_session_payload_file` | staged KV payload for disk I/O |
| `ds4_token_score` | token id + logit + logprob tuple |

## Engine Lifecycle

```
ds4_engine_open() → lock → detect shape → bind weights
→ init backend → return engine
ds4_engine_close() → free resources → unlock
ds4_engine_create_with_gpu_config() → same + multi-GPU placement
```

Process lock prevents multiple concurrent model loads (instance lock via `flock` on PID file). `ds4_engine_create_with_gpu_config()` accepts an optional `ds4_gpu_config` for multi-GPU pipeline-parallel placement; passing NULL is identical to `ds4_engine_open()`.

## Engine Queries

| Function | Returns |
|---|---|
| `ds4_engine_summary()` | print model shape + memory to stderr |
| `ds4_engine_model_name()` | model name string |
| `ds4_engine_vocab_size()` | vocabulary count |
| `ds4_engine_layer_count()` | number of transformer layers |
| `ds4_engine_embd_dim()` | hidden dimension |
| `ds4_engine_model_bytes()` | total weight bytes on disk |
| `ds4_engine_model_id()` | stable id for KV cache compatibility |
| `ds4_engine_is_glm_dsa()` | true if model is GLM DSA variant |
| `ds4_engine_routed_quant_bits()` | bits per weight for routed experts |
| `ds4_engine_has_output_head()` | true if model has separate output head |
| `ds4_engine_has_mtp()` | true if model has MTP draft head |
| `ds4_engine_mtp_draft_tokens()` | number of MTP draft tokens |
| `ds4_engine_power()` | current GPU power limit |
| `ds4_engine_prefill_chunk()` | max tokens per prefill chunk |
| `ds4_engine_tp_vocab_split()` | vocab half for TP rank |
| `ds4_engine_tp_gate_schedule()` | decode gate schedule for TP transport |
| `ds4_engine_layer_compress_ratio()` | KV compression ratio per layer |
| `ds4_engine_hidden_f32_values()` | number of float32 values in hidden state |
| `ds4_engine_glm_layer_payload_bytes()` | per-layer payload byte count (GLM) |

## Generation

```
ds4_engine_generate_argmax(prompt, n_predict, ctx_size,
                           emit, done, progress, cancel)
    → full greedy generation loop, calls emit per token
```

High-level convenience: creates session, syncs prompt, loops argmax + eval until EOS or `n_predict` reached, then frees session. Progress and cancellation callbacks are forwarded to the internal session.

```
ds4_engine_collect_imatrix(dataset_path, output_path, ...)
    → compute importance matrix from a dataset
```

## Session Lifecycle

```
ds4_session_create() → allocate KV cache → init graph → return session
ds4_session_free() → free KV cache → destroy session

ds4_session_sync(prompt) → extend/rebuild graph state to match prompt
ds4_session_eval(token) → one decode step
ds4_session_sample() → sample next token from logits
ds4_session_eval_speculative_argmax(first_token, max_tokens, eos)
    → speculative decode block: draft then verify

ds4_session_invalidate() → mark session unusable (TP error recovery)
ds4_session_rewind(pos) → truncate KV cache to position
ds4_session_pos() → current token position
ds4_session_ctx() → maximum context size
ds4_session_prefill_cap() → max prefill tokens per chunk
ds4_session_tokens() → pointer to live token array
```

## Sync Strategy

`ds4_session_sync()` finds common prefix between current state and new prompt, rebuilds only divergent suffix. Avoids recomputing shared prefix KV cache.

Internal rewrite API exposes three return codes:

```
common = ds4_session_common_prefix(session, prompt)
switch ds4_session_rewrite_from_common(session, prompt, common, err, errlen):
    case DS4_SESSION_REWRITE_OK (0):
        // prefix matched, suffix evaluated in place
        break
    case DS4_SESSION_REWRITE_REBUILD_NEEDED (1):
        // live checkpoint cannot be safely rewritten;
        // caller should restore an older checkpoint then sync
        break
    case DS4_SESSION_REWRITE_ERROR (-1):
        // invalid state or parameters; check err string
        break
```

`ds4_session_rewrite_requires_rebuild(live_len, canonical_len, common)` checks whether a rebuild is needed before calling the rewrite function.

## Sampling

```
ds4_session_argmax()                — greedy (top-1)
ds4_session_argmax_excluding(id)    — greedy excluding one token
ds4_session_sample(temperature, top_k, top_p, min_p, rng)
                                    — temperature + top-k + top-p + min-p
ds4_sample_logits(logits, n_vocab, temperature, top_k, top_p, min_p, rng)
                                    — stateless sampler (for testing)
ds4_session_top_logprobs(out, k)    — top-k token scores with logprobs
ds4_session_token_logprob(token)    — logprob of a specific token
ds4_session_copy_logits(out, cap)   — raw logits copy to caller buffer
ds4_session_set_logits(logits, n)   — inject logits from external source
```

## Rewrite API (low-level)

| Function | Purpose |
|---|---|
| `ds4_session_common_prefix()` | length of common prefix between checkpoint and prompt |
| `ds4_session_rewrite_from_common()` | rewrite session from common prefix, returns `ds4_session_rewrite_result` |
| `ds4_session_rewrite_requires_rebuild()` | predicate: does rewrite require a full rebuild? |

Return values for `ds4_session_rewrite_from_common()`:

| Code | Constant | Meaning |
|---|---|---|
| -1 | `DS4_SESSION_REWRITE_ERROR` | invalid state or parameters |
| 0 | `DS4_SESSION_REWRITE_OK` | session rewritten in place |
| 1 | `DS4_SESSION_REWRITE_REBUILD_NEEDED` | cannot rewrite safely; restore checkpoint then sync |

`ds4_session_sync()` itself returns:

| Code | Constant | Meaning |
|---|---|---|
| 0 | — | success |
| 1 | — | generic error |
| 2 | `DS4_SESSION_SYNC_INTERRUPTED` | cancelled by `ds4_session_set_cancel()` callback |

## Payload Persistence

KV cache payloads can be saved to / loaded from disk for session migration or checkpointing.

| Function | Purpose |
|---|---|
| `ds4_session_save_snapshot()` | save full KV state to `ds4_session_snapshot` struct |
| `ds4_session_load_snapshot()` | restore full KV state from snapshot |
| `ds4_session_snapshot_free()` | free snapshot memory |
| `ds4_session_save_payload(fp)` | write KV payload to an open FILE* |
| `ds4_session_load_payload(fp, bytes)` | read KV payload from an open FILE* |
| `ds4_session_payload_bytes()` | byte count of serialized KV payload |
| `ds4_session_stage_payload()` | stage payload to temp file for async write |
| `ds4_session_write_staged_payload()` | write staged payload to a FILE* |
| `ds4_session_payload_file_free()` | free staged payload file metadata |
| `ds4_session_layer_payload_bytes(start, end)` | byte count of a layer range |
| `ds4_session_save_layer_payload(fp, start, end)` | write layer-range payload |
| `ds4_session_load_layer_payload(fp, bytes, tokens, n, start, end)` | read layer-range payload |

Payload file constants:

| Constant | Value |
|---|---|
| `DS4_SESSION_PAYLOAD_MAGIC` | `0x34565344` ("DSV4") |
| `DS4_SESSION_PAYLOAD_VERSION` | 2 |
| `DS4_SESSION_LAYER_PAYLOAD_MAGIC` | `0x4c565344` ("DSVL") |
| `DS4_SESSION_LAYER_PAYLOAD_VERSION` | 1 |

## Batch Eval

| Function | Purpose |
|---|---|
| `ds4_sessions_eval_batch(items, count)` | advance multiple independent sessions by one token each |
| `ds4_sessions_eval_batch_with_prefill(items, count, prefill_session, prefill_prompt)` | combined prefill + decode batch step |

`ds4_decode_item` pairs a session with the next token. Backends without native batching use a correctness-first sequential fallback.

## Distributed / TP

| Function | Purpose |
|---|---|
| `ds4_session_is_distributed()` | true if session uses distributed inference |
| `ds4_session_distributed_route_ready()` | 1 = route ready, 0 = incomplete, -1 = error |
| `ds4_session_gpu_warmup()` | pay one-time GPU submission cost (TP worker) |
| `ds4_session_tp_spec_cycle()` | TP worker side of mirrored speculative-verify block |
| `ds4_engine_tp_bind(tp)` | bind TP transport gates to engine |
| `ds4_session_layer_slice_reset()` | reset layer-slice graph state |
| `ds4_session_eval_layer_slice()` | evaluate a contiguous layer range |
| `ds4_session_eval_output_head_from_hc()` | evaluate output head from hidden state |

## Configuration

| Function | Purpose |
|---|---|
| `ds4_engine_set_power()` | GPU power limit (0-100%) |
| `ds4_engine_prefill_chunk()` | max tokens per prefill chunk |
| `ds4_session_set_progress()` | progress callback for UI |
| `ds4_session_set_display_progress()` | fine-grained UI-only progress (not a KV checkpoint boundary) |
| `ds4_session_set_cancel()` | cooperative cancellation (returns `DS4_SESSION_SYNC_INTERRUPTED`) |
| `ds4_session_power()` | current per-session power limit |
| `ds4_session_set_power()` | per-session power limit |
| `ds4_session_report_progress()` | fire progress callback directly |

## Utility

| Function | Purpose |
|---|---|
| `ds4_engine_dump_tokens()` | print token ids to stderr |
| `ds4_engine_head_test()` | run a head-only forward pass |
| `ds4_engine_first_token_test()` | first-token latency test |
| `ds4_engine_metal_graph_test()` | Metal graph compilation + execution test |
| `ds4_engine_metal_graph_full_test()` | full Metal graph test |
| `ds4_engine_metal_graph_prompt_test()` | Metal graph prompt-mode test |
| `ds4_dump_text_tokenization()` | tokenize text and print token ids |
| `ds4_backend_name()` | string name for a backend enum |
| `ds4_think_mode_enabled()` | true if think mode is available |
| `ds4_think_mode_name()` | string name for a think mode |
| `ds4_think_max_prefix()` | think-max reasoning prefix text |
| `ds4_glm_reasoning_effort_text()` | GLM reasoning effort prompt |
| `ds4_think_max_min_context()` | minimum context for think-max mode |
| `ds4_think_mode_for_context()` | select think mode based on context size |
| `ds4_context_memory_estimate()` | memory estimate for a backend + context |
| `ds4_log()` | structured log output |

## Invariants

- One engine per process (instance lock enforced).
- Multiple sessions per engine (independent KV caches).
- Session not thread-safe; callers serialize access.
- Engine thread-safe for read-only queries after init.

## See Also

- [session-batch-decode.md](../concepts/session-batch-decode.md) — batch decode architecture and session grouping
- [session-rewrite-invalidation.md](../concepts/session-rewrite-invalidation.md) — rewrite strategy, invalidation, and error recovery
- [session-snapshots.md](session-snapshots.md) — session payload serialization for save/restore

[← Back to Index](../README.md)
