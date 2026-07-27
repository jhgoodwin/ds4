# Session Batch Decoding

## Definition

Batch decoding advances multiple independent sessions by one token each in a single call. The engine exposes two APIs:

- `ds4_sessions_eval_batch(items, count, err, errlen)` — uniform decode batch. `count==1` is an exact alias for `ds4_session_eval()`.
- `ds4_sessions_eval_batch_with_prefill(items, count, prefill_session, prefill_prompt, err, errlen)` — mixed operation: one session resumes a prefill suffix while a decode batch runs concurrently on the same GPU.

## Why It Exists

Multiple concurrent sessions (server, agent) each want one token per decode step. Without batching, each session pays full kernel launch overhead and leaves GPU underutilized on decode-bound workloads. Batching amortizes launch latency and maximizes arithmetic utilization by concatenating independent sequences into one kernel dispatch.

## Where It Appears

- `ds4.h` — public API declarations (`ds4_sessions_eval_batch`, `ds4_sessions_eval_batch_with_prefill`, `ds4_decode_item`).
- `ds4.c` — dispatch logic, validation, sequential fallback, Metal batch path.
- `ds4_cuda.cu` — CUDA batch kernel dispatcher and GPU attention row kernels.
- `ds4_gpu.h` — `ds4_gpu_attention_decode_row` struct, `DS4_GPU_ATTENTION_DECODE_BATCH_MAX` (32).
- `ds4_server.c` — server batch scheduler (`server_eval_token`, `decode_worker_main`).

## Batch Architecture

### `ds4_sessions_eval_batch` — Implementation Flow

1. **Validate inputs.** Rejects null `items`, `count <= 0`, null session/engine.
2. **Shortcut `count==1`.** Forwards to `ds4_session_eval()` — identical behavior.
3. **Cross-check all items.** Every session must share one engine (`items[i].session->engine == first->engine`). Rejects cross-engine, duplicate sessions (full pairwise check), invalid tokens (`< 0` or `>= DS4_N_VOCAB`), and sessions at context limit (`checkpoint.len >= ctx_size`).
4. **CUDA path** (`ds4_sessions_eval_batch_cuda`). Gated on `backend == DS4_BACKEND_CUDA`. Two modes controlled by env `DS4_CUDA_SESSION_BATCH_INTERLEAVE`:
   - **Pipeline interleave mode** (default on; gated by env `DS4_CUDA_SESSION_BATCH_INTERLEAVE` — unset or non-`"0"` enables, `"0"` disables): encodes all decode rows as one interleaved graph via `metal_graph_encode_session_pipeline_batch()`. Requires `first->graph.placement` (graph placement tier array) — without it, falls through to serial graph mode even if interleave is on. Different stages fill GPU pipeline slots concurrently.
   - **Serial graph mode** (env `DS4_CUDA_SESSION_BATCH_INTERLEAVE=0`, or `graph.placement` unavailable): encodes each session independently via `metal_graph_encode_token_raw_swa()` in a loop.
   - Both modes: `ds4_gpu_begin_commands()`, encode, `ds4_gpu_end_commands()`, then readback logits per session via `ds4_gpu_tensor_read(metal_graph_logits(&s->graph), ..., s->logits)`. Failures mark all sessions invalid.
5. **Metal path** (`ds4_sessions_eval_batch_metal`). Gated on `ds4_sessions_eval_batch_metal_supported()`.
6. **Fallback.** Serializes via `ds4_session_eval()` per item. All-or-nothing: any failure invalidates every session via `ds4_session_invalidate()`.

### `ds4_sessions_eval_batch_metal_supported` — Gate

Returns true when:

- Backend is `DS4_BACKEND_METAL`.
- `support_kind == DS4_SUPPORT_NONE` (no MTP/DSpark).
- No graph dump prefix (`DS4_METAL_GRAPH_DUMP_PREFIX`).
- No decode stage profile (`DS4_METAL_DECODE_STAGE_PROFILE`).
- For GLM sessions: `glm_graph_ready` true, no SSD streaming, no debug hidden dump, no `glm_mtp` or `DS4_GLM_MTP_PROBE` env.
- For non-GLM sessions: no SSD streaming.
- All sessions non-CPU, non-distributed, checkpoint valid.

On Metal, the batch path has two sub-paths:

- **Native shared batch** (`metal_graph_native_session_batch_shared_supported`): encodes all sessions into one interleaved Metal graph. Accepts a `bool batch_qkv` flag (gated by `metal_graph_native_session_batch_qkv_supported`). When `true`, also batches attention Q/K/V projections across sessions via an internal code path (stages `decode-to-qkv`, `gather-qkv`, `qkv`, `decode-from-qkv`). Available when sessions share compatible Q8_0 shared-expert layouts and (for QKV) Q8_0 attention projections.

When native batching is unavailable, Metal falls back to encoding each session's graph independently in a loop — still batched into one `begin_commands`/`end_commands` epoch, but without cross-session kernel fusion.

### `ds4_sessions_eval_batch_with_prefill` — Implementation Flow

1. **Validate prefill session.** Must have valid checkpoint, prompt must extend checkpoint (`prompt->len > checkpoint.len`), must not exceed context size, prompt must start with checkpoint tokens (`ds4_tokens_starts_with`).
2. **Validate decode items.** All sessions must share prefill session's engine, no duplicates or self-match, all must have valid checkpoints and valid tokens.
3. **CUDA path** (`ds4_sessions_eval_batch_with_prefill_cuda`). Checks `DS4_CUDA_MIXED_PREFILL_DECODE` env and `metal_graph_mixed_prefill_decode_supported()`:
   - Checks: `decode_count >= 3`, prefill rows in valid range (`metal_graph_resume_prefill_min_tokens()` to `metal_graph_mixed_routed_max_prefill_rows()`), placement graph with CUDA TP/EP, compatible workspaces, and all batch sub-checks (MoE, shared, FFN pre, attn pre).
   - On success: `metal_graph_eval_mixed_prefill_decode()` — encodes prefill + decode in one interleaved graph. Advances both prefill checkpoint (copy full prompt) and decode checkpoints (push token).
   - On failure: invalidates all sessions.
4. **Metal path** (`ds4_sessions_eval_batch_with_prefill_metal`). Gated on `ds4_sessions_eval_batch_with_prefill_metal_supported()`. Interleaves one prefill chunk and decode rows by layer in a single command epoch.
5. **Fallback.** Serialized: first `ds4_session_sync(prefill_session, prefill_prompt)`, then `ds4_sessions_eval_batch(decode_items)`. Prefill invalidated if decode batch fails.

### GPU Attention Decode Row Kernels

#### `ds4_gpu_attention_decode_row` Struct

Fields: `raw_kv`, `comp_kv`, `topk` (device pointers), `pos`, `n_raw`, `raw_cap`, `raw_start`, `n_comp`, `top_k`, `window`, `ratio`, `indexed`.

#### `ds4_gpu_attention_decode_rows_rope_tensor`

Dispatcher for batched decode attention. Validates: `n_rows` in `[2, DS4_GPU_ATTENTION_DECODE_BATCH_MAX]`, `head_dim == 512`, valid device pointers. Accepts only the default score-split path (`DS4_CUDA_EXACT_SCORE_SPLIT_DECODE=1`, no split-KV, no graph, no ldg, no vec4, no dim2 variants). Launches three kernel groups:

1. **Scoring** — `attention_decode_score_split_scores_tile512_rows_kernel`. Grid `(ceil(max_dense_score/tile_rows), ceil(n_head/tile_heads), n_rows)`. Computes tiled QK dot products over raw + compressed KV for each dense (non-indexed) row. Uses `cudaFuncAttributeMaxDynamicSharedMemorySize` for tile shared memory. Skips indexed rows (`dsc.indexed` check).
2. **Finalize** — `attention_decode_score_split_finalize_rows_kernel`. Grid `(n_rows, n_head)`. Softmax over scores, weighted sum into output heads. Skips indexed rows.
3. **Indexed path** — `attention_indexed_mixed_decode_rows_kernel`. Grid `(n_rows, n_head)`. For rows with `dsc.indexed != 0`: reads top-k indices, gathers KV from compressed store, computes attention. For non-indexed rows: reads standard dense output.
4. **RoPE** — `rope_tail_decode_rows_kernel`. Applies inverse RoPE to all rows.

## Session Grouping

All sessions in a batch must share the same engine (same model weights, same backend). Sessions may have different KV cache positions — no alignment requirement. Duplicate sessions rejected (pairwise comparison). Invalid tokens and context-limit sessions rejected at entry.

Mixed prefill+decode permits at most one prefill session per call. The CUDA path requires `decode_count >= 3`.

Batch size bounded by `DS4_GPU_ATTENTION_DECODE_BATCH_MAX` (32).

| Backend | Batching | Mechanism |
|---|---|---|
| CUDA | native | `metal_graph_encode_session_pipeline_batch` (interleaved) or per-session `metal_graph_encode_token_raw_swa` (serial graphs) |
| ROCm | native | Same as CUDA |
| Metal | native (with fallback) | `metal_graph_encode_native_session_batch_shared(items, count, model, weights, batch_qkv)` where `batch_qkv` is a boolean flag (when supported) enabling optional QKV projection batching within the same function; falls back to per-session encode |
| CPU | sequential only | `ds4_session_eval()` per item |
| Mixed | CUDA native | `metal_graph_eval_mixed_prefill_decode` in one interleaved graph |
| Mixed | Metal native | Per-layer interleaved prefill + decode rows |

See [server.md](../modules/server.md) for server batch decode scheduling and slot management.

## Relationship

- **Depends on**: [gpu-tensor-primitives.md](gpu-tensor-primitives.md) (batch kernel variants), `ds4_gpu_graph` encode/decode lifecycle, [kv-cache-lifecycle.md](kv-cache-lifecycle.md), attention decode row table, server's coalescing scheduler.
- **Used by**: server (concurrent client requests via `decode_worker_main`), non-server multi-session use (agent, evaluation harnesses).
- **Alternatives**: sequential per-session eval (no batching overhead, less GPU utilization), split-KV speculation (different decode dispatch).

## Notes

- Sequential fallback is correctness-first: runs each session eval in isolation with no shared state. Results bit-identical to native batch mode on identical input.
- Mixed prefill+decode hides prefill latency behind concurrent decode: the prefill runs on the same GPU while decode sessions wait, so the scheduler sees one step instead of two.
- Metal native batch mode enables shared expert fusion across sessions. Requires Q8_0 shared-expert tensors and compatible graph layouts.
- CUDA batch interleave mode (`DS4_CUDA_SESSION_BATCH_INTERLEAVE`, default on) places all sessions' decode graphs into one pipeline graph. The per-session KV order and one-token kernels are preserved; only command submission changes.
- CUDA attention row dispatcher (`ds4_gpu_attention_decode_rows_rope_tensor`) bails out if non-default attention paths are active (split-KV, graph variants, score4/score8, alternative score kernels). Only the tiled tile512 rows + finalize rows + indexed mixed path is batched.
- Server coalescing is purely time-based, not count-based. A low-traffic server may batch only 1-2 items per decode cycle. Coalesce window adjustable but never exceeds 100ms.

[← Back to Index](../README.md)
