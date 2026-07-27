# Session Rewrite & Invalidation

## Files

- `ds4.c` — `ds4_session_rewrite_from_common`, `ds4_session_rewrite_requires_rebuild`, `ds4_session_common_prefix`, `ds4_session_sync`, `ds4_session_sync_internal`, `ds4_session_invalidate`, `ds4_session_rewind`.
- `ds4.h` — result codes enum `ds4_session_rewrite_result`, public API declarations, `DS4_SESSION_SYNC_INTERRUPTED` constant.
- `ds4_tp.h` — `DS4_TP_FRAME_REWIND` (3), `DS4_TP_FRAME_INVALIDATE` (4), `ds4_tp_send_sync`, `ds4_tp_send_rewind`, `ds4_tp_send_invalidate`, `ds4_tp_wait_command_ack`, `ds4_tp_recv_logits_half`.
- `ds4_tp.c` — TP send implementations.

## Definition

Session rewrite adjusts the live KV cache to match a new prompt without fully rebuilding from scratch. The engine finds the common prefix between the current checkpoint and the target prompt, then evaluates only the divergent suffix. Invalidation discards the entire cache state; rewind truncates to a specific position.

A DS4 session checkpoint (`s->checkpoint`, type `ds4_tokens`) records the token sequence that the live backend state (raw SWA rows, compressed KV rows, indexer rows, compressor frontiers) matches. Replacing any part of the live tail requires restoring the whole frontier first. Extending exactly at the live end is safe in-place; rewriting behind it is not an in-place operation.

## Why It Exists

Server and agent code change the prompt mid-conversation (user edits, tool call results appended, system prompt changes). Full recompute from scratch costs tokens * tokens time. Common-prefix rewrite saves the shared KV work — the bulk of prefill cost.

## Where It Appears

| File | Symbol | Role |
|---|---|---|
| `ds4.h` | `DS4_SESSION_REWRITE_ERROR` | `-1`: internal error (missing session, empty prompt, context overflow, checkpoint invalid, prefix mismatch) |
| `ds4.h` | `DS4_SESSION_REWRITE_OK` | `0`: live cache matches target after rewrite |
| `ds4.h` | `DS4_SESSION_REWRITE_REBUILD_NEEDED` | `1`: must restore older checkpoint + full sync |
| `ds4.h` | `ds4_session_rewrite_result` | Return type enum for rewrite API |
| `ds4.h` | `DS4_SESSION_SYNC_INTERRUPTED` | `2`: cooperative cancellation fired during sync |
| `ds4.h` | `ds4_session_sync` | Synchronize session to full prompt token prefix |
| `ds4.h` | `ds4_session_rewrite_requires_rebuild` | Predicate: can the cache be rewritten in-place? |
| `ds4.h` | `ds4_session_rewrite_from_common` | Main rewrite entry point |
| `ds4.h` | `ds4_session_common_prefix` | Token-by-token common prefix scan |
| `ds4.h` | `ds4_session_invalidate` | Clear all cached state |
| `ds4.h` | `ds4_session_rewind` | Truncate to position |
| `ds4.c` | `ds4_session_rewrite_requires_rebuild` | Returns `common < live_len` |
| `ds4.c` | `ds4_session_rewrite_from_common` | Validates inputs, checks checkpoint validity, delegates |
| `ds4.c` | `ds4_session_common_prefix` | Simple token comparison |
| `ds4.c` | `ds4_session_sync` | TP mirror + local sync + drain split logits |
| `ds4.c` | `ds4_session_sync_internal` | CPU decode suffix or full re-decode |
| `ds4.c` | `ds4_tokens_starts_with` | Prefix match guard for sync |
| `ds4.c` | `ds4_session_invalidate` | Send INVALIDATE frame, reset all cached state |
| `ds4.c` | `ds4_session_rewind` | Send REWIND frame, truncate checkpoint |
| `ds4_tp.h` | `DS4_TP_FRAME_REWIND` | `3` — TP frame type for rewind |
| `ds4_tp.h` | `DS4_TP_FRAME_INVALIDATE` | `4` — TP frame type for invalidate |
| `ds4_tp.h` | `ds4_tp_send_sync` | Send SYNC frame to TP workers |
| `ds4_tp.h` | `ds4_tp_send_rewind` | Send REWIND frame to TP workers |
| `ds4_tp.h` | `ds4_tp_send_invalidate` | Send INVALIDATE frame to TP workers |
| `ds4_tp.h` | `ds4_tp_wait_command_ack` | Wait for worker ACK after mirror sync |
| `ds4_tp.h` | `ds4_tp_recv_logits_half` | Drain worker's split logits after sync |

## Common-Prefix Rewrite

### Result Codes

| Code | Value | Meaning |
|---|---|---|
| `DS4_SESSION_REWRITE_ERROR` | `-1` | Internal error. Preconditions failed: no session, empty prompt, context overflow, no checkpoint, prefix mismatch, or unreachable state. Also returned when `ds4_session_sync` returns non-zero during the `common == checkpoint.len` path. |
| `DS4_SESSION_REWRITE_OK` | `0` | Live checkpoint now matches the target prompt. |
| `DS4_SESSION_REWRITE_REBUILD_NEEDED` | `1` | Backend state cannot be rewritten in-place. Caller must restore an older checkpoint (if available) and call `ds4_session_sync`, or fall back to a full KV rebuild. No mutation to session state. |
| `DS4_SESSION_SYNC_INTERRUPTED` | `2` | Cooperative cancellation fired during sync. Maintained as bare `#define` (not enum) because `ds4_session_sync` returns `int`, not the rewrite result type. |

### `ds4_session_rewrite_requires_rebuild(live_len, canonical_len, common)`

Pure predicate. Returns `true` when `common < live_len` — the canonical prompt wants to replace already-sampled tokens. Also catches negative or out-of-range values. Only returns `false` when `common == live_len`, meaning the prompt extends exactly at the live end (safe in-place).

Cannot perform in-place rewrite behind the live end because the backend state contains raw SWA rows, compressed KV, indexer rows, and compressor frontiers — all tied to the token position, not replaceable without a full frontier restore.

### `ds4_session_rewrite_from_common(s, prompt, common, err, errlen)`

Main rewrite entry point. Validates inputs in order:

1. Null session or prompt → `DS4_SESSION_REWRITE_ERROR` ("missing session or prompt")
2. Empty prompt → `DS4_SESSION_REWRITE_ERROR` ("empty prompt")
3. Prompt exceeds context (needs one token of generation room) → `DS4_SESSION_REWRITE_ERROR`
4. No valid checkpoint → `DS4_SESSION_REWRITE_ERROR` ("session has no valid checkpoint")
5. Common out of range, or `checkpoint.v[i] != prompt->v[i]` for any `i < common` → `DS4_SESSION_REWRITE_ERROR`

After validation:

- **`common == checkpoint.len`**: prompt extends at live end. Delegates to `ds4_session_sync`. Returns `DS4_SESSION_REWRITE_OK` on success, `DS4_SESSION_REWRITE_ERROR` on sync failure.
- **`requires_rebuild` returns true**: Returns `DS4_SESSION_REWRITE_REBUILD_NEEDED`. No mutation of session state.
- **Unreachable**: Returns `DS4_SESSION_REWRITE_ERROR` ("unexpected canonical rewrite state").

```c
if (common == s->checkpoint.len) {
    return ds4_session_sync(s, prompt, err, errlen) == 0 ?
        DS4_SESSION_REWRITE_OK : DS4_SESSION_REWRITE_ERROR;
}

if (ds4_session_rewrite_requires_rebuild(
        s->checkpoint.len, prompt->len, common)) {
    snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
             common, s->checkpoint.len, prompt->len);
    return DS4_SESSION_REWRITE_REBUILD_NEEDED;
}

return DS4_SESSION_REWRITE_ERROR;
```

### `ds4_session_common_prefix(s, prompt)`

Token-by-token comparison of `s->checkpoint.v[i] == prompt->v[i]` up to the shorter length. Returns 0 when `s->checkpoint_valid` is false. No heap allocation, no side effects.

```c
if (!s->checkpoint_valid) return 0;
int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
int i = 0;
while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
return i;
```

## Rewind

### `ds4_session_rewind(s, pos)`

Truncate the live checkpoint to a token position. Under tensor parallelism, the leader mirrors the operation to workers so both engines execute the same graph sequence and per-layer gates pair up.

```
ds4_session_rewind:
  → if leader && !tp_failed:
      ds4_tp_send_rewind(tp, session_id, pos)
  → checkpoint.len = pos (clamped to [0, checkpoint.len])
  → invalidate local draft/DSpark/GLM state
```

```c
if (ds4_session_tp_leader(s) &&
    !ds4_tp_failed(s->engine->tp.ctx)) {
    (void)ds4_tp_send_rewind(s->engine->tp.ctx, s->tp_session_id, pos);
}
if (pos < 0) pos = 0;
if (pos > s->checkpoint.len) pos = s->checkpoint.len;
s->checkpoint.len = pos;
s->mtp_draft_valid = false;
ds4_session_dspark_capture_invalidate(s);
#ifndef DS4_NO_GPU
ds4_session_glm_cap_dense_cache(s);
#endif
```

On the worker side, `ds4_tp_dispatch` dispatches `DS4_TP_FRAME_REWIND` to the same `ds4_session_rewind` on the mirrored session.

## Invalidation

### `ds4_session_invalidate(s)`

Clear all cached session state. Under tensor parallelism, the leader mirrors to workers first.

```
ds4_session_invalidate:
  → if leader && !tp_failed:
      ds4_tp_send_invalidate(tp, session_id)
  → checkpoint_valid = false
  → checkpoint.len = 0
  → invalidate local draft/DSpark/GLM state
```

```c
if (!s) return;
if (ds4_session_tp_leader(s) &&
    !ds4_tp_failed(s->engine->tp.ctx)) {
    (void)ds4_tp_send_invalidate(s->engine->tp.ctx, s->tp_session_id);
}
s->checkpoint_valid = false;
s->checkpoint.len = 0;
s->mtp_draft_valid = false;
ds4_session_dspark_capture_invalidate(s);
#ifndef DS4_NO_GPU
ds4_session_glm_reset_dense_cache(s);
#endif
```

`ds4_session_invalidate` is the common failure recovery path in `ds4_session_sync` under TP: if local sync, worker ACK, or logits drain fails, the leader invalidates both sides to keep state consistent.

On the worker side, `ds4_tp_dispatch` dispatches `DS4_TP_FRAME_INVALIDATE` to the same `ds4_session_invalidate` on the mirrored session.

## Sync & TP Mirror

### `ds4_session_sync_internal`

Returns `DS4_SESSION_SYNC_INTERRUPTED` (2) if `ds4_session_cancelled(s)` fires before prefill begins.

Checks `ds4_tokens_starts_with(prompt, &s->checkpoint)` — the checkpoint must be a prefix of the prompt (not just equal). If true, extends by decoding only the suffix tokens. Otherwise discards the checkpoint and prefills from token zero.

For the CPU path, suffix tokens decoded one at a time via `forward_token_raw_swa_cpu_decode_scratch`. Cancellation checked between each token; partial progress preserved (`checkpoint_valid = true` on interrupt so session remains in a defined state).

GPU path follows same prefix-extend or full-rebuild logic but dispatches batch metal graph operations instead of per-token CPU loops.

### TP Sync Mirror

```
ds4_session_sync(s, prompt):
  if leader:
    ds4_tp_send_sync(tp, session_id, prompt tokens)
  rc = ds4_session_sync_internal(s, prompt, err, errlen)
  if rc == 0:
    GLM debug dump (if env vars set)
  if leader:
    worker_ok = ds4_tp_wait_command_ack(tp, session_id, ...)
    if worker_ok && vocab_split:
      ds4_tp_recv_logits_half(tp, s->logits + vhalf, vhalf)
    if rc != 0 || !worker_ok || !logits_ok:
      ds4_session_invalidate(s)
  return rc
```

On any failure (local sync error, worker ACK timeout, missing logits half), the leader calls `ds4_session_invalidate(s)` which propagates to the worker via `ds4_tp_send_invalidate`. This keeps both sides consistent.

TP sync mirror must drain worker split logits via `ds4_tp_recv_logits_half` even when the local sync fails. Without draining, the worker's subsequent logits frame would be misinterpreted as the next command. `DS4_TP_FRAME_SYNC_ACK` (8) is sent by workers after completing the mirrored prefill; the leader waits for this via `ds4_tp_wait_command_ack` before proceeding to decode.

## Lifecycle

```
User edits prompt → server computes common prefix:
  ds4_session_common_prefix(s, new_prompt) → common N

server calls rewrite:
  ds4_session_rewrite_from_common(s, new_prompt, N, err, errlen)

  case DS4_SESSION_REWRITE_OK:
    // checkpoint matches, may have decoded suffix
    server proceeds to decode

  case DS4_SESSION_REWRITE_REBUILD_NEEDED:
    // cannot rewrite in place
    server restores older checkpoint (if available from disk KV)
    → ds4_session_sync(s, new_prompt, err, errlen)

  case DS4_SESSION_REWRITE_ERROR:
    // precondition failed or sync failure
    caller error handling

User abandons conversation:
  ds4_session_invalidate(s)
  // all cached state cleared

User trims conversation:
  ds4_session_rewind(s, pos)
  // cache truncated to pos, future prefills extend from there
```

## Relationship

- **Depends on**: [kv-cache-lifecycle.md](kv-cache-lifecycle.md) (checkpoint records live backend state), sync strategy (`ds4_session_sync` with `ds4_tokens_starts_with` prefix detection), [multi-gpu-pipeline.md](multi-gpu-pipeline.md) / [distributed-protocol.md](distributed-protocol.md) (TP mirroring via `ds4_tp_send_*` for REWIND/INVALIDATE/SYNC frames), [dspark.md](dspark.md) (capture via `ds4_session_dspark_capture_invalidate`), [glm-model-path.md](glm-model-path.md) (dense cache via `ds4_session_glm_reset_dense_cache`, `ds4_session_glm_cap_dense_cache`).
- **Used by**: server (branching conversations), agent (tool call insertion), CLI (edit history).
- **Alternatives**: full session destroy+create (always works, always rebuilds from scratch), disk KV checkpoint restore (older checkpoint from disk before falling back to full replay in `DS4_SESSION_REWRITE_REBUILD_NEEDED` path).

## Notes

- Rewrite only works when the live checkpoint and canonical prompt share a real prefix — the tokens must be identical, not just at the same positions. `ds4_session_rewrite_from_common` validates this with explicit token-by-token comparison for `i < common`.
- `DS4_SESSION_SYNC_INTERRUPTED` (2) is a bare `#define`, not in the `ds4_session_rewrite_result` enum. Returned only by `ds4_session_sync`, not by rewrite functions.
- When `ds4_session_sync` is interrupted (`DS4_SESSION_SYNC_INTERRUPTED`), `s->checkpoint_valid` remains `true` and partial progress is preserved. Callers must restore an older checkpoint and retry for full consistency.

[← Back to Index](../README.md)
