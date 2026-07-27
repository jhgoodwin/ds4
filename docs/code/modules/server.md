# Server

## Files

- `ds4_server.c` — HTTP API, worker queue, streaming, tool calls, disk KV cache policy

## Purpose

OpenAI/Anthropic-compatible HTTP API. Handles multi-session management, request queuing, streaming responses, tool-call parsing.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `/v1/chat/completions` | OpenAI-compatible chat |
| `/v1/completions` | OpenAI-compatible text |
| `/v1/messages` | Anthropic-compatible |
| `/health` | Readiness probe |

## Architecture

```
HTTP listener → request parser → session lookup/creation →
client thread enqueues job → worker (batched: decode coordinator + slot workers / non-batched: dequeuer + worker_main) → engine inference → streaming response
```

## Batch Decode Scheduling

In `ds4_server.c`, the server uses a dedicated decode thread with coalescing.

### Server Fields
- `decode_pending` — global count of pending decode items.
- `active_generations` — number of active generation slots.
- `model_busy` — flag held while batch is executing.
- `model_cv` — condition variable for decode thread synchronization.

### Server Slot Fields
- `decode_pending`, `decode_in_flight`, `decode_done` — state flags for slot's decode lifecycle.
- `decode_token` — token to decode.
- `decode_rc`, `decode_err` — result stored by decode worker.

### `server_eval_token` (non‑batched path)

Two modes:
- **Non-batched** (`!s->batched_mode`): `ds4_session_eval()` directly under `inference_mu`.
- **Batched**: Sets `slot->decode_pending=true`, increments `s->decode_pending`, broadcasts `model_cv`, waits on `slot->decode_done` via `pthread_cond_wait`. Reads `slot->decode_rc` after wake.

### Batched mode thread structure

Two thread pools coordinate under `model_mu` / `model_cv`:

1. **Decode coordinator** (`decode_worker_main`): single thread. Waits on `model_cv` until `decode_pending > 0`, coalesces pending slots, calls `ds4_sessions_eval_batch()`, stores results per-slot, broadcasts `model_cv`.
2. **Slot workers** (`slot_worker_main`): one per slot. Each waits on `s->cv` for job assignment, runs `generate_job()` which calls `server_eval_token()` per token. In batched mode, `server_eval_token` sets `slot->decode_pending`, increments `s->decode_pending`, broadcasts `model_cv`, then waits on `slot->decode_done` via `pthread_cond_wait`. After decode completes, the slot worker signals `job->done` and re-dispatches queued jobs.

**Communication**: slot → coordinator via `decode_pending` flag + `model_cv` broadcast. Coordinator → slot via per-slot `decode_done` flag + `model_cv` broadcast. No per-slot condition variable; all wait on `model_cv` with a check of their own `decode_done`.

### `decode_worker_main`

Dedicated decode thread:

1. Wait on `model_cv` until `s->decode_pending > 0`.
2. **Coalesce** (`coalesce_us`, default 2000µs, configurable via `DS4_SERVER_DECODE_COALESCE_US`). If pending slots < total slots and pending < active_generations, wait with `pthread_cond_timedwait` up to `coalesce_us` to let more slots enqueue.
3. Drain all pending slots into `ds4_decode_item[]`. Transition each from `decode_pending` → `decode_in_flight`.
4. Set `model_busy = true`, unlock `model_mu`.
5. Call `ds4_sessions_eval_batch(items, count)` under `inference_mu`.
6. Re-lock `model_mu`, set `model_busy = false`.
7. For each completed slot: set `decode_rc`, copy error string, set `decode_done = true`, clear `decode_in_flight`.
8. Broadcast `model_cv` to wake waiting slot threads.

Coalescing trades small latency for larger batches: waiting 2ms can double or triple batch size on low-concurrency workloads, improving GPU utilization.

## Session Management

- One session per conversation
- Disk KV cache eviction on disk budget overrun (budget_bytes)
- Configurable max concurrent sessions
- Session reuse across requests in same conversation

## Streaming

SSE for token-by-token output. Supports:

- Token text chunks
- Tool call deltas
- Finish reason
- Usage statistics

## Tool Calls

- Parse tool definitions from OpenAI/Anthropic format
- Canonicalize to internal representation
- Execute server-side
- Stream tool call deltas

## Dependencies

- **Imports from**: `ds4.h`, `ds4_distributed.h`, `ds4_gpu_args.h`, `ds4_help.h`, `ds4_kvstore.h`, `rax.h`
- **Does not import**: `ds4_web.h` (web browser tool is a separate subsystem; server routes tool name strings only)
- **Uses**: libc networking (`<sys/socket.h>`, `<netinet/in.h>`), pthread worker pool

## Invariants

- Session pinned to one worker thread during inference.
- KV cache eviction serialized (not concurrent with session access).
- Streaming buffer flushed every token (inline per-delta send_all; no periodic flush timer).

## Configuration

### Mode selection

- **Batched mode**: activated when `--batched-sessions N` (`N > 0`). Creates `N` resident session slots with dedicated slot worker threads and a decode coordinator thread.
- **Non-batched mode**: activated when `--batched-sessions` is omitted or `0`. Uses a single session slot (index 0) with one dequeuer-worker thread (`worker_main`).

### Environment variables

| Variable | Default | Role |
|---|---|---|
| `DS4_SERVER_DECODE_COALESCE_US` | `2000` | Decode coalesce window in µs (0 disables) |
| `DS4_SERVER_BATCH_LOG` | unset | When set, log per-batch slot membership |
| `DS4_SERVER_DISABLE_THINK_TOOL_RECOVERY` | unset | When set, skip thinking-token recovery for tool calls |
| `DS4_SERVER_PREFILL_QUANTUM` | `2048` | Prefill token chunk size (no active generation) |
| `DS4_SERVER_MIXED_PREFILL_QUANTUM` | `128` | Prefill token chunk size (during active generation) |
| `DS4_MTP_SPEC_DISABLE` | unset | When set, disable MTP speculative decoding |

## See Also

- [Engine API](../engine/engine-api.md)
- [KV Cache](../engine/kv-cache.md)
- [Session Batch Decode](../concepts/session-batch-decode.md)
- [KV Cache Lifecycle](../concepts/kv-cache-lifecycle.md)

[← Back to Index](../README.md)
