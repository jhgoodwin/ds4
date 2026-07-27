# Distributed Inference

## Files

- `ds4_distributed.c` — pipeline-parallel coordinator/worker protocol
- `ds4_distributed.h` — public API for distributed mode

## Purpose

Glue multiple machines via pipeline parallelism to sum RAM and run larger models. One coordinator owns tokenization, sampling, and layers 0..N. One or more workers own layers N+1..output.

## Dependencies

- **Imports from**: `ds4.h` (engine, session, tokens), system headers (socket, pthread, poll, netdb)
- **Exports to**: `ds4.c` (session delegation), `ds4.h` (option types), CLI `ds4.c` (flag parsing, `ds4_dist_run()`)
- **Init order**: after `ds4_engine_open()`, before first `ds4_session_sync()`. Coordinator opens listener during `ds4_dist_session_create()`; worker connects during `ds4_dist_run()`.

## Architecture

Coordinator → network → Worker. Activations sent over TCP. Workers return logits to coordinator. Network transport transparent to inference code — same `ds4_session` API on both ends. Coordinator and worker run same binary with different flags.

## Key Types

| Type | Role |
|---|---|
| `ds4_distributed_options` | CLI-parsed role, layer range, listen/coordinator address, prefill tuning, activation bits |
| `ds4_distributed_layers` | Layer range descriptor: start, end, has_output |
| `ds4_dist_generation_options` | Standalone coordinator generation config (prompt, sampling, logit dumps) |
| `ds4_dist_session` | Opaque coordinator session wrapping state, listener, accept thread, route plan |
| `ds4_dist_worker_entry` | Registered worker metadata: fd, host, port, model id, quant bits, layer range |
| `ds4_dist_coordinator_state` | Coordinator runtime: engine ref, worker registry (linked list), activation bits, mutex |
| `ds4_dist_worker_state` | Worker runtime: engine ref, layer range, listen fd, session list (linked list) |
| `ds4_dist_worker_upstream` | Worker upstream connection to coordinator (fd, write mutex, forwarder list) |
| `ds4_dist_worker_forwarder` | Worker-to-worker forwarder for multi-hop routes (host, port, fd, thread, pending queue) |
| `ds4_dist_worker_job_queue` | Bounded worker work queue with condvar signaling |
| `ds4_dist_route_plan` | Resolved route for one forward pass (ordered entries + blob) |
| `ds4_dist_route_entry` | Single route hop (host, port, layer range, flags, fd) |
| `ds4_dist_frame_header` | Wire frame header (magic `DS4D`, type, byte count) |
| `ds4_dist_hello_fixed` | Worker hello payload: model id, quant bits, layer range, has_output/hidden, ctx size |
| `ds4_dist_work_fixed` | Work frame: session/request id, token range, prefix/result hash, layer range, flags, activation metadata |
| `ds4_dist_result_fixed` | Result frame: request id, status, result kind (ACK/hidden/logits), telemetry, payload |
| `ds4_dist_route_fixed` | Route entry in wire format: host, port, layer range, flags |
| `ds4_dist_snapshot_req_fixed` | Snapshot save request: model/session/request id, token hash, layer range |
| `ds4_dist_snapshot_begin_fixed` | Snapshot begin: metadata + payload size + status |
| `ds4_dist_snapshot_chunk_fixed` | Snapshot data chunk: request id, chunk byte count |
| `ds4_dist_snapshot_done_fixed` | Snapshot completion: request id, status, message |
| `ds4_dist_telemetry_fixed` | Per-hop telemetry: eval time, downstream wait, send time, input/output bytes |
| `ds4_dist_kv_layout` | KV cache layout metadata (ctx, prefill cap, raw/comp cap, head dim, vocab) |
| `ds4_dist_prefill_sender` | Coordinator prefill pipeline sender thread state (slot ring buffer, condvars) |
| `ds4_dist_prefill_result_reader` | Coordinator prefill result collector thread state (progress tracking, result buffer) |
| `ds4_dist_pending_request` | Pending forward request tracking for worker forwarders |
| `ds4_dist_worker_session` | Per-session KV state on worker, keyed by coordinator session id |

### Message Types (enum)

| Constant | Value | Direction | Purpose |
|---|---|---|---|
| `DS4_DIST_MSG_HELLO` | 1 | Worker → Coordinator | Handshake with capabilities |
| `DS4_DIST_MSG_ERROR` | 2 | Bidirectional | Error notification |
| `DS4_DIST_MSG_WORK` | 3 | Coordinator → Worker | Forward pass work item |
| `DS4_DIST_MSG_RESULT` | 4 | Worker → Coordinator | Forward pass result |
| `DS4_DIST_MSG_SNAPSHOT_SAVE_REQ` | 5 | Coordinator → Worker | Request KV snapshot save |
| `DS4_DIST_MSG_SNAPSHOT_BEGIN` | 6 | Worker → Coordinator | Snapshot transfer start |
| `DS4_DIST_MSG_SNAPSHOT_CHUNK` | 7 | Worker → Coordinator | Snapshot data chunk (8 MiB max) |
| `DS4_DIST_MSG_SNAPSHOT_DONE` | 8 | Worker → Coordinator | Snapshot transfer complete |
| `DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN` | 9 | Coordinator → Worker | Initiate KV snapshot restore |

## API Surface

### Creation / Teardown

- `ds4_dist_options_create()` — allocates zero-initialized `ds4_distributed_options`. Caller owns. (ds4_distributed.h:44, .c:8124)
- `ds4_dist_options_free()` — frees options struct. (ds4_distributed.h:45, .c:8128)
- `ds4_dist_session_create()` — creates coordinator session: opens TCP listener, spawns accept thread, initializes route state. Returns opaque `ds4_dist_session*`. (ds4_distributed.h:68, .c:5402)
- `ds4_dist_session_free()` — shuts down listener, signals shutdown to worker threads, frees route plan. Session object kept process-lifetime to avoid racing detached threads. (ds4_distributed.h:76, .c:5483)

### Core Operations

- `ds4_dist_session_sync()` — synchronizes distributed KV state to requested prompt timeline. Handles checkpoint reuse (suffix eval), pipelined prefill, and full rebuild. Returns 0 on success, 1 on error, 2 on interrupt. (ds4_distributed.h:84, .c:5517)
- `ds4_dist_session_eval()` — evaluates one additional token across the current distributed route. Sends WORK frame to downstream workers, collects results. (ds4_distributed.h:94, .c:5650)
- `ds4_dist_session_save_payload()` — saves full KV payload (coordinator + all worker shards) to a single DSV4 stream. Workers send shards via snapshot protocol; coordinator interleaves them. (ds4_distributed.h:106, .c:5092)
- `ds4_dist_session_load_payload()` — loads KV payload, splits shards across currently registered route. Sends each shard to the owning worker via snapshot load protocol. (ds4_distributed.h:112, .c:5225)
- `ds4_dist_run()` — standalone distributed mode. Workers stay in event loop; coordinator one-shot mode runs prompt and exits. (ds4_distributed.h:123, .c:8403)
- `ds4_dist_prepare_engine_options()` — applies distributed layer-loading choices to `ds4_engine_options` before model load: sets `load_slice`, `load_layer_start/end`, `load_output`. (ds4_distributed.h:59, .c:8358)

### Query / Introspection

- `ds4_dist_enabled()` — returns true if role is not `DS4_DISTRIBUTED_NONE`. (ds4_distributed.h:43, .c:8120)
- `ds4_dist_session_route_ready()` — returns 1 when full layer route is available (all workers registered), 0 when incomplete, -1 for error. (ds4_distributed.h:81, .c:5502)
- `ds4_dist_usage()` — prints distributed CLI flags to fp. (ds4_distributed.h:46, .c:8132)
- `ds4_dist_parse_cli_arg()` — parses one CLI argument for distributed options. Returns matched/not-matched/error. Handles `--role`, `--layers`, `--listen`, `--coordinator`, `--dist-prefill-chunk`, `--dist-prefill-window`, `--dist-activation-bits`, `--dist-replay-check`, `--debug`. (ds4_distributed.h:47, .c:8156)

## Data Flow

```
                        ┌──────────────────────────────────────────────────┐
                        │                  Coordinator                     │
                        │                                                  │
                        │  ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
  Prompt ───────────────►│  Tokenizer │──►│ Layers   │──►│  Output Head │─┼──► Logits
                        │  & Sample  │   │ 0..N     │   │  (if local)  │ │
                        │  └──────────┘   └────┬─────┘   └──────────────┘ │
                        │                      │                          │
                        │              Activation Tensor                  │
                        │              (quantized 8/16/32-bit)            │
                        └──────────────────────┼──────────────────────────┘
                                               │ TCP
                                               ▼
                        ┌──────────────────────────────────────────────────┐
                        │                   Worker                         │
                        │                                                  │
                        │  ┌────────────────┐   ┌──────────────────────┐  │
                        │  │  Layers        │   │  KV Cache (per-      │  │
                        │  │  N+1..output   │──►│  session, token-     │  │
                        │  │                │   │  hash verified)      │  │
                        │  └───────┬────────┘   └──────────────────────┘  │
                        │          │                                       │
                        │    Logits / Hidden State                        │
                        └──────────┼──────────────────────────────────────┘
                                   │ TCP
                                   ▼
                        Coordinator receives result
                        (logits or hidden state for next hop)

  Multi-hop route (3+ machines):
  Coordinator ──TCP──► Worker A ──TCP──► Worker B ──TCP──► ... ──TCP──► Coordinator
                       (forwarder)      (forwarder)                 (final result)

  KV snapshot flow:
  Coordinator ──SNAPSHOT_SAVE_REQ──► Worker
  Worker      ──SNAPSHOT_BEGIN──────► Coordinator (metadata + payload size)
  Worker      ──SNAPSHOT_CHUNK──────► Coordinator (8 MiB chunks, streamed)
  Worker      ──SNAPSHOT_DONE───────► Coordinator (status + optional error msg)

  KV restore flow:
  Coordinator ──SNAPSHOT_LOAD_BEGIN──► Worker (metadata + payload)
  Coordinator ──SNAPSHOT_CHUNK───────► Worker (8 MiB chunks, streamed)
  Worker      ──SNAPSHOT_DONE────────► Coordinator (status)
```

## Configuration

### CLI Flags

| Flag | Role | Default | Description |
|---|---|---|---|
| `--role` | both | — | `coordinator` or `worker` |
| `--layers` | both | — | Inclusive layer slice, e.g. `10:20` or `21:output` |
| `--listen` | coordinator | — | TCP listen address (`HOST PORT`) |
| `--coordinator` | worker | — | Coordinator TCP address (`HOST PORT`) for worker to dial |
| `--dist-prefill-chunk` | coordinator | session prefill cap (normally 4096) | Coordinator prefill pipeline chunk size. Non-default values experimental — can change logits unless validated. |
| `--dist-prefill-window` | coordinator | workers+2, capped at 8 | Max end-to-end prefill chunks in flight |
| `--dist-activation-bits` | coordinator | 32 | Hidden-state transport width: 32, 16, or 8 |
| `--dist-replay-check` | coordinator | off | Diagnostic: reset and replay prompt, compare logits |
| `--debug` | both | off | Print coordinator route/debug logs |

### Environment Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `DS4_DIST_PREFILL_SEND_DEPTH` | uint 1..8 | 2 | Coordinator prefill send pipeline depth |
| `DS4_DIST_SOCKET_BUFFER_MB` | uint 0..512 | 128 | TCP socket send/recv buffer size in MB. 0 = system default. |
| `DS4_DIST_SOCKET_TIMEOUT_SEC` | uint 1..3600 | 60 | TCP send timeout |
| `DS4_DIST_SOCKET_RECV_TIMEOUT_SEC` | uint 1..3600 | unset (no timeout) | TCP receive timeout. Unset by default — worker control sockets can be idle during KV snapshot data transfers. |
| `DS4_DIST_WORKER_PREFETCH_DEPTH` | uint 1..8 | 2 | Worker work prefetch queue depth |
| `DS4_DIST_WORKER_FORWARD_WINDOW` | uint 1..64 | 4 | Worker-to-worker forwarder pending request window |
| `DS4_DIST_DECODE_PROFILE` | (any) | off | Enable decode profiling telemetry |
| `DS4_DIST_CONNECT_TRACE` | (any) | off | Log connection candidate details for debugging |
| `DS4_DIST_CONNECT_BIND_HOST` | string | unset | Source address for outbound connections |
| `DS4_DIST_CONNECT_BIND_IF` | string | unset | Source interface for outbound connections |
| `DS4_DIST_DISABLE_PREFILL_PIPELINE` | (any) | off | Disable pipelined prefill, fall back to serial |
| `DS4_DIST_DISABLE_PREFILL_ACK_ONLY` | (any) | off | Disable ack-only prefill send slots |
| `DS4_DIST_DISABLE_WORKER_PREFETCH` | (any) | off | Disable worker work prefetch |

## KV Snapshot Protocol

KV cache snapshots are transferred between coordinator and workers using a dedicated sub-protocol over separate TCP data connections (each worker's `listen_port` from its HELLO). Snapshots are topology-independent: save gathers worker-owned layer tensors into a normal DSV4 payload; load splits a DSV4 payload across the currently registered route.

### Save Flow

1. Coordinator opens a data connection to the worker's listen port.
2. Coordinator sends `DS4_DIST_MSG_SNAPSHOT_SAVE_REQ` with `ds4_dist_snapshot_req_fixed` (model/session/request id, token hash, layer range).
3. Worker responds with `DS4_DIST_MSG_SNAPSHOT_BEGIN` carrying `ds4_dist_snapshot_begin_fixed` (metadata + payload size + status). Status 0 = ready; non-zero = error with message.
4. Worker streams the KV shard data in `DS4_DIST_MSG_SNAPSHOT_CHUNK` frames, each up to `DS4_DIST_SNAPSHOT_CHUNK_BYTES` (8 MiB). Chunk header: `ds4_dist_snapshot_chunk_fixed` (request id, chunk_bytes).
5. Worker sends `DS4_DIST_MSG_SNAPSHOT_DONE` with `ds4_dist_snapshot_done_fixed` (request id, status, optional message).
6. Coordinator interleaves received shards into a single DSV4 output stream.

### Load Flow

1. Coordinator opens a data connection to the worker's listen port.
2. Coordinator sends `DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN` with `ds4_dist_snapshot_begin_fixed` (metadata + payload size + tokens).
3. Coordinator streams the KV shard data in `DS4_DIST_MSG_SNAPSHOT_CHUNK` frames, 8 MiB each.
4. Worker loads the shard into its local KV cache.
5. Worker sends `DS4_DIST_MSG_SNAPSHOT_DONE` confirming success or reporting failure.

### Wire Format

All snapshot frames use the standard `ds4_dist_frame_header` (magic `DS4D`, type, byte count). Chunk size fixed at 8 MiB (`DS4_DIST_SNAPSHOT_CHUNK_BYTES`). Token lists in snapshot frames are encoded as `uint32_t` in network byte order. Error messages are ASCII strings appended after the fixed struct.

## Invariants

- Same `ds4_session` API on coordinator and worker.
- Coordinator and worker run same binary, distinguished by flags.
- Network transport transparent to caller.
- **P0: Pipeline parallelism uses TCP exclusively — no RDMA, no shared memory.**
- **P0: Full frame header with magic (`DS4D` = `0x44533444`), type, and byte count exists on every wire frame.**
- **P0: Message types are correct: HELLO(1), ERROR(2), WORK(3), RESULT(4), SNAPSHOT_SAVE_REQ(5), SNAPSHOT_BEGIN(6), SNAPSHOT_CHUNK(7), SNAPSHOT_DONE(8), SNAPSHOT_LOAD_BEGIN(9).**
- **P0: Activation quantization supports 8-bit (E4M3), 16-bit (IEEE half), and 32-bit (full float).**
- Coordinator prefill pipeline uses chunked send with ack-only slots; chunk boundaries are deterministic for a given prompt and route.
- Worker KV sessions are keyed by coordinator-provided session id and token hash; mismatched hash causes rebuild.
- Route plans are built from the worker registry once all layers are covered; route is immutable for the lifetime of a generation.
- Coordinator accept thread is detached; session object kept process-lifetime to avoid races during shutdown.

## Notes

- **P1: Protocol wire format details (framing, activation quantization, KV snapshot sub-protocol) are documented in [distributed-protocol.md](../concepts/distributed-protocol.md).**
- Worker listen port is communicated in the HELLO message and used for out-of-band data connections (KV snapshots). The control connection (from coordinator's accept) carries WORK/RESULT frames.
- Multi-hop routes (3+ machines) use worker-to-worker forwarders: each intermediate worker opens a TCP connection to the next hop and relays activations downstream.
- Prefill pipeline uses a ring-buffer of send slots with ack tracking. The coordinator sends chunks ahead while collecting results asynchronously.
- `DS4_DIST_CONNECT_BIND_HOST` and `DS4_DIST_CONNECT_BIND_IF` are useful when machines have multiple network interfaces and need to pin distributed traffic to a specific path.
- Decode profile (`DS4_DIST_DECODE_PROFILE`) emits per-hop telemetry including eval time, downstream wait, forward send time, and input/output byte counts.

## See Also

- [Distributed Protocol](../concepts/distributed-protocol.md)
- [Multi-GPU Pipeline](../concepts/multi-gpu-pipeline.md)

[← Back to Index](../README.md)
