# Tensor Parallelism

## Files

- `ds4_tp.c` — transport layer: RDMA over Thunderbolt UC QP, TCP fallback, lockstep control plane, gate exchange
- `ds4_tp.h` — `ds4_tp` handle, `ds4_tp_identity`, gate schedule API, frame protocol, worker entry

## Purpose

Split routed experts across two identical machines, each holding one contiguous half.  Both machines load the same shared (non-routed) layers; each holds half the experts.  Halves the per-machine RAM requirement for large models.  Both ranks execute the same graph sequence in lockstep; partial expert outputs are exchanged every layer.

## Architecture

Two ranks: rank 0 (leader) runs a normal frontend session; rank 1 (worker) mirrors every `ds4_session_sync()` / `ds4_session_eval()` call over a TCP control socket.  Both engines execute identical graph sequences.

Two transport planes:

- **Control socket** — TCP stream, framed commands.  Used for session mirroring (sync, eval, rewind, invalidate, stop), identity exchange, and logit-half transfer.  Created during bring-up.
- **Gate socket** — second TCP socket (same ports) dedicated to gate payloads so control frames never interleave with data.  Under RDMA, the actual exchange goes over the Thunderbolt UC queue pair; the TCP socket carries only the header handshake for verify-block and big gates.

Bring-up sequence:

1. Leader listens; worker dials with retry.
2. Both exchange `ds4_tp_identity` (model shape, gate schedule, RDMA capability) — mismatch aborts before inference.
3. RDMA device opened and slab registered (if both sides capable and transport not forced TCP).
4. Second socket created for gate traffic.
5. Engine calls `ds4_engine_tp_bind()` to register the gate-slab callback and start the worker loop.

Worker entry: `ds4_tp_worker_run()` receives mirrored commands in a loop, dispatches to the local engine, and ships logits half back to the leader after each sync.

## Sharding Strategy

Expert splitting is static and contiguous.  Each rank loads the same GGUF file; the engine skips non-local expert weights at load time based on the TP rank.  Rank 0 owns expert indices `[0, n_expert/2)`, rank 1 owns `[n_expert/2, n_expert)`.  Non-routed layers (attention, embeddings, output) are duplicated on both machines — no split.

During a decode token, each rank runs the full shared layers locally, then only its half of routed experts.  The `out` slab holds partial outputs; the `in` slab receives the peer's partials.

The slab layout (single contiguous GPU-visible allocation):

```
out vectors   S * vec_bytes     written by local GPU kernels
in  vectors   S * vec_bytes     RDMA/TCP-written with peer partials
in  seq flags S * 8             written strictly after each in vector
token slot    16                {seq u64, token i32, pad} leader→worker
```

Where `S = n_layer * 2` (one slot per attention gate + one per FFN gate), `vec_bytes = n_embd * 4` (f32 partials, never quantized on wire).

For speculative decode, additional batch regions exist: `n_layer * BATCH_MAX_ROWS * vec_bytes` each for out and in, holding row partials for the verify-block gates.

## Collective Ops

Three gate-exchange primitives, all symmetric (both sides send then receive):

### Decode gate (`ds4_tp_gate_exchange`)

Exchanges one token's partial output for one layer/gate pair.  Per-token, the engine fires attention and FFN gates for every layer.

- **RDMA path**: lookahead recv-window posted on first gate.  Each gate posts send of the local out-slot, waits for the peer's in-slot completion, then reposts one recv to slide the window.  UC queue pair delivers in-order; vectors larger than the driver's 16KB message cap split into contiguous chunks.
- **TCP path**: symmetric `writev` (header + payload) then `read` of peer's header + payload.  16KB per direction fits socket buffers; write-then-read order prevents deadlock.

### Verify-block batch gate (`ds4_tp_batch_gate_exchange`)

Bulk exchange of `rows` row partials for one layer across all speculative-block rows at once.

- **RDMA path**: TCP header handshake, drain the decode recv-window with dummy sends on both sides, then bulk RDMA exchange over the latency QP.
- **TCP path**: single `writev` of header + all rows, then read of peer's data.

### Prefill batch gate (`ds4_tp_big_gate_exchange`)

Arbitrary-size symmetric payload exchange for prompt prefill.

- **RDMA path**: same drain-and-swap as verify-block, but over the registered slab path.
- **TCP path**: interleaved 2MB write/read rounds (socket buffers absorb one round, preventing deadlock).

### Logit half transfer

After every sync and eval, the worker sends its half of the output logit vector to the leader over the control socket.  The leader interleaves both halves into the full output.

### Hash check (`ds4_tp_hash_check`)

Debug lockstep verification: both sides send their hidden-state hash for a token and compare.  Returns on mismatch.

## Invariants

- Both machines must load the identical model (GGUF bytes, layer count, embedding dim, vocab size, quant bits all match — validated in hello).
- Both must agree on the gate schedule (slot start/step/gates-per-token).
- Gate exchange ordered by sequence number; desync (wrong layer/gate/seq in header) marks TP as failed.
- TP transport initialized before `ds4_engine_tp_bind()`.
- Network latency adds to decode time; overlapped compute mitigates.
- Plausible speculative blocks (≤5) keep batch gate row count within `DS4_TP_BATCH_MAX_ROWS`.
- Worker discards its own logit output on vocab-split models; leader combines both halves.
- RDMA path requires both sides to have a functional verbs device.  Forced RDMA (`--transport rdma`) errors out if missing.

## See Also

- [Distributed Protocol](../concepts/distributed-protocol.md)
- [Multi-GPU Pipeline](../concepts/multi-gpu-pipeline.md)
- [GPU Tensor Primitives](../concepts/gpu-tensor-primitives.md)

[← Back to Index](../README.md)
