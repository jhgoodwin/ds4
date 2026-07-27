# Distributed Protocol (Pipeline + TP)

## Definition

DS4 supports two distributed inference modes: pipeline parallelism (split layers across machines) and tensor parallelism (split experts across machines). Both use network transport to coordinate.

## Why It Exists

Single machine RAM insufficient for largest models. Pro (1.6T) requires ~900 GB at Q4. Pipeline parallelism sums RAM across machines. TP reduces per-machine expert memory by half.

## Where It Appears

| File | Role |
|---|---|
| `ds4_distributed.c` | pipeline-parallel protocol |
| `ds4_distributed.h` | public API |
| `ds4_tp.c` | tensor-parallel transport |
| `ds4_tp.h` | TP identity, gate schedule |

## Protocol

Two modes share a common transport layer but differ in coupling.

**Pipeline parallelism:**
Coordinator tokenizes, runs layers 0..N, sends activation over network, receives logits, samples. Worker receives activation, runs layers N+1..end, sends logits back. Activation wire format supports variable bit widths: default float32, with packing to 16-bit or 8-bit per element to reduce bandwidth. Wire size = `n_embd * n_hc * (activation_bits / 8)` bytes. TCP sufficient — transfers infrequent between pipeline stages.

**Tensor parallelism:**
Each machine runs all layers but owns half the experts. Gate schedule controls when expert mid-activations exchange fires per layer. Overlaps compute with network. RDMA preferred with TCP fallback — tight coupling demands ~1 µs latency.

Transport characteristics:

| Transport | Latency | Bandwidth | Setup |
|---|---|---|---|
| RDMA (TB5) | ~1 µs | ~40 GB/s | direct cable, IPv4 on interface |
| TCP | ~100 µs | ~1 GB/s | standard network |

## Framing

Every frame starts with a 12-byte header:

```
[4 bytes: magic 0x44533444 ("DS4D")] [4 bytes: message type] [4 bytes: payload size]
```

Types are numeric constants (`ds4_distributed.c`):

| Value | Constant | Direction | Purpose |
|---|---|---|---|
| 1 | `DS4_DIST_MSG_HELLO` | bidirectional | model identity exchange |
| 2 | `DS4_DIST_MSG_ERROR` | bidirectional | error signaling |
| 3 | `DS4_DIST_MSG_WORK` | coordinator → worker | activation payload (hc/logits request) |
| 4 | `DS4_DIST_MSG_RESULT` | worker → coordinator | hidden states, logits, or ack |
| 5 | `DS4_DIST_MSG_SNAPSHOT_SAVE_REQ` | coordinator → worker | KV snapshot save request |
| 6 | `DS4_DIST_MSG_SNAPSHOT_BEGIN` | worker → coordinator | KV snapshot begin |
| 7 | `DS4_DIST_MSG_SNAPSHOT_CHUNK` | worker → coordinator | KV snapshot chunk transfer |
| 8 | `DS4_DIST_MSG_SNAPSHOT_DONE` | worker → coordinator | KV snapshot done |
| 9 | `DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN` | coordinator → worker | KV snapshot load begin |

Work messages carry work flags classifying the activation kind (`DS4_DIST_RESULT_HIDDEN_STATE`, `DS4_DIST_RESULT_LOGITS`, `DS4_DIST_RESULT_ACK`). Result frames carry the same classification.

## KV Snapshot Sub-Protocol

Workers save/load KV cache snapshots over the same TCP connection using five message types:

- **Save**: coordinator sends `SNAPSHOT_SAVE_REQ` → worker responds with `SNAPSHOT_BEGIN` → `SNAPSHOT_CHUNK` (8 MB each) → `SNAPSHOT_DONE`.
- **Load**: coordinator sends `SNAPSHOT_LOAD_BEGIN` → worker responds with the same chunk sequence.

Chunk size fixed at 8 MB (`DS4_DIST_SNAPSHOT_CHUNK_BYTES`). Layers and sessions framed within the snapshot payload via nested headers.

## Pipeline Flow

Coordinator: bind → accept → hello → loop: {compute → send WORK → recv RESULT}
Worker: connect → hello → loop: {recv WORK → compute → send RESULT}

## Relationship

- **Depends on**: engine API (same `ds4_session` calls)
- **Used by**: server, CLI (multi-machine config)
- **Alternatives**: single-machine (no network)
- **KV snapshot**: workers persist/restore KV cache via five `SNAPSHOT_*` message types over the same TCP connection

[← Back to Index](../README.md)
