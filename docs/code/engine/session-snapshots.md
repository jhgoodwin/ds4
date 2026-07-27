# Session Snapshots (Disk KV Cache)

## Files

- `ds4.c` — payload serialization
- `ds4_kvstore.c` — disk KV store (file management)

## Purpose

Serialize session state to disk so server can evict/restore sessions across requests.  Stores minimum state needed for next token to match a session that just finished prefilling.

## Snapshot Format

```
fixed 13-field u32 header (magic, version, ctx_size, prefill_cap, raw_cap, raw_window, comp_cap, token_count, layers, head_dim, indexer_head_dim, vocab, raw_live)
→ checkpoint tokens (the prompt)
→ last logits
→ per-layer compressed row counts (n_comp per layer)
→ per-layer indexer row counts (n_index_comp per layer; 0 when compress ratio < 4)
→ raw SWA rows (last logical window only)
→ compressed attention rows
→ compressor/indexer frontiers
```

Header field layout (0-indexed):

| Field | Name | Purpose |
|-------|------|---------|
| 0 | magic | `DS4_SESSION_PAYLOAD_MAGIC` — identity marker |
| 1 | version | `DS4_SESSION_PAYLOAD_VERSION` — format version |
| 2 | ctx_size | total context window (tokens) |
| 3 | prefill_cap | prefill capacity (tokens) |
| 4 | raw_cap | raw ring-buffer capacity |
| 5 | raw_window | raw SWA window size |
| 6 | comp_cap | compressed cache capacity |
| 7 | token_count | current token position |
| 8 | layers | number of transformer layers |
| 9 | head_dim | raw attention head dimension |
| 10 | indexer_head_dim | indexer head dimension |
| 11 | vocab | vocabulary size |
| 12 | raw_live | number of live raw rows (GLM: `full_live`) |

### Magic and Version Constants

Defined in `ds4.h`:

```c
#define DS4_SESSION_PAYLOAD_MAGIC    UINT32_C(0x34565344)  /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION  UINT32_C(2)
```

- **Magic** (`0x34565344` = ASCII `"DSV4"`): Identifies a DS4 session payload. Checked at load time to reject non-DS4 data or corrupted files.
- **Version** (`2`): Format version for forward-compatibility gating. Incremented when serialization layout changes. Checked alongside magic.

Both checked in `ds4_session_load_payload()`:

```c
if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
    payload_set_err(err, errlen, "unsupported session payload version");
    return 1;
}
```

A layer-level variant (`DS4_SESSION_LAYER_PAYLOAD_MAGIC` / `DS4_SESSION_LAYER_PAYLOAD_VERSION`) exists for per-layer partial payloads used in distributed pipeline-parallel saves.

### Compatibility

| Scenario | Behavior |
|----------|----------|
| Magic mismatch | Load fails with `"unsupported session payload version"`. File is not a DS4 payload or is corrupt. |
| Version mismatch | Load fails with same error. Format has changed and is not forward-compatible. |
| Header shape mismatch | Payload loads but engine may produce incorrect output. Not explicitly validated — callers must ensure model architecture matches. |

**Backward/forward compatibility:** None. Payload format is model-specific and version-locked. A snapshot saved by one version of the engine can only be loaded by the same or a carefully wire-compatible version. The magic+version check is a hard gate: if either mismatches, load is rejected entirely.

The payload is **not self-describing**. The header stores shape fields (`layers`, `head_dim`, `indexer_head_dim`, `vocab`) that identify the model architecture, not a model ID. If these shapes differ between save and load, behavior is undefined — the caller must ensure the session was created with the same model.

### Three-way Engine Branch

The save path branches on engine type:

- **GPU (default):** Header uses `DS4_N_HEAD_DIM` (raw head dim), `DS4_N_INDEXER_HEAD_DIM`. Serializes raw ring (logical order), compressed rows + compressor state (attn_state_kv, attn_state_score), and indexer rows + indexer state (ratio=4 only).
- **CPU:** Same header fields as GPU. Raw cache serialized from ring buffer via start-of-window offset.
- **GLM:** Header uses `DS4_N_KEY_MLA` / `DS4_N_VALUE_MLA` instead of `DS4_N_HEAD_DIM` / `DS4_N_INDEXER_HEAD_DIM`; uses `g->normal_layers` instead of `DS4_N_LAYER`; `raw_live` field stores `full_live` (from `session_glm_full_live_rows`). Serializes full KV cache plus compact KV cache (`kv_lora`, `k_rope`) and indexer key cache (for layers using full indexer).

## Save/Restore Lifecycle

See [kvstore.md](../modules/kvstore.md) for snapshot save/load API and lifecycle. See [kv-cache-lifecycle.md](../concepts/kv-cache-lifecycle.md) for KV cache lifecycle and compression triggers.

The engine stages payloads to temp files before handing to the KV store for writing. This two-phase approach (stage vs save) avoids holding serialization state while the KV store opens its target file. Chunked I/O: `DS4_SESSION_IO_CHUNK` (8 MB) per write.

Raw cache serialized as last logical window only (suffix prefill writes own rows). Compressed caches serialized up to live row counts (sparse attention may select from full prefix).

### stage_payload vs save_snapshot

Two distinct functions with different ownership and lifecycle:

| Aspect | `ds4_session_stage_payload` | `ds4_session_save_snapshot` |
|--------|----------------------------|----------------------------|
| **Scope** | Engine-level | Module-level (calls into kvstore) |
| **Output** | Temp file on disk (`ds4_session_payload_file`) | In-memory buffer (`ds4_session_snapshot`) |
| **Storage** | At-rest staging (manages staging buffers) | Passed to kvstore for durable write |
| **Ownership** | Engine owns the temp file path | Caller owns the buffer pointer |
| **Cleanup** | `ds4_session_payload_file_free()` | `ds4_session_snapshot_free()` |
| **When used** | HTTP/agent code stages payload, then writes staged file to KV store | Direct in-memory snapshot (e.g. tests, embedded use) |

**`ds4_session_stage_payload`** serializes session state to a temp file (`/tmp/ds4-session-payload.XXXXXX`) via `mkstemp`. Returns a `ds4_session_payload_file` struct with path and byte count. The caller then reads this file and writes it to the KV store (or any persistent backend). This decouples serialization from I/O — the engine does not need to know about the KV store's file system.

**`ds4_session_save_snapshot`** serializes session state into an in-memory buffer (`ds4_session_snapshot`). Used when the caller wants the payload bytes directly (e.g. for kvstore integration, testing, or custom persistence). The snapshot struct holds `ptr`, `len`, `cap`.

Both call the same internal `ds4_session_save_payload()` — the difference is the output medium (file vs memory) and the lifecycle contract.

### Distributed path

Distributed (pipeline-parallel) sessions save per-GPU layer payloads independently. `ds4_session_save_payload` dispatches to `ds4_dist_session_save_payload` which serializes each GPU's layers into a combined payload. Snapshot save (`ds4_session_save_snapshot`) returns an error for distributed sessions — snapshots are not yet supported in distributed mode.

### Format Details

Payload is model-specific, not self-describing. Header shape fields (`layers`, `head_dim`, `indexer_head_dim`, `vocab`) identify the model architecture, not a model id. Field 12 (`raw_live`) stores the number of live raw rows — for GLM it stores `full_live` (from `session_glm_full_live_rows`).

## See Also

- [distributed-protocol.md](../concepts/distributed-protocol.md) — distributed session snapshot serialization across GPUs

[← Back to Index](../README.md)
