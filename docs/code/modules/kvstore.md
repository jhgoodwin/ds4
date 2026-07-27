# KV Store

## Files

- `ds4_kvstore.c` — disk KV checkpoint file management
- `ds4_kvstore.h` — public interface

## Purpose

Persist session graph checkpoints to disk for reuse across prompts, agent turns, and process restarts. Provides byte-prefix-addressed file storage with density-weighted eviction, text-override variants (visible transcript, thinking), and optional protocol-specific trailers. Exists as a separate compilation unit because the file format, eviction policy, and SHA1 naming are shared between `ds4-server` (automatic byte-prefix cache) and `ds4-agent` (explicit session checkpoints).

## Dependencies

- **Imports from**: `ds4.h` (ds4_engine, ds4_session, ds4_tokens, ds4_token_text, ds4_tokenize_rendered_chat), `ds4_engine.h` (ds4_engine_routed_quant_bits, ds4_engine_model_id), `ds4_session.h` (ds4_session_tokens, ds4_session_ctx, ds4_session_stage_payload, ds4_session_write_staged_payload, ds4_session_load_payload, ds4_session_invalidate), `<math.h>` (exp2), `<dirent.h>`, `<sys/stat.h>`, `<unistd.h>`
- **Exports to**: `ds4-server.c` (automatic cache policy), `ds4_agent.c` (explicit session management), `ds4_agent_session.c`
- **Init order**: Called by server/agent main after engine and session init; no static constructors.

## Key Types

| Type | Role |
|---|---|
| `ds4_kvstore` | Opaque cache handle: directory path, budget bytes, in-memory entry table, store policy options, logger |
| `ds4_kvstore_entry` | One cached file: SHA1 hex name, path, header fields (quant_bits, model_id, reason, tokens, hits, ctx_size, ext_flags, timestamps, payload_bytes, text_bytes, file_size) |
| `ds4_kvstore_options` | Tunable policy: min_tokens, cold_max_tokens, continued_interval_tokens, boundary_trim_tokens, boundary_align_tokens |
| `ds4_kvstore_eviction_context` | Incoming checkpoint metadata passed into eviction: text/text_len, model_id, quant_bits, ctx_size, reject_different_quant |
| `ds4_kvstore_trailer_hooks` | Protocol-specific trailer read/write callbacks: serialized_size, write, load; ext_flag marks ext_flags bit for this trailer type |
| `ds4_kvstore_load_result` | Load outcome: tokens, text_bytes, quant_bits, ext_flags, load_ms, consumed (file deleted if over cold_max_tokens), path |
| `ds4_kvstore_reason` | Enum: UNKNOWN=0, COLD=1, CONTINUED=2, EVICT=3, SHUTDOWN=4, AGENT_SYSTEM=5, AGENT_SESSION=6 |
| `ds4_kvstore_log_type` | Enum: DEFAULT, KVCACHE, WARNING |

## API Surface

### Creation / Teardown

- `ds4_kvstore_open()` — constructor. Creates cache directory (`mkdir -p`), sets `budget_bytes` from `budget_mb` (default 4096 MiB), scans existing `.kv` files via `kv_cache_refresh()`, runs initial eviction. Not an "open existing for read" — it's a full init that may create files and evict immediately.
- `ds4_kvstore_close()` — destructor. Frees entry table via `ds4_kvstore_clear()`, frees directory string, zeroes the struct.
- `ds4_kvstore_clear()` — frees all in-memory entries. Does **not** delete `.kv` files on disk. Resets table to empty.
- `ds4_kvstore_entry_free()` — frees `e->path`, zeroes the entry struct.
- `ds4_kvstore_load_result_free()` — frees `result->path`, zeroes the result struct.

### Store Operations

- `ds4_kvstore_store_live_prefix()` — write a new checkpoint from the live token prefix. Renders `store_len` tokens to text, computes SHA1 hex hash, checks for existing compatible file (same model/quant/ctx, matching text), stages session payload, evicts if needed, writes atomically via `.tmp.<pid>` + rename. Returns false if tokens < `min_tokens` or budget exceeded.
- `ds4_kvstore_store_live_prefix_text()` — like `store_live_prefix` but uses a caller-provided text override instead of rendered token text. Used for visible transcript, thinking, session title variants. Sets `cache_text_ext` flags in header `ext_flags`.
- `ds4_kvstore_maybe_store_continued()` — periodic waypoint store. Checks if `live_tokens` hit the `continued_interval_tokens` boundary. If so, calls `store_live_prefix` with reason `"continued"` and records via `ds4_kvstore_note_store()`.
- `ds4_kvstore_try_load_text()` — load a checkpoint matching prompt text prefix. Scans entries via `ds4_kvstore_find_text_prefix()`, validates header/text hash/byte prefix match, loads session payload from file, optionally loads trailer via hooks. If loaded tokens exceed `cold_max_tokens`, deletes the file (consumed). Otherwise touches file (increments hits, updates last_used). Returns loaded token count or 0.
- `ds4_kvstore_file_size_fits()` — budget check before committing. Computes file size from text_bytes + payload_bytes + trailer_bytes + fixed header. Applies 1% slack for accounting headroom. Returns false if file would exceed budget.

### Eviction

- `ds4_kvstore_entry_eviction_score()` — compute density-weighted decay score for one entry. Lower score = better eviction candidate.
- `ds4_kvstore_evict()` — run scored eviction until `total <= budget - extra_bytes`. Calls `kv_cache_refresh()` first to sync in-memory table with filesystem. Scans all entries, picks lowest-score victim (tiebreak: older `last_used` wins), `unlink()` the file, removes from table. Logs each eviction at `DS4_KVSTORE_LOG_KVCACHE` level.

### Eviction Scoring Formula

```
effective_hits = hits * 2^(-elapsed / half_life)   # half_life = 6h
clamp(effective_hits) → [0, ∞), floor at 0.01

score = (effective_hits + 1) * tokens / file_size

# Anchor bonus: cold/evict/shutdown checkpoints
if reason in {COLD, EVICT, SHUTDOWN}:
    score *= 2.0   # KV_CACHE_ANCHOR_REASON_SCORE_FACTOR

# Continued prefix discount: superseded waypoints
if incoming supersedes this continued checkpoint (same model/quant, smaller text, larger or equal ctx):
    h = effective_hits > 0 ? effective_hits / (effective_hits + 1) : 0
    score *= 0.05 + 0.45 * h   # ranges 0.05× (never hit) to 0.5× (many hits)
```

**Tiebreaker**: equal scores → older `last_used` wins.

Invariant: every store call runs `ds4_kvstore_evict()` before writing, so the store never overflows budget.

### Store Policy Helpers

- `ds4_kvstore_default_options()` — returns default options struct: `min_tokens=512`, `cold_max_tokens=30000`, `continued_interval_tokens=10000`, `boundary_trim_tokens=32`, `boundary_align_tokens=2048`.
- `ds4_kvstore_store_len()` — compute store token count: trim `boundary_trim_tokens` from tail, align down to `boundary_align_tokens`, clamp at `min_tokens`.
- `ds4_kvstore_chat_anchor_pos()` — find the last user-marker token before the first assistant-marker token. Used for cold checkpoint positioning. Returns position or -1 if below `min_tokens`.
- `ds4_kvstore_continued_store_target()` — check if `live_tokens` hit the continued interval boundary and haven't been stored yet. Returns target token count or 0.
- `ds4_kvstore_note_store()` — record that a store happened at `tokens`. Updates `continued_last_store_tokens`.
- `ds4_kvstore_suppress_continued_store()` — suppress the next continued store at `tokens`. Returns old `continued_last_store_tokens` for restore, or -1 if not at target.
- `ds4_kvstore_restore_suppressed_continued()` — undo a suppression if `continued_last_store_tokens` still matches the suppressed value.
- `ds4_kvstore_reason_code()` — map reason string (`"cold"`, `"continued"`, `"evict"`, `"shutdown"`, `"agent-system"`, `"agent-session"`) to `ds4_kvstore_reason` enum.
- `ds4_kvstore_key_kind()` — map `ext_flags` to human-readable key kind: `"responses-visible"`, `"thinking-visible"`, or `"token-text"`.

### Text Matching / Token Helpers

- `ds4_kvstore_render_tokens_text()` — detokenize a `ds4_tokens` array to rendered UTF-8 bytes via `ds4_token_text()` per token. Caller frees returned buffer.
- `ds4_kvstore_byte_prefix_match()` — byte-level prefix comparison: `memcmp(text, prefix, prefix_len)` with length guard.
- `ds4_kvstore_tokens_copy_prefix()` — copy first `n` tokens from source to destination token array.
- `ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix()` — compose restored prefix tokens + suffix text (tokenized via `ds4_tokenize_rendered_chat`) into one token array. Used after text-prefix cache load to reconstruct the full prompt.

### File Format

Each `.kv` file has this layout:

| Offset | Field | Size | Notes |
|---|---|---|---|
| 0 | magic | 3 | `KVC` |
| 3 | version | 1 | `1` |
| 4 | quant_bits | 1 | 2 or 4 |
| 5 | reason code | 1 | ds4_kvstore_reason value |
| 6 | ext_flags | 1 | bitmask: tool_map=bit0, responses_visible=bit1, thinking_visible=bit2, session_title=bit3 |
| 7 | model_id | 1 | 0 = unknown (backward compat: Flash=0) |
| 8 | tokens | 4 | LE uint32 — token count |
| 12 | hits | 4 | LE uint32 — load hit count |
| 16 | ctx_size | 4 | LE uint32 — context window size |
| 20 | payload_abi | 1 | graph-payload serialization version (currently 2) |
| 21-23 | reserved | 3 | zero |
| 24 | created_at | 8 | LE uint64 unix timestamp |
| 32 | last_used | 8 | LE uint64 unix timestamp |
| 40 | payload_bytes | 8 | LE uint64 |
| **48** | **text_bytes** | **4** | LE uint32 — byte length of rendered text |
| 52 | text | text_bytes | rendered byte prefix (UTF-8) |
| 52+text_bytes | payload | payload_bytes | session graph state (DS4_SESSION_PAYLOAD_VERSION) |
| end | trailer | variable | optional; presence indicated by ext_flags bit via trailer hooks |

### SHA1 Naming Scheme

Files named by `SHA1(rendered_text_bytes)` → 40 lowercase hex chars + `.kv` suffix. The hash identifies the byte prefix of the prompt, enabling fast prefix-match lookup without tokenization. `ds4_kvstore_sha1_bytes_hex()` computes the digest inline (no OpenSSL dependency).

- `ds4_kvstore_sha1_bytes_hex()` — inline SHA1 digest of arbitrary bytes → 40-char hex + NUL. Uses custom `sha1_ctx` (no OpenSSL).
- `ds4_kvstore_sha_hex_name()` — validate a `.kv` filename: 43 chars, 40 hex digits + `.kv`. Extracts hex portion into `sha[41]`. Returns false on mismatch.
- `ds4_kvstore_path_join()` — join directory + filename with `/` separator. Allocates result.
- `ds4_kvstore_path_for_sha()` — build full path from cache directory + SHA1 hex + `.kv`.

### Low-Level I/O

- `ds4_kvstore_read_header()` — read and validate the 48-byte fixed header + 4-byte text_bytes from an open `FILE *`. Populates `ds4_kvstore_entry` fields. Validates magic/version, payload_abi, quant_bits. Returns false on read failure or validation error.
- `ds4_kvstore_read_entry_file()` — stat + open + read_header for a given path. Validates file size >= header + text_bytes + payload_bytes. Populates full `ds4_kvstore_entry` including SHA, path, file_size.
- `ds4_kvstore_fill_header()` — write all 48 fixed header bytes into a buffer given field values. Used by both write and touch paths.
- `ds4_kvstore_touch_file()` — open existing `.kv` file for update, re-read header, write back with incremented hits and updated `last_used`. Used after a cache hit.
- `ds4_kvstore_le_put32()` / `ds4_kvstore_le_get32()` — little-endian 32-bit encode/decode. Static helpers for 64-bit (`kv_le_put64`/`kv_le_get64`) are file-internal.

## Data Flow

```
[engine produces tokens] → [render tokens to text] → [SHA1(text)] → [check existing .kv file]
  ↓ (miss)
[session stage payload] → [evict until budget fits] → [write atomically via .tmp + rename]
  ↓
[incoming prompt text] → [SHA1 prefix bytes] → [scan entries for matching sha]
  ↓ (hit)
[validate header/model/quant/ctx] → [validate text hash + byte prefix] → [load payload] → [reconstruct prompt]
```

## Invariants

- `.kv` files are self-contained: header carries model_id, quant_bits, ctx_size. Cross-model restore rejected by `ds4_kvstore_find_text_prefix()`.
- **Not thread-safe internally.** Zero mutex or atomic operations in `ds4_kvstore.c` or `ds4_kvstore.h`. Caller must externally synchronize all accesses (e.g., `ds4-server.c` wraps kvstore calls in `pthread_mutex_lock`).
- Every store call runs `ds4_kvstore_evict()` before writing, guaranteeing budget compliance for single-threaded access.
- File writes are atomic: write to `.tmp.<pid>`, then `rename()` to final name.
- `ds4_kvstore_clear()` frees in-memory entries but does **not** delete files. To clear disk state, iterate entries and `unlink()` before calling `clear()`.
- The in-memory entry table can drift from filesystem state (e.g., external deletion). `ds4_kvstore_evict()` and `ds4_kvstore_find_text_prefix()` call `kv_cache_refresh()` to rescan the directory before operating.
- Header byte 7 (model_id) defaults to 0 for backward compatibility with older cache files where this reserved byte was always written as zero.

## Configuration

- Budget: `budget_mb` parameter to `ds4_kvstore_open()` (default 4096 from `DS4_KVSTORE_DEFAULT_MB`).
- Default options: `ds4_kvstore_default_options()` sets `min_tokens=512`, `cold_max_tokens=30000`, `continued_interval_tokens=10000`, `boundary_trim_tokens=32`, `boundary_align_tokens=2048`.
- Hit half-life: `DS4_KVSTORE_HIT_HALF_LIFE_SECONDS` = 21600 (6 hours).
- No environment variables or compile-time flags. All configuration is parameter-based.

## Notes

- The eviction formula is density-weighted: `(effective_hits + 1) * tokens / file_size`. A checkpoint with more tokens per byte (denser) gets a higher score and survives longer. Hits amplify density but decay exponentially (6h half-life).
- Anchor bonus (2×) gives cold/evict/shutdown checkpoints a soft prior. These are intentional snapshots, not automatic waypoints, so they survive comparable continued entries under mild pressure.
- Continued prefix discount (0.05× to 0.5×) makes never-hit waypoints cheap eviction victims while keeping frequently-reused ones. A superseded continued checkpoint has a smaller text prefix than the incoming store but same model/quant/equal-or-larger context.
- SHA1 naming is purely byte-addressable — no tokenization needed for lookup. This means two different token sequences that render to the same bytes map to the same file, which is correct for prefix matching but means the payload carries the exact token sequence, not the hash.
- Text-override variants (visible transcript, thinking visible, session title) share the same SHA1 namespace as token-text checkpoints because they use different rendered text. They are distinguished by `ext_flags` bits in the header.
- The `payload_abi` field (byte 20) is separate from the outer file version. The KVC envelope can remain stable while the serialized `ds4_session` internals become unsafe to restore across runtime changes.
- `ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix()` uses `ds4_tokenize_rendered_chat()` for the suffix text because the suffix may start with DS4 chat markers such as `user` or `assistant`.
- `ds4-kvstore` is also used by `ds4-agent` for explicit session checkpoints with its own policy in `ds4_agent.c`. Protocol-specific extras (e.g., server tool-id → exact DSML trailer) are attached through `ds4_kvstore_trailer_hooks`.

## See Also

- [Session Snapshots](../engine/session-snapshots.md)
- [KV Cache Lifecycle](../concepts/kv-cache-lifecycle.md)

[← Back to Index](../README.md)
