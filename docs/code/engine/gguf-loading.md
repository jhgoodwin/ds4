# GGUF Loading

## Files

- `ds4.c` — GGUF parsing, mmap, tensor directory, type definitions (`ds4_model`, `ds4_cursor`, `ds4_kv`, `ds4_tensor`, `ds4_weights`). All loading types live in `ds4.c`, not `ds4.h`.

## Purpose

Map model file once, parse header + metadata + tensor directory, keep tensor data in mmap. No eager copy of weight bytes. Metal wraps mmap slices as no-copy MTLBuffers.

## Loading Pipeline

```
model_open(path, metal_mapping, prefetch_cpu)
  → mmap → verify magic + version → parse_metadata → parse_tensors
```

The mmap flag (`MAP_SHARED` vs `MAP_PRIVATE`) is decided from `metal_mapping` before any parsing begins. Three phases inside `model_open`:

1. **mmap + verify** — open file, `mmap` with flag from `metal_mapping`, read magic + version, fail on wrong magic or unsupported version.
2. **`parse_metadata`** — read KV pairs into offset table (values stay in mmap).
3. **`parse_tensors`** — read tensor directory, compute absolute offsets, validate bounds.

See [gguf-format.md](../concepts/gguf-format.md) for the metadata KV table and file structure.

## Key Types

### `ds4_model`

Mmap state and parsed directory. Allocated on the caller's stack; fields populated by `model_open`.

| Field | Type | Role |
|---|---|---|
| `fd` | `int` | Open file descriptor (-1 when closed) |
| `map` | `const uint8_t *` | Mmap base pointer |
| `size` | `uint64_t` | Mmap region size |
| `version` | `uint32_t` | GGUF format version (must be 3) |
| `n_kv` | `uint64_t` | Count of metadata KV pairs |
| `n_tensors` | `uint64_t` | Count of tensors in directory |
| `alignment` | `uint64_t` | Tensor data alignment (32) |
| `tensor_data_pos` | `uint64_t` | Absolute offset where tensor payload begins |
| `max_tensor_bytes` | `uint64_t` | Largest single tensor in bytes |
| `kv` | `ds4_kv *` | Metadata KV array (heap) |
| `tensors` | `ds4_tensor *` | Tensor directory array (heap) |

### `ds4_cursor`

Bounded reader over the mmap region. Stack-allocated; zero-copy for strings (points into mmap).

| Field | Type | Role |
|---|---|---|
| `base` | `const uint8_t *` | Start of readable region |
| `size` | `uint64_t` | Region size in bytes |
| `pos` | `uint64_t` | Current read position |
| `error` | `char[256]` | Error buffer, set on read failure |

### `ds4_kv`

Single metadata key-value entry. Values stay in mmap; `value_pos` points into the mapping.

| Field | Type | Role |
|---|---|---|
| `key` | `ds4_str` | Key string (points into mmap) |
| `type` | `uint32_t` | GGUF value type enum |
| `value_pos` | `uint64_t` | Offset into mmap where value bytes begin |

### `ds4_tensor`

Single parsed tensor entry from the directory.

| Field | Type | Role |
|---|---|---|
| `name` | `ds4_str` | Tensor name (points into mmap) |
| `ndim` | `uint32_t` | Number of dimensions |
| `dim` | `uint64_t[8]` | Dimension sizes |
| `type` | `uint32_t` | GGUF tensor type enum |
| `rel_offset` | `uint64_t` | Offset relative to `tensor_data_pos` |
| `abs_offset` | `uint64_t` | Absolute offset into mmap |
| `elements` | `uint64_t` | Total element count |
| `bytes` | `uint64_t` | Total byte size |

## Cursor I/O

`ds4_cursor` wraps the mmap region with checked reads. Every read advances `pos`; all fail if `pos + n > size`:

- `cursor_at(m, pos)` — create cursor positioned at byte `pos` of model `m`
- `cursor_u32(c, &v)` — read 4-byte LE uint32
- `cursor_u64(c, &v)` — read 8-byte LE uint64
- `cursor_string(c, &s)` — read length-prefixed string into `ds4_str` (points into mmap, no copy)
- `cursor_skip(c, n)` — advance cursor by n bytes
- `cursor_has(c, n)` — check n bytes remain without consuming
- `cursor_read(c, dst, n)` — copy n bytes to caller buffer
- `cursor_error(c, msg)` — set `c->error` (first error wins, no-overwrite)

## Error States

All fatal errors call `ds4_die(msg)` which prints `"ds4: <msg>\n"` to stderr and `exit(1)`. File-level errors call `ds4_die_errno(what, path)` which includes `strerror`. Cursor underflows set `c->error` and are propagated by the caller via `ds4_die(c->error)`.

| Condition | Error message | How handled |
|---|---|---|
| Wrong magic | `"model is not a GGUF file"` | `ds4_die` → exit(1) |
| Wrong version (≠ 3) | `"only GGUF v3 is supported"` | `ds4_die` → exit(1) |
| Metadata OOM | `"out of memory while allocating metadata table"` | `ds4_die` → exit(1) |
| Tensor table OOM | `"out of memory while allocating tensor table"` | `ds4_die` → exit(1) |
| Tensor dim overflow | `"tensor element count overflow"` | `ds4_die` → exit(1) |
| Offset overflow | `"tensor offset overflow"` | `ds4_die` → exit(1) |
| Out-of-bounds tensor | `"tensor points outside GGUF file"` | `ds4_die` → exit(1) |
| Cursor underflow | `c->error` set by `cursor_error()` → `"truncated GGUF file at byte N"` | caller passes to `ds4_die` → exit(1) |
| Unsupported ndim | `"tensor has an unsupported number of dimensions"` | `ds4_die` → exit(1) |
| Cannot open | `"cannot open model '<path>': <strerror>"` | `ds4_die_errno` → exit(1) |
| Cannot mmap | `"cannot mmap model '<path>': <strerror>"` | `ds4_die_errno` → exit(1) |
| File too small | `"model file is too small to be GGUF"` | `ds4_die` → exit(1) |

## API Surface

All loading functions are `static` — they live in `ds4.c` and are called through the engine lifecycle, not exported.

### Loading / Teardown

- `model_open(ds4_model *m, const char *path, bool metal_mapping, bool prefetch_cpu)` — mmap + parse full GGUF. `metal_mapping=true` uses `MAP_SHARED` for no-copy GPU buffers; `metal_mapping=false` uses `MAP_PRIVATE`. Tokenizer-only callers pass `prefetch_cpu=false` to skip walking tensor payload.
- `model_close(ds4_model *m)` — munmap, close fd, free KV + tensor arrays. Resets `m` to zero state with `fd = -1`.

### Parsing (called by `model_open`)

- `parse_metadata(ds4_model *m, ds4_cursor *c)` — read `n_kv` key-value pairs into heap-allocated `m->kv` array. Values stay in mmap; only offset (`value_pos`) is stored. Sets `m->alignment = 32`.
- `parse_tensors(ds4_model *m, ds4_cursor *c)` — read `n_tensors` entries into heap-allocated `m->tensors` array. Validates dimensions, computes element count and byte size, converts relative offsets to absolute `abs_offset`. Sets `m->tensor_data_pos` and `m->max_tensor_bytes`.

### Metadata Lookup

- `model_find_kv(m, key)` — linear scan of `m->kv` for matching key name. Returns `ds4_kv *` or NULL.
- `model_get_u32(m, key, &v)` — lookup u32 metadata value, returns false if missing or wrong type.
- `model_get_string(m, key, &out)` — lookup string metadata value, returns false if missing or wrong type.
- `model_get_token_id(m, key, &id)` — lookup tokenizer token id (accepts UINT32 or INT32 values).
- `model_get_u32_any(m, keys[], nkeys, &v)` — try multiple key names, return first match.
- `model_get_u32_array_any(m, keys[], nkeys, &out)` — try multiple key names for array values.

### Cursor (stack-allocated, used during parse)

- `cursor_at(m, pos)` — construct a cursor at byte `pos` of model `m`.
- `cursor_u32`, `cursor_u64`, `cursor_string`, `cursor_skip`, `cursor_has`, `cursor_read`, `cursor_error` — see [Cursor I/O](#cursor-io) above.

## mmap Policy

The mmap flag is a single binary decision based on `metal_mapping`:

- `metal_mapping == true` → `MAP_SHARED` (Metal and CUDA backends)
- `metal_mapping == false` → `MAP_PRIVATE` (CPU and SSD streaming backends)

The flag is chosen before any parsing and stays constant for the model lifetime. Rationale:

| `metal_mapping` | Flag | Backends | Reason |
|---|---|---|---|
| `true` | `MAP_SHARED` | Metal, CUDA | GPU needs no-copy access (Metal MTLBuffers, CUDA page migration via PCIe) |
| `false` | `MAP_PRIVATE` | CPU, SSD streaming | CPU reads only; avoids Darwin VM panic observed with shared mappings on very large files |

## Invariants

- Tensor data never copied at load time for Metal backend.
- Metadata keys read via `model_get_u32()` / `model_get_string()` — return false if missing or type mismatch.
- SSD streaming reads expert blob offsets from in-memory `ds4_tensor` pointers, no directory re-parse.
- Cursor reads never overrun mmap boundary.
- `model_close` is idempotent: safe to call on zeroed struct.

## Notes

Tensor name matching, shape validation, and layer struct population happen during weight binding, after the loading pipeline completes. See [weight-binding.md](weight-binding.md) for tensor name matching and shape validation.

See [model-shape-detection.md](../concepts/model-shape-detection.md) for the model shape detection and profile selection logic.

[← Back to Index](../README.md)
