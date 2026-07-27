# Help

## Files

- `ds4_help.c` — help text generation
- `ds4_help.h` — help API

## Purpose

Print usage info and command-line option descriptions.

## Key Types

| Type | Role |
|---|---|
| `ds4_help_tool` | Enum identifying the tool requesting help: `DS4_HELP_DS4`, `DS4_HELP_SERVER`, `DS4_HELP_AGENT`, `DS4_HELP_BENCH`, `DS4_HELP_EVAL` |

## Dependencies

- **Imports from**: `stdio.h` (FILE, fprintf, fputs, fputc), `stdbool.h`, `string.h` (strcmp, strlen), `unistd.h` (isatty, fileno)
- **Exports to**: `ds4.c`, `ds4-server.c`, `ds4-agent.c`, `ds4-bench.c`, `ds4-eval.c` (each tool calls `ds4_help_print()` from its `--help` / `-h` path)
- **Init order**: No explicit init — help text and color detection are purely call-site driven

## Topic Dispatch

Help is dispatched by topic string via `ds4_help_print(FILE *fp, ds4_help_tool tool, const char *topic)`:

1. `help_make_colors(fp)` — detect TTY color support once per call
2. Print tool name banner, summary, and usage line
3. If `topic` is non-NULL: dispatch to `print_topic()` which maps topic strings to section printers via `if/else if` chain
4. If `topic` is NULL: run `print_default()` (compact view) then `print_more_info()` (topic quick-reference)
5. Always append `print_examples()` at the end

### Help topics by tool

| Tool | Available topics |
|---|---|
| `ds4` | `all`, `runtime`, `distributed`, `sampling`, `steering`, `diagnostics`, `commands` |
| `ds4-server` | `all`, `runtime`, `distributed`, `steering`, `api`, `kv-cache`, `thinking` |
| `ds4-agent` | `all`, `runtime`, `distributed`, `sampling`, `steering`, `sessions`, `commands`, `tools` |
| `ds4-bench` | `all`, `runtime`, `distributed`, `benchmark` |
| `ds4-eval` | `all`, `runtime`, `distributed`, `sampling`, `evaluation` |

Topics `runtime`, `distributed` are available for every tool. Topic `all` prints every section available for that tool.

## API Surface

- `ds4_help_print(fp, tool, topic)` — print help text for the given tool and optional topic. Caller owns `fp`; output is flushed per-fprintf. Never blocks on I/O beyond the underlying `FILE *`.

## Invariants

- Help text compiled into binary (no external file dependency).
- Caller chooses output stream via `FILE *fp` parameter (no hardcoded stdout).

[← Back to Index](../README.md)
