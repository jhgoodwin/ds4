# Evaluation

## Files

- `ds4_eval.c` — standalone program (no `.h`). All functions static except `main()`.
- No `ds4_eval.h` exists. The module exports no library API; it is a CLI entry point.

## Purpose

Embedded multiple-choice QA regression harness. Loads the real model, renders chat prompts, prefills through `ds4_session_sync()`, samples token by token, and grades the final answer. Four benchmark sources are hardcoded as `static const eval_case eval_cases[]`:

| Source | Type | Count | License |
|---|---|---|---|
| GPQA Diamond | Multiple-choice (4 choices) | 20 | CC BY 4.0 |
| SuperGPQA | Multiple-choice (10 choices) | 20 | ODC-BY |
| AIME 2025 | Integer answer | 20 | MIT |
| COMPSEC | Vulnerability line-number localization | 15 | Audited subset |

Not a generic eval framework. No perplexity, multi-turn conversation, or tool call accuracy measurement. No JSONL dataset input — all test cases are embedded at compile time.

## Dependencies

- **Imports from**: `ds4_engine` (model load, tokenize, encode), `ds4_session` (sync, sample, eval), `ds4_distributed` (coordinator/worker route), `ds4_help` (usage text), `ds4_ssd` (SSD streaming constants)
- **Exports to**: none — this is a `main()` program, not a library
- **Init order**: standalone — `main()` calls `parse_options()`, `ds4_engine_open()`, `ds4_session_create()`, then runs the grading loop

## Key Types

| Type | Role |
|---|---|
| `eval_case` | One benchmark question: source, id, domain, title, question text, up to 10 choices, expected answer string |
| `eval_config` | CLI-parsed runtime configuration (~30 fields: model path, backend, context size, temperature, think mode, distributed options, etc.) |
| `eval_status` | Per-case lifecycle state: `PENDING`, `PREFILL`, `THINKING`, `SKIPPED`, `STOPPED`, `PASSED`, `FAILED` |
| `eval_run_result` | Per-case outcome: `OK`, `ERROR`, `SWITCH`, `QUIT` |
| `eval_think_close_kind` | How thinking ended: `NONE`, `NATURAL`, `SOFT` (model had `</think>` near top-k), `HARD` (budget exhausted) |
| `eval_think_close_info` | Record of how thinking closure was forced: kind, token index, remaining budget, rank |
| `eval_ui` | Full TUI state: terminal dimensions, pane layout, per-case status/guess/token counts, streaming buffer, input state |
| `byte_buf` | Growable byte buffer for streaming model output |
| `style_buf` | Growable ANSI style buffer paired with `byte_buf` for TUI rendering |

## API Surface

No public functions. The module exposes only `main()`. All helpers are `static`:

### Program Entry

- `main()` — parses CLI args, opens engine + session, runs grading loop, prints report, returns exit code

### Configuration Parsing

- `parse_options()` — parses argc/argv into `eval_config` (long options + short aliases)
- `usage()` — prints help text for a given topic
- `parse_int_arg()`, `parse_nonnegative_int_arg()`, `parse_u64_arg()`, `parse_float_arg()` — typed argument helpers
- `parse_backend()`, `default_backend()` — backend selection
- `need_arg()` — consumes required positional argument

### Prompt Building

- `eval_system_prompt()` — returns static system prompt string
- `build_question_prompt()` — renders question + choices + answer format instruction into a single prompt string
- `ds4_encode_chat_prompt()` — (from ds4 engine) tokenizes system + user prompt with think mode

### Context Sizing

- `eval_max_prompt_tokens()` — finds the largest prompt across all cases for a given context size
- `eval_auto_context_size()` — iteratively sizes context to fit largest prompt + generation budget (up to `EVAL_MAX_CONTEXT` = 1,000,000)
- `eval_warn_context_budget()` — warns if prompts don't fit or generation budget is tight
- `eval_warn_think_max_downgraded()` — warns if `--think-max` is set below its minimum context

### Grading Loop (per case)

- `run_one_case()` — full question→answer→grade cycle: prefill via `ds4_session_sync()`, sample via `ds4_session_sample()`, eval via `ds4_session_eval()`, extract answer via `find_case_answer()`, compare via `answer_matches()`
- `next_pending_case()` — scans for next case with `EVAL_PENDING` status
- `parse_case_sequence()` — parses `--case` argument (comma/range syntax like `1,3-5`)
- `mark_case_pending()` — resets a case for re-run after switch

### Answer Extraction

- `find_case_answer()` — dispatches to `find_answer_letter()`, `find_integer_answer()`, or `find_compsec_answer()` based on case type
- `find_answer_letter()` — scans generated text for `Answer: <letter>` pattern, returns uppercase letter or `?`
- `find_integer_answer()` — extracts first integer from generated text after last `Answer:` marker
- `find_compsec_answer()` — extracts line-number specification (single, range, or comma-separated)
- `normalize_integer_answer()` — strips leading zeros for canonical comparison
- `normalize_compsec_line_spec()` — canonicalizes line number specs
- `answer_matches()` — compares extracted answer to expected (letter match, integer compare, or COMPSEC set containment via `compsec_answer_matches()`)
- `compsec_answer_matches()` — checks that all model-supplied line numbers fall within the accepted range

### Interactive TUI

- `tui_start()` — initializes terminal (alternate screen, raw mode), spawns input thread
- `tui_free()` — cleans up TUI state
- `tui_restore()` — restores original terminal settings
- `tui_refresh()` — full-frame redraw: title, left pane (question list), right pane (streaming output + status)
- `tui_draw_left()` — renders scrollable case list with status colors
- `tui_draw_stream()` — renders model output with ANSI colors (thinking vs. answer)
- `tui_draw_right_status()` — renders status bar with phase name, token counts, speed
- `tui_run_clock_start/stop/tick()` — wall-clock timing for per-case display
- `tui_consume_input()` — processes queued keyboard events (up/down, enter, p/pause, q/quit)
- `tui_wait_if_paused()` — blocks while paused, returns pause duration
- `tui_has_quit_request()`, `tui_has_switch_request()` — non-blocking input checks
- `stream_append_token_text()` — appends decoded token text to streaming buffer with think/answer style tracking
- `tui_reset_stream()` — clears streaming buffer for a new case
- `tui_signal_restore()` — SIGINT/SIGTERM handler restores terminal

### Input Thread

- `input_thread_main()` — raw stdin reader in a background thread, decodes ANSI escape sequences, queues state
- `tui_start_input()` / `tui_stop_input()` — manages raw mode and thread lifecycle
- `input_queue_key()` — pushes key events into `eval_input` struct under mutex

### Trace / Regrade

- `trace_write_header()` — writes trace metadata (model, config, date)
- `trace_write_case()` — writes per-case record with counted blocks for robust regrading
- `trace_write_block()` — writes a length-prefixed content block
- `regrade_trace_file()` — re-reads a trace file, re-extracts answers from saved model output, compares, reports changes
- `trace_find_next_case()` / `trace_find_block_begin()` / `trace_get_line_field()` — trace parser helpers
- `trace_copy_model_output()` — extracts saved model output from trace block
- `find_eval_case_by_source_id()` — looks up embedded case by source + id for regrading
- `read_text_file()` — slurps entire trace file

### Self-Test

- `run_extractor_self_tests()` — validates answer extraction functions against known inputs (returns 0 on success)

### Reporting

- `print_eval_report()` — prints summary table: per-case status, token counts, given vs. correct answer

### Distributed / Logging

- `wait_distributed_route()` — blocks coordinator until worker routes are established
- `log_context_memory()` — prints context memory estimate for SSD streaming configurations
- `eval_prefill_progress()` — progress callback for `ds4_session_set_progress()`

## Data Flow

```
CLI args
   ↓
parse_options() → eval_config
   ↓
ds4_engine_open() → engine
   ↓
ds4_session_create() → session
   ↓
for each eval_case (in sequence or interactive order):
   │
   ├─ build_question_prompt(tc) → prompt string
   ├─ ds4_encode_chat_prompt(engine, system, question, think_mode) → token array
   ├─ ds4_session_sync(session, &prompt) → prefill KV cache
   │
   ├─ for each generation step (up to max_tokens):
   │   ├─ ds4_session_sample(session, temperature, top_p, min_p) → token
   │   ├─ ds4_token_is_stop(engine, token)? → break
   │   ├─ ds4_session_eval(session, token) → extend KV cache
   │   ├─ ds4_token_text(engine, token) → decoded text
   │   ├─ stream to TUI (or stdout in non-TTY mode)
   │   └─ think-close controller: soft/hard limit checks
   │
   ├─ find_case_answer(tc, generated_text) → extracted answer
   ├─ answer_matches(tc, extracted) → pass/fail
   └─ trace_write_case() + status update
   ↓
print_eval_report() → summary table
```

## Invariants

- All test cases are embedded at compile time in `static const eval_case eval_cases[]`. No runtime dataset loading.
- Results are deterministic for the same input, seed, and model weights.
- Output structured trace written to `--trace` file (if specified). Summary printed to stdout.
- Each eval run is independent — no state carryover between cases (session is re-synced each case).
- COMPSEC and SuperGPQA slices are audited: bad rows replaced, not locally re-keyed.
- The input thread never writes to the terminal — all rendering is owned by the main thread.
- Signal handlers (SIGINT, SIGTERM) restore terminal before exit.

## Configuration

### CLI options (all parsed by `parse_options()`)

| Flag | Field | Default |
|---|---|---|
| `--model` / `-m` | `model_path` | (required) |
| `--mtp` | `mtp_path` | NULL |
| `--trace` | `trace_path` | NULL |
| `--regrade-trace` | `regrade_trace_path` | NULL |
| `--case` | `case_sequence` | NULL (interactive/all) |
| `--backend` | `backend` | auto-detected |
| `--threads` / `-t` | `threads` | 0 |
| `--ctx` | `ctx_size` | 0 (auto-size) |
| `--tokens` | `max_tokens` | 8192 |
| `--questions` | `question_limit` | 0 (all) |
| `--temp` | `temperature` | 0.0 |
| `--top-p` | `top_p` | 1.0 |
| `--min-p` | `min_p` | 0.0 |
| `--seed` | `seed` | time+pid+clock |
| `--pause` | `pause_ms` | 0 |
| `--power` | `power_percent` | 0 |
| `--prefill-chunk` | `prefill_chunk` | 0 |
| `--ssd-streaming-cache-experts` | `ssd_streaming_cache_experts` | 0 |
| `--ssd-streaming-cache-bytes` | `ssd_streaming_cache_bytes` | 0 |
| `--ssd-streaming-full-layers` | `ssd_streaming_full_layers` | 0 |
| `--ssd-streaming-preload-experts` | `ssd_streaming_preload_experts` | 0 |
| `--simulate-memory` | `simulate_used_memory_bytes` | 0 |
| `--soft-limit-reply-budget` | `soft_limit_reply_budget` | 0 |
| `--hard-limit-reply-budget` | `hard_limit_reply_budget` | 0 |
| `--soft-limit-think-close-rank` | `soft_limit_think_close_rank` | 0 |
| `--think-mode` | `think_mode` | DS4_THINK_DISABLED |
| `--plain` | `plain` | false |
| `--warm-weights` | `warm_weights` | false |
| `--quality` | `quality` | false |
| `--ssd-streaming` | `ssd_streaming` | false |
| `--ssd-streaming-cold` | `ssd_streaming_cold` | false |
| `--self-test-extractors` | `self_test_extractors` | false |
| Distributed options | `dist` | zero-initialized |

### Compile-time constants

- `EVAL_MAX_CHOICES` = 10 (max answer choices per question)
- `EVAL_ANSWER_MAX` = 32 (max length of extracted answer string)
- `EVAL_MAX_CONTEXT` = 1,000,000 (max auto-sized context tokens)
- `AUTH_STACK_MAX` = 256 (COMPSEC scratch buffer size — in embedded code snippets, not eval itself)

## Notes

- **No header file.** Unlike other modules, `ds4_eval.c` is a standalone program. The public interface is the CLI and the trace file format.
- **Interactive TUI** uses ANSI escape sequences only — no ncurses. Two-pane layout: left scrollable case list with status indicators, right streaming model output with think/answer color coding.
- **Keyboard controls** (interactive mode): up/down arrows to select case, Enter to run, `p` to pause/resume, `q` to quit.
- **Think-close controller** prevents the model from exhausting the generation budget inside ` response` tokens to force a visible answer. Two tiers: soft (model's own ` response` is near top-k) and hard (absolute reserve).
- **Regrade mode** (`--regrade-trace`) re-reads a previous trace file, re-extracts answers from saved model output, and reports changes — useful for validating extraction logic changes.
- **Self-test mode** (`--self-test-extractors`) runs unit tests on answer extraction functions and exits.
- **COMPSEC** questions embed reduced C/C++ function snippets with known vulnerabilities. The model identifies the bug location by line number. Expected answers are audited ranges (not single lines) where multiple adjacent lines are equivalent bug locations.

## See Also

- [Engine API](../engine/engine-api.md)
- [Tokenizer](../engine/tokenizer.md)

[← Back to Index](../README.md)
