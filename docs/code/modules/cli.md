# CLI

## Files

- `ds4_cli.c` — command-line entry point, REPL, interactive transcript handling
- (no separate public header; `ds4.h` is the public API)

## Purpose

Command-line entry point. Parses CLI flags, loads a model via `ds4_engine_open`, and dispatches to one of several modes: interactive REPL, one-shot generation, diagnostic tests, perplexity scoring, importance-matrix collection, or model introspection.

## Dependencies

- **Imports from**: `ds4.h` (engine API), `ds4_distributed.h` (distributed inference), `ds4_gpu_args.h` (GPU vram/device parsing), `ds4_tp.h` (tensor parallelism), `ds4_help.h` (usage text), `linenoise.h` (REPL line editing)
- **Exports to**: direct call — the CLI is the entry point; no other module calls into it
- **Init order**: first — `main()` parses options, opens engine, then dispatches

## Key Types

| Type | Role |
|---|---|
| `cli_config` | Top-level parsed configuration: engine options, generation options, distributed options, GPU placement args |
| `cli_generation_options` | Generation parameters: prompt, system prompt, sampling params, test flags, dump paths |
| `repl_chat` | REPL chat state: live session, token transcript, context size, think-prefix position/length |
| `token_printer` | Streaming output helper: handles thinking-mode colorization and token boundary buffering |
| `cli_prefill_progress` | Progress callback state for prefill phase |

### cli_config fields

| Field | Type | Description |
|---|---|---|
| `engine` | `ds4_engine_options` | Engine-level options (model path, backend, threads, etc.) |
| `dist` | `ds4_dist_options *` | Distributed inference options (heap-allocated) |
| `gen` | `cli_generation_options` | Generation-specific options |
| `prompt_owned` | `char *` | Owned copy of prompt text (from `--prompt-file`) |
| `inspect` | `bool` | `--inspect` flag: print model summary and exit |
| `gpu_vram_arg` | `const char *` | Raw `--gpu-vram` value, resolved post-parse |
| `gpu_devices_arg` | `const char *` | Raw `--gpu-devices` value, resolved post-parse |

### cli_generation_options fields

| Field | Type | Description |
|---|---|---|
| `prompt` | `const char *` | Prompt text (borrowed or from `prompt_owned`) |
| `system` | `const char *` | System prompt |
| `raw_prompt` | `bool` | `--raw` flag: skip chat template wrapping |
| `n_predict` | `int` | Max tokens to generate |
| `ctx_size` | `int` | Context size |
| `temperature` | `float` | Sampling temperature |
| `top_p` | `float` | Top-p sampling threshold |
| `min_p` | `float` | Min-p sampling threshold |
| `temperature_set` | `bool` | Whether temperature was explicitly set |
| `top_p_set` | `bool` | Whether top_p was explicitly set |
| `min_p_set` | `bool` | Whether min_p was explicitly set |
| `seed` | `uint64_t` | RNG seed (0 = auto) |
| `dump_tokens` | `bool` | `--dump-tokens` flag |
| `dump_logits_path` | `const char *` | `--dump-logits` output path |
| `dump_logprobs_path` | `const char *` | `--dump-logprobs` output path |
| `dump_logprobs_top_k` | `int` | Top-k for logprobs dump |
| `decode_consistency_tokens` | `int` | `--decode-consistency` token count |
| `perplexity_file_path` | `const char *` | `--perplexity-file` input path |
| `imatrix_dataset_path` | `const char *` | `--imatrix-dataset` path |
| `imatrix_output_path` | `const char *` | `--imatrix-out` path |
| `imatrix_max_prompts` | `int` | Max prompts for imatrix collection |
| `imatrix_max_tokens` | `int` | Max tokens for imatrix collection |
| `think_mode` | `ds4_think_mode` | Thinking mode (none/high/max) |
| `head_test` | `bool` | `--head-test` flag |
| `first_token_test` | `bool` | `--first-token-test` flag |
| `metal_graph_test` | `bool` | `--metal-graph-test` flag |
| `metal_graph_full_test` | `bool` | `--metal-graph-full-test` flag |
| `metal_graph_prompt_test` | `bool` | `--metal-graph-prompt-test` flag |

### repl_chat fields

| Field | Type | Description |
|---|---|---|
| `session` | `ds4_session *` | Live inference session (owns KV cache) |
| `transcript` | `ds4_tokens` | Full token transcript of the conversation |
| `ctx_size` | `int` | Current context size (may change via `/ctx`) |
| `think_prefix_pos` | `int` | Position of thinking prefix in transcript |
| `think_prefix_tokens` | `int` | Length of thinking prefix in tokens |

## API Surface

### Creation / Teardown

- `main(argc, argv)` — program entry point. Calls `parse_options()`, opens engine, dispatches to mode, closes engine.
- `parse_options(argc, argv)` — parses all CLI flags into a `cli_config`. Calls `exit(2)` on error.
- `run_repl(engine, cfg)` — enters interactive REPL loop. Returns when user types `/quit` or EOF.

### Core Operations

- `run_generation(engine, cfg)` — builds prompt, dispatches to test/dump/generation sub-routines.
- `run_sampled_generation(engine, cfg, prompt)` — session-based generation with sampling, speculative decoding, progress callbacks.
- `run_chat_turn(engine, cfg, chat, user_text)` — one REPL turn: extends transcript, syncs session, generates response.
- `run_perplexity_file(engine, cfg)` — scores a text file token-by-token, prints perplexity.
- `run_logits_dump(engine, cfg, prompt)` — dumps full logit vector after prefill to JSON.
- `run_logprob_dump(engine, cfg, prompt)` — dumps per-step top-k logprobs during generation to JSON.
- `run_decode_consistency(engine, cfg, prompt)` — compares logits after sequential decode vs. fresh prefill.
- `ds4_engine_collect_imatrix(engine, ...)` — collects importance matrix data from a dataset.

### Query / Introspection

- `usage(fp, topic)` — prints help text via `ds4_help_print()`.
- `ds4_engine_summary(engine)` — prints model architecture summary (`--inspect`).

## Data Flow

```
argv → parse_options() → cli_config
                              │
                    ┌─────────┼──────────┬─────────────┬──────────────┐
                    ▼         ▼          ▼             ▼              ▼
              inspect?   imatrix?  perplexity?    prompt==NULL?   prompt set?
                    │         │          │             │              │
                    ▼         ▼          ▼             ▼              ▼
           ds4_engine_  ds4_engine_  run_perplexity_  run_repl()  run_generation()
           summary()   collect_      file()                        │
                       imatrix()                              ┌─────┼──────┐
                                                              ▼     ▼      ▼
                                                       tests/    session  argmax
                                                       dumps    (sampled) (stateless)
```

### Dispatch branching

1. `--inspect` → `ds4_engine_summary()`, exit
2. `--imatrix-out` + `--imatrix-dataset` → `ds4_engine_collect_imatrix()`, exit
3. `--perplexity-file` → `run_perplexity_file()`, exit
4. No prompt → `run_repl()` (interactive REPL)
5. Prompt provided → `run_generation()` which further dispatches:
   - `--metal-graph-test` / `--metal-graph-full-test` / `--metal-graph-prompt-test` → graph diagnostic, exit
   - `--dump-logits` → `run_logits_dump()`, exit
   - `--dump-logprobs` → `run_logprob_dump()`, exit
   - `--decode-consistency` → `run_decode_consistency()`, exit
   - `--head-test` / `--first-token-test` / `--dump-tokens` → diagnostic run, exit
   - Otherwise → `run_sampled_generation()` (session-based) or `ds4_engine_generate_argmax()` (stateless), depending on temperature, MTP, distributed role, and TP role

### Command-line parsing flow

```
argv → parse_options()
  ├── ds4_dist_parse_cli_arg()    (distributed flags)
  ├── ds4_tp_parse_cli_arg()      (tensor-parallel flags)
  ├── per-flag handlers           (model, prompt, sampling, tests, etc.)
  ├── ds4_tp_adopt_distributed_options()
  ├── ds4_dist_prepare_engine_options()
  └── ds4_tp_validate_engine_options()
```

### REPL flow

```
run_repl()
  ├── repl_chat_init()       — build initial transcript, create session
  ├── linenoise loop         — read commands, dispatch
  │   ├── /help              — print commands
  │   ├── /think             — set think mode to high
  │   ├── /think-max         — set think mode to max (with ctx check)
  │   ├── /nothink           — disable thinking
  │   ├── /ctx N             — resize context, recreate session
  │   ├── /power N           — set GPU duty cycle
  │   ├── /read FILE         — read prompt from file, run turn
  │   ├── /quit, /exit       — exit REPL
  │   └── <text>             — run_chat_turn()
  └── repl_chat_free()       — free session and transcript
```

## Configuration

### Environment variables

| Variable | Effect |
|---|---|
| `DS4_CUDA_SPLITKV_SPEC` | Enable split-KV speculative decoding on CUDA |
| `DS4_CUDA_NO_SPLITKV_SPEC` | Disable split-KV speculative decoding on CUDA |
| `DS4_CUDA_GREEDY_SPLITKV` | Enable greedy split-KV attention optimization |
| `DS4_CUDA_NO_GREEDY_SPLITKV` | Disable greedy split-KV attention optimization |
| `DS4_CUDA_GREEDY_VEC4` | Enable greedy vec4 optimization |
| `DS4_CUDA_NO_GREEDY_VEC4` | Disable greedy vec4 optimization |
| `DS4_MTP_SPEC_DISABLE` | Disable MTP speculative decoding |
| `DS4_CLI_FORCE_SESSION` | Force session-based generation path (for TP validation) |
| `HOME` | Used for REPL history file location (`~/.ds4_history`) |

### Compile-time flags

- `DS4_NO_GPU` — forces CPU backend as default
- `DS4_ROCM_BUILD` — enables ROCm (AMD GPU) backend

### CLI flags

Configured entirely through CLI flags. No config file.

## Invariants

- `parse_options()` exits the process on any parse error (exit code 2) — never returns with invalid config.
- REPL mode is single-threaded. The REPL loop blocks on `linenoise()` and `run_chat_turn()`.
- Session-based generation (`run_sampled_generation`) uses the live KV cache and supports interrupt (SIGINT) with safe rollback on zero-token interrupt.
- Stateless generation (`ds4_engine_generate_argmax`) does not create a session; it processes the prompt and generates tokens in one shot.
- The `repl_chat` transcript owns the token sequence; `ds4_session_sync()` decides whether to reuse or rebuild KV state.
- Think-prefix tokens live at a fixed position in the transcript (after BOS); changing think mode rewrites them in-place and invalidates the session.
- Server mode is a separate binary (`ds4-server`); `--server` in the CLI prints an error and exits.
- Piping input without `-p`/`--prompt` or `--prompt-file` hits `isatty()` fatal exit — no implicit pipe detection.

## Notes

- The CLI deliberately keeps policy (parsing, dispatch, REPL) and leaves graph/cache mechanics inside the engine API (`ds4.h`).
- `-t` / `--threads` controls CPU thread count, **not** test mode. Test flags are `--head-test`, `--first-token-test`, `--metal-graph-test`, `--metal-graph-full-test`, `--metal-graph-prompt-test`.
- Token streaming uses a `token_printer` helper that handles thinking-mode colorization (grey text for thinking tokens) and token-boundary buffering to avoid splitting ` thinking` / ` response` markers.
- Distributed and tensor-parallel options are parsed by separate modules (`ds4_distributed.c`, `ds4_tp.c`) and merged into the engine options.
- GPU placement (`--gpu-vram`, `--gpu-devices`) is resolved after option parsing via `parse_gpu_vram_arg()` and uses `ds4_engine_create_with_gpu_config()` instead of `ds4_engine_open()`.
- MTP speculative decoding (`--mtp-draft`) is active when temperature <= 0 and `DS4_MTP_SPEC_DISABLE` is unset.

## See Also

- [Engine API](../engine/engine-api.md)
- [Tokenizer](../engine/tokenizer.md)
- [Help](help.md)
- [Server](server.md)
- [Agent](agent.md)

[← Back to Index](../README.md)
