# Agent

## Files

- `ds4_agent.c` — self-contained module (no separate `.h`)

## Purpose

AI agent loop — manages user interaction (REPL), LLM inference, DSML/GLM tool-call parsing, streaming token renderer, bash job lifecycle, and session persistence.

## Dependencies

- **Imports from**: `ds4.h` (engine, session, token/text helpers), `ds4_distributed.h` (distributed CLI parsing), `ds4_gpu_args.h` (GPU config), `ds4_help.h` (usage text), `ds4_kvstore.h` (session cache paths), `ds4_web.h` (browser-based web tools), `linenoise.h` (REPL)
- **Exports to**: Entry point (`main`) — no other module calls agent functions directly
- **Init order**: `main()` → `parse_options()` → `ds4_engine_open()` → `agent_worker_init()` → worker thread starts → REPL loop (`run_agent` / `run_agent_non_interactive`)
- **Teardown**: REPL exits → `agent_worker_free()` (joins worker thread, kills bash jobs, frees session/web/KV) → `ds4_engine_close()`

## Key Types

| Type | Role |
|---|---|
| `agent_config` | ~50 CLI flags: model path, backend, GPU devices, generation params (temperature, top_p, min_p, context size, think mode), distributed options, tool config |
| `agent_worker` | Two-thread shared state: worker thread owns session, KV, bash jobs; UI thread owns terminal I/O. Synchronized via mutex/condvar + wake pipe |
| `agent_status` | Read-only snapshot published by worker for UI footer: state enum, prefill/generation progress, TPS, context usage, power percent |
| `agent_generation_options` | Subset of config: prompt, system, n_predict, ctx_size, sampling params, think mode |
| `agent_bash_job` | One per running bash command: pid, pipe fd, temp file path, start time, timeout, byte count, exit status, linked-list next pointer |
| `agent_stream_renderer` | State machine for streaming model output: owns `agent_token_renderer` (ANSI markdown/code highlighting), `agent_dsml_parser` (tool call parsing), `agent_tool_visualizer` (inline 🛠️ rendering) |
| `agent_token_renderer` | Token-by-token ANSI renderer: markdown bold/italic/code highlighting, fence syntax coloring, UTF-8 safe output |
| `agent_dsml_parser` | Two-mode parser (DSML or GLM syntax): extracts `agent_tool_call` list from model output tokens |
| `agent_tool_call` | Parsed tool invocation: name, argument list (name + value + is_string flag) |
| `agent_tool_visualizer` | Renders inline tool-call visualization (ANSI-colored, 🛠️ prefix) during streaming |
| `agent_editor` | Linenoise wrapper with scroll-region support: model/tool output scrolls above a bottom-anchored prompt and status footer |
| `agent_prompt_queue` | FIFO queue for user input that arrives while the model is busy; replayed when worker becomes idle |

## API Surface

### Creation / Teardown

- `main(argc, argv)` — parses options, opens engine, initializes worker, enters REPL loop (`run_agent`) or headless mode (`run_agent_non_interactive`), then shuts down
- `agent_worker_init(w, engine, cfg)` — allocates session, web module, creates wake pipe, spawns worker thread
- `agent_worker_free(w)` — stops worker thread, kills all bash job process groups, frees session, web, KV, transcript, wake fds, trace file

### Core Operations

- `worker_main(arg)` — worker thread entry. Waits on condvar for commands (user text, save, compact, power change). Calls `worker_run_turn()` to generate + execute tools in a loop
- `worker_run_turn(w, user_text)` — one user → model → tool(s) → model → ... cycle. Appends user message, calls engine generation, streams tokens through renderer/parser, executes parsed tool calls, injects results, repeats until model finishes (EOS or max tokens)
- `worker_submit(w, text)` — UI thread enqueues user input; signals worker via condvar
- `worker_interrupt(w)` — sets interrupt flag; worker checks after each token and tool execution
- `worker_consume(w, &out, &out_len, &status)` — UI thread drains rendered output buffer under mutex, then writes to terminal outside the lock
- `agent_execute_tool_call(w, call)` — dispatch to one of: read, write, edit, search, list, more, bash, bash_status, bash_stop, google_search, visit_page

### Tool Execution Functions

| Tool | Implementation |
|---|---|
| `read` | `fopen()` with range/rlimit, context-size bounded by default |
| `write` | `fopen()` + `fwrite()` |
| `edit` | File read + old/new validation + rewrite |
| `search` | Recursive directory search with glob, context lines, max results |
| `list` | `opendir()`/`readdir()` |
| `more` | Continue previous read from saved offset |
| `bash` | `fork()` + `execl("/bin/sh", "sh", "-c", cmd, NULL)` — no sandbox (see Invariants) |
| `bash_status` | Poll running job: drain pipe, check `waitpid()` WNOHANG, return latest output |
| `bash_stop` | `kill(-job->pid, SIGKILL)` + waitpid — terminates process group |
| `google_search` | Launches Chrome via `ds4_web`, runs Google search in visible browser, returns compact Markdown |
| `visit_page` | Opens URL in visible browser via `ds4_web`, returns rendered Markdown |

### Bash Job Lifecycle

1. `agent_bash_start()` — fork, setpgid (own process group), `/dev/null` stdin, pipe stdout/stderr, temp file for capture. Returns `agent_bash_job`
2. `agent_bash_poll()` — drain pipe, check `waitpid(WNOHANG)`, enforce timeout (`kill(-pgid, SIGKILL)`)
3. `agent_bash_finalize()` — close pipe/tmpfd, store exit status (WIFEXITED/WIFSIGNALED→128+signo)
4. `agent_bash_job_free()` — SIGKILL if still running, waitpid, close fds, free cmd string

### Query / Introspection

- `worker_get_status(w, &status)` — lock-copy of current `agent_status` for UI footer rendering
- `worker_is_idle(w)` — returns true when `state == AGENT_WORKER_IDLE` and no `cmd_text` pending
- `agent_worker_effective_ctx_size(w)` — session context size or config fallback

## Data Flow

### Two-Thread Architecture

```
┌─────────────────────────────────────────────┐
│  UI Thread (main)                           │
│  · stdin → linenoise → worker_submit()      │
│  · poll(wake_fd, STDIN_FILENO)             │
│  · worker_consume() → write to terminal     │
│  · ANSI scroll region: output above prompt  │
│  · SIGINT → worker_interrupt()              │
└──────────────┬──────────────────────────────┘
               │ mutex + condvar + pipe
               ▼
┌─────────────────────────────────────────────┐
│  Worker Thread                              │
│  · wait on condvar for cmd_text             │
│  · worker_run_turn():                       │
│    1. Append user msg to transcript         │
│    2. engine_generate() → streaming tokens  │
│    3. Token → renderer (ANSI) → publish()   │
│    4. Token → DSML parser (tool calls)      │
│    5. Execute tools → inject results        │
│    6. Repeat until EOS or max tokens        │
│  · Deferred: save, compact, power change    │
└─────────────────────────────────────────────┘
```

### Streaming Token Pipeline

```
Model tokens ─→ agent_stream_renderer
                    ├── agent_token_renderer (ANSI markdown/code/syntax)
                    ├── agent_dsml_parser (DSML/GLM tool calls)
                    └── agent_tool_visualizer (inline 🛠️ rendering)
                         ↓
                    agent_publish() → worker.out buffer
                         ↓
                    UI thread: worker_consume() → terminal write
```

### Agent Loop (one turn)

1. User types message, presses Enter
2. `worker_submit()` signals worker thread
3. Worker appends user message to transcript
4. Worker calls `ds4_engine_generate()` — produces tokens one by one
5. Each token flows through stream renderer → ANSI text published to output buffer
6. DSML parser extracts tool calls from token stream
7. After EOS: execute parsed tool calls sequentially
8. Each tool result appended back to transcript as a rendered text block
9. If tool calls were made, continue generation with results in context
10. Repeat from step 4 until model outputs a final answer (no tool calls) or hits `n_predict`

## Invariants

- **Single-process**: worker thread and UI thread share the same address space. Synchronization via mutex + condvar + level-triggered wake pipe. The worker never writes to the terminal directly; all output goes through `agent_publish()` → `worker_consume()`.
- **Bash is NOT sandboxed**: `fork()` + `execl("/bin/sh")` with full host filesystem, network, and process access. No `seccomp`, `chroot`, `landlock`, or container isolation. A bash command can read/write any file the process owner can, open sockets, and spawn background processes.
- **Process group isolation**: each bash job gets its own process group (`setpgid`) so `bash_stop` and timeout kill grandchildren, not just the shell wrapper.
- **Temporary output files**: bash stdout/stderr go to a named pipe and a temp file (`/tmp/ds4_agent_output_*`). Temp files are not cleaned up on hard crash.
- **Full network and filesystem access**: `google_search` and `visit_page` launch a visible Chrome instance. File tools (`read`/`write`/`edit`/`search`) use standard POSIX I/O on any path the process can access.
- **Worker owns session + KV**: only the worker thread reads/writes the session transcript and KV store. The UI thread submits commands and drains output.
- **Line editor owns terminal**: linenoise runs in raw mode on the UI thread. The worker must never write escape sequences or read stdin. When a bash child changes terminal mode (via `/dev/tty`), the worker sets a flag and the UI restores raw mode at the next poll cycle.
- **System prompt is trusted DS4 control text**: the built-in tool prompt is tokenized via `ds4_tokenize_rendered_chat()` so DSML markers become dedicated model tokens. User-supplied `-sys` text is appended as plain chat content to avoid control-token injection.
- **System prompt reminders**: if the transcript grows by ~50K tokens since the last system prompt, the worker re-injects the tool definitions to keep the model aware of available tools.

## Configuration

### CLI flags (~50, parsed in `parse_options()`)

| Category | Flags |
|---|---|
| Model | `-m`/`--model`, `--mtp`, `--mtp-draft`, `--mtp-margin`, `--glm-mtp`, `--glm-mtp-timing` |
| Backend | `--backend`, `--metal`, `--cuda`, `--cpu`, `--gpu-vram`, `--gpu-devices`, `--cuda-tensor-parallel` |
| Generation | `-n`/`--tokens`, `-c`/`--ctx`, `--temp`, `--top-p`, `--min-p`, `--seed`, `--think`/`--think-max`/`--nothink` |
| Prompt | `-p`/`--prompt`, `-sys`/`--system`, `--raw`/`--raw-prompt`, `--non-interactive` |
| Performance | `-t`/`--threads`, `--quality`, `--power`, `--prefill-chunk`, `--warm-weights`, `--ssd-streaming` variants |
| Steering | `--dir-steering-file`, `--dir-steering-ffn`, `--dir-steering-attn` |
| Distributed | `--dis-role`, `--dis-connect`, `--dis-listen`, `--dis-workers`, `--dis-name` (delegated to `ds4_dist_parse_cli_arg`) |
| Other | `--chdir`, `--trace`, `--dspark` variants, `--simulate-used-memory` |

### Environment variables

- `HOME` — used for session history path (`~/.ds4_agent_history`) and web module home directory
- `LINENOISE_ASSUME_TTY` — if set, forces linenoise to treat stdin/stdout as a TTY even when not a terminal (e.g., inside tools that pipe input)

### Compile-time flags

- `DS4_NO_GPU` — forces CPU backend as default instead of Metal (macOS) or CUDA (Linux)

## Notes

- **No separate header**: `ds4_agent.c` is entirely self-contained. All supporting types and helper functions are `static` within the translation unit. The only non-static symbol is `main`.
- **DSML vs GLM syntax**: the agent auto-detects the model type via `ds4_engine_is_glm_dsa()` and switches between DSML (`<｜DSML｜tool_calls>`) and GLM (`<tool_call>`) tool-call syntax. Builds different system prompts and uses different parsers accordingly.
- **Session persistence**: sessions are saved to `~/.cache/ds4/agent/sessions/` via KV store. The `/save` and `/compact` slash commands trigger deferred saves/compaction on the worker thread.
- **Headless mode**: `--non-interactive` runs the same worker without linenoise. With `-p` it is one-shot; without it reads piped input until stdin goes quiet for 200 ms.
- **History**: linenoise persists command history to `~/.ds4_agent_history` (up to 512 entries).
- **Tool result size limit**: results that would push the transcript beyond context capacity are truncated and a note is appended.
- **No split-pane display**: the terminal uses a single linenoise line with ANSI scroll regions. Streaming model/tool output scrolls above the prompt line. No ncurses panes.

## See Also

- [Engine API](../engine/engine-api.md)
- [Tokenizer](../engine/tokenizer.md)
- [Session Batch Decode](../concepts/session-batch-decode.md)

[← Back to Index](../README.md)
