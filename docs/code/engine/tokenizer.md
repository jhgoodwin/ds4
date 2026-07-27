# Tokenizer & Chat Encoding

## Files

- `ds4.c` — BPE tokenizer, hash tables, chat prompt encoding

## Purpose

Load JoyAI-style byte-level BPE tokenizer from GGUF, tokenize user text, encode chat prompts with special tokens.

## BPE Tokenizer

DeepSeek V4 Flash stores tokenizer data in GGUF metadata:

- Token strings (byte-level BPE)
- Merge ranks
- Special token IDs (EOS, BOS, user, assistant, thinking control, tool/DSML markers)

One open-addressed hash table for encode, one flat array for decode:

```
token_to_id (str_i32_table):   token string -> token ID (for encode)
vocab->token[id] (ds4_str[]):  token ID -> token string (for decode, direct index)
```

Token strings stored in mmap, not copied.

BPE encoding:

1. **Pre-tokenization (JoyAI rules):** `bpe_tokenize_text` splits text into pieces using JoyAI rules, not GPT-2 regex. For DeepSeek V4 models, the split rules are:

   | Priority | Rule | Example |
   |---|---|---|
   | 1 | Digit runs up to 3 digits (`\p{N}{1,3}`) | `123` -> one piece, `1234` -> `123` + `4` |
   | 2 | CJK/Hiragana/Katakana clusters | `你好世界` -> one piece |
   | 3 | Punctuation/symbol followed by alpha (`[P/S][A-Za-z]+`) | `!hello` -> one piece |
   | 4 | Letter-like sequences with optional single leading non-letter | `hello`, `'em` |
   | 5 | Space + punctuation/symbol run + trailing newlines | ` >;\n` -> one piece |
   | 6 | Punctuation/symbol run + trailing newlines | `!!!\n` -> one piece |
   | 7 | Whitespace runs - newlines split at the boundary; leading space joins next word/punct run | `    int` -> `   ` + ` int` |
   | 8 | Single character (fallback) | any unmatched byte |

   For GLM models, `bpe_tokenize_text` delegates to `bpe_tokenize_text_glm4`, which uses a different split strategy: possessive/contraction suffixes (`'s`, `'t`, `'re`, `'ve`, `'ll`, `'m`, `'d`), letter clusters, digit runs (<=3), punctuation runs, and whitespace runs with newline handling.

2. **Byte encode:** Each pre-tokenization piece is GPT-2 byte-encoded: bytes 0-255 are mapped to Unicode codepoints 256-511. This ensures any arbitrary byte sequence maps to a known token in the vocabulary.

3. **BPE merge:** Starting with the byte-encoded symbols, iteratively find the adjacent pair with the lowest merge rank (from the GGUF merge table), merge them into a new symbol, and repeat until no mergeable pairs remain.

4. **Emit tokens:** Each final symbol is looked up in `token_to_id`. Single bytes that never merged fall back to byte-level tokens (which always exist in the vocabulary).

## Chat Encoding

```
ds4_chat_begin()                  -> BOS token (plus SOP for GLM)
ds4_chat_append_message()         -> role header + content + separator
ds4_chat_append_assistant_prefix() -> assistant header (for generation)
ds4_chat_append_max_effort_prefix() -> reasoning effort prefix
ds4_encode_chat_prompt()          -> full prompt (system + prompt + think_mode)
```

Note: Tool definitions are handled via `ds4_chat_append_message()` with role `"tool"` (or `"function"`), not via `ds4_encode_chat_prompt()`. The `"tool"` role wraps content in `<tool_result>`...`</tool_result>` for DeepSeek V4 or `<tool_response>`...`</tool_response>` for GLM.

`ds4_chat_append_message()` behavior by role:

| Role | DeepSeek V4 | GLM |
|---|---|---|
| `"system"` / `"developer"` | BPE-tokenize content directly | Push `<\|system\|>`, tokenize via `tokenize_rendered_chat_vocab` |
| `"assistant"` | Push `assistant_id`, push `think_end_id` (if not already thinking/response), BPE-tokenize content | Push `assistant_id`, push `think_start_id`+`think_end_id` (if not already thinking/response), tokenize via `tokenize_rendered_chat_vocab` |
| `"tool"` / `"function"` | Push `user_id`, BPE-tokenize `<tool_result>`, tool result content, `</tool_result>` | Push `observation_id`, tokenize `<tool_response>`, tool response content, `</tool_response>` |
| `"user"` (default) | Push `user_id`, BPE-tokenize content | Push `user_id`, BPE-tokenize content |

## Tokenizer API Functions

```
void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
```
BPE-tokenize a plain text string. Wraps `bpe_tokenize_text`. Allocates per call.

```
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
```
Tokenize pre-rendered chat text, substituting special token sequences (e.g. `"<\|user\|>"`, `"<\|assistant\|>"`, `"<tool_call>"`, `"<sop>"`, `"<|observation|>"`) with their token IDs inline. Uses `tokenize_rendered_chat_vocab` which scans for known special token strings and BPE-tokenizes the spans between them. This is the entry point for parsing already-formatted chat strings (e.g. from a cache or external renderer).

```
char *ds4_token_text(ds4_engine *e, int token, size_t *len);
```
Decode a token ID back to its text representation. For byte-level tokens, reverses the GPT-2 byte encoding. Returns a malloc'd string.

```
int ds4_token_eos(ds4_engine *e);
```
Return the EOS (end-of-sequence) token ID.

```
int ds4_token_user(ds4_engine *e);
```
Return the user role token ID.

```
int ds4_token_assistant(ds4_engine *e);
```
Return the assistant role token ID.

```
bool ds4_token_is_stop(ds4_engine *e, int token);
```
Return true if the token is a generation stop signal (EOS, or role markers for GLM).

```
bool ds4_token_is_thinking_control(ds4_engine *e, int token);
```
Return true if the token is a thinking control marker (`" thinking"` or `" response"`).

```
bool ds4_token_is_stop_for_think_mode(ds4_engine *e, int token, ds4_think_mode mode);
```
Return true if the token should halt generation for the given think mode. Includes EOS/role stops, plus thinking control markers when in no-thinking mode.

```
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
```
Push the BOS token (and SOP token for GLM models) onto the token vector.

```
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
```
Build a complete chat prompt: BOS -> think prefix -> system text -> user marker -> user prompt -> assistant marker -> think start/end marker. For GLM, uses `<\|system\|>` tags and SOP. Does not accept tool definitions - those go through `ds4_chat_append_message`.

```
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
```
Append a single chat message: role marker + content + optional wrappers. Role can be `"system"`, `"developer"`, `"assistant"`, `"tool"`, `"function"`, or `"user"` (default).

```
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);
```
Append the assistant header for generation: `assistant_id` + think start/end marker depending on mode.

```
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
```
Append the "Reasoning Effort: Max" prefix prompt before the user message. Used only for `DS4_THINK_MAX` mode.

## Special Tokens

| Token | Field | DeepSeek V4 | GLM |
|---|---|---|---|
| BOS | `bos_id` | `""` (empty string) | `""` |
| EOS | `eos_id` | `""` | `"<\|endoftext\|>"` |
| SOP (start of prompt) | `sop_id` | - | `"<sop>"` |
| System | `system_id` | - | `"<\|system\|>"` |
| User | `user_id` | `"<\|User\|>"` / `"用户"` | `"<\|user\|>"` |
| Assistant | `assistant_id` | `"<\|Assistant\|>"` / `"助手"` | `"<\|assistant\|>"` |
| Observation | `observation_id` | - | `"<\|observation\|>"` |
| Think start | `think_start_id` | `"<begin_of_thinking>"` (alias `" thinking"`) | `"<begin_of_thinking>"` |
| Think end | `think_end_id` | `"<end_of_thinking>"` (alias `" response"`) | `"<end_of_thinking>"` |
| Tool call start | `tool_call_start_id` | - | `"<tool_call>"` |
| Tool call end | `tool_call_end_id` | - | `"</tool_call>"` |
| Tool response start | `tool_response_start_id` | - | `"<tool_response>"` |
| Tool response end | `tool_response_end_id` | - | `"</tool_response>"` |
| Arg key start | `arg_key_start_id` | - | `"<arg_key>"` |
| Arg key end | `arg_key_end_id` | - | `"</arg_key>"` |
| Arg value start | `arg_value_start_id` | - | `"<arg_value>"` |
| Arg value end | `arg_value_end_id` | - | `"</arg_value>"` |
| DSML | `dsml_id` | `"<dsml>"` (alias `"<|dsml|>"`) | - |

## Invariants

- Token strings stored in mmap (not copied).
- BPE merge table sized for worst-case (model-dependent).
- BPE encoding allocates per call (byte_encode buffer, symbol array, BPE merge buffers, token_vec realloc).
- Thread-safe after initialization (read-only hash tables).

## See Also

- [gguf-format.md](../concepts/gguf-format.md) — GGUF metadata keys storing tokenizer data (token strings, merge ranks, special token IDs)

[← Back to Index](../README.md)
