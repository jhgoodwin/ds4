# DwarfStar Code Docs

Decompose the ds4 codebase into digestible, indexed modules.  Goal: reduce effort to learn abstract ideas (model mechanics, data flow, backend architecture) without reading ~64K-line files top-to-bottom.

## Layout

```
docs/code/
├── README.md              ← this file: index, conventions, navigation
├── templates/
│   ├── MODULE.md          ← template: one .c/.h file or logical unit
│   └── CONCEPT.md         ← template: cross-cutting idea (e.g. attention)
├── engine/                ← ds4.c internal sections (the big file)
│   ├── model-shapes.md    ─  model shape profiles (Flash/Pro/GLM52)
│   ├── gguf-loading.md    ─  GGUF parsing, mmap, metadata
│   ├── weight-binding.md  ─  tensor binding & validation
│   ├── quant-kernels.md   ─  CPU quantized math (Q8_0, Q2_K, IQ2_XXS)
│   ├── attention.md       ─  QKV projections, RoPE, compressed attn
│   ├── moe-ffn.md         ─  MoE: router, shared/routed experts, SwiGLU
│   ├── hc-transforms.md   ─  hyper-connection 4-stream state
│   ├── kv-cache.md        ─  raw/compressed KV cache, compressors
│   ├── metal-graph.md     ─  Metal release graph: alloc, decode, prefill
│   ├── tokenizer.md       ─  BPE tokenizer, chat encoding
│   ├── engine-api.md      ─  ds4_engine / ds4_session public API
│   ├── directional-steering.md
│   ├── imatrix.md         ─  importance matrix collection
│   └── session-snapshots.md  ─  disk KV cache checkpoint format
├── modules/               ← separate .c/.h files
│   ├── cli.md             ─  ds4_cli.c: REPL, command line
│   ├── server.md          ─  ds4_server.c: HTTP API, streaming
│   ├── agent.md           ─  ds4_agent.c: coding agent TUI
│   ├── distributed.md     ─  ds4_distributed: pipeline parallelism
│   ├── tp.md              ─  ds4_tp: tensor parallelism
│   ├── web.md             ─  ds4_web: web UI
│   ├── kvstore.md         ─  ds4_kvstore: disk KV cache store
│   ├── ssd-streaming.md   ─  ds4_ssd: SSD streaming for routed experts
│   ├── layer-pack.md      ─  ds4_layer_pack: expert layer packing
│   ├── bench.md           ─  ds4_bench: benchmarking harness
│   ├── eval.md            ─  ds4_eval: quality evaluation
│   └── help.md            ─  ds4_help: help text
├── backends/
│   ├── metal.md           ─  ds4_metal.m + metal/ kernels
│   ├── cuda.md            ─  ds4_cuda.cu, GPU tensor ops
│   ├── rocm.md            ─  ROCm specifics, Strix Halo
│   └── cpu.md             ─  CPU reference backend
└── concepts/
    ├── attention-compression.md         ─  CSA/HCA compressed attention
    ├── moe-routing.md                   ─  expert selection, top-k, IQ2_XXS
    ├── hc-state.md                      ─  4-stream hyper-connection design
    ├── gguf-format.md                   ─  GGUF quant formats used
    ├── distributed-protocol.md          ─  pipeline + TP wire protocol
    ├── kv-cache-lifecycle.md            ─  raw → compressed → indexer flow
    ├── mtp.md                           ─  Multi-Token Prediction (speculative decode)
    ├── dspark.md                        ─  DeepSeek V4 multi-stage speculative decode
    ├── multi-gpu-pipeline.md            ─  Multi-GPU pipeline parallelism (wave 2)
    ├── gpu-tensor-primitives.md         ─  GPU tensor API, command buffer, primitives
    ├── indexer-subsystem.md             ─  Compressed attention indexer (score/topk)
    ├── session-batch-decode.md          ─  Batch decoding across sessions
    ├── glm-model-path.md                ─  GLM architecture: KV LoRA, compact KV
    ├── fp8-kv-quantization.md           ─  E4M3 FP8 KV cache quantization
    ├── gpu-expert-streaming-cache.md    ─  GPU LRU for SSD-streamed experts
    ├── session-rewrite-invalidation.md  ─  Common-prefix rewrite, rewind, invalidate
    ├── process-instance-lock.md         ─  flock singleton engine lock
    ├── layer-packing-engine.md          ─  Monotonic GPU/CPU layer placement
    └── model-shape-detection.md         ─  Architecture detect: Flash/Pro/DSpark/GLM
```

## Conventions

### Language

Strive for concise writing. Omit filler (just/really/basically) and hedging. Prefer fragments where clear. Use precise technical terms.

Pattern: `[thing] [action] [reason]. [next concept].`

### Structure

Every doc follows one of two templates (see `templates/`):

- **MODULE.md** — for a single file or logical unit. Covers purpose, API surface, data flow, key types, invariants, dependencies.
- **CONCEPT.md** — for cross-cutting ideas. Covers definition, why it exists, how it appears in code, variants, lifecycle.

### Updates

Prefer append/new files over edits.  When a function changes, add a note at the bottom of the relevant doc with date and change description.  When a new concept emerges, create a new file in the appropriate folder.  Only edit existing docs when the documented interface or invariant actually changes (not implementation details).

### Indexing

Each doc starts with a `## Files` line listing every source file it touches.  Cross-reference by file path, not by concept name — grep for file names to find all related docs.

### Navigation

- `README.md` at each folder level links to children
- Aspirational: each doc links back to parent `../README.md`
- Aspirational: cross-file links use relative paths: `../engine/attention.md`

## How To Read This

1. Start with `concepts/` for the ideas (attention compression, MoE routing, HC state).
2. Then `engine/` for how ds4.c implements them.
3. Then `modules/` for the CLI/server/agent that wrap the engine.
4. Then `backends/` for GPU specifics.

Or: grep for a file name in these docs to find which docs reference it.

## Undocumented Topics (Code Exists, No Doc Yet)

5 of 10 topics have partial coverage in sibling docs (noted inline). The rest lack any documentation. These abstractions exist in the source but lack dedicated docs.  Each represents a significant code surface worth documenting:

1. **GPU Q8_0 Cache** — `ds4_gpu_cache_q8_f16_range()` pre-quantizes weight ranges to Q8_0 for decode matmuls. Suppressed via `ds4_gpu_q8_cache_suppressed()`. Lives in `ds4_gpu.h` cache API.

2. **CUDA Tensor-Parallel Sliced Projections** — `ds4_gpu_matmul_q8_0_kslice_tensor()` and variants for k-sliced matvec across TP ranks. Metal-only today; CUDA multi-row variants exist.

3. **Decode Attention Fast Paths** — `ds4_gpu_set_decode_fast_attention()` and `ds4_gpu_set_decode_score_vec4()` toggle decode attention optimizations. Internal heuristics for when to use each.

4. **Speculative Verify Batches (TP)** — `ds4_session_tp_spec_cycle()` drives the TP worker side of speculative verify. The leader sends draft token blocks; the worker runs its half and awaits commit or rollback. Lives in `ds4.c` eval dispatch and `ds4_tp.c`.

5. **Expert Handoff Packing** — `ds4_gpu_moe_handoff_pack_tensor()` packs selected expert outputs into contiguous rows for the down-projection. Pre-requisite for batched expert eval.

6. **Q4_K Routed Expert MVPs** — CPU-side Q4_K quantized matvec for routed experts. Separate from Q8_0 path; slower but lower memory.

7. **Metal Diagnostic Comparisons** — Debug-only hooks that compare Metal GPU results against CPU reference, dump tensor values, validate bit-exactness. Gated behind `metal_graph_test` option.

8. **GPU Model Map Span Registration** — `ds4_gpu_set_model_map_spans()` and friends let callers register non-contiguous model file ranges per device. Used by multi-GPU selective caching.

9. **Slice Loading** — `load_layer_start`/`load_layer_end`/`load_output` in `ds4_engine_options` load only a contiguous slice of layers. Used for distributed worker setups.

10. **GLM Streaming Prefill** — `ds4_gpu_set_glm_streaming_prefill_full_layer()` controls whether GLM streaming prefill loads full layers. GLM-specific SSD streaming variant.
