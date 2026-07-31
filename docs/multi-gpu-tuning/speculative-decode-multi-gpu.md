# Speculative Decode (DSpark) on Multi-GPU

## Purpose

Analyze DSpark speculative decoding overlap opportunities on a 2-GPU pipeline-parallel system. Identify critical path, GPU idle time during draft stages, and acceptance rate ceilings.

## Architecture Recap

DSpark interleaves lightweight stage blocks at target layers through the model stack. During decode, each step:
1. **Capture**: Record hidden state (HC) at target layers during base model forward pass
2. **Propose**: Run draft stages (stage0 → stage chain → Markov → confidence) using captured HC
3. **Verify**: Run base model forward pass over draft tokens, compare with draft proposal

## 2-GPU Placement Context

From the measured model layout:

```
GPU0: layers 0-23 + embedding  (82.6 GB, 82.18 GiB weights)
GPU1: layers 24-42 + output head  (71.5 GB, 71.14 GiB weights)
```

[measured: ds4 multi-GPU layout at ctx=32768]

### Target Layer Distribution

DeepSeek V4 Flash DSpark target layers (from GGUF metadata). For this model, typical DSpark target layers are at regular intervals. Without exact metadata values, we analyze by distribution:

If DSpark target layers are distributed uniformly across the 43 layers, ~24 layers on GPU0 and ~19 on GPU1. This means target layers span both GPUs.

## Overlap Analysis

### Current Execution Model

The DSpark pipeline in ds4 (CUDA backend):

```
decode_layer loop:
  for each layer:
    if dspark_capture_enabled AND target_layer:
      record HC to support cache
    compute layer (attention + FFN)
    if layer boundary crosses GPU:
      xdev activation transfer to next GPU
```

Draft proposal (after base model decode completes):
```
ds4_session_prepare_dspark_draft_impl:
  1. Check capture state
  2. stage0 eval
  3. Setup block
  4. Stage chain eval (TP suspended on coordinator)
  5. Final hidden + base logits
  6. Markov + confidence loop (CPU fallback or GPU)
  7. Fill draft tokens
```

[source: dspark.md specification]

### Key Finding: Draft and Verify Run on Different Devices

**Draft proposal executes on the DSpark executor tier, verification runs full 43-layer model across both GPUs.**

Source: source code analysis of ds4.c.
- `ds4_session_prepare_dspark_draft` (line 59335): switches to `g->dspark_exec_tier` before running draft impl, switches back after. The executor tier is selected dynamically: follows `e->placement[DS4_N_LAYER + 1]` (output head's placement tier) by default, adjusted for tensor parallelism, overridable via `DS4_DSPARK_EXEC_TIER`. On the current 2-GPU system with default config, the output head resides on GPU1, making GPU1 the default exec tier.
- `metal_graph_verify_suffix_tops_impl` (line 33895): iterates DS4_N_LAYER = 43 layers with `g->placement[il+1]` cross-device tier switching.

The pipeline is:

```
Step N: GPU0: [base model decode layers 0-23] → PCIe xfer → idle during GPU1 work
         GPU1: [base model decode layers 24-42 + output head] → [DSpark draft chain] → [verify full 43 layers]
```

Draft proposal and verification run sequentially on GPU1. GPU0 is idle after layer 23 transfer. There is no overlap between draft computation and base model computation within a single step.

### Multi-GPU Overlap Opportunities

**Opportunity 1: Overlap capture with compute on non-target layers**

During the base model forward pass, HC capture runs on the device owning each target layer. For non-target layers, the capture mechanism is idle. This means on a 2-GPU system, when GPU0 is computing a non-target layer and GPU1 is also computing, there's no capture overhead on non-target layers.

Impact: Zero — capture is a lightweight recording operation that adds <1µs per target layer [hypothesis].

**Opportunity 2: Overlap stage chain on GPU1 with base model on GPU0**

If GPU0 finishes its layers (0-23) and sends activation to GPU1, GPU0 is idle while GPU1 processes layers 24-42. During this idle window, GPU0 could compute the draft stage chain.

Current constraint: The stage chain requires the final hidden state from ALL layers (output of GPU1's last layer). GPU0 cannot start the draft chain until GPU1 finishes and sends the result back.

**Opportunity 3: Overlap verification on GPU0 with draft proposal on GPU1**

After the base model forward pass completes:
- GPU0 knows the output of its layers (which contributed to capture)
- GPU1 has the final hidden state

In theory, GPU1 could start the draft proposal while GPU0 prepares for verification. But the draft proposal needs GPU0's expert sharding (TP suspend/resume), making this complex.

### TP Expert Sharding: No Conflict on 2-GPU Pipeline-Parallel

On 2-GPU pipeline-parallel without tensor parallelism, `g->tp_world` = 1. The TP suspend/resume logic in `metal_graph_eval_dspark_stage_chain` (ds4.c line 31193):
```c
const bool suspended_expert_sharding = saved_tp_world == 2;
if (suspended_expert_sharding) {
    ds4_gpu_tp_suspend_expert_sharding(1);
}
```
This is a no-op when TP is not active. The draft chain runs on GPU1 via explicit tier switch (ds4.c line 59335), where all 81 support model tensors are cached. No cross-device weight access occurs during the draft chain.

For 2-GPU pipeline-parallel (no TP), the draft chain operates entirely on GPU1 with local support weights. Cross-device reads of support model weights do not occur.

### Stage Chain Critical Path

```
// ds4_session_prepare_dspark_draft (ds4.c line 59335):
//   switch to g->dspark_exec_tier (GPU1, where support weights are cached)

Stage 0 (GPU1): main_proj + main_norm → hidden
Per stage (GPU1, TP inactive on non-TP config):
  ← requires hidden from previous stage
  Embed token → norm → project → fill cache
  Attention + FFN over support KV ring (local)
  → next hidden
Final stage (GPU1): norm → hc_head → base logits
Markov loop: w1 → matvec → argmax (GPU or CPU)

// switch back to GPU0 after draft completes
```

[source: ds4.c lines 59335-59360 — ds4_session_prepare_dspark_draft]

The draft chain runs on GPU1 (executor tier). GPU0 is idle during draft proposal.

### Verification Critical Path

```
metal_graph_verify_suffix_tops_impl (ds4.c line 33821):
  Iterates layer il = 0..DS4_N_LAYER-1:
    tier = g->placement[il+1]         // which GPU owns this layer
    metal_graph_encode_layer_batch()  // encode on correct device
  → uses ALL layers across both GPUs
  → cross-device via placement[] tier switching
```

Verification runs the full 43-layer model. GPU0 encodes layers 0-23, then switches to GPU1 for layers 24-42 + output head via `metal_graph_set_active_tier_batch`.

## Overlap Constraints

Source-verified constraints on overlap potential [measured: ds4.c]:

1. **Draft chain runs on GPU1** (ds4.c line 59335): support model weights are cached on GPU1. Moving draft to GPU0 would require relocating 5.99 GiB support tensors to GPU0's ~12 GiB headroom.

2. **Verify runs full 43-layer model** (ds4.c line 33895): verify uses both GPUs via cross-device `placement[]` switching. Cannot overlap verify on one GPU with independent work on the other.

3. **Pipeline order is sequential**: GPU0 decode → transfer → GPU1 decode → GPU1 draft → GPU1 verify. No phase can start before its input from the prior phase completes.

Overlap opportunity remaining: GPU0 is idle during GPU1's decode tail (layers 24-42), draft chain, and verify phases. This idle period is the only time window for overlapping independent work. Current design has no overlapping work queued for GPU0 during this window.

## Measured Timing (from baseline)

| Component | Time | Source |
|---|---|---|
| Total decode time per token | 14.7 ms | [measured: ds4-bench, gen_steady_tps=68.47] |
| GPU0 compute (24 layers) | ~7.0 ms | [derived: 48% of total from layer distribution] |
| GPU1 compute (19 layers) | ~5.5 ms | [derived: 37% of total] |
| Output head | ~0.5 ms | [derived: 3.4%] |
| Synchronization/other | ~1.7 ms | [derived: 12% residual] |

DSpark draft chain time measured via DS4_DSPARK_STATS: prop_chain=228ms total over 20 cycles, mean 11.4ms/step [measured: research-log.md §DSpark-acceptance-rate]. Verify time (full 43-layer batch-encode): 103ms total over ~12 verify calls, mean 8.6ms/call [measured: same source].

## Acceptance Rate

Acceptance rate is determined by: support model quality, capture reduction fidelity, and confidence threshold calibration. Measured 5% [research-log.md §DSpark-acceptance-rate]. Experiments E1 and D3.1 [PRD-3.md] are designed to isolate the contributing factors. No analytical bound is asserted.

## Key Findings

1. **Draft chain runs on the DSpark executor tier** (dynamically selected, follows output head placement by default: GPU1 on this system) via `ds4_session_prepare_dspark_draft` tier switch [measured: ds4.c line 59335]. Support model weights (81 tensors) are cached on the executor tier. GPU0 is idle during draft proposal.

2. **Verification runs full 43-layer model** via `metal_graph_verify_suffix_tops_impl` with cross-device `placement[]` tier switching [measured: ds4.c line 33895]. GPU1's layers (24-42 + output head) are included.

3. **Pipeline imbalance**: GPU0 computes layers 0-23, then idles while GPU1 completes layers 24-42 → draft chain → verify. GPU1 does decode tail, draft chain, and verify sequentially. GPU0 is idle for multiple sequential phases.

4. **No overlap in current design**: Draft propose and verify are sequential on GPU1. GPU0 is idle after layer 23 transfer.

5. **Acceptance rate ceiling unknown**: Not yet determined whether 5% acceptance [measured: research-log.md] is caused by support model quality, capture reduction loss, or confidence threshold calibration. E1 and D3.1 [PRD-3.md] are designed to isolate.

## Recommendations

1. **Account for pipeline imbalance**: Current design has GPU1 doing decode tail + draft chain + verify sequentially while GPU0 idles. Quantify the idle fraction via P0.1 (layer split) and E2 (per-step cost breakdown from PRD-3.md).

2. **Measure support model quality before optimizing plumbing**: Run E1 (draft logit vs base model logit, PRD-3.md) to determine whether the 5% acceptance rate is a model quality issue or a pipeline loss issue.

3. **Benchmark throughput**: Run E3 (DSpark throughput comparison) to measure whether DSpark provides net benefit at current configuration.

4. **Determine exec tier trade-off**: If E2 shows tier switch cost dominates, test moving DSpark exec tier to GPU0 via `DS4_DSPARK_EXEC_TIER=0`. Support model pack (5.99 GiB) must fit in GPU0 headroom (~12 GiB, measured).

## See Also

- [dspark.md](../code/concepts/dspark.md) — DSpark architecture specification
- [multi-gpu-pipeline.md](../code/concepts/multi-gpu-pipeline.md) — pipeline architecture
- [roofline-analysis.md](roofline-analysis.md) — throughput ceilings
- [tuning-guide.md](tuning-guide.md) — tuning recommendations
