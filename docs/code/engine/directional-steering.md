# Directional Steering (DS)

## Files

- `ds4.c` — steering file load, CPU/Metal steering hooks
- `ds4_cli.c` — `--dir-steering-file`, `--dir-steering-ffn`, `--dir-steering-attn` flags
- `ds4_server.c` — same flags (daemon entry)
- `ds4_agent.c` — same flags (agent entry)
- `ds4_gpu.h` — `ds4_gpu_directional_steering_project_tensor` declaration

## Purpose

Edit selected block outputs in-place to remove or amplify specific directions in activation space.  Opt-in; zero scales skip steering tensor allocation entirely.

## Steering Vectors

One normalized 4096-wide direction vector per layer.  Text file with float values, one vector per line.

## Application

```
y = y - scale * v * dot(v, y)
```

Positive scale: remove direction from activation.  Negative scale: add direction.

Applied to **both** attention and FFN outputs, each with its own scale:
- Attention output: steered with `attn_scale`, before post-attention HC add
- FFN output: steered with `ffn_scale`, before residual add

Direction vectors are class P (per-tier replicated); same host buffer written to every tier slot.  Read-only after init; never re-sync.

## Invariants

- Zero scales → steering tensor not allocated, no inference overhead.
- Direction vectors must be pre-normalized in the steering file (loaded as raw binary, no runtime normalization).
- Projection formula `y = y - scale * v * dot(v, y)` assumes `||v|| = 1`. Non-normalized vectors change effective scale — a vector with norm k applies scale × k² instead of scale.
- Two global scalars (`attn_scale`, `ffn_scale`), uniform across all layers.

## See Also

- [hc-state.md](../concepts/hc-state.md) — HC state structure, lifecycle, and residual stream layout
- [gpu-tensor-primitives.md](../concepts/gpu-tensor-primitives.md) — GPU tensor operations used for steering projection

[← Back to Index](../README.md)
