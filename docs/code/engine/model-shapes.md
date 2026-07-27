# Model Shape Profiles

## Files

- `ds4.c` — shape constants, profile tables

## Purpose

Define the three supported model architectures as compile-time constant tables.  Weight binder and metadata validator select one profile after GGUF header validation.

## Shape Profiles

See [model-shape-detection.md](../concepts/model-shape-detection.md) for the model shape constant table with verified values.

## Key Dimensions

| Constant | Value | Meaning |
|---|---|---|
| `DS4_MAX_LAYER` | 79 | worst-case (GLM52) array size |
| `DS4_MAX_EXPERT` | 384 | worst-case expert count |
| `DS4_MAX_EXPERT_USED` | 8 | worst-case top-k |
| `DS4_N_HC` | profile-dependent | hyper-connection streams (Flash/Pro=4, GLM52=0) |
| `DS4_N_HEAD` | profile-dependent | attention heads per layer |
| `DS4_N_HEAD_DIM` | profile-dependent | head dimension (Flash/Pro=512, GLM52=576) |
| `DS4_N_VALUE_DIM` | 512 | value dimension (all profiles) |
| `DS4_N_ROT` | 64 | rotary dimension (all profiles) |
| `DS4_N_LORA_Q` | profile-dependent | Q low-rank projection (Flash=1024, Pro=1536, GLM52=2048) |

## See Also

- [gguf-format.md](../concepts/gguf-format.md) — GGUF metadata keys used during profile selection

## Invariants

- Profile selected once at engine open, immutable after.
- Arrays reserve max (Pro) dimensions; hot loops read active profile.
- GLM52 shares Flash's `DS4_N_HEAD` = 64 but has different `DS4_N_EMBD` = 6144 and different `DS4_N_HEAD_DIM` = 576.

[← Back to Index](../README.md)
