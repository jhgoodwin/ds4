# DSpark Profile — Not Run

## Status: Skipped

DSpark speculative decoding profiling was not run on GPU. This experiment requires:
1. A DSpark support GGUF model loaded and active during inference
2. DSpark-specific env vars (`DS4_DSPARK_PROBE`, `DS4_DSPARK_SPEC_LOG`) enabled
3. Measuring draft chain latency, acceptance rate, and overlap opportunities

## Abandonment Rationale

Per GROUND-RULES.md §3 (Abandonment Criteria):

- **Plateau rule**: The analytical model in speculative-decode-multi-gpu.md already constrains the design space. Expected overlap savings (7-20%) and coordinator-only restriction are derived from the ds4 pipeline architecture specification and measured baseline, without running DSpark directly.

- **Learning rate** [per §4.1]: Running a DSpark profile would require loading the 5.6 GiB DSpark support GGUF, configuring capture stages, and measuring draft chain timing. The expected info gain is low for the setup cost — the analytical bounds are tight enough to guide design decisions.

- **Phase budget** [per §3.3]: Later phases may revisit if DSpark becomes active and the analytical predictions need validation.

## If Run Later

Enable: `DS4_DSPARK_PROBE=1` and `DS4_DSPARK_SPEC_LOG=1` during ds4 decode. Measure:
- Stage chain latency per step (GPU0 coordinator)
- Acceptance rate vs confidence threshold
- Overlap opportunity: GPU0 draft chain time vs GPU1 compute time
