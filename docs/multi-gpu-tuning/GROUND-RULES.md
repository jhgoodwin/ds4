# Ground Rules — Experimental Methodology

## Purpose

Define the epistemology and efficiency constraints governing all experiments in this research project. Every document, experiment, and analysis inherits these rules by reference. Do not repeat them in topic-specific documents.

---

## 1. Knowledge-Building Cycle

Every claim must follow this cycle. If any element is missing, the claim is not actionable.

```
hypothesis → test → predicted outcome → measurement → compare → if mismatch → new hypothesis
```

### 1.1 Required Elements

| Element | Definition | Example |
|---|---|---|
| Hypothesis | Testable statement about the system | "PCIe Gen 5 x8 achieves ≥28 GB/s bidirectional with aligned 256 KB transfers" |
| Test | Reproducible procedure | "Run cudaMemcpyPeerAsync GPU0→GPU1 and GPU1→GPU0 concurrently, 256 KB, 100 iterations" |
| Predicted outcome | Quantitative bound or direction | "Measured ≥28 GB/s aggregate bidirectional" |
| What outcome informs | Which decision or model the result feeds | "If confirmed, PCIe is not bottleneck for ≤14 KB activation transfers. If refuted, cross-device transfer is a bottleneck candidate." |
| Iteration trigger | Condition that induces next experiment | "If measured <25 GB/s, run topology discovery + IOMMU-disabled test to isolate cause" |

### 1.2 No Orphan Claims

Every numerical statement, estimate, or threshold in any document must be tagged with one of:

| Tag | Meaning |
|---|---|
| `[measured: <ref>]` | Value from experiment on this system |
| `[datasheet: <vendor doc>]` | Value from hardware specification |
| `[derived: <formula>]` | Value computed from measured values |
| `[hypothesis: <rationale>]` | Value to be tested — must have corresponding experiment planned |

Untagged values are forbidden.

---

## 2. High-Contrast Requirement

Every experiment must be designed so its outcome eliminates at least one competing explanation.

### 2.1 Test

Before running, enumerate the possible outcomes. For each outcome, state which hypothesis(es) it rules out. If all outcomes are consistent with the same set of hypotheses, the experiment is low-contrast — redesign or skip.

**Low-contrast example**: "Measure decode throughput at batch sizes 1, 2, 4, 8, 16, 32" without a hypothesis about the scaling law. The sweep produces a curve but doesn't rule out any specific bottleneck.

**High-contrast example**: "If decode throughput doubles from batch 1→2, bottleneck is launch overhead. If throughput stays flat, bottleneck is memory bandwidth." One test, two possible outcomes, each eliminates the opposite cause.

### 2.2 Matrix Sweeps

Sweeping N parameters × M values without per-point hypotheses is forbidden. Use instead:

- **Binary search** — find inflection point (e.g., transfer size where p2p beats host bounce)
- **Adaptive sampling** — test at predicted boundaries of competing models
- **Factorial design with hypothesis per cell** — each (param, value) pair maps to a distinct prediction

---

## 3. Abandonment Criteria

Stop testing a dimension when additional tests stop producing information. Do not exhaustively explore plateaus.

### 3.1 Plateau Rule

If two consecutive measurements on the same dimension produce results within margin of error, terminate that line unless the test cost is below threshold.

```
margin_of_error = measurement_noise * 2   (estimated from 3 repeat runs)
plateau_trigger = |result[n] - result[n-1]| ≤ margin_of_error
```

When plateau triggers:
1. Log the plateau value and range
2. Document that further testing on this dimension is unlikely to change the result
3. Move to next experiment
4. Exception: if test cost < 30 seconds, run one more confirmation, then stop

### 3.2 Model Refinement Limit

If a model's prediction error remains >20% after 3 refinement cycles:
1. Document the residual as "unexplained gap"
2. Estimate its impact on final tuning decisions
3. Move on — do not chase the last 5% unless that gap is the dominant bottleneck

### 3.3 Phase Time Budget

Each research phase has a hard time budget (active testing hours). If the budget is exhausted before all planned experiments complete:
1. Log which experiments were not run
2. Note the priority of each skipped experiment
3. Advance to next phase
4. Skipped experiments may be revisited only if later findings show they are critical

---

## 4. Learning Rate Optimization

Optimize for information gained per hour of active experimentation. Some error is acceptable.

### 4.1 Pre-Experiment Estimate

Before each experiment block, estimate:

```
learning_rate = expected_info_gain / expected_wall_time
```

Where:
- `expected_info_gain` = number of hypotheses the experiment can rule out (weighted by how central they are to the research questions)
- `expected_wall_time` = setup + execution + analysis time in hours

### 4.2 Prioritization

- High learning rate (>1 hypothesis ruled out per hour) → run first
- Medium learning rate (0.25–1) → run if time permits
- Low learning rate (<0.25) → skip unless it gates a higher-rate experiment

### 4.3 Diminishing Returns

If the highest-rate experiment available has learning rate < 0.1 (would take >10 hours to rule out one hypothesis), stop active testing and switch to analytical work (model building, document writing) until a higher-rate experiment becomes feasible.

---

## 5. Assumption Hygiene

### 5.1 Source Tagging

Every numerical value must carry a source tag as defined in §1.2. Review for untagged values before any document is considered final.

### 5.2 Confidence Bounds

Every measurement must be reported with:
- Central estimate (mean or median)
- Spread (standard deviation or 5th/95th percentile)
- Number of trials
- Whether the system was in steady state (warm vs cold)

### 5.3 Context Validity Check

Before each experiment, verify that the premises it depends on still hold:

- Is the system configuration unchanged since last test? (GPU clocks, power limit, VM pinning, storage contention)
- Are the software components the same version?
- Is the test environment free of known confounds (other GPU workloads, background I/O)?

If any premise is violated, either:
1. Fix the premise and retest, or
2. Abandon the experiment and log the violation

Do not run an experiment on a system whose state is unknown.

---

## 6. Falsification Protocol

When measured outcome does not match predicted outcome:

### 6.1 Classification

| Category | Condition | Action |
|---|---|---|
| Measurement error | Outcome within 3σ of prediction but opposite direction | Increase trials, check for systematic bias |
| Parameter misestimate | Outcome differs by <2× from prediction | Refine parameter estimates, re-run |
| Missing factor | Outcome differs by >2× from prediction | Design isolation experiment for each candidate missing factor |
| Wrong model | No candidate factor explains gap | Document as unexplained, estimate impact, move on |

### 6.2 Iteration Limit

After 3 refinement cycles without converging to <20% error, classify as "Wrong model" per above. Do not iterate indefinitely.

---

## 7. Phase Advancement Gates

### 7.1 Gating Criteria

Advance to next phase only when:
1. Current phase time budget not exhausted, OR
2. Plateau detected on all remaining high-priority experiments, OR
3. Remaining experiments have learning rate < 0.25

### 7.2 Phase Log

At each phase transition, log:
- Experiments completed / planned
- Key findings (hypotheses confirmed, refuted, or abandoned)
- Unexplained gaps carried forward
- Skipped experiments and why

---

## 8. Research Log Format

Every experiment entry in `research-log.md` must follow:

```markdown
### YYYY-MM-DD — Experiment: [short name]

**Hypothesis**: [one sentence]
**Context premises**: [system state, versions, configuration]
**Predicted outcome**: [quantitative bound or direction]
**What it informs**: [which decision or model this feeds]
**Method**: [reproducible steps]
**Raw result**: [central estimate, spread, n_trials, steady-state flag]
**Compare to prediction**: [match / mismatch / classification per §6.1]
**Iteration trigger**: [what next experiment this induces, or "none — plateau"]
**Learning rate**: [info gain / wall time]
```

---

## References

- [PRD.md](PRD.md) — Research objectives and topics (all experiments governed by these ground rules)
- [research-log.md](research-log.md) — Experiment entries following §8 format
