# Seed-0 Diagnostic Screening Report

## Protocol

CIFAR-100 continual learning with 5 tasks and 20 classes per task.
Primary training conditions:

- `none`
- `head_masked`
- `head_masked_frozen_old`

Probe suite:

- learned classifier head
- refit NCM
- all-seen-class linear probe
- task-restricted linear probe
- chance baseline

## Seed-0 screening

| Condition | Final old-task head | Final old-task linear probe | 1.5x criterion | 10-pp retention |
|---|---:|---:|---|---|
| `none` | 0.0000 | 0.1894 | Pass | Fail |
| `head_masked` | 0.1200 | 0.2169 | Pass | Fail |
| `head_masked_frozen_old` | 0.1208 | 0.2155 | Pass | Fail |

The final old-task linear probe is at least 1.5x the learned-head accuracy in all three conditions.

The strict retention criterion fails in all three conditions because the all-seen linear-probe drop exceeds 10 percentage points for old tasks 0 and 1.

## Task-restricted diagnostic

Final-boundary task-restricted probe accuracy:

- `none`: 0.3730, 0.3830, 0.4260, 0.4210
- `head_masked`: 0.4255, 0.4220, 0.4665, 0.4515
- `head_masked_frozen_old`: 0.4270, 0.4205, 0.4655, 0.4500

Final-boundary all-seen chance baseline: 0.01.

The task-restricted probe remains substantially higher than the all-seen probe, indicating that class-space expansion materially affects the all-seen diagnostic.

## Decision

The decoder-dominant branch is not selected under the written preregistered criterion because both magnitude and retention criteria must hold, and the strict retention criterion fails.

The trunk-dominant branch is not selected.

The router/expert four-cell decomposition was not run.

**Outcome: No dominant MoE-specific mechanism at this scale.**

## Reproducibility

The probe-enabled `none`, seed-0 run exactly matched the historical canonical 20-class/task accuracy matrix.

The diagnostic RNG-state issue was fixed for both probe and drift evaluation, with the full test suite passing after the fixes.
