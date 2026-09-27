# Decoupled Panel Preregistration v2

## Scope

This document fixes the multi-seed aggregation rule for the decoupled
router panel before seed 1 is run.

The primary metric is old-task NCM at boundary 4 using the bounded
715-example replay buffer.

## Per-seed decision rules

### Active

The router effect is classified as active only if both hold:

- D2 - D1 > 0.010
- D2 - D3 > 0.010

### Inert

The router effect is classified as inert only if both hold:

- |D2 - D1| < 0.005
- |D2 - D3| < 0.005

Otherwise the seed is classified as ambiguous.

## Multi-seed aggregation

Seeds 0, 1, and 2 are evaluated independently using the same evaluator,
memory capacity, and primary metric.

The final three-seed classification uses a 2-of-3 replication rule:

- Active: the per-seed Active criterion must fire in at least 2 of 3 seeds.
- Inert: the per-seed Inert criterion must fire in at least 2 of 3 seeds.
- Ambiguous: otherwise.

A seed classified as ambiguous does not count toward either Active or
Inert replication.

No across-seed mean or post-hoc threshold is used to replace this rule.

## Secondary observation

Task-0 learned-head diagonal accuracy is recorded separately as a
descriptive observation. It is not part of the primary router-effect
decision rule.

Task 0 occurs before continual-learning interventions activate, so
task-0 accuracy is not used to classify the router effect on old-task
recoverability.

## Current observations before seed 1

Seed 0:

- D1 old-task NCM = 0.06325
- D2 old-task NCM = 0.05925
- D3 old-task NCM = 0.10150
- D2 - D1 = -0.00400
- D3 - D2 = +0.04225
- Per-seed classification: ambiguous

Seed 2:

- D1 old-task NCM = 0.05100
- D2 old-task NCM = 0.08275
- D3 old-task NCM = 0.08225
- D2 - D1 = +0.03175
- D3 - D2 = -0.00050
- Per-seed classification: ambiguous

Seed 1 is run under this fixed rule.

## Interpretation constraint

The final result will be reported according to the rule above without
changing thresholds, aggregation method, or classification criteria
after observing seed 1.
