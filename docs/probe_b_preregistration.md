# Probe B — Cross-Boundary Newest-Task NCM (Preregistered)

Status: FROZEN

## 1. Question

At boundary 4, replay training produces higher Task-4 NCM accuracy than
plain CE training under the bounded 715-example decoder protocol.

This probe tests whether that effect is:

(a) generic to the task that was just trained,
(b) specific to boundary 4,
(c) persistent after the task ceases to be newest, or
(d) transient and decaying once the task ages.

## 2. Measurements

For each seed in {0, 1, 2} and each training regime in
{replay_715, plain_ce}:

### M1 — newest task at boundary 3

- Checkpoint: boundary 3 (`task_3.pt`)
- Evaluated task: Task 3
- Decoder fit: 715-example fixed probe buffer containing only Tasks 0–3
- Expected classes: 80
- Metric: Task-3 NCM accuracy

The boundary-3 probe buffer is evaluation-only. It is constructed
deterministically from the Tasks 0–3 training data and contains no
Task-4 examples.

### M2 — newest task at boundary 4

- Checkpoint: boundary 4 (`task_4.pt`)
- Evaluated task: Task 4
- Decoder fit: the existing seed-specific Path-1 A buffer
- Expected classes: 100
- Metric: Task-4 NCM accuracy

### M3 — recent but no longer newest at boundary 4

- Checkpoint: boundary 4 (`task_4.pt`)
- Evaluated task: Task 3
- Decoder fit: the existing seed-specific Path-1 A buffer
- Expected classes: 100
- Metric: Task-3 NCM accuracy

### M4 — old tasks at boundary 4

Already measured in Path-1:

- Checkpoint: boundary 4
- Evaluated tasks: Tasks 0–2
- Decoder fit: seed-specific Path-1 A buffer
- Metric: mean old-task NCM accuracy

## 3. Checkpoint inventory

All replay_715 and plain-CE boundary-3 and boundary-4 checkpoints
for seeds 0–2 are available.

Seed-0 replay checkpoints were regenerated solely to fill the missing
boundary-checkpoint artifacts and were accepted only after their full
accuracy matrix exactly matched the archived seed-0 replay run.

## 4. Decision rules

For each measurement X, define:

`gap_X = replay NCM - plain CE NCM`

### H1 — Generic newest-task effect

M1 and M2 show approximately the same effect:

`abs(gap_M1 - gap_M2) < 0.020`

for each seed.

### H2 — Boundary-4-specific effect

For all three seeds:

`gap_M1 < 0.020`
and
`gap_M2 > 0.040`

### H3 — Persistent recent-task effect

For all three seeds:

`gap_M3 > 0.040`

### H4 — Transient recent-task effect

For all three seeds:

`gap_M3 < 0.020`

H3 and H4 are mutually exclusive.

H1 and H2 are mutually exclusive.

The following combinations may co-occur:

- H1 + H4: newest-task effect that decays after the task ages
- H1 + H3: newest-task effect that persists for at least one boundary
- H2 + H4: boundary-4-specific effect that is transient
- H2 + H3: boundary-4-specific effect that also appears on the recent task

## 5. Thresholds

Small gap: `< 0.020`

Large gap: `> 0.040`

Ambiguous: `0.020` to `0.040`

A gap is called present in all seeds only when the sign is consistent
and the magnitude threshold is satisfied in all three seeds.

## 6. Execution protocol

All decoder evaluations use the same NCM implementation and held-out
CIFAR-100 test split.

M1 uses a dedicated boundary-3 715-example probe buffer containing only
Tasks 0–3. It is not the final boundary-4 replay buffer and contains
no Task-4 examples.

M2 and M3 use the already archived seed-specific Path-1 A buffer.

No new training is performed for Probe B.

## 7. What this probe does not claim

- No causal mechanism for any observed NCM difference.
- No generalization beyond boundaries 3 and 4.
- No training-time memory-matched comparison.
- No conclusion about boundaries 1 or 2.
- No claim about tasks other than those explicitly evaluated.

## 8. Reproducibility

Seeds: 0, 1, 2

Boundary checkpoints:

- `task_3.pt`
- `task_4.pt`

Decoder capacity: 715 examples

Evaluation: held-out CIFAR-100 test split

Evaluator controls:

- `--eval-task`
- `--expected-classes`

Any deviation from this protocol is recorded in a dated
`probe_b_preregistration_v2.md` rather than modifying this document.
