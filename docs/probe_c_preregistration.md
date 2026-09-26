# Probe C — Class-Subset-Restricted NCM (Preregistered)

Status: FROZEN
Frozen on: 2026-09-26
Supersedes: none

## 1. Motivation

Probe B measured newest-task NCM at boundaries 3 and 4:

| Measurement | Replay mean | Plain CE mean | Gap |
|---|---:|---:|---:|
| M1: Task 3 at boundary 3 | 0.1715 | 0.2023 | -0.0308 |
| M2: Task 4 at boundary 4 | 0.1875 | 0.1280 | +0.0595 |
| M3: Task 3 at boundary 4 | 0.1482 | 0.1137 | +0.0345 |

The M2 gap is preregistered Claim 2 (confirmed). The M1 gap has the
opposite sign. The class space grows from 80 classes at boundary 3 to
100 classes at boundary 4, and per-task NCM accuracy depends on how
well the target task's classes compete against the current distractor
set. The M1/M2 sign flip may therefore reflect a class-space
inflation effect rather than a difference in intra-task feature
geometry.

Probe C removes distractor competition by restricting the NCM
evaluation argmax to the target task's 20 classes. The NCM fit is
unchanged (fit on all seen classes from the buffer). Only the
evaluation argmax is restricted. This isolates intra-task feature
geometry from distractor competition.

## 2. Measurements

For each seed in {0, 1, 2} and each training regime in {replay_715,
plain_ce}, using boundary-3 and boundary-4 checkpoints, with buffer A
(the replay_715 boundary-4 buffer) and the same NCM protocol as
Probe B, but with evaluation argmax restricted to the target task's
20 class prototypes:

| ID | Checkpoint | Task evaluated | Analog of |
|---|---|---|---|
| R1 | boundary 3 | Task 3 (restricted to 20) | M1 |
| R2 | boundary 4 | Task 4 (restricted to 20) | M2 |
| R3 | boundary 4 | Task 3 (restricted to 20) | M3 |
| R4 | boundary 4 | Tasks 0-2 (restricted to 20) | M4 |

R2 is the primary. R1, R3, R4 are descriptive.

## 3. Primary decision rule (R2)

Let gap_R2 = replay - plain CE on R2, averaged over seeds.

- **H_a — Class-space inflation explains the M2 gap:**
  |gap_R2| < 0.020 in all three seeds.

- **H_b — Intra-task feature geometry explains the M2 gap:**
  gap_R2 > 0.030 in all three seeds.

- **H_amb — Neither:**
  Any seed with 0.020 <= |gap_R2| <= 0.030, or inconsistent signs.

## 4. Descriptive rule for R1, R3, R4

Report gaps per seed. Do not decide between readings based on these.
The purpose is to record the restricted version of the full Probe B
matrix so that the sign flip on M1 can be seen with the class-space
effect removed. No preregistered decision is attached to R1, R3, R4.

## 5. Thresholds

- "Small gap": |gap| < 0.020
- "Ambiguous": 0.020 <= |gap| <= 0.030
- "Large gap": |gap| > 0.030

A claim is "present in all seeds" only if the sign is consistent and
the magnitude threshold is met in 3 of 3 seeds.

## 6. Predictions to be tested

| Reading | R2 gap prediction | R1 gap prediction |
|---|---|---|
| Class-space inflation (H_a) | < 0.020 | closer to 0 than M1's -0.031 |
| Feature geometry (H_b) | > 0.030 | sign and magnitude roughly preserved |

The two readings make different predictions for R1. If H_a holds on
R2 but R1 stays strongly negative, the two effects are not the same
mechanism and the paper should not describe them as one.

## 7. What this does not claim

- Any specific mechanism for why replay produces different features.
- Generalization beyond boundaries 3 and 4.
- Any training-memory-matched comparison.
- Any statement about tasks 0-2 beyond what R4 records.
- Any decomposition of "class-space inflation" into component causes
  (distractor count, distractor-class similarity, feature scale).

## 8. Conditions of execution

Probe C runs only after:

1. This document is committed.
2. The boundary-3 and boundary-4 checkpoints for both regimes and all
   three seeds are available (confirmed by Probe B).
3. The buffer at `replay_715_seed0_buffer.pt` is loadable, and its
   counterparts for seeds 1 and 2 exist.
4. The NCM evaluation code accepts a `restrict_to_classes` argument
   and the restriction is unit-tested on a toy NCM.

Any deviation is recorded as `probe_c_preregistration_v2.md` with a
dated note describing the deviation and what was already known.

## 9. Order of operations

1. Run R1, R2, R3, R4 for seed 0 in both regimes.
2. Report the seed-0 matrix before running seeds 1-2.
3. If the seed-0 R2 result falls in H_a or H_b unambiguously, run
   seeds 1-2 for confirmation.
4. If the seed-0 R2 result is H_amb, stop and report H_amb. Do not
   expand seeds to resolve an ambiguous result.
