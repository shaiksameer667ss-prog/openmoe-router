# RCR Seed-0 Outcome Record

**Date:** 2026-09-26  
**Protocol:** CIFAR-100 Phase 1 / Milestone 6  
**Primary comparison:** A = replay_722 vs C = RCR_722, ß = 1.0

## Seed-0 result

A (`replay_722`):
- Boundary-4 old-task learned-head mean: 0.040625
- Task-4 diagonal: 0.3920
- Task-0 diagonal: 0.326

C (`RCR_722`, ß=1.0):
- Boundary-4 old-task learned-head mean: 0.045000
- ? vs A: +0.004375
- Task-4 diagonal: 0.3735
- Task-0 diagonal: 0.326
- Mean RCR loss across 1000 recorded steps: 0.030048158844932914

## ß sweep

| ß | Old-task mean | ? vs A | Task-4 diagonal | Mean RCR loss |
|---:|---:|---:|---:|---:|
| 0.1 | 0.038250 | -0.002375 | 0.3885 | 0.1083375211 |
| 1.0 | 0.045000 | +0.004375 | 0.3735 | 0.0300481588 |
| 10.0 | 0.048000 | +0.007375 | 0.3440 | 0.0147607281 |

## Validity checks

All three RCR runs:
- Task-0 diagonal = 0.326
- Replay capacity = 722
- RCR active = true
- Method-state memory = 8,906,400 bytes
- Method-state relation = under target
- Stability active = false
- RCR loss values finite

## Preregistered interpretation

The primary ß=1.0 comparison is **null** because ? = +0.004375 is below the +0.005 weak-positive threshold.

ß=10.0 reaches the weak-positive band, but it is not the primary condition and has a 0.048 decrease in Task-4 diagonal relative to A.

No confirmation seeds or RCR_723 runs are triggered by the preregistered screening rule.

## Interpretation boundary

The observed result concerns the RCR package: replay plus compressed per-class historical routing references and the routing-consistency objective. The experiment does not isolate the routing-consistency objective from the compressed routing-reference representation.
