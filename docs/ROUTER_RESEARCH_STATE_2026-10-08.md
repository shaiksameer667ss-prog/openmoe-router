# Router Research State — 2026-10-08

## RM1 / RM1b status

### T3 margin convention — LOCKED

For the T3 checkpoint, use the natural continual-learning definition:

`m_T3(x) = max(logits[0:60]) - max(logits[60:80])`

That is: three previously learned tasks (classes 0–59) versus the most recently learned task (classes 60–79).

This convention is supported by the archived RM1 T3 margin mean of approximately +7.365677. The RM1 T3 analysis therefore remains canonical.

For final checkpoints F and E-BOTH v4, use:

`m_F(x) = max(logits[0:80]) - max(logits[80:100])`

`m_E-BOTH(x) = max(logits[0:80]) - max(logits[80:100])`

The corrected RM1b run matched the archived F and E-BOTH margin means to numerical precision, confirming those target definitions.

### RM1b first execution

Status: WITHDRAWN / INVALID.

Reason: the first runner used a fixed T3-style class range for every checkpoint and therefore did not implement the checkpoint-relative target at F/E-BOTH.

Its prior Case 1 classification is withdrawn.

### RM1b corrected run

Status: PARTIALLY VALID.

Valid scientific results:
- F full16: A R² 0.051198; B R² 0.051789; C R² 0.038009; C permutation p 0.000999; D mean R² 0.057905; E mean R² 0.033056.
- E-BOTH full16: A R² 0.037822; B R² 0.039278; C R² 0.017198; C permutation p 0.000999; D mean R² 0.052810; E mean R² 0.011985.

These F/E-BOTH results use the correct final-task margin definition and are eligible for the preregistered decision.

The T3 values from that run are invalid for comparison because that execution used 0:40 vs 40:60.

### RM1b preregistered decision

Decision: NO_PRE_REGISTERED_CASE.

Case 1 requires C R² >= 0.02 at both F and E-BOTH with permutation p < 0.01 at both. F passes (0.038009), E-BOTH misses the threshold (0.017198) despite p = 0.000999.

Case 2, Case 3, and Case 4 do not fire.

Do not relax the 0.02 threshold post hoc.

### Interpretation

The valid F/E-BOTH results establish detectable class- and task-controlled routing-to-margin signal at both checkpoints, but the signal strength is weaker at E-BOTH than at F and does not meet the preregistered Case 1 threshold at E-BOTH.

The LOTO means (F 0.033056; E-BOTH 0.011985) are secondary evidence that cross-task transfer of the routing-to-margin mapping is limited.

No causal claim is made that a specific freeze intervention caused the signal reduction; F and E-BOTH differ in their training conditions.

## Next actions

1. Run the T3-only RM1b analysis with the locked T3 target 0:60 vs 60:80. This resolves the remaining metric comparability issue without rerunning the already-valid F/E-BOTH analysis.
2. RM1c may proceed using the settled F/E-BOTH margins and the same checkpoint-relative T3 convention once the T3 rerun is available.
3. RM2 remains conditional. The current RM1b evidence does not satisfy the preregistered Case 1 gate.

## Artifacts

- `experiments/results/RM1/rm1_results.json`
- `experiments/results/RM1b/rm1b_invalid_run.json`
- `experiments/results/RM1b_corrected/rm1b_partial_valid_F_E.json`
- `experiments/preregistrations/RM1b-v1.0-amendment-A3.md`
- `scripts/rm1b_corrected.py`
