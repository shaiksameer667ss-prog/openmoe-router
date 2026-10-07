# RM1b-v1.0 Amendment A1 — Checkpoint-relative margin indexing

Date: 2026-10-08
Status: locked before corrected rerun

## Reason for amendment

The first execution of RM1b completed computationally but was invalidated because the runner used the T3 class split (old 0:59, new 60:79) for every checkpoint.

This amendment corrects that indexing error. It does not change:
- samples
- routing signatures
- CV procedure
- ridge alpha
- permutation count
- seed
- decision thresholds
- Case priority order
- interpretation of D or E

The invalid run is preserved separately and is not used as evidence.

## Correct checkpoint-relative target

For each checkpoint, the current task is the task immediately learned at that checkpoint.

T3:
max(logits[0:60]) - max(logits[60:80])

F:
max(logits[0:80]) - max(logits[80:100])

E-BOTH v4:
max(logits[0:80]) - max(logits[80:100])

The same checkpoint-relative target is used for A, B, C, D, and E.

## RM1 audit requirement

The corrected run reports checkpoint-specific margin means and compares them with the archived RM1 values:

T3 archived margin mean: 7.365677
F archived margin mean: -3.285769
E-BOTH archived margin mean: -2.827610

Any mismatch remains an audit finding and is not silently reconciled.

## Decision rule

The original RM1b Case 1–4 thresholds are unchanged. The corrected run is the only RM1b run eligible for scientific decision.

The prior run is retained as INVALID_RUN for provenance only.
