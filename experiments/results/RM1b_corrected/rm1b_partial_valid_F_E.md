# RM1b corrected run — partial validity record

## Scientific decision

**NO_PRE_REGISTERED_CASE**

This decision is valid because the preregistered decision uses the F checkpoint as primary and E-BOTH as replication. Both checkpoints used the correct checkpoint-relative target:

- F: old 0:79 vs new 80:99
- E-BOTH: old 0:79 vs new 80:99

The F/E-BOTH checkpoint-specific margin means match the archived RM1 values to numerical precision.

## Full16 results

| Checkpoint | A R² | B R² | C R² | C p | D mean R² | D range | E mean R² | Status |
|---|---:|---:|---:|---:|---:|---|---:|---|
| T3 | 0.077273 | 0.069692 | 0.037734 | 0.000999 | 0.083840 | [0.063428, 0.098406] | 0.035228 | INVALID TARGET |
| F | 0.051198 | 0.051789 | 0.038009 | 0.000999 | 0.057905 | [0.034799, 0.094968] | 0.033056 | VALID |
| E-BOTH | 0.037822 | 0.039278 | 0.017198 | 0.000999 | 0.052810 | [0.020941, 0.083073] | 0.011985 | VALID |

## Decision rule

Case 1 requires C R² >= 0.02 at both F and E-BOTH.

Observed:
- F C R² = 0.038009
- E-BOTH C R² = 0.017198

Therefore Case 1 **does not pass**.

Cases 2–4 also do not match their preregistered conditions.

Final decision: **NO_PRE_REGISTERED_CASE**.

## T3 issue

The T3 portion of this run used the incorrect range 0:40 vs 40:60. The correct T3 target is 0:60 vs 60:80. Therefore the T3 values from this run are preserved only as invalid provenance and are not scientifically interpreted.

No training occurred; no checkpoints were modified.
