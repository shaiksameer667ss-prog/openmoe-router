# RM1b — Task- and Class-Controlled Routing→Margin Test

Protocol ID: RM1b-v1.0
Status: preregistered before execution
Training: none
Decision variables: F checkpoint primary; E-BOTH replication
Dataset: Split-CIFAR-100, final-task old-task evaluation
Samples: exactly 8,000 examples from source tasks T0–T3 (2,000/task)
Signatures: pooled8, block0_8, block1_8, full16
CV: 5-fold, shuffled, random_state=0
Permutations: 1,000 matched global target permutations
Ridge alpha: 1.0, intercept fitted
Primary target: m(x) = max(logits[0:60]) - max(logits[60:80])

## Target residualization

m'(x) = m(x) - mean(m | source task)

m''(x) = m(x) - mean(m | source class)

Classes are globally indexed and nested within source tasks. m'' is therefore the class-level residual target.

The class/task residualization is a preregistered nuisance transformation. For permutation nulls, the raw target is permuted first and the corresponding residual target is recomputed using the same grouping rule.

## Conditions

A. Pooled regression: routing signature → m(x), 5-fold CV.

B. Task-residualized regression: routing signature → m'(x), 5-fold CV.

C. Task-and-class-controlled regression: routing signature → m''(x), 5-fold CV.

D. Within-task CV: for each source task separately, routing signature → m'(x), 5-fold CV within that task. Report per-task R², mean and range. Permutation null is the mean of the four within-task R² values after matched permutation.

E. Leave-one-task-out: train on 3 source tasks and test on the held-out task, using m'' as the target. Report held-out-task R² and mean LOTO R². The same class-residualized target definition is applied before the LOTO split; this is an inferential diagnostic, not a deployment-predictive pipeline. Permutation nulls are matched to the same procedure.

For every condition/signature report:
- observed R²
- permutation-null mean/std
- null 95th and 99th percentiles
- observed-minus-null z
- empirical permutation p = (1 + count(null >= observed)) / 1001

For D report each task plus mean/range. For E report each held-out task plus mean.

## Pre-registered decision rules

Priority order:

1. Case 1 — Per-example signal confirmed
   CV R²(C) >= 0.02 at F and E-BOTH
   AND permutation p(C) < 0.01 at both checkpoints
   => RM2 Margin-Budget Router pilot is justified.

2. Case 2 — Class-level signal only
   CV R²(B) >= 0.02 at F
   AND CV R²(C) < 0.01 at F
   => RM2 must be redesigned as a class-conditional router.

3. Case 3 — Task-level signal only
   CV R²(A) >= 0.03 at F
   AND CV R²(B) < 0.01 at F
   => RM1 was task-confounded; per-example RM2 is not justified on this architecture.

4. Case 4 — No residual signal
   BOTH R²(B) < 0.01 and R²(C) < 0.01 at F
   => router pivot on the current architecture is CLOSED.

5. Otherwise
   => NO_PRE_REGISTERED_CASE; do not reinterpret thresholds post hoc.

LOTO is secondary and is not itself a kill criterion. Positive mean LOTO with high task variance is interpreted as a task-conditional design constraint.

## RM1 replication audit

The archived RM1 numerical output is preserved, but the original RM1 source implementation was not preserved. RM1b therefore reruns condition A and records an explicit audit against the archived point estimates:

T3 full16: 0.13364613
F full16: 0.05240563
E-BOTH full16: 0.03784760

A mismatch is reported; it is not silently reconciled or overwritten. The archived RM1 numbers remain the historical record.

## Expected artifacts

- experiments/results/RM1b/rm1b_results.json
- experiments/results/RM1b/rm1b_summary.md
- full stdout log
- git commit + checkpoint SHA-256 provenance

No checkpoint, optimizer state, model parameters, or replay files may be modified.
