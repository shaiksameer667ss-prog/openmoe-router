# Router Research State — 2026-10-08

## RM1 / RM1b status

### T3 RM1 target — UNREPRODUCED

The archived RM1 target is:

`m_T3(x) = max(logits[0:60]) - max(logits[60:80])`

Archived RM1 reported:
- N = 8000
- margin mean = +7.365677
- margin std = 1.514167
- theta3 SHA = `26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc`

The current audit used the exact same theta3 SHA and the explicit T3 formula above, but produced:
- N = 8000
- margin mean = -2.526397
- margin std = 2.329262
- mean difference = -9.892074
- std difference = +0.815095

Per-source-task current means were:
- T0: -2.340171
- T1: -2.306948
- T2: -1.969035
- T3: -3.489433

No archived 8000-element RM1 margin vector was found in the accessible local artifacts. The archived RM1 JSON contains aggregate provenance/metrics only; it does not contain sample IDs or the per-example target vector. The historical `scripts/rm1.py` source is also not present in accessible git history.

Therefore the archived RM1 T3 target cannot currently be reproduced elementwise, and its root cause is NOT_ESTIMABLE from the accessible evidence. Possible domains remain sample population/order, preprocessing/transform path, model/logit extraction path, or historical implementation differences; none is established as the cause.

Do not run or interpret another T3 RM1b regression until the archived sample mapping and/or per-example target vector becomes available.

### RM1b first execution

Status: WITHDRAWN / INVALID.

Reason: the first runner used a fixed T3-style class range for every checkpoint and therefore did not implement the checkpoint-relative target at F/E-BOTH.

Its prior Case 1 classification is withdrawn.

### RM1b corrected run

Status: PARTIALLY VALID.

Valid scientific results:
- F full16: A R² 0.051198; B R² 0.051789; C R² 0.038009; C permutation p 0.000999; D mean R² 0.057905; E mean R² 0.033056.
- E-BOTH full16: A R² 0.037822; B R² 0.039278; C R² 0.017198; C permutation p 0.000999; D mean R² 0.052810; E mean R² 0.011985.

These F/E-BOTH results use the correct final-task margin definitions and are eligible for the preregistered decision.

The T3 values from that run are withdrawn from comparison because that execution used 0:40 vs 40:60.

### RM1b T3 final run

Status: AUDIT-PENDING / NOT INTERPRETABLE.

Current T3 run used the explicit target `max(logits[0:60]) - max(logits[60:80])` and obtained margin mean -2.526397, which does not reproduce archived RM1 +7.365677.

The regression/permutation outputs are therefore not scientific results and must not be used for RM1b decision-making.

### RM1b preregistered decision

Decision: NO_PRE_REGISTERED_CASE.

Case 1 requires C R² >= 0.02 at both F and E-BOTH with permutation p < 0.01 at both. F passes (0.038009), E-BOTH misses the threshold (0.017198) despite p = 0.000999.

Case 2, Case 3, and Case 4 do not fire.

Do not relax the 0.02 threshold post hoc.

### Interpretation

The valid F/E-BOTH results establish detectable class- and task-controlled routing-to-margin signal at both checkpoints, but the signal strength is weaker at E-BOTH than at F and does not meet the preregistered Case 1 threshold at E-BOTH.

The LOTO means (F 0.033056; E-BOTH 0.011985) are secondary evidence that cross-task transfer of the routing-to-margin mapping is limited.

No causal claim is made that a specific freeze intervention caused the signal reduction; F and E-BOTH differ in their training conditions.

The T3 RM1 result itself is a reproducibility-blocked historical measurement, not evidence for or against the routing-to-margin hypothesis.

## Reproduction audit

Audit ID: `RM1-T3-REPRO-AUDIT-v1.0`

Status: `ARCHIVE_VECTOR_UNAVAILABLE`

Verified:
- repo: `/kaggle/working/openmoe-router`
- git HEAD: `f10fd09b1b88a51561f1912b502635d5cc692d9a`
- Torch: 2.11.0+cu128
- GPU: Tesla T4
- theta3 SHA: exact match to archived RM1
- current sample count: 8000
- no training
- no regression
- no permutations
- no checkpoint modification

Unavailable:
- archived RM1 sample IDs/index mapping
- archived per-example 8000-value margin vector
- recoverable historical `scripts/rm1.py` implementation

Current audit artifacts on Kaggle:
- `experiments/results/RM1_T3_REPRO_AUDIT/audit.json`
- `experiments/results/RM1_T3_REPRO_AUDIT/audit.md`
- `experiments/results/RM1_T3_REPRO_AUDIT/current_t3_samples.npz`

### Reproducibility conclusion

RM1 T3 is currently UNREPRODUCED and should remain so in the ledger. The exact archived number +7.365677 is retained as a historical reported measurement, but not as a currently reproducible target.

## Next actions

1. Stop T3 RM1b reruns unless a recoverable archived sample mapping/vector or historical implementation is found.
2. Treat F/E-BOTH RM1b results as the only current preregistered RM1b scientific results.
3. RM1c may proceed on the valid F/E-BOTH data, but any T3-dependent conclusion must be excluded or explicitly marked unavailable.
4. RM2 remains conditional on the preregistered logic; do not promote it from a T3 reproduction failure.

## Artifacts

- `experiments/results/RM1/rm1_results.json`
- `experiments/results/RM1b/rm1b_invalid_run.json`
- `experiments/results/RM1b_corrected/rm1b_partial_valid_F_E.json`
- `experiments/preregistrations/RM1b-v1.0-amendment-A3.md`
- `scripts/rm1b_corrected.py`
- `experiments/results/RM1_T3_REPRO_AUDIT/audit.json`
- `experiments/results/RM1_T3_REPRO_AUDIT/audit.md`
- `experiments/results/RM1_T3_REPRO_AUDIT/current_t3_samples.npz`
