# RM1b Run — INVALIDATED

The executed RM1b run completed computationally, but its scientific decision is invalid.

## Protocol error

The runner hard-coded:

`old = logits[:, :60]`

`new = logits[:, 60:80]`

for every checkpoint.

That is correct for T3, where task 3 is the current task and classes 0–59 are old while 60–79 are new.

It is **not** correct for F or E-BOTH, where the final current task is task 4. There the preregistered checkpoint-relative margin must compare:

- F: old classes 0–79 vs new classes 80–99
- E-BOTH: old classes 0–79 vs new classes 80–99

Therefore the reported F/E-BOTH C values and the Case 1 classification cannot be used as evidence for the Margin-Budget Router pivot.

The run is preserved for provenance only. It is not to be cited as a valid RM1b result.

## Observed audit symptom

The run reported margin means:

- T3: -2.526397
- F: 0.178587
- E-BOTH: 0.425811

The archived RM1 run reported:

- T3: 7.365677
- F: -3.285769
- E-BOTH: -2.827610

The mismatch is consistent with the class-indexing error and should be resolved by the corrected rerun.

## Corrected checkpoint-relative definition

- T3: `max(logits[0:60]) - max(logits[60:80])`
- F: `max(logits[0:80]) - max(logits[80:100])`
- E-BOTH: `max(logits[0:80]) - max(logits[80:100])`

No scientific conclusion is taken from the invalid run.
