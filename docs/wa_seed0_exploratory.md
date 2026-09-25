# Seed-0 Weight Aligning Exploratory Diagnostic

This analysis is exploratory and post-hoc. It does not alter the preregistered
seed-0 branch decision and does not modify the archived checkpoints.

## Scope

Condition:
- `none`, seed 0
- final task-4 checkpoint
- 80 old classes, 20 newly introduced classes

## Weight-norm alignment

Mean classifier-row norms at the final boundary:

- old classes: 0.688271
- new classes: 0.798425
- WA scale gamma: 0.862036

After applying the WA scale to an in-memory copy of the new-class rows:

- aligned new-class mean norm: 0.688272

## Accuracy effect

Final-boundary old-task learned-head accuracy:

- Task 0: 0.0000 -> 0.0000 (+0.00 pp)
- Task 1: 0.0000 -> 0.0000 (+0.00 pp)
- Task 2: 0.0000 -> 0.0000 (+0.00 pp)
- Task 3: 0.0000 -> 0.0010 (+0.10 pp)

The saved checkpoint was not modified.

## Old-vs-new logit competition

Mean old-minus-new maximum-logit margin:

| Task | Original | WA |
|---|---:|---:|
| 0 | -5.5041 | -4.4611 |
| 1 | -5.5661 | -4.5206 |
| 2 | -5.4228 | -4.3729 |
| 3 | -5.2620 | -4.2167 |

New-class win rate remained approximately 99.75%-100.0% after WA.

## Interpretation

Weight-norm imbalance is present, and WA moves the old-versus-new logit
competition in the expected direction. However, the argmax changes almost
not at all and old-task learned-head accuracy remains essentially zero.

Therefore, simple classifier weight-norm imbalance is insufficient to explain
the observed learned-head collapse in this seed-0 `none` checkpoint.

This result is exploratory only. It does not change the preregistered
diagnostic conclusion:

**No dominant MoE-specific mechanism at this scale.**
