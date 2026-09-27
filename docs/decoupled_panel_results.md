# Decoupled Router Panel — Three-Seed Results

## Primary metric

Primary metric: old-task NCM at boundary 4 using the bounded 715-example
replay buffer.

Per-seed classification:

- Active if D2 - D1 > 0.010 AND D2 - D3 > 0.010
- Inert if |D2 - D1| < 0.005 AND |D2 - D3| < 0.005
- Otherwise ambiguous.

Final aggregation rule, preregistered before seed 1:

- Active requires the per-seed Active criterion in at least 2 of 3 seeds.
- Inert requires the per-seed Inert criterion in at least 2 of 3 seeds.
- Otherwise the final result is ambiguous.

## Results

| Seed | D1 old-task NCM | D2 old-task NCM | D3 old-task NCM | D2-D1 | D3-D2 | Classification |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.06325 | 0.05925 | 0.10150 | -0.00400 | +0.04225 | Ambiguous |
| 1 | 0.09475 | 0.09475 | 0.092125 | 0.00000 | -0.002625 | Inert |
| 2 | 0.05100 | 0.08275 | 0.08225 | +0.03175 | -0.00050 | Ambiguous |

### Three-seed outcome

- Active: 0/3 seeds
- Inert: 1/3 seeds
- Required for either classification: 2/3
- Final classification: AMBIGUOUS

The seed-0 D3-D2 difference of +0.04225 did not replicate at seed 2
(-0.00050) or seed 1 (-0.002625).

The seed-2 D2-D1 difference of +0.03175 also did not reproduce the
same directional pattern at seed 1 (0.00000).

No post-hoc reclassification is applied.

## Secondary observation: task-0 seed variance

Task-0 learned-head diagonal accuracy is recorded separately from the
primary router-effect test.

D1 task-0 diagonal:

- Seed 0: 0.326
- Seed 1: 0.3155
- Seed 2: 0.159

D2 task-0 diagonal:

- Seed 1: 0.301
- Seed 2: 0.3415

D3 task-0 diagonal:

- Seed 1: 0.301
- Seed 2: 0.3415

D2 and D3 are identical at task 0 because router_frozen is applied at
the continual-learning boundary rather than during task-0 training.

The D1 task-0 diagonal varies substantially across seeds (0.326, 0.3155,
0.159). This is recorded as a descriptive observation, not as part of
the primary hypothesis test.

## Interpretation

Under the frozen preregistered aggregation rule, the panel does not
establish either an active or inert router effect on old-task NCM.

The observed D3 > D2 effect at seed 0 is not replicated across the
three seeds. The primary metric also exhibits variation large enough
to cross the preregistered decision thresholds.

The appropriate result for this panel is therefore:

**AMBIGUOUS — no replicated router effect detected under the
preregistered thresholds and three-seed protocol.**

This result should not be relabeled as "router inert."
