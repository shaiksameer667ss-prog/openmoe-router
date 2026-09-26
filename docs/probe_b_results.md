# Probe B Results: Cross-Boundary Newest-Task NCM

## Purpose

Probe B tests whether the replay-vs.-plain-CE decoder recovery gap is specific to the final boundary or instead reflects a generic newest-task effect.

The preregistered measurements are:

- **M1:** Task 3 NCM at boundary 3, when Task 3 is the newest task.
- **M2:** Task 4 NCM at boundary 4, when Task 4 is the newest task.
- **M3:** Task 3 NCM at boundary 4, when Task 3 is recent but no longer newest.

For each measurement, the reported quantity is:

`gap = replay - plain CE`

A positive gap means replay has higher NCM accuracy under the matched decoder evaluation; a negative gap means plain CE is higher.

## Preregistered hypotheses

The preregistered thresholds were:

- **small:** `gap < 0.020`
- **large:** `gap > 0.040`
- **ambiguous:** `0.020 <= gap <= 0.040`

The hypotheses were:

- **H1, generic newest-task:** `abs(gap_M1 - gap_M2) < 0.020` for every seed.
- **H2, boundary-4-specific:** `gap_M1 < 0.020` and `gap_M2 > 0.040` for every seed.
- **H3, persistent recent:** `gap_M3 > 0.040` for every seed.
- **H4, transient:** `gap_M3 < 0.020` for every seed.

H1 and H2 are mutually exclusive under the preregistered rules. H3 and H4 are also mutually exclusive.

## Results

### M1: Task 3 at boundary 3

| Seed | Replay | Plain CE | Gap |
|---|---:|---:|---:|
| 0 | 0.1605 | 0.2060 | -0.0455 |
| 1 | 0.1640 | 0.2000 | -0.0360 |
| 2 | 0.1900 | 0.2010 | -0.0110 |
| **Mean gap** | | | **-0.03083** |

Replay is below plain CE on the newest task at boundary 3 in all three seeds.

M1 used a dedicated deterministic 715-example boundary-3 buffer containing only Tasks 0-3 (80 classes), with no Task 4 examples.

### M2: Task 4 at boundary 4

| Seed | Replay | Plain CE | Gap |
|---|---:|---:|---:|
| 0 | 0.1885 | 0.1210 | +0.0675 |
| 1 | 0.1775 | 0.1310 | +0.0465 |
| 2 | 0.1965 | 0.1320 | +0.0645 |
| **Mean gap** | | | **+0.05950** |

Replay is above plain CE on the newest task at boundary 4 in all three seeds.

M2 uses the seed-specific Path-1 A-buffer and the same 715-example decoder capacity.

### M3: Task 3 at boundary 4

| Seed | Replay | Plain CE | Gap |
|---|---:|---:|---:|
| 0 | 0.1370 | 0.1300 | +0.0070 |
| 1 | 0.1360 | 0.1020 | +0.0340 |
| 2 | 0.1715 | 0.1090 | +0.0625 |
| **Mean gap** | | | **+0.03450** |

The seed-level M3 gaps span all three preregistered regions: small, ambiguous, and large.

## Preregistered outcome

| Hypothesis | Outcome |
|---|---|
| H1 generic newest-task | **Not supported** |
| H2 boundary-4-specific | **Supported** |
| H3 persistent recent | **Not supported** |
| H4 transient recent | **Not supported** |

### Why H1 is not supported

The boundary-3 and boundary-4 newest-task gaps differ substantially in every seed:

- Seed 0: `|-0.0455 - 0.0675| = 0.1130`
- Seed 1: `|-0.0360 - 0.0465| = 0.0825`
- Seed 2: `|-0.0110 - 0.0645| = 0.0755`

All exceed the preregistered 0.020 threshold.

### Why H2 is supported

For every seed, M1 is below the small-gap threshold and M2 is above the large-gap threshold:

- M1: `[-0.0455, -0.0360, -0.0110]`
- M2: `[+0.0675, +0.0465, +0.0645]`

Thus the preregistered H2 criterion is met in all three seeds.

### Why H3 and H4 are not supported

M3 does not satisfy either all-seed threshold:

- M3: `[+0.0070, +0.0340, +0.0625]`

Seed 0 is below 0.020, seed 1 is ambiguous, and seed 2 is above 0.040. Therefore neither the preregistered persistent-recent criterion (H3) nor the transient criterion (H4) is satisfied.

## Interpretation

The most informative feature of Probe B is the **sign flip across boundaries**.

At boundary 3, replay is worse than plain CE on the newest task under the NCM decoder. At boundary 4, replay is better than plain CE on the newest task under the same evaluation framework.

This is evidence against a generic claim that replay simply improves the newest task. Instead, under this experiment, the newest-task effect depends strongly on the boundary at which it is measured.

One possible descriptive reading is that the transition from 80 seen classes at boundary 3 to 100 seen classes at boundary 4 changes the decoder's discrimination problem. In that reading, replay may be more robust to the larger competing class set at boundary 4. Another possibility is that the boundary-4 result is specific to the Task-4 class subset or ordering.

Probe B does **not** distinguish those explanations.

## Scope and limitations

This experiment does not establish a mechanism for the boundary-specific result.

In particular, it does not test:

- a causal mechanism for why replay changes the feature geometry;
- generalization beyond boundaries 3 and 4;
- training-memory matching between replay and plain CE;
- whether the effect is specific to Task 4;
- whether class-count growth or task-specific properties explain the sign flip.

The M2 values were already established in the Path-1 results and were not re-run for Probe B. M1 uses a dedicated boundary-3 buffer with only the first 80 classes; M2 and M3 use the seed-specific Path-1 A-buffer at the final boundary.

The decoder evaluation is bounded to 715 retained examples. The decoder state is therefore evaluated under the same retained-example capacity used by the preceding matched-memory analysis; this does not imply matched training-time memory usage between replay and plain CE.

## Reproducibility

The raw Probe B artifacts and aggregate summary are archived in:

`experiments/results/bounded_decoder/`

The aggregate machine-readable result is:

`experiments/results/bounded_decoder/probe_b_results_summary.json`

The dedicated boundary-3 decoder buffer is:

`experiments/results/bounded_decoder/probe_b_boundary3_buffer.pt`

The Probe B archive was committed in Git as:

`9ead157 Archive Probe B results`

This report is a documentation-only follow-up to that archived result set.
