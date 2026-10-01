# Coarse Superclass vs Benchmark Task-ID Interpretation

## Status

CLOSED — interpretation diagnostic completed with a freshly regenerated
D2/512 seed-0 checkpoint.

## Task construction audit

`openmoe/data/streams.py::split_class_ranges(num_classes=100,tasks=5)`
returns native contiguous CIFAR-100 fine-label ranges:

- Task 0 = fine labels 0–19
- Task 1 = fine labels 20–39
- Task 2 = fine labels 40–59
- Task 3 = fine labels 60–79
- Task 4 = fine labels 80–99

No class permutation occurs in task construction.

Therefore the benchmark task-ID target is an arbitrary partition of the native
CIFAR-100 fine-label ordering, rather than five semantically coherent visual
domains.

## Probe protocol

All linear probes use:

- StandardScaler
- LogisticRegression
- solver = LBFGS
- C = 1.0
- max_iter = 2000
- chance normalization = `(accuracy - chance) / (1 - chance)`
- final transformer representation = final root LayerNorm followed by mean
  pooling over tokens

The trained D2/512 model uses:

- hidden_dim = 512
- depth = 2
- patch_size = 4
- 4 heads
- FFN = 1024
- 8 experts
- Top-2
- continual router
- no decomposition
- replay capacity = 715
- replay matching = sample_matched
- 200 steps/task
- seed = 0

## x0 input result

Raw pixels -> PCA-256.

Reference:

| Target | Accuracy | Chance-normalized |
|---|---:|---:|
| CIFAR-100 coarse superclass, 20-way | 26.62% | 22.7579% |
| Benchmark task-ID, 5-way | 29.61% | 12.0125% |

Normalized gap:

**+10.7454 pp coarse superclass minus benchmark task-ID**

Strict-4:

| Target | Accuracy | Chance-normalized |
|---|---:|---:|
| Coarse superclass, 20-way | 28.5625% | 24.8026% |
| Benchmark task-ID, 4-way | 36.2375% | 14.9833% |

Strict-4 normalized gap:

**+9.8193 pp**

## Trained D2/512 result

Freshly regenerated D2/512 seed-0 checkpoint:

`experiments/results/coarse_probe_d2_512/checkpoints/task_4.pt`

Reference:

| Target | Accuracy | Chance-normalized |
|---|---:|---:|
| CIFAR-100 coarse superclass, 20-way | 35.18% | 31.7684% |
| Benchmark task-ID, 5-way | 35.24% | 19.0500% |

Normalized gap:

**+12.7184 pp coarse superclass minus benchmark task-ID**

Change relative to x0:

**+1.9730 pp**

Strict-4:

| Target | Accuracy | Chance-normalized |
|---|---:|---:|
| Coarse superclass, 20-way | 36.5625% | 33.2237% |
| Benchmark task-ID, 4-way | 42.0375% | 22.7167% |

Strict-4 normalized gap:

**+10.5070 pp**

## Interpretation

The semantic-vs-arbitrary distinction exists before the transformer and
survives the trained D2 representation.

The normalized coarse-superclass advantage is:

- x0: **+10.7454 pp**
- trained D2/512: **+12.7184 pp**
- strict-4 trained D2/512: **+10.5070 pp**

Therefore the earlier ~34% benchmark task-ID probe should not be interpreted
as a generic measurement of semantic task identity.

The supported conclusion is narrower:

> The benchmark's arbitrary native fine-label partition is less linearly
> recoverable than CIFAR-100's semantic superclass structure, and this
> distinction persists through the trained D2/512 representation.

This does NOT establish that the arbitrary benchmark partition is the sole
cause of the end-to-end CIL limitation. The trained representation still
contains substantial recoverable structure for both targets.

The class-prediction frontier, double dissociation, decoder measurements,
and other previously archived CIL findings are unaffected by this diagnostic.

## Provenance note

The original D2/512 seed-0 checkpoint was lost during the working-tree
recovery. The checkpoint used here is a fresh regeneration under the same
intended protocol, not binary recovery of the original checkpoint.

The fresh run reached:

- Task 0 = 31.30%
- Task 1 = 19.95%, 27.50%
- Task 2 = 5.55%, 9.65%, 37.40%
- Task 3 = 4.10%, 2.85%, 7.65%, 40.80%
- Task 4 = 1.65%, 0.90%, 2.35%, 3.95%, 45.10%

The final 45.10% T4 accuracy differs from the earlier archived lost D2/512
seed-0 run (40.95%), so this result is a fresh protocol replication.

## Closed decision

No new router is justified by this diagnostic.

The next unresolved question remains end-to-end historical decision realization:
how to convert information retained in the representation into correct
old-task predictions without paying an equivalent current-task plasticity cost.
