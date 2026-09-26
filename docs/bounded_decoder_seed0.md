# Bounded Decoder Result — Seed 0

## Status

Preregistered diagnostic. Not yet replicated. Seed 0 only.

## Conditions

- Training checkpoint: replay_722 family, capacity 715 in this run
- Decoder fit data: replay buffer only (memory-bounded)
- Evaluation: held-out CIFAR-100 test split
- Boundary: 4
- Classes: 100/100 represented

## Result

| Decoder | Old-task mean | Ratio vs learned head |
|---|---:|---:|
| Learned head | 0.046375 | 1.00x |
| Replay-fitted NCM | 0.092875 | 2.00x |
| Replay-fitted ridge | 0.091125 | 1.96x |

## Memory accounting

| Component | NCM | Ridge |
|---|---:|---:|
| Replay buffer | 8,797,360 B | 8,797,360 B |
| Decoder state | 102,400 B | 103,600 B |
| Total method state | 8,899,760 B | 8,900,960 B |
| Target | 8,911,776 B | 8,911,776 B |
| Headroom | 12,016 B | 10,816 B |

Both decoder configurations fit under the audited target.

The decoder state is 102,400 B for NCM and 103,600 B for ridge, corresponding to approximately 1.15% and 1.16% of the 8,911,776 B target, respectively.

## What this establishes

- A 102 KB class-mean decoder, fit only on the retained replay buffer, recovers 2.00x the learned head's old-task accuracy at boundary 4.
- The recovery is not an artifact of full-training-set access. The earlier approximately 0.189 oracle probe remains diagnostic only.
- Replay for the trunk plus a small prototype/linear decoder fits within the same audited method-state target.

## What this does not establish

- Multi-seed replication. Seed 0 only.
- Generalization beyond this protocol.
- That the learned head's failure is uniquely decoder-side. Criterion 2 showed substantial probe drift under the full-data/oracle probe. The bounded refit measured here does not yet separate representation drift from decoder misalignment; the frozen-vs-refit comparison is the next measurement.

## Caveats

- The bounded probe run used capacity 715, not 722. The learned-head baseline quoted here (0.046375) is from the same 715 run. Comparisons to the 722 checkpoint require matching on the same checkpoint.
- NCM outperforms the ridge probe on task 4 by 12.35 percentage points, while ridge is competitive on old tasks 0–2. This motivates retaining NCM as the primary decoder candidate for the next comparison, without treating that comparison as a multi-seed method ranking.
- The existing full-data probe remains an oracle-like diagnostic and is not memory-matched.
