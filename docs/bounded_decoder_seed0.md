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

| Decoder | Old-task mean | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 (new) |
|---|---:|---:|---:|---:|---:|---:|
| Learned head | 0.046375 | — | — | — | — | 0.391 |
| Frozen NCM (introduction-time prototypes) | 0.012375 | 0.0025 | 0.0020 | 0.0020 | 0.0430 | 0.3710 |
| Replay-refit NCM | 0.092875 | 0.0620 | 0.0800 | 0.0925 | 0.1370 | 0.1885 |
| Replay-fitted ridge | 0.091125 | 0.0680 | 0.0890 | 0.0955 | 0.1120 | 0.0650 |

The bounded refit result is **2.00x** the learned head's old-task mean. Frozen introduction-time NCM reaches only **0.0124** on old tasks.

The frozen-to-refit old-task gap is **0.0805**. By task age, the refit-minus-frozen gaps are 0.0595, 0.0780, 0.0905, and 0.0940 for Tasks 0–3. This is direct evidence of substantial representation drift: the class structure remains recoverable in the current feature space, while introduction-time prototype coordinates no longer align with it. The monotone age pattern is descriptive at seed 0 and is not by itself a proof of a pure translation model.

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

The frozen-NCM reference means come from `DriftState` introduction-time references and are a diagnostic comparison, not a memory-matched deployed decoder state.

## What this establishes

- A memory-bounded NCM decoder, fit only on the retained 715-example replay buffer, recovers 2.00x the learned head's old-task accuracy at boundary 4.
- The recovery is not an artifact of full-training-set access. The earlier approximately 0.189 full-data probe remains diagnostic only and is not a memory-matched candidate result.
- Introduction-time frozen prototypes fail strongly on old tasks, while replay-refit prototypes recover substantially higher accuracy using the current feature coordinates.
- The frozen-to-refit old-task gap of 0.0805 provides a direct measurement of substantial feature-space drift under this protocol.
- Replay plus a small current-feature decoder fits within the same audited method-state target.

## What this does not establish

- Multi-seed replication. Seed 0 only.
- Generalization beyond this protocol.
- That the representation change is a pure translation, rather than a more general deformation of feature coordinates.
- That refit NCM alone is responsible for the recovery independent of training-time replay.
- That the learned head's failure is uniquely decoder-side. The prior oracle probe showed substantial recoverable decoder degradation, but the bounded comparison still does not by itself isolate all representation and optimization effects.

## Caveats

- The bounded probe run used capacity 715, not 722. The learned-head value quoted here is from the same 715 run.
- NCM outperforms the ridge probe on the newest task by 12.35 percentage points, while ridge is competitive on old tasks. This motivates NCM as the primary decoder candidate for subsequent testing without treating that as a multi-seed method ranking.
- The existing full-data probe remains an oracle-like diagnostic and is not memory-matched.
- The frozen NCM uses introduction-time `DriftState` reference means; its reference images/state are diagnostic infrastructure and are not counted as part of the deployed memory-matched decoder budget.

## Next ablation

The next planned test is the same memory-bounded decoder refit on a `none`/stability checkpoint with a matched-size stored replay buffer, but without training-time replay. This distinguishes recovery attributable to current-feature decoder refitting from recovery attributable to replay having preserved the representation during continual training.
