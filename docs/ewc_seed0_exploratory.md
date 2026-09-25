# Seed-0 Exploratory EWC Sweep

This is an exploratory post-hoc analysis of dense parameter-space EWC for the
CIFAR-100 continual-learning setup. It does not change the canonical
configuration, preregistered branch criteria, or base-phase conclusions.

## Sweep

All runs use seed 0, 5 tasks of 20 classes, 200 steps per task, decomposition
`none`, and the existing continual router configuration. Probe and drift
diagnostics were disabled for this sweep.

| EWC weight | Final accuracy `[T0,T1,T2,T3,T4]` | Mean dense EWC | Weighted EWC |
|---:|---|---:|---:|
| 1 | `[0.0, 0.0, 0.0, 0.0, 0.3995]` | 0.000327622 | 0.000327622 |
| 10 | `[0.0, 0.0, 0.0, 0.0, 0.3995]` | 0.000319075 | 0.003190748 |
| 100 | `[0.0, 0.0, 0.0, 0.0005, 0.4155]` | 0.000258796 | 0.025879630 |
| 1000 | `[0.0, 0.0, 0.0, 0.003, 0.4355]` | 0.000109033 | 0.109033342 |
| 10000 | `[0.0, 0.0, 0.0, 0.0055, 0.388]` | 0.000027358 | 0.273581247 |

Mean routing-KL decreased from `0.153530108` at EWC 1 to `0.110740145`
at EWC 10000. Mean task loss decreased from `2.440142137` to
`2.366240374`.

## Interpretation

Low EWC weights (1 and 10) are effectively inert relative to the dominant
training losses. Higher weights materially constrain the dense EWC objective,
but they do not recover meaningful old-task retention in this seed-0 run.

Final old-task accuracy remains near zero throughout the sweep. The largest
observed old-task recovery is task 3 at EWC weight 10000, where final accuracy
is only `0.0055` (0.55%).

Current-task accuracy improves from `0.3995` at weights 1-10 to `0.4355` at
weight 1000, then falls to `0.388` at weight 10000.

Therefore, for this seed-0 exploratory sweep, increasing the current dense EWC
coefficient does not provide evidence of an effective retention regime. The
sweep is stopped at 10000 rather than extending to arbitrarily larger
coefficients.

## Status

- Exploratory only.
- Canonical `configs/cifar100_milestone6.yaml` is unchanged.
- Preregistered branch criteria are unchanged.
- No branch-selection decision is made from this sweep.
