# Path-1 Results — Three Seeds

## Preregistered claims

### Claim 1

At boundary 4, old-task mean NCM accuracy does not distinguish replay
from plain CE training under the preregistered practical-difference
threshold:

`|mean(plain CE - replay)| < 0.005`

Observed:

`mean(plain CE - replay) = -0.002250`

`|mean difference| = 0.002250`

**Status: CONFIRMED**

### Claim 2

At boundary 4, Task-4 NCM accuracy is higher for replay training than
plain CE training by more than 0.030 in every seed.

Observed replay minus plain-CE gaps:

- Seed 0: `+0.0675`
- Seed 1: `+0.0465`
- Seed 2: `+0.0645`

Minimum gap:

`0.0465`

**Status: CONFIRMED**

## Three-seed results

| Seed | Replay old-task NCM | Plain CE + A old-task NCM | Plain CE + C old-task NCM |
|---:|---:|---:|---:|
| 0 | 0.092875 | 0.101250 | 0.093250 |
| 1 | 0.091125 | 0.081750 | 0.084125 |
| 2 | 0.097250 | 0.091500 | 0.087250 |
| Mean | 0.093750 | 0.091500 | 0.088208 |

| Seed | Replay Task-4 NCM | Plain CE + A Task-4 NCM | Plain CE + C Task-4 NCM |
|---:|---:|---:|---:|
| 0 | 0.1885 | 0.1210 | 0.1175 |
| 1 | 0.1775 | 0.1310 | 0.1180 |
| 2 | 0.1965 | 0.1320 | 0.1295 |
| Mean | 0.1875 | 0.1280 | 0.1217 |

## Findings

1. Old-task NCM recoverability does not show a practically meaningful
   advantage for replay over plain CE under the preregistered
   three-seed threshold. The seed-level replay-minus-plain differences
   are `-0.008375`, `+0.009375`, and `+0.005750`, equivalently
   plain-CE-minus-replay differences of `+0.008375`, `-0.009375`,
   and `-0.005750`.

2. Task-4 NCM recoverability is consistently higher for replay training.
   The three-seed mean is `0.1875` for replay versus `0.1280` for
   plain CE using the corresponding A buffer, for a mean difference
   of `+0.0595`. Every seed exceeds the preregistered `+0.030`
   threshold.

3. The plain-CE A/C comparison does not support buffer invariance as a
   universal claim. Old-task NCM changes from A to C by `-0.008000`,
   `+0.002375`, and `-0.004250` across seeds 0–2. Task-4 NCM changes
   by `-0.003500`, `-0.013000`, and `-0.002500`, respectively.

4. The learned-head Task-4 diagonal does not explain the NCM Task-4
   result by itself. For seed 2, for example, the learned-head
   diagonal is `0.3895` for replay versus `0.3960` for plain CE,
   while Task-4 NCM is `0.1965` versus `0.1320`.

## Interpretation

The results support a boundary-4 distinction between learned-head
retention and recoverable feature geometry.

The preregistered old-task result does not show a practically
meaningful replay advantage under the bounded NCM readout. The large
difference between replay and plain CE in the learned classifier's
old-task performance is therefore not reproduced as a corresponding
difference in NCM-recoverable old-task information.

In contrast, replay consistently increases Task-4 NCM recoverability.
This is an observed representation/readout signal: the current-task
feature geometry is more recoverable by NCM after replay training.

"Representation sharpening" is an interpretation of this geometry,
not a directly measured mechanism. The present experiments do not
establish why replay produces the Task-4 difference.

## Memory scope

The decoder evaluation uses a 715-example buffer containing
8,797,360 bytes plus 102,400 bytes of NCM state.

The Path-1 replay/plain-CE decoder comparison is matched on the
evaluation decoder budget, but training-time memory is not matched:
replay uses its 715-example buffer during training, whereas plain CE
uses no training-time replay or stability state.

The none-with-stability diagnostic condition is also not a
training-memory-matched control for replay because its stability state
is approximately 8.9 MB.

## Claim scope

- Boundary 4 only.
- CIFAR-100, 5 class-incremental tasks.
- 715-example bounded decoder.
- Post-hoc NCM refit on held-out CIFAR-100 test evaluation.
- Seeds 0, 1, and 2.
- Single model architecture and training protocol.

## What is not claimed

- No general causal mechanism for the Task-4 representation effect.
- No claim that replay harms representation quality.
- No universal buffer-invariance claim.
- No conclusions about boundaries 1, 2, or 3.
- No generalization beyond this architecture, dataset, protocol, and
  decoder budget.

