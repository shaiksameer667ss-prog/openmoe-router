# Matched-Memory Decoder Recovery — Seed 0

## Question

Is the memory-bounded NCM recovery causally dependent on training-time replay?

## Conditions

| Training | Decoder buffer | Old-task NCM | Task-4 NCM |
|---|---|---:|---:|
| replay_715 | A | 0.092875 | 0.1885 |
| none (stability) | A | 0.099750 | 0.1155 |
| none (stability) | C | 0.092000 | 0.1130 |
| plain CE | A | 0.101250 | 0.1210 |
| plain CE | C | 0.093250 | 0.1175 |

## Findings

1. Old-task NCM recovery falls in a relatively narrow range across
   the five seed-0 conditions (0.092000–0.101250). The replay-trained
   checkpoint does not show a higher old-task NCM at boundary 4.

2. The buffer draw has a small effect on the plain-CE checkpoint:
   A vs C changes old-task NCM from 0.101250 to 0.093250.
   Buffer invariance is therefore not claimed.

3. Task-4 NCM is substantially higher for replay training:
   0.1885 versus 0.1155 for none-with-stability and
   0.1210/0.1175 for plain CE. This is an observed representation
   signal, but its mechanism is not established by this single seed.

4. The seed-0 Path-1 matched-memory comparison is:
   replay + A = 0.092875 old-task NCM versus plain CE + A =
   0.101250. Plain CE + fresh C = 0.093250. Thus the replay-trained
   checkpoint does not show a representation-side old-task advantage
   under this bounded NCM probe.

## Interpretation

The base-phase replay advantage should not be interpreted as evidence
of superior old-task representation preservation at this boundary.
The learned classifier and the feature geometry are separable: the
learned head collapses on old tasks in the no-replay conditions, while
post-hoc NCM recovers substantial old-task accuracy.

Replay does, however, produce substantially higher newest-task NCM
recoverability in seed 0. Whether this reflects a reproducible
representation effect rather than seed variation remains unresolved.

## Memory accounting

The decoder buffer contains 715 CIFAR-100 training examples and uses
8,797,360 bytes. The NCM state is 102,400 bytes, so the plain-CE
evaluation-time decoder footprint is 8,899,760 bytes.

The plain-CE training condition has no training-time stability or
replay state. Its evaluation uses the bounded decoder buffer and NCM
state.

The none-with-stability condition is not memory-matched to replay:
its training-time stability state is approximately 8.9 MB, while adding
the 715-example decoder buffer at evaluation creates an additional
8.8 MB footprint.

## Caveats

- Single seed; not replicated.
- A and C are deterministic decoder-buffer conditions, but their exact
  quantitative difference is not treated as a general buffer effect.
- The higher Task-4 replay NCM is an observed association. A causal
  claim about representation sharpening requires replication.
- These results apply to this model, CIFAR-100 class-incremental
  protocol, boundary 4, and 715-example decoder budget.

## Reproducibility

Seed: 0

Boundary: 4

Decoder capacity: 715 examples

C buffer seed: 12345

Plain-CE boundary-4 checkpoint:
`experiments/results/bounded_decoder/plain_ce_seed0_for_match_checkpoints/task_4.pt`

Seed-0 Path-1 manifest:
`experiments/results/bounded_decoder/path1_plain_ce_seed0_manifest.json`

