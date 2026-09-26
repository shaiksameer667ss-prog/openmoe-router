# Path-1 Multi-Seed Claim (Preregistered)

## Claim under test

At boundary 4, under a memory-bounded NCM decoder refit on 715 images:

1. Old-task mean NCM accuracy does not distinguish replay training
   from plain CE training under the preregistered practical threshold.
2. Task-4 NCM accuracy is higher for replay training than plain CE
   training by a practically meaningful margin.

## Thresholds

Claim 1 is confirmed if:

`|mean(plain CE - replay)| < 0.005`

over seeds 0–2.

This is a preregistered practical-difference threshold, not a
statistical equivalence test.

Claim 2 is confirmed if:

`min over seeds (replay - plain CE) > 0.030`

Claim 2 is weakened rather than confirmed if:

`mean over seeds (replay - plain CE) > 0.030`

but fewer than 3 of 3 seeds show a gap greater than 0.030.

## Conditions

The comparison is made at boundary 4 using:

- replay_715 + its own 715-example decoder buffer (A)
- plain CE + the corresponding replay buffer (A)
- plain CE + a deterministic fresh 715-example decoder buffer (C)

The decoder is post-hoc NCM refitting only.

## What is not claimed

- No memory-matched comparison against the stability stack at
  training time.
- No general statement about representation preservation beyond
  this model, protocol, boundary, and decoder budget.
- No claim about boundaries 1, 2, or 3.
- No claim that replay is harmful when the seed-level Task-4 effect
  is absent.
- No causal mechanism claim from the seed-0 or seed-1 observations
  alone.

## Replication protocol

Seeds 0, 1, and 2 use the same training configuration, 200 steps per
task, CIFAR-100 5-task class-incremental stream, boundary 4, and
715-example bounded decoder capacity.

Seed 0 and seed 1 results are treated as already observed before this
preregistration. Seed 2 is the remaining preregistered replication.

## Decision rule

After seed 2 is complete, compute the three-seed quantities directly
from the archived per-seed results:

- mean `(plain CE old-task NCM - replay old-task NCM)`
- per-seed `(replay Task-4 NCM - plain CE Task-4 NCM)`
- minimum Task-4 gap
- mean Task-4 gap

Apply the thresholds above without changing them after observing seed 2.
