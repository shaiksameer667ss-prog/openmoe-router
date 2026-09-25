# Routing-Consistent Replay (RCR) — Preregistered Protocol

Status: FROZEN
Frozen on: 2026-09-26
Supersedes: none

## 1. Motivation

Three independent findings from the base and diagnostic phases:

1. **Replay beats the stability stack at matched memory.** Across three seeds, byte-matched replay (724 examples, 8,908,096 B) achieves 0.112 accuracy / 0.267 forgetting, vs. the routing-KL stability configuration at 0.081 / 0.366 (8,911,776 B).
2. **The routing reference is over-resolved by ~160×.** `task_routing_reference` occupies 1,048,576 B to store per-token routing distributions. The routing diagnostic shows per-class means are highly stable across boundaries: selection-probability cosine 0.978–0.996, top-2 frequency cosine 0.930–0.993.
3. **Router drift is real and class-structured under `none`.** Fixed-input gate KL 0.766, top-2 overlap 0.358, with layer-1 KL reaching 0.84–1.10 at boundaries 2–4.

RCR tests whether compressing the routing reference to per-class means, and applying a routing-consistency term only on replayed classes, improves replay at matched byte budget.

## 2. Method

### 2.1 Routing memory

Per seen class `c`, per MoE layer `l`:

```text
mu_c[l] = mean routing distribution over samples of class c
          at the boundary where c was introduced
```

Shape: `[num_classes, num_layers, num_experts]`.

Cost: 100 classes × 2 layers × 8 experts × 4 bytes = 6,400 B.

This replaces `task_routing_reference` (1,048,576 B).

### 2.2 Training objective

Baseline replay loss on the mixed batch:

```text
L = CE(current_batch) + CE(replay_batch)
```

RCR adds, on the replay batch only:

```text
L_rc = beta * mean_{c in replayed classes} KL(p_hat_c || mu_c)
```

where `p_hat_c` is the current model's mean routing distribution for replayed examples of class `c`.

The current-model routing computation remains attached to the autograd graph so that `L_rc` can update trainable router parameters. Historical `mu_c` references are detached/frozen constants.

Applied only to replayed classes. New-task classes are unconstrained.

### 2.3 Hyperparameter

`beta ∈ {0.1, 1.0, 10.0}`, grid-searched on seed 0. Primary result at `beta = 1.0`.

## 3. Memory accounting

Target stability method-state memory: 8,911,776 B.

| Component | Stability stack | RCR |
| --- | ---: | ---: |
| Replay images | — (uses task_images) | 8,883,488 B (722 images) |
| Routing reference | 1,048,576 B (per-token) | 6,400 B (per-class means) |
| Fisher + param ref | 4,700,960 B | 0 B |
| Router buffers | 16,512 B | 16,512 B |
| **Total** | **8,911,776 B** | **8,906,400 B** |

Under the stated byte accounting, 722 is the largest whole-example replay capacity that remains strictly under the stability target:

```text
722 * 12,304 = 8,883,488 B
```

Correct complete-state calculation:

```text
722 * 12,304 = 8,889,? 
```

The complete RCR state is:

```text
722 * 12,304 + 6,400 + 16,512
= 8,906,400 B
```

Therefore:

- `RCR_722`: 722 replay examples + routing memory. Total 8,906,400 B, strictly under the stability budget by 5,376 B.
- `RCR_723`: 723 replay examples + routing memory. Total 8,918,704 B, 6,928 B above the stability budget (+0.078%).
- `RCR_724`: 724 replay examples + routing memory. Total 8,931,008 B, 19,232 B above the stability budget (+0.216%).

For a strict memory-matched replay control, `replay_722` is therefore the appropriate comparator for the primary RCR condition.

## 4. Preregistered comparators

All seed 0, 200 steps/task, CIFAR-100, continual router, post-`1aa8d56` code.

| ID | Condition | Purpose |
| --- | --- | --- |
| A | `replay_722` | Strict memory-matched replay control |
| B | `replay_724` | Archived comparator (have) |
| C | `RCR_722, beta=1.0` | Primary RCR result |
| D | `RCR_723, beta=1.0` | Secondary, slight budget overrun |
| E | `replay_722 + routing_consistency, beta=1.0` | Isolates routing term from memory compression |

Condition E is the clean two-way test: same replay capacity as A, plus routing-consistency term. If E > A, the routing term is doing work. If E ≈ A, the routing term is inert and RCR reduces to memory compression.

## 5. Preregistered thresholds

Evaluated on the old-task mean at boundary 4 (tasks 0–3).

**Positive result:** `RCR_722` old-task mean ≥ `replay_722` old-task mean + 0.010 absolute.

**Weak positive:** `+0.005 ≤ Δ < +0.010`.

**Null:** `|Δ| < 0.005`.

**Negative:** `Δ < −0.005`.

**Falsification of the routing-consistency component:** if `E ≤ A` (routing term does not help at matched replay capacity), the routing-consistency mechanism is rejected, regardless of C's performance.

**Confirmation requirement:** any positive result at seed 0 triggers seeds 1 and 2 for conditions A, C, and E only. No other conditions are replicated.

## 6. What would falsify RCR

- `RCR_722` old-task mean within noise of `replay_722` → routing-consistency adds nothing, memory compression does not translate to retention.
- `RCR_722` forgetting higher than `replay_722` → routing constraint harms plasticity, reject.
- Condition E fails while C succeeds → improvement comes from byte reallocation, not the routing term. RCR is a memory result, not a mechanism result. Report as such.
- β sweep shows monotone degradation with β → the routing constraint is mis-specified and should not be tuned further.

## 7. What RCR is not

- Not a response to a proven causal router failure. The router causal thread is unresolved, and RCR does not depend on it.
- Not a class-signature retrieval method. RCR uses per-class routing signatures only on replayed batches, with no query-time retrieval.
- Not a replacement for replay. It is replay plus a compressed routing constraint.
- Not coupled to head masking. `head_masked` is excluded from all RCR runs because the condition has a test-time distribution shift (task-4 diagonal collapse).

## 8. Claim scope

These results attribute the effect of a routing-consistency term and a routing-memory compression scheme within the specified 2-layer, 8-expert, continual CIFAR-100 protocol. They do not establish:

- that routing drift is causally responsible for forgetting in general,
- that RCR generalizes to larger models, more tasks, or other datasets,
- that the routing-consistency term is the mechanism by which any observed improvement occurs.

If RCR succeeds and E fails, the improvement is a memory-reallocation result, not a routing-mechanism result, and must be reported as such.

## 9. Conditions of execution

RCR runs only after:

1. The RCR implementation passes a unit test verifying the routing-consistency term computes correctly on a toy class-incremental stream, including a gradient check that the current routing computation remains differentiable while the historical `mu_c` reference is detached.
2. `replay_722` (condition A) is produced and archived under identical code, seed, and step count.
3. This document is committed to the repository.

Any deviation from the comparators or thresholds above must be recorded as a new document, `rcr_preregistration_v2.md`, with a dated note describing the deviation and what result was already known at revision time.

---

## Open causal-thread note

Do not use the `head_masked_router_frozen` result to conclude anything about router causality. The B→C comparison is non-diagnostic because both conditions share the head-masked task-4 diagonal collapse. The head-masked family is a diagnostic condition with a known test-time distribution shift, not a method baseline.

A separate causal closure test may freeze `router.proj` only while allowing router bias and memory updates to continue. That test is outside the RCR protocol and must be preregistered separately.
