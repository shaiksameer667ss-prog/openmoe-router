# Seed-0 Routing Diagnostic

## Scope

CIFAR-100, 5 tasks, 20 classes/task, seed 0, `none`, `head_masked`, and
`head_masked_frozen_old` conditions. Existing boundary checkpoints only.
The diagnostic is read-only.

## Main findings

### 1. Routing drift

For the canonical `none` condition, fixed-input routing drift is large:

- mean gate-preference KL: 0.7656
- mean selection KL: 0.6518
- mean top-2 overlap: 0.3576

The largest drift occurs in layer 1, where fixed-input gate KL reaches
0.839–1.100 and top-2 overlap is approximately 0.23–0.25 at boundaries 2–4.

By contrast:

- `head_masked`: gate KL 0.0459, selection KL 0.0399, top-2 overlap 0.8095
- `head_masked_frozen_old`: gate KL 0.0461, selection KL 0.0401, top-2 overlap 0.8092

The two head-masked conditions are nearly identical.

### 2. Four-cell decomposition

All canonical checkpoints freeze expert parameters after task-0 warmup.
Consequently, the expert-side counterfactual is identically zero in these
checkpoint pairs:

- expert main effect = 0
- interaction = 0
- routing share = 1.0 by construction

Therefore the four-cell result does not independently establish router-vs-
expert dominance for this protocol.

### 3. Router-component isolation

For `none`, changes in unbiased router logits are much larger than changes
from memory affinity or routing bias across boundaries 2–4.

The component-swap test showed:

- projection-only KL: approximately 0.42–0.99
- memory-only KL: approximately 0.0007–0.0023
- bias-only KL: approximately 0.0030–0.0062

Thus the observed fixed-input routing drift is primarily associated with the
learned router projection, not the continual prototype-memory or bias terms.

### 4. Class dependence

Projection-drift magnitude varies by class and is moderately consistent
across boundaries.

For example, cross-boundary class-rank Spearman correlations were:

- Layer 0: 0.5878, 0.7567, 0.7981
- Layer 1: 0.5786, 0.7023, 0.6809

This indicates structured class-dependent drift, but does not by itself
establish a recoverable historical class signature.

### 5. Historical route-signature stability

Historical class-conditioned routing signatures are highly stable across
adjacent checkpoints:

- selection-probability cosine: 0.9775–0.9963
- top-2 frequency cosine: 0.9299–0.9926

However, current inputs do not reliably retrieve their corresponding
historical class signatures.

Full 8-way selection-distribution retrieval:

- Boundary 2: 12.42% (L0), 10.62% (L1)
- Boundary 3: 8.48% (L0), 8.18% (L1)
- Boundary 4: 6.33% (L0), 6.05% (L1)

Top-2 route-frequency retrieval:

- Boundary 2: 11.00% (L0), 8.60% (L1)
- Boundary 3: 8.17% (L0), 7.13% (L1)
- Boundary 4: 6.94% (L0), 5.15% (L1)

The true-vs-best-other similarity margin is negative throughout.

## HARM gate

### Passed

A router-specific intervention is scientifically motivated for the canonical
`none` condition because:

1. fixed-input routing drift is large;
2. the drift is primarily associated with learned router-projection change;
3. the drift is class-dependent and temporally structured.

### Not passed

The proposed class-conditioned HARM mechanism based on nearest historical
route-signature retrieval is not yet justified because query-side retrieval
is weak despite high temporal stability of the stored signatures.

## Decision

Do not implement the proposed alpha-gated historical class-signature HARM
yet.

The next router investigation should target stabilization of learned
router-projection drift rather than adding the existing prototype-memory
mechanism as HARM.

This diagnostic does not modify the canonical training protocol or archived
checkpoints.
