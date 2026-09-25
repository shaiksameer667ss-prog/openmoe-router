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

## HARM / routing causal gate

### Observed

The canonical `none` condition has large fixed-input routing drift, while
`head_masked` and `head_masked_frozen_old` have much smaller drift. The drift
is primarily associated with learned router-projection change and is
class-dependent and temporally structured.

### Causal interpretation

These observations do not establish router drift as an independent cause of
forgetting. The large difference between `none` and the head-masked
conditions is produced by a change in the training objective.

The earlier `router_frozen` decomposition also showed essentially unchanged
forgetting relative to `none`. Therefore, the existing evidence is
consistent with router drift being downstream of the training objective
rather than a load-bearing cause of forgetting. A post-fix causal test is
required before treating router stabilization as a mechanism target.

### HARM retrieval gate

Historical class-conditioned routing signatures are temporally stable, but
current inputs have weak retrieval of their corresponding historical
signatures. Therefore the proposed alpha-gated historical route-signature
HARM is not justified by the current evidence.

The feature-based similarity gate proposed for HARM was not tested by this
diagnostic, so that specific retrieval mechanism is not ruled out
independently.

## Decision

Do not implement HARM or another router stabilizer yet.

Run the causal test `head_masked_router_frozen`: freeze the router at the
Task-1 transition while retaining the existing head-logit masking objective.
Use seed 0, the canonical 200 steps/task protocol, and compare against the
archived `head_masked` and `none` results.

If `head_masked_router_frozen` retains the `head_masked` benefit, router drift
is not required for that benefit and the routing thread closes. If the benefit
disappears, router drift is a candidate causal intermediate and a
projection-level intervention becomes justified.

This diagnostic does not modify the canonical training protocol or archived
checkpoints.

This diagnostic does not modify the canonical training protocol or archived
checkpoints.
