# OpenMoE-Router — Phase 0 Revision 3

## Research boundary

The project should not claim that “routing drift in continual MoE is unsolved.” 2026 work now directly studies routing stability in expandable MoE and router/expert co-drift in fixed-capacity MoE-LoRA settings. The defensible gap is narrower:

> **A fixed expert pool with frozen expert weights, task-agnostic continual router adaptation, explicit routing-memory constraints, and systems-level routing evaluation remains a distinct experimental setting.**

The project tests whether preserving the routing function itself can improve backward transfer without simply moving forgetting into shared dense parameters.

## Core model

For token representation h and expert e:

`s_e(h) = z_e(h) + lambda m_e(h) + b_e`

where z is learned router affinity, m is non-parametric expert-memory affinity, and b is a non-gradient load-balancing bias used only for top-k selection.

The actual mixture gate is computed from the **unbiased** z values on the selected experts.

## Stability objective

`L = L_task + alpha L_load + beta L_entropy + gamma_1 L_route-KL + gamma_2 L_dense-EWC + delta L_z + eta L_spec`

The mandatory decomposition separates routing drift from dense/shared-parameter drift.

## New stretch contribution: Counterfactual Route Regret

Periodically select a small subset of tokens and evaluate a small candidate set of frozen experts that were not necessarily executed by the main sparse route. Define:

`R(x) = u_best(x) - u_chosen(x)`

where u is task utility measured using frozen expert outputs. This is a hypothesis and diagnostic, not a claimed solved method.

## Experimental tiers

### Tier A — must run

- Dense Transformer baseline
- Frozen single expert baseline
- Top-1 MoE
- Top-2 MoE
- Continual router
- Continual + bias
- Continual + route-KL
- Continual + dense EWC
- Continual + both stability terms

### Tier B — high signal

- z-loss on/off
- specialization loss on/off
- shared-expert architecture
- Expert Choice
- capacity factor sweep
- reference dispatch vs block-sparse dispatch

### Tier C — research stretch

- counterfactual route regret
- learned utility estimator
- route-regret-triggered router update scheduling
- optional progressive expert growth as an out-of-scope comparison

## Primary metrics

- final average accuracy
- average forgetting / backward transfer
- forward transfer
- routing-consistency KL
- route regret
- routing entropy
- expert-load Gini
- dead experts
- token drop rate
- router bias magnitude/drift
- cross-expert activation cosine similarity
- peak VRAM during dispatch
- dispatch latency
- end-to-end tokens/s

## Falsification criteria

The project is successful as a research artifact even if the central method does not win, provided measurements establish at least one measurable mechanism or trade-off described above.
