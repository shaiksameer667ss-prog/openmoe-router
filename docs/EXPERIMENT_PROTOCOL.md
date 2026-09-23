# Experiment Protocol

This document is the reproducibility contract for OpenMoE-Router.

## Rule 1 — Baseline first

Every new mechanism is compared against the simplest matching baseline. Do not compare multiple simultaneous upgrades and then attribute the entire delta to one component.

## Rule 2 — Fixed task order

The main protocol uses a deterministic class-incremental stream. The first public result should use the same class order, train budget, optimizer family, and seed set across all router variants.

## Rule 3 — Compute matching

When comparing routing methods, report the number of experts evaluated per token and the same token/sample budget. Kernel comparisons report both microbenchmark latency and end-to-end training throughput.

## Rule 4 — Preserve raw measurements

Each run writes its configuration, seed, accuracy matrix, timing, and telemetry to `experiments/results/`. Raw outputs are immutable research records; plots are derived artifacts.

## Rule 5 — Falsify the hypothesis

A negative result is useful. The project should report regimes where routing stability fails, where dense shared parameters dominate forgetting, and where balancing improves utilization without improving downstream accuracy.

## Required primary comparisons

1. Dense Transformer vs. Top-1 MoE.
2. Top-1 vs. Top-2.
3. Continual router vs. standard Top-2 with matched capacity.
4. Continual + bias vs. auxiliary balancing loss.
5. Route-KL vs. dense EWC vs. both.
6. Reference dispatch vs. Triton/block-sparse dispatch.
