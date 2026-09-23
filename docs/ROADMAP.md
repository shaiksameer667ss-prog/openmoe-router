# OpenMoE-Router Roadmap

## Phase 1 — Correctness baseline

- [x] Router contract
- [x] Top-1 / Top-2
- [x] Non-gradient balancing bias
- [x] Routing memory
- [x] Reference sparse dispatch
- [x] Tiny Transformer backbone
- [x] Synthetic continual stream
- [ ] Split-CIFAR-100 full protocol
- [ ] W&B telemetry
- [ ] checkpoint/resume

## Phase 2 — Continual-learning research

- [ ] old-task routing reference buffer
- [ ] routing-KL stability loss
- [ ] Fisher estimation and dense EWC
- [ ] specialization diagnostics
- [ ] router-vs-dense forgetting decomposition
- [ ] task-order robustness

## Phase 3 — Systems

- [ ] capacity-aware dispatch
- [ ] token dropping telemetry
- [ ] block-sparse reference implementation
- [ ] Triton dispatch kernel
- [ ] numerical equivalence tests
- [ ] PyTorch profiler traces
- [ ] GPU benchmark harness

## Phase 4 — Research stretch

- [ ] shared experts
- [ ] Expert Choice
- [ ] counterfactual route regret
- [ ] lightweight utility estimator
- [ ] regret-triggered router updates

## Publication-quality artifacts

- [ ] architecture diagram
- [ ] benchmark table generated from raw JSON
- [ ] ablation plots
- [ ] engineering write-up
- [ ] reproducible Docker image
