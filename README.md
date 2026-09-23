# OpenMoE-Router

**Continual-learning Mixture-of-Experts framework with a systems-first routing research agenda.**

OpenMoE-Router is a from-scratch research codebase for studying whether a **fixed pool of experts** can preserve prior task competence while a **task-agnostic router** adapts to new tasks. The project separates router drift from dense/shared-parameter drift and treats the MoE dispatch path as a first-class systems benchmark.

## Research question

Can frozen experts + adaptive routing improve continual-learning backward transfer without merely moving forgetting into the shared Transformer parameters?

## Repository map

```text
openmoe-router/
├── openmoe/                  # Library code
│   ├── routers/              # Top-1 / Top-2 / continual routing
│   ├── models/               # Transformer + sparse MoE
│   ├── kernels/              # Reference + future Triton backend
│   ├── continual/            # Metrics
│   ├── losses/               # Stability / specialization losses
│   ├── data/                 # Synthetic + Split-CIFAR-100 stream
│   └── training/             # Training/evaluation utilities
├── experiments/              # Raw results and checkpoints
├── benchmarks/               # Routing/dispatch microbenchmarks
├── configs/                  # Reproducible experiment configs
├── docs/                     # Protocol, roadmap, architecture
├── tests/                    # Unit + smoke tests
├── scripts/                  # CLI experiment entry points
└── .github/workflows/        # Continuous integration
```

## Current implementation boundary

The included code is the **Phase 1 correctness scaffold**. It intentionally does not pretend the GPU kernel or continual-learning results already exist.

Implemented:

- Top-1 and Top-2 token-choice routing
- selection-vs-gating separation
- non-gradient expert balancing bias
- routing-memory prototypes
- frozen-expert control
- correctness-first sparse dispatch
- Tiny ViT-style Transformer
- deterministic synthetic continual stream
- optional Split-CIFAR-100 loader
- experiment JSON outputs
- unit tests and CI

Staged next:

- old-task routing reference buffer
- route-KL loss
- Fisher/EWC implementation
- full Split-CIFAR-100 protocol
- W&B telemetry
- capacity-aware dispatch
- Triton kernel
- block-sparse benchmark
- counterfactual route regret

## Quick start in VS Code

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -e .
pytest
```

Run the CPU smoke experiment:

```bash
python scripts/train_baseline.py \
  --config configs/smoke_cpu.yaml \
  --router top2 \
  --steps 10 \
  --data synthetic \
  --output experiments/results/smoke.json
```

Run the reference dispatch benchmark:

```bash
python benchmarks/dispatch_benchmark.py
```

For CIFAR-100:

```bash
pip install -e '.[data]'
python scripts/train_baseline.py --config configs/phase1.yaml --router continual --data cifar100
```

## Git workflow

Initialize the repository from the VS Code terminal:

```bash
bash scripts/setup_git.sh
git commit -m "chore: initialize OpenMoE-Router research scaffold"
```

Do not hand-edit experiment results. Every run should generate its own JSON artifact under `experiments/results/` and commit only the source/configuration required to reproduce the result unless a binary checkpoint is intentionally part of a release.

## Research discipline

This repository distinguishes **established mechanisms** from **open hypotheses**. In particular, balancing bias, router z-loss, shared experts, and Expert Choice are treated as prior techniques; the experimental contribution is the fixed-pool continual setting, routing-memory/stability decomposition, and systems measurements.

See:

- `docs/PHASE0_V3.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/ROADMAP.md`
- `docs/architecture.png`

## License

MIT
