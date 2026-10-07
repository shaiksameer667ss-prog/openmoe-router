# RM1 — Routing-to-Margin Predictability

## Status
Complete. No training performed; no checkpoint modified.

## Provenance
- GPU: Tesla T4
- Torch: 2.11.0+cu128
- Theta3 SHA-256: `26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc`
- F final SHA-256: `a011b0f5ae4fb691d6cc8173d14acb10e96e415b1384d60a38829c945262a5c4`
- E-BOTH v4 final SHA-256: `1cc2548309f5f61e86c9960eabab0ff2d96f1a98989ac152af85493aafd2aa90`
- Canonical identity checks: PASS
- Deterministic old-task loaders: 4
- Samples per checkpoint: 8,000
- 5-fold CV; 1,000 global label permutations

## Target
`m(x) = max_old_logits - max_new_logits`

## CV R² / permutation p
| Checkpoint | Signature | CV R² | Permutation p |
|---|---|---:|---:|
| T3 | pooled8 | 0.03585542 | 0.000999 |
| T3 | block0_8 | 0.09298929 | 0.000999 |
| T3 | block1_8 | 0.04841426 | 0.000999 |
| T3 | full16 | 0.13364613 | 0.000999 |
| F | pooled8 | 0.04329553 | 0.000999 |
| F | block0_8 | 0.01643079 | 0.000999 |
| F | block1_8 | 0.03854904 | 0.000999 |
| F | full16 | 0.05240563 | 0.000999 |
| E-BOTH | pooled8 | 0.02663877 | 0.000999 |
| E-BOTH | block0_8 | 0.00670214 | 0.000999 |
| E-BOTH | block1_8 | 0.03331054 | 0.000999 |
| E-BOTH | full16 | 0.03784760 | 0.000999 |

## Strongest signature
Full16 is strongest for all three checkpoints:
- T3: 0.13364613
- F: 0.05240563
- E-BOTH: 0.03784760

## Interpretation
RM1 establishes statistically detectable but modest routing-to-margin predictability in all three checkpoints. Because random CV mixes tasks, RM1 does not control for task-identity confounding. The planned RM1b follow-up is task-controlled within-task residual prediction and/or leave-one-task-out evaluation.

## Reproducibility note
The raw console output used to create this artifact is preserved in the conversation attachment `Pasted markdown(20261007-232742).md`. The structured values in `rm1_results.json` are transcribed directly from that run; no values were recomputed.
