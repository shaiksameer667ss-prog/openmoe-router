# Experiment F — Combined B0_attn + E_patch Freeze

**Protocol ID:** `EXP-F-COMBINED-B0-EPATCH-FREEZE-v1.0`

**Status:** PREREGISTERED BEFORE EXECUTION

## Scientific question

Do the independently tested B0 attention and E_patch interventions combine to produce at least a 2 pp reduction in old-task forgetting relative to Arm-A-prime?

## Hypothesis

If B0_attn and E_patch contribute partially independent harmful update leverage, jointly freezing them should reduce old-task forgetting more than either single-group intervention.

## Primary endpoint

`Δforget_F = F_F - F_Arm-A′`

Decision gates:

- `Δforget_F <= -2.0 pp` → **CAUSAL SUPPORT**
- `-2.0 pp < Δforget_F < +2.0 pp` → **INCONCLUSIVE**
- `Δforget_F >= +2.0 pp` → **COUNTER-SUPPORT**

These are mechanistic decision gates, not statistical significance thresholds.

## Locked intervention

Freeze exactly these six tensors:

```text
blocks.0.attn.in_proj_weight
blocks.0.attn.in_proj_bias
blocks.0.attn.out_proj.weight
blocks.0.attn.out_proj.bias
patch_embed.weight
patch_embed.bias
```

Frozen tensor count: **6**

Frozen parameter count: **1,075,712**

Expected trainable tensors: **83**

The frozen tensors remain in persistent AdamW.

**Optimizer state must not be cleared.**

## Locked training protocol

- Split-CIFAR100
- T0 = classes 0–19
- T1 = classes 20–39
- T2 = classes 40–59
- T3 = classes 60–79
- T4 = classes 80–99
- exact replay = 715 examples
- exact replay SHA256 = `de9136168116970af6051e73e64b208ee776ac127367f52862daabcaf65da8ed`
- AdamW learning rate = `3e-4`
- AdamW weight decay = `0.05`
- persistent optimizer
- 64 current + 64 replay per T4 batch
- 200 T4 steps
- 10 warmup + 190 adaptation
- continual router enabled
- router updates enabled
- CE + z-loss
- no stability regularization
- no advanced regularization
- canonical seed = `0`

## Reconstruction rule

Canonical theta3 alone is not sufficient for exact T4 optimizer continuation.

Therefore:

1. Reuse the existing reconstructed live T3 bundle when available.
2. Otherwise reconstruct T0→T3.
3. Verify reconstruction fidelity.
4. Run the matched Arm-A-prime control from the reconstructed live T3 state.
5. Require forgetting drift ≤ **2.0 pp** before interpreting Experiment F.

The exact canonical 715-example replay must not be reconstructed.

## Locked single-intervention references

```text
Arm-A′ forgetting = 32.0375 pp
D B0_attn         = 31.1625 pp
E E_patch         = 30.5500 pp

D effect          = -0.8750 pp
E effect          = -1.4875 pp

Additive reference = -2.3625 pp
```

## Secondary additivity readout

`interaction = Δforget_F - (-2.3625 pp)`

This is a descriptive point estimate only.

No post-hoc additivity threshold will be introduced.

## Plasticity gate

Arm-A′ T4 accuracy = **45.10%**

Plasticity collapse threshold:

`T4 accuracy < 30.10%`

This corresponds to a drop greater than **15 pp**.

## Reference checkpoint hashes

- theta0 SHA256 = `66dc3ac61695a96da3f23352d7c63c2bdd3e24f02056b94ed71c39a495c51ef7`
- theta3 SHA256 = `26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc`

## Required audits

- exact replay SHA256
- theta0 SHA256
- theta3 SHA256
- reconstruction fidelity
- frozen parameter invariance
- frozen optimizer-state invariance
- persistent AdamW membership
- non-frozen learning
- continual-router updates
- correct 64 + 64 batch composition
- exactly 200 T4 steps
- no theta3 mutation
- no optimizer-state clearing
- clean Git worktree

## Explicit closure

Experiment C is **closed** and must not be rerun.

Experiment D is **closed** and must not be rerun.

Experiment E is **closed** and must not be rerun.

The missing FWR `D_X` cross-reference remains **NOT_ESTIMABLE** and must not be regenerated for Experiment F.

## Execution lock

**NO T4 TRAINING MAY BEGIN UNTIL THIS PREREGISTRATION HAS BEEN COMMITTED.**
