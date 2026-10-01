# Step 1 — Task-0 Routing Pathology Across Configurations

## Experimental identity

- Repository: `/kaggle/working/openmoe-router`
- Dataset: Split-CIFAR-100
- Task 0: classes 0–19
- Task-0 training images: 10,000
- Transformer depth: 2
- Hidden dimension: 512
- Attention heads: 4
- FFN dimension: 1024
- Experts: 8
- Top-k: 2
- Tokens/image: 64
- Top-2 assignments/layer: 1,280,000
- Seed: 0

## Conditions

### Arm A
ContinualRouter + replay, canonical trainable-expert condition.

Checkpoint:

`/kaggle/working/openmoe-router/experiments/results/crr_seed0/arm_A/checkpoints/task_0.pt`

### Margin λ=0
MarginRouter, `margin_loss=false`, `margin_weight=0.0`.

Checkpoint:

`/kaggle/working/trainable_router_m000_milestone6_seed0_checkpoints/task_0.pt`

### Margin λ=0.05
MarginRouter, `margin_loss=true`, `margin_weight=0.05`.

Checkpoint:

`/kaggle/working/trainable_router_m005_milestone6_seed0_checkpoints/task_0.pt`

## Measurement protocol

Routing was measured from the actual forward path:

`SparseMoE._last_routing.indices`

No router-selection formula was reconstructed manually.

All three conditions processed exactly 10,000 Task-0 images and exactly 1,280,000 Top-2 assignments per layer.

## Results

### Arm A

#### Layer 0

Counts:

`[202464, 298214, 192255, 538, 15401, 50522, 0, 520606]`

Active experts: `7/8`

Active experts: `E0, E1, E2, E3, E4, E5, E7`

Dead experts: `E6`

Firing rates:

`[0.158175, 0.23297969, 0.15019922, 0.00042031, 0.01203203, 0.03947031, 0.0, 0.40672344]`

Dominant expert: `E7`

Dominant assignments: `520,606`

#### Layer 1

Counts:

`[152059, 638947, 22416, 174769, 15544, 276265, 0, 0]`

Active experts: `6/8`

Active experts: `E0, E1, E2, E3, E4, E5`

Dead experts: `E6, E7`

Firing rates:

`[0.11879609, 0.49917734, 0.0175125, 0.13653828, 0.01214375, 0.21583203, 0.0, 0.0]`

Dominant expert: `E1`

Dominant assignments: `638,947`

### Margin λ=0

#### Layer 0

Counts:

`[189476, 887, 252495, 1608, 8423, 591184, 1419, 234508]`

Active experts: `8/8`

Dead experts: none

Firing rates:

`[0.14802813, 0.00069297, 0.19726172, 0.00125625, 0.00658047, 0.4618625, 0.00110859, 0.18320938]`

Dominant expert: `E5`

Dominant assignments: `591,184`

#### Layer 1

Counts:

`[32834, 637039, 206474, 271042, 49357, 64763, 14890, 3601]`

Active experts: `8/8`

Dead experts: none

Firing rates:

`[0.02565156, 0.49768672, 0.16130781, 0.21175156, 0.03856016, 0.05059609, 0.01163281, 0.00281328]`

Dominant expert: `E1`

Dominant assignments: `637,039`

### Margin λ=0.05

#### Layer 0

Counts:

`[189476, 887, 252495, 1608, 8423, 591184, 1419, 234508]`

Active experts: `8/8`

Dead experts: none

Firing rates:

`[0.14802813, 0.00069297, 0.19726172, 0.00125625, 0.00658047, 0.4618625, 0.00110859, 0.18320938]`

Dominant expert: `E5`

Dominant assignments: `591,184`

#### Layer 1

Counts:

`[32834, 637039, 206474, 271042, 49357, 64763, 14890, 3601]`

Active experts: `8/8`

Dead experts: none

Firing rates:

`[0.02565156, 0.49768672, 0.16130781, 0.21175156, 0.03856016, 0.01163281, 0.01163281, 0.00281328]`

Dominant expert: `E1`

Dominant assignments: `637,039`

## Three-way dead-expert result

| Condition | L0 active | L0 dead | L1 active | L1 dead |
|---|---:|---|---:|---|
| Arm A | 7/8 | E6 | 6/8 | E6, E7 |
| Margin λ=0 | 8/8 | none | 8/8 | none |
| Margin λ=0.05 | 8/8 | none | 8/8 | none |

## L0/L1 asymmetry

Arm A:

- L0 active: 7/8
- L1 active: 6/8
- L0 dead: E6
- L1 dead: E6, E7

Margin λ=0:

- L0 active: 8/8
- L1 active: 8/8

Margin λ=0.05:

- L0 active: 8/8
- L1 active: 8/8

## E6 / E7 checks

### Arm A

- L0 E6: 0 assignments
- L1 E6: 0 assignments
- L0 E7: 520,606 assignments = 0.40672344
- L1 E7: 0 assignments

### Margin λ=0

- L0 E6: 1,419 assignments
- L1 E6: 14,890 assignments
- L0 E7: 234,508 assignments = 0.18320938
- L1 E7: 3,601 assignments = 0.00281328

### Margin λ=0.05

- L0 E6: 1,419 assignments
- L1 E6: 14,890 assignments
- L0 E7: 234,508 assignments = 0.18320938
- L1 E7: 3,601 assignments = 0.00281328

## Interpretation supported by this diagnostic

The canonical Arm-A continual-router condition exhibits Task-0 expert under-utilization before continual drift is measured: E6 receives zero assignments in both MoE layers, while E7 is additionally inactive in Layer 1.

The MarginRouter reproductions activate all eight experts in both layers.

The λ=0 and λ=0.05 Margin conditions have identical Task-0 routing counts.

This diagnostic establishes a routing-pathology contrast; it does not by itself establish a continual-learning performance improvement.

## Provenance

The historical original Margin checkpoint binaries were unavailable in the current Kaggle session.

The λ=0 and λ=0.05 checkpoints used for this diagnostic were therefore reproduced from the recorded canonical experiment definitions under seed 0.

They should be described as reproduction-based routing diagnostics, not measurements from the original unavailable binaries.

## Integrity

- 10,000 Task-0 images processed for each condition.
- 1,280,000 Top-2 assignments per layer for each condition.
- No files were modified by the routing diagnostic itself.

