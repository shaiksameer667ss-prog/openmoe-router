# OpenMoE-Router — Final Kaggle Research Checkpoint

Date: 2026-10-05

## Canonical setup

Split-CIFAR100:
- T0 = classes 0–19
- T1 = classes 20–39
- T2 = classes 40–59
- T3 = classes 60–79
- T4 = classes 80–99

Model:
- hidden = 512
- heads = 4
- FFN = 1024
- experts = 8
- Top-2
- depth = 2
- image = 32x32
- patch = 4
- tokens/image = 64

Exact replay:
- total = 715
- T0 = 179
- T1 = 179
- T2 = 179
- T3 = 178
- T4 = 0
- tokens = 45,760

## Exact checkpoint identities

Theta0:
66dc3ac61695a96da3f23352d7c63c2bdd3e24f02056b94ed71c39a495c51ef7

Prospective task-4 step10:
e69fb593e6b158db1b5db16ffe26c75da0113bcc8ee9fa9d9c8abd8077823041

Exact replay file:
de9136168116970af6051e73e64b208ee776ac127367f52862daabcaf65da8ed

Exact replay payload:
ac54b7af81500046ef1472d2236eb6eb7aa3e738aa17903e6d4ed7976bdc4963

## FWR Cell 2

Forced pair:
(3,4)

eta:
0.001

Sequential steps:
20

First-order quantity:
DeltaA_1st,k = -eta <grad q_k, grad L_k^(S)>

Result:
- Spearman rho = +1.000000
- sign agreement = 20/20
- median relative error = 4.113310e-4
- maximum relative error = 7.337738e-3
- cumulative residual = -1.346006e-6

Trajectory:
- no early-valid / late-collapse regime observed
- prefix rho = +1.0 for n=3 through n=20

Technical correction:
L1 h is [B,64,512].
Router dense gates are [B*64,8].
Correct alignment is [45760,512] with [45760,8].

Forced patch target:
L1 ContinualRouter.

## Scientific decision

FWR multi-step predictor validity:
PASS

Router training:
NOT YET JUSTIFIED

Next gate:
FWR robustness/generalization and causal task-level retention benefit.

## Important archival rule

This checkpoint preserves the negative/closed mechanisms as well as positive results. No previous experiment should be deleted merely because it failed. Failed experiments are part of the scientific record.
