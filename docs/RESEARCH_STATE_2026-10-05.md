# OpenMoE-Router Research State — 2026-10-05

## Purpose

This file is the durable research checkpoint for the OpenMoE-Router project. It records the canonical continual-learning setup, the diagnostic evidence accumulated before router training, the mechanisms that have been closed, the current FWR evidence, exact artifact identities, and the current scientific decision.

## Canonical setup

- Dataset: Split-CIFAR100 continual learning.
- Tasks:
  - T0 = classes 0–19
  - T1 = classes 20–39
  - T2 = classes 40–59
  - T3 = classes 60–79
  - T4 = classes 80–99
- Test examples/task: 2,000.
- Exact replay: 715 total; T0=179, T1=179, T2=179, T3=178, T4=0.
- Model: ViT-style sparse MoE, hidden=512, heads=4, FFN=1024, experts=8, Top-2, depth=2, image=32x32, patch=4, 64 tokens/image.
- Expert: Linear 512->1024 -> GELU -> Linear 1024->512.
- Block: norm1/attention/residual -> norm2/MoE/residual; final LayerNorm -> mean pooling -> classifier.
- Canonical continual router:
  - learned logits z
  - memory affinity a
  - routing bias b
  - selection = z + 0.25*a + b
  - Top-2 on selection
  - gate weights = softmax over original learned logits z restricted to selected experts
- Canonical router settings:
  - memory_lambda=0.25
  - memory_momentum=0.99
  - z_loss=5e-7
  - bias_lr=0.001
  - temperature=1.0
- Optimizer for canonical continual training: AdamW, lr=3e-4, wd=0.05, 200 steps/task, optimizer persists.

## Canonical endpoint and baselines

Canonical endpoint:
- T0 = 1.65%
- T1 = 0.90%
- T2 = 2.35%
- T3 = 3.95%
- T4 = 45.10%
- Old-task mean = 2.2125%

No-replay comparison:
- Old-task mean = 8.083%
- Reported forgetting = 36.588 pp.

Restricted historical evaluation:
- T0 = 22.05%
- T1 = 23.95%
- T2 = 29.45%
- T3 = 35.20%
- Old-task mean = 27.6625%.

These values are the project baseline used for mechanism diagnosis and should not be silently replaced.

## Exact checkpoint identities

Task checkpoints:
- task0 SHA256 = 66dc3ac61695a96da3f23352d7c63c2bdd3e24f02056b94ed71c39a495c51ef7
- task1 SHA256 = 9de3853eab02d95d77143864fa45a692304d268b2f7c9cb437ad8fcf639818b7
- task2 SHA256 = c02d343af7bd8ed466cd0535616b3278ce2cfe3dea91b12594969ae0045e0909
- task3 SHA256 = 26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc
- task4 SHA256 = 9504f80fda16fd56e8bc1e913d57b9713f2224d9285aa78f17f2d237adf79f95

All listed task checkpoints are 76,165,345 bytes.

Prospective step-10 checkpoint:
- SHA256 = e69fb593e6b158db1b5db16ffe26c75da0113bcc8ee9fa9d9c8abd8077823041
- path in the diagnostic tree: experiments/results/diagnostics/prospective_refit_endpoint_trajectory_56C72R/task_4_step_010.pt

Exact replay:
- file SHA256 = de9136168116970af6051e73e64b208ee776ac127367f52862daabcaf65da8ed
- payload SHA256 = ac54b7af81500046ef1472d2236eb6eb7aa3e738aa17903e6d4ed7976bdc4963
- shape = [715,3,32,32]
- labels shape = [715]
- task_ids shape = [715]
- payload keys = images, labels, task_ids
- task counts = [179,179,179,178,0]

## Routing drift and identity evidence

Raw routing drift:
- Top-2 overlap ≈ 0.3576
- gate KL ≈ 0.7656
- selection KL ≈ 0.6518

Representation-controlled drift:
- Top-2 overlap ≈ 0.809
- controlled gate KL ≈ 0.046
- controlled selection KL ≈ 0.040

Expert identity:
- Hungarian permutation = [0,1,2,3,4,5,6,7]
- identity p = 0.0005.

Historical expert-function drift:
- pathway/function preservation L0 = 0.168431
- pathway/function preservation L1 = 0.095288
- post-GELU tangent similarity L0 = 0.591880
- post-GELU tangent similarity L1 = 0.534104
- Taylor residual L0 ≈ 0.9020
- Taylor residual L1 ≈ 0.8896

## Mechanisms tested and closed

The following generic mechanisms were experimentally rejected/closed under the project protocol:

- ACR
- ROR / Test2R
- XPORT
- route regret
- coherent selection/gating
- image routing coherence
- router-expert coupling
- fixed-dispatch aggregation
- task-ID routing
- route priors
- static expert direction
- head-wise routing
- PAPA
- FARP
- general expert-function mechanism
- representation oracle
- interpolation / rollback
- generic routing preservation
- RADR / downstream-Jacobian regime routing
- zero-head regime
- optimizer hysteresis (frozen diagnostic)
- CMED / common-mode drift
- covariance-aware / GDCR

Selected quantitative closures:
- coherent selection/gating ≈ -0.525 pp
- image routing coherence ≈ -0.175 pp
- fixed-dispatch aggregation: L0 −2.083 pp, L1 −1.736 pp
- head-wise routing: 37.80% vs random 37.96%

Historical expert-function oracle:
- L0 only: +0.563 pp
- L1 only: +3.350 pp
- L0+L1: +6.388 pp
- conditional L0 given historical L1 on T0: +18.25 pp
- T1: −0.15 pp
- T2: −1.00 pp
- T3: −4.95 pp

These closures are important: they prevent the project from regressing into another generic additive routing heuristic.

## Fixed historical retention bridge

Fixed historical expert-indexed rank-4 correction:
- old-task gain = +4.850 pp

Controls:
- shared rank-4 = +4.875 pp
- shuffled = +4.775 pp
- full shared = +5.650 pp
- mean-only oracle = +5.400 pp
- fitted mean-only = +5.350 pp
- deviation-only = −0.050 pp

Geometry:
- cosine(mean correction, L1 oracle) ≈ 0.9811
- norm ≈ 9.0034

Retention SVD:
- singular values ≈ 16.1885, 1.8165, 1.5953, 1.4122
- uncentered rank-1 explained fraction = 97.095621%
- rank-2 = 98.318188%
- per-task cosine to b_hat:
  - T0 = 0.98788
  - T1 = 0.98617
  - T2 = 0.98343
  - T3 = 0.98392

LOO gate-weighted drift alignment:
- T0 = 0.990975
- T1 = 0.985798
- T2 = 0.980360
- T3 = 0.981878
- pooled = 0.984757

Interpretation: the historical retention signal is highly population/shared, but this alone does not identify a causal router mechanism.

## Online estimability diagnostics

Q2 source artifact:
- gate_weighted_drift_bridge_715_loo.json
- reference norms:
  - T0 = 9.003648
  - T1 = 8.918308
  - T2 = 9.049150
  - T3 = 9.023508
- prospective endpoint trajectory checkpoints available at steps:
  10, 25, 50, 100, 150, 200

Cosine to canonical fixed retention direction:
- step 10 = 0.672548
- step 25 = 0.712919
- step 50 = 0.815496
- step 100 = 0.895391
- step 150 = 0.907244
- step 200 = 0.930119

Q2-A, cross 0.90 by step 100:
- BLOCKED / not satisfied because step 100 = 0.895391 and step 75 was unavailable.

Q2-B, gate weighting vs uniform:
- margin = +0.035002
- FAIL versus preregistered >0.10 requirement.

Q2-C, population vs tokenwise:
- margin = +0.437057
- PASS.

Canonical theta4:
- mean cos(mu_g,b_fixed) = 0.997597
- per-LOO = [0.998189, 0.997441, 0.997194, 0.997565]

Endpoint construction matches the gate-weighted-drift-bridge target more closely than the causal target:
- GWDA target ≈ 0.984757
- causal target ≈ 0.932770

Decision: these observations do not justify router training.

## Q3 local drift

Local boundary diagnostics:
- theta3 -> step10: norm 3.220294, mean cosine 0.801462
- step10 -> 25: norm 1.423171, mean cosine 0.166538
- 25 -> 50: norm 1.575472, mean cosine 0.363076
- 50 -> 100: norm 1.839351, mean cosine 0.544774
- 100 -> 150: norm 1.033762, mean cosine 0.468103
- 150 -> 200: norm 0.927778, mean cosine 0.141129

Chained local aggregate:
- norm = 6.206011
- forward/reverse identical

The chain does not telescope because each local boundary uses a different h_t/g_t reference. Conclusion: local inter-boundary drift is not uniformly informative; the cumulative signal is path-level rather than a persistent per-boundary observable. Do not train.

## Q4 frame decomposition

Frames:
- qF = fixed-function frame
- qH = current-function frame
- qHG = current-function/current-gate frame

qF telescoped exactly:
- direct theta3 -> step200 norm = 5.1447944641
- absolute telescoping error = 5.058e-7

Focal alignment with canonical b_fixed (LOO):
- 50 -> 100: qF 0.616390, qH 0.520813, qHG 0.544774
- 100 -> 150: qF 0.198140, qH 0.485810, qHG 0.468103
- 150 -> 200: qF 0.087470, qH 0.295217, qHG 0.141129

Conclusion: qF collapses late; qH remains larger; qHG tracks qH. This establishes reference-distribution sensitivity, not gate causality.

Q4B matched-target current cumulative b_c^cum:
- step10 norm 7.113515
- step25 norm 6.723919
- step50 norm 6.124149
- step100 norm 6.364429
- step150 norm 6.350389
- step200 norm 6.300228

LOO qH / qHG alignment:
- theta3 -> 10: 0.618812 / 0.690395
- 10 -> 25: 0.354310 / 0.305296
- 25 -> 50: 0.199859 / 0.197437
- 50 -> 100: 0.394425 / 0.455549
- 100 -> 150: 0.426180 / 0.458679
- 150 -> 200: 0.230788 / 0.099915

Matched qHG - qH:
- +0.071584
- −0.049014
- −0.002422
- +0.061124
- +0.032500
- −0.130873

Conclusion: re-anchoring the reference distribution to each checkpoint does not yield a persistent strong local observable.

## Q5 per-image identity decomposition

For B = C + Delta:
- theta3 -> 10:
  - ||B|| = 7.113515
  - ||Delta|| = 3.220294
  - ||C|| = 5.411044
- 10 -> 25:
  - ||B|| = 6.723919
  - ||Delta|| = 1.423170
  - ||C|| = 6.437395
- 25 -> 50:
  - ||B|| = 6.124149
  - ||Delta|| = 1.575472
  - ||C|| = 6.030265
- 50 -> 100:
  - ||B|| = 6.364429
  - ||Delta|| = 1.839351
  - ||C|| = 5.791286
- 100 -> 150:
  - ||B|| = 6.350389
  - ||Delta|| = 1.033762
  - ||C|| = 5.944879
- 150 -> 200:
  - ||B|| = 6.300228
  - ||Delta|| = 0.927778
  - ||C|| = 6.281218

Identity reconstruction error:
- approximately 2.384e-7 to 4.768e-7.

cos(Delta,B):
- 0.691756845
- 0.302867502
- 0.187762395
- 0.442072809
- 0.461135536
- 0.094089724

Pooled geometry (C,B), (Delta,B), (C,Delta), C/B ratio, Delta/B ratio, alpha:
- theta3 -> 10: 0.902941, 0.691757, 0.314270, 0.760671, 0.452701, 0.187033
- 10 -> 25: 0.977552, 0.302868, 0.095269, 0.957387, 0.211658, 0.021062
- 25 -> 50: 0.966514, 0.187762, -0.070575, 0.984670, 0.257256, -0.018439
- 50 -> 100: 0.958561, 0.442073, 0.168217, 0.909946, 0.289005, 0.053427
- 100 -> 150: 0.988024, 0.461136, 0.318699, 0.936144, 0.162787, 0.055419
- 150 -> 200: 0.989129, 0.094090, -0.053332, 0.996983, 0.147261, -0.007878

Significant alpha cancellations were observed, including:
- 25 -> 50 T3 CI entirely negative [-0.054758, -0.030814]
- 150 -> 200 T0 CI [-0.019394, -0.000823]
- 150 -> 200 T1 CI [-0.029622, -0.012739]

Conclusion: after the first transition the historical cumulative direction is predominantly carried forward; local innovations are smaller and unstable. This is not a separately stored router memory state.

## M1/U4/U5 controls

M1 memory selection/gate desynchronization:
- memory-added slots = 1.49%
- g_added = 0.2550
- displaced = 0.2493
- ratio = 1.02298

Conclusion: memory was structurally inactive enough that it cannot by itself explain the failure.

U4 archival directional control:
- old baseline archival = 19.7203
- B_true = 21.2587 (+1.5385)
- random = +0.1399
- random-negative = −0.8392
- orthogonal = +1.8182
- cos(B_true,b_hat) = −0.529

This baseline is archival and not used as a primary causal result.

U5A corrected before-state protocol:
- 1,789 images
- 0/12 paired tests significant after Bonferroni correction
- smallest p = 0.307456
- null not rejected.

## FWR mechanism: Cell 1

Functional Write Routing (FWR) asks whether routing experts that receive current-task updates changes their future functional-drift trajectory in a Pareto-meaningful way.

Locked first-order quantity:
- U_e = eta * ||grad_e L_T4||^2
- q_e = E[g_e(h) <E_e(h), mu_hat>]
- DeltaA_1st = -eta <grad q, grad L>

Corrected forced-pair first-order screen at canonical theta4:
- replay = 715
- tokens = 45,760
- ||mu|| = 8.875405563
- target pair (4,6):
  - corrected DeltaA = −1.13960225e-4
  - rank 21/28 by DeltaA
  - rank 24/28 by U
- First-order Pareto frontier:
  - (0,6)
  - (2,6)
  - (0,5)
  - (0,2)
  - (2,7)
  - (2,3)
  - (3,7)
- positive predicted DeltaA:
  - (2,3) +6.39507925e-6
  - (3,7) +2.08109668e-5
- Existing finite-validation ordering:
  - DeltaA: (4,7) > (4,6) > (1,4)
  - Utility: (1,4) > (4,7) > (4,6)
- Spearman for both orderings = 1.0.

Training-frontier persistence using fixed first128 T4 images:
- theta3: 28/28 positive; frontier (0,7),(5,7)
- step10: 6/28 positive; candidates:
  - (3,4) +2.371220959e-4
  - (3,6) +1.328225801e-4
  - (4,6) +1.139400835e-4
  - (0,3) +8.771266352e-5
  - (0,6) +5.743813542e-5
  - (0,4) +1.015356149e-5
- step25: 0/28 positive
- step50: 0/28 positive
- Criterion A (same pair positive at >=3/4 checkpoints): none.
- Criterion B (positive pair at every checkpoint): false.
- Classification: NO_A_B_CRITERION_MET.
- Frontier recurrence:
  - (2,4): frontier 3/4
  - (3,4): frontier 3/4
  - sign patterns +--- and ++--
- Interpretation: high-utility recurrence exists, but not FWR persistence.

Corrected finite step10 confirmation for selected positive candidates:
- pair (3,4):
  - loss before 3.632120370865
  - loss after 3.558289289474
  - loss drop 0.07383108139038
  - U 0.0746871950247
  - DeltaA_1st +2.514987564491e-4
  - finite +2.514130012556e-4
- pair (3,6):
  - loss drop 0.05516052246094
  - U 0.05638946098024
  - DeltaA_1st +1.860294239713e-4
  - finite +1.859455883920e-4
- pair (4,6):
  - loss drop 0.04963636398315
  - U 0.05054719085631
  - DeltaA_1st +6.049882693875e-6
  - finite +5.988438021053e-6
- pair (0,3):
  - loss drop 0.09161305427551
  - U 0.09087210215253
  - DeltaA_1st +2.491106761137e-4
  - finite +2.490616318854e-4
- pair (0,6):
  - loss drop 0.03850507736206
  - U 0.03905220772997
  - DeltaA_1st +5.953502333388e-5
  - finite +5.949740981931e-5
- pair (0,4):
  - loss drop 0.05103230476379
  - U 0.05234172577133
  - DeltaA_1st +5.315729547187e-6
  - finite +5.268800508726e-6
- sign agreement = 6/6
- Spearman write vs finite = +1.0
- utility vs loss-drop = +1.0
- positive finite DeltaA = 6/6
- immutability = PASS

Canonical theta4 regression:
- pair (4,6):
  - finite DeltaA = −1.139604857752e-4
  - first-order = −1.139602339364e-4
- pair (4,7):
  - finite = −3.297458161841e-5
  - first-order = −3.296647510571e-5
- pair (1,4):
  - finite = −1.780266372057e-4
  - first-order = −1.780213427611e-4
- sign agreement = 3/3
- Spearman functional-write = +1.0
- Spearman utility vs loss-drop = +1.0
- archive regression = PASS
- immutability = PASS

## FWR mechanism: Cell 2 — multi-step

Protocol:
- start from exact prospective task-4 step10 checkpoint
- forced pair = (3,4)
- K = 20 sequential SGD steps
- eta = 0.001
- at every k:
  1. recompute natural h_k/g_k on the exact 715-example replay
  2. recompute mu_k from theta0 -> theta_k
  3. compute DeltaA_1st,k = -eta <grad q_k, grad L_k^(S)>
  4. apply theta{k+1} = theta_k - eta grad L_k^(S)
  5. measure finite DeltaA using the same pre-update h_k/g_k/mu_hat_k

Technical correction captured in this checkpoint:
- L1 h is [B,64,512]
- router gates are [B*64,8]
- h must be flattened to [B*64,512] before token alignment.
- Exact replay therefore gives 45,760 aligned tokens.
- The forced patch is applied to the L1 ContinualRouter, not the whole TinyMoETransformer.

20-step result:
- Spearman rho = +1.000000
- p-value = 0
- sign agreement = 20/20
- sum DeltaA_1st = −3.712977038049e-3
- sum DeltaA_finite = −3.714323043823e-3
- cumulative residual = −1.346005774394e-6
- mean per-step residual = −6.730028871971e-8
- SD per-step residual = 1.950945780578e-7
- mean absolute residual = 1.574394900672e-7
- median relative error = 4.113310e-4
- maximum relative error = 7.337738e-3
- all 20 signs matched.

Prefix Spearman:
- n=3 through n=20: rho = +1.0 throughout.
- early (steps 0..4) rho = +1.0
- first10 (steps 0..9) rho = +1.0
- late10 (steps 10..19) rho = +1.0

Trajectory:
- k=0: first +2.514987718314e-4; finite +2.515316009521e-4
- k=1: first +1.391399709973e-4; finite +1.387596130371e-4
- k=2: first +5.668270750903e-5; finite +5.626678466797e-5
- k=3: first −1.006377260637e-5; finite −1.001358032227e-5
- k=4: first −6.471338565461e-5; finite −6.508827209473e-5
- k=5: first −1.130984164774e-4; finite −1.130104064941e-4
- k=6: first −1.544553379063e-4; finite −1.541376113892e-4
- k=7: first −1.899721100926e-4; finite −1.902580261230e-4
- k=8: first −2.163069875678e-4; finite −2.162456512451e-4
- k=9: first −2.415829658275e-4; finite −2.413988113403e-4
- k=10: first −2.630269154906e-4; finite −2.630949020386e-4
- k=11: first −2.818156208377e-4; finite −2.819299697876e-4
- k=12: first −2.982773003168e-4; finite −2.985000610352e-4
- k=13: first −3.098197339568e-4; finite −3.098249435425e-4
- k=14: first −3.209745045751e-4; finite −3.210306167603e-4
- k=15: first −3.288959851488e-4; finite −3.287792205811e-4
- k=16: first −3.354343061801e-4; finite −3.355741500854e-4
- k=17: first −3.384808078408e-4; finite −3.385543823242e-4
- k=18: first −3.438499697950e-4; finite −3.437995910645e-4
- k=19: first −3.495303681120e-4; finite −3.496408462524e-4

Largest relative linearization errors:
- k=2: 7.337738e-3
- k=4: 5.793028e-3
- k=3: 4.987422e-3
- k=1: 2.733635e-3
- k=6: 2.057077e-3

Interpretation:
- FWR first-order prediction remains rank-valid and sign-valid across the full 20-step path.
- There is no observed early-valid / late-collapse regime in this experiment.
- Local linearization error stays small (<1% in every step).
- This establishes multi-step predictor validity for the forced (3,4) trajectory.
- It does NOT establish that routing by FWR causally improves continual-learning retention.

## Current scientific decision

Status: DO NOT TRAIN THE ROUTER YET.

What is now established:
1. Generic routing mechanisms tested so far do not explain the failure.
2. The historical retention signal is real and largely population/shared.
3. FWR has a mathematically defined functional-write quantity.
4. FWR first-order predictions were validated at step10 and theta4 single-step tests.
5. FWR multi-step prediction remained valid for 20 sequential SGD steps on forced pair (3,4).

What remains open before any router training:
- pair-level robustness beyond (3,4)
- independent T4-batch robustness
- whether FWR ranking predicts a task-level retention benefit
- intervention width / selection bandwidth
- whether the signal survives across more than one checkpoint/trajectory.

A router should only be trained after these causal/generalization gates are satisfied.

## Reproducibility / repository notes

The code repository contains nested duplicate OpenMoE trees under the working directory. Exact SHA matching was used to select the intended checkpoints rather than relying on filenames alone.

Source-of-truth repository implementation used:
- openmoe/models/moe.py
- openmoe/models/transformer.py
- openmoe/routers/base.py
- openmoe/routers/continual.py
- openmoe/routers/topk.py
- openmoe/data/streams.py
- configs/cifar100_milestone6_512.yaml
- scripts/train_baseline.py

The corrected Cell 2 logic requires the existing Cell-1 helper:
- install_forced_pair_forward(router, pair)

Diagnostic invariants:
- router observation disabled
- eval mode
- no optimizer state
- direct SGD only for the forced trajectory
- canonical checkpoints not mutated
- exact replay identity checked by SHA256
- token alignment checked at 45,760 x 512 / 45,760 x 8.

## This checkpoint

Created on 2026-10-05.
Purpose: freeze the scientific state before proceeding to the next FWR robustness/generalization gate.
