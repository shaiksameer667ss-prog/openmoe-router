# OpenMoE-Router — Master Research Archive
## Complete project state, experiments, results, diagnostics, falsification record, and GitHub archive status
### Snapshot date: 2026-10-05

---

## 0. Purpose of this document

This is the durable, self-contained research record for the OpenMoE-Router project as of 2026-10-05.

The project goal is to invent a genuinely new, mechanism-grounded sparse-MoE continual-learning router rather than another heuristic or additive routing score. The research protocol has therefore been diagnostic-first:

1. define the continual-learning failure precisely;
2. recover the exact canonical state and exact replay;
3. test mechanisms one diagnostic at a time;
4. preregister falsification criteria where possible;
5. close generic explanations before proposing a new causal mechanism;
6. do not train a new router unless frozen diagnostics establish a mechanism that is both observable and causally testable.

A failed or closed mechanism is part of the scientific record and must not be deleted just because it did not work.

This document records both positive and negative evidence.

---

# 1. Executive scientific status

## 1.1 Current conclusion

**DO NOT TRAIN THE ROUTER YET.**

The project has reached a strong diagnostic checkpoint:

- the historical retention signal is real and largely population/shared;
- raw routing drift is substantial, but most of it is explained by representation drift;
- expert identity remains stable;
- generic routing-preservation, routing-coherence, representation, optimizer-hysteresis, and related explanations have been experimentally closed under the project's protocols;
- a functional-write routing mechanism (FWR) has now been defined mathematically;
- FWR first-order predictions have been validated against finite interventions at step 10 and at the canonical endpoint;
- FWR was additionally validated over a 20-step sequential forced trajectory with perfect rank/sign agreement and sub-1% per-step linearization error;
- however, that is **predictor validity**, not yet evidence that training a router with FWR causally improves task-level continual-learning retention.

The next gate is:

**FWR robustness/generalization + causal task-level retention benefit**

Specifically:
- more forced expert pairs;
- independent T4 batches;
- task-level retention causal tests;
- intervention-width tests;
- more than one checkpoint / trajectory where feasible.

No router training should begin before these gates are passed.

---

# 2. Project identity

Repository:

    https://github.com/shaiksameer667ss-prog/openmoe-router

Default branch:

    main

Research branches currently important:

    research/checkpoint-2026-10-05-fwr
    research/master-project-archive-2026-10-05
    research/final-kaggle-archive-2026-10-05   (local Kaggle branch; push was still failing at snapshot time)

The GitHub repository is not archived and the linked account has push permission.

---

# 3. Research philosophy and protocol

The central methodological rule has been:

**do not turn a descriptive correlation into a routing rule.**

The project repeatedly distinguishes:

- selection from gating;
- raw routing drift from representation-controlled routing drift;
- expert identity from expert function;
- population-level retention directions from tokenwise local drift;
- first-order causal predictors from finite intervention outcomes;
- predictor validity from actual continual-learning performance gain.

This separation is important because the canonical router already contains multiple signals:

- learned routing logits z;
- memory affinity a;
- routing bias b.

The project explicitly avoids inventing another arbitrary weighted sum merely because a score correlates with retention.

---

# 4. Canonical continual-learning setup

## 4.1 Dataset

Split-CIFAR100 continual learning.

Tasks:

- T0 = classes 0–19
- T1 = classes 20–39
- T2 = classes 40–59
- T3 = classes 60–79
- T4 = classes 80–99

Test examples:

    2,000 per task

## 4.2 Exact replay

Total replay:

    715 examples

Per task:

- T0 = 179
- T1 = 179
- T2 = 179
- T3 = 178
- T4 = 0

Replay tensor:

    [715, 3, 32, 32]

Replay payload keys:

    images, labels, task_ids

Task counts:

    [179, 179, 179, 178, 0]

Exact replay file SHA256:

    de9136168116970af6051e73e64b208ee776ac127367f52862daabcaf65da8ed

Exact replay payload SHA256:

    ac54b7af81500046ef1472d2236eb6eb7aa3e738aa17903e6d4ed7976bdc4963

Replay token count:

    715 × 64 = 45,760 tokens

## 4.3 Model

ViT-style sparse MoE:

- hidden size = 512
- attention heads = 4
- FFN width = 1024
- experts = 8
- Top-2 routing
- depth = 2
- image = 32×32
- patch = 4
- tokens per image = 64

Expert:

    Linear(512 → 1024)
    GELU
    Linear(1024 → 512)

Transformer block:

    norm1 → attention → residual
    norm2 → MoE → residual

Final representation:

    final LayerNorm
    mean pooling
    classifier

## 4.4 Canonical continual router

Learned logits:

    z

Memory affinity:

    a

Routing bias:

    b

Selection logits:

    selection = z + 0.25 a + b

Top-2 selection is applied to the selection logits.

Gate weights are then computed by softmax over the **original learned logits z**, restricted to the selected experts.

This selection/gate split is central to the analysis.

Canonical router settings:

- memory_lambda = 0.25
- memory_momentum = 0.99
- z_loss = 5e-7
- bias_lr = 0.001
- temperature = 1.0

Canonical continual-training optimizer:

- AdamW
- learning rate = 3e-4
- weight decay = 0.05
- 200 steps/task
- optimizer state persists across tasks

---

# 5. Exact canonical checkpoint identities

Task checkpoints:

T0:

    66dc3ac61695a96da3f23352d7c63c2bdd3e24f02056b94ed71c39a495c51ef7

T1:

    9de3853eab02d95d77143864fa45a692304d268b2f7c9cb437ad8fcf639818b7

T2:

    c02d343af7bd8ed466cd0535616b3278ce2cfe3dea91b12594969ae0045e0909

T3:

    26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc

T4:

    9504f80fda16fd56e8bc1e913d57b9713f2224d9285aa78f17f2d237adf79f95

All listed task checkpoints:

    76,165,345 bytes

Prospective task-4 step-10 checkpoint:

    e69fb593e6b158db1b5db16ffe26c75da0113bcc8ee9fa9d9c8abd8077823041

Prospective step-10 diagnostic path:

    experiments/results/diagnostics/prospective_refit_endpoint_trajectory_56C72R/task_4_step_010.pt

---

# 6. Canonical performance baseline

Canonical endpoint:

| Task | Accuracy |
|---|---:|
| T0 | 1.65% |
| T1 | 0.90% |
| T2 | 2.35% |
| T3 | 3.95% |
| T4 | 45.10% |
| Old-task mean | 2.2125% |

No-replay comparison:

- old-task mean = 8.083%
- reported forgetting = 36.588 pp

Restricted historical evaluation:

| Task | Historical accuracy |
|---|---:|
| T0 | 22.05% |
| T1 | 23.95% |
| T2 | 29.45% |
| T3 | 35.20% |
| Old-task mean | 27.6625% |

These canonical values are the project reference and should not be silently replaced with later variants.

---

# 7. Initial broad evidence: routing drift

## 7.1 Raw routing drift

Measured across the continual trajectory:

- Top-2 overlap ≈ 0.3576
- gate KL ≈ 0.7656
- selection KL ≈ 0.6518

This shows substantial apparent routing change.

## 7.2 Representation-controlled routing drift

After holding representation effects under control:

- Top-2 overlap ≈ 0.809
- controlled gate KL ≈ 0.046
- controlled selection KL ≈ 0.040

Interpretation:

A large fraction of the apparent routing drift is explained by representation drift rather than an independent router-pathology effect.

This closed a naïve line of explanation in which raw route change itself is treated as the causal failure.

---

# 8. Expert identity versus expert function

## 8.1 Expert identity

Hungarian matching found:

    permutation = [0,1,2,3,4,5,6,7]

identity p-value:

    p = 0.0005

Interpretation:

Expert identities are stable. The phenomenon is not simple expert permutation / role-swapping.

## 8.2 Historical expert-function drift

Pathway/function preservation:

- L0 = 0.168431
- L1 = 0.095288

Post-GELU tangent similarity:

- L0 = 0.591880
- L1 = 0.534104

Taylor residual:

- L0 ≈ 0.9020
- L1 ≈ 0.8896

Interpretation:

Experts retain identity while their functional maps drift. This motivated explicit functional-pathway diagnostics and ultimately the functional-write perspective.

---

# 9. Generic mechanism screen: mechanisms closed or rejected

The following families were tested/closed under the project's diagnostic protocol:

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

This list is an important research artifact because the project is intended to avoid rediscovering an already-closed generic mechanism.

## 9.1 Selected quantitative closures

Coherent selection/gating:

    ≈ −0.525 pp

Image routing coherence:

    ≈ −0.175 pp

Fixed-dispatch aggregation:

- L0 = −2.083 pp
- L1 = −1.736 pp

Head-wise routing:

    37.80% vs random 37.96%

Interpretation:

Simple route-consistency or coarse routing-structure controls do not explain the retention failure.

---

# 10. Historical expert-function oracle

Historical function oracles produced:

- L0 only = +0.563 pp
- L1 only = +3.350 pp
- L0 + L1 = +6.388 pp

Conditional L0 given historical L1:

- T0 = +18.25 pp
- T1 = −0.15 pp
- T2 = −1.00 pp
- T3 = −4.95 pp

Interpretation:

The L1 historical function is materially relevant, but the effect is not a uniform task-independent scalar. This argues for a mechanism that is functionally structured rather than a generic static expert rule.

---

# 11. Fixed historical retention bridge

A fixed historical expert-indexed rank-4 correction produced:

    old-task gain = +4.850 pp

Controls:

- shared rank-4 = +4.875 pp
- shuffled = +4.775 pp
- full shared = +5.650 pp
- mean-only oracle = +5.400 pp
- fitted mean-only = +5.350 pp
- deviation-only = −0.050 pp

Geometry:

- cosine(mean correction, L1 oracle) ≈ 0.9811
- mean-correction norm ≈ 9.0034

Interpretation:

The retention signal is not strongly dependent on expert identity indexing. Much of the effect is explainable by a shared population direction.

This result is useful because it rules out a simple “each expert owns its own unique retention vector” explanation.

---

# 12. Population structure of the historical retention signal

Retention SVD singular values:

    16.1885
    1.8165
    1.5953
    1.4122

Uncentered variance-like explained fractions:

- rank-1 = 97.095621%
- rank-2 = 98.318188%

Per-task cosine to shared direction b_hat:

- T0 = 0.98788
- T1 = 0.98617
- T2 = 0.98343
- T3 = 0.98392

Interpretation:

The historical retention direction is very close to a single population/shared mode.

This is a central clue but not, by itself, a causal routing mechanism.

---

# 13. LOO gate-weighted drift bridge

Leave-one-out alignment of gate-weighted drift with the historical shared direction:

- T0 = 0.990975
- T1 = 0.985798
- T2 = 0.980360
- T3 = 0.981878
- pooled = 0.984757

The project treated this as a strong descriptive bridge but deliberately did not convert it directly into a router rule.

The distinction is:

    historical association ≠ causal routing intervention

---

# 14. Q2: online estimability of the shared retention direction

Source artifact:

    gate_weighted_drift_bridge_715_loo.json

Historical reference norms:

- T0 = 9.003648
- T1 = 8.918308
- T2 = 9.049150
- T3 = 9.023508

Available prospective checkpoints:

    step 10, 25, 50, 100, 150, 200

Mean cosine to canonical fixed retention direction:

| Step | Mean cosine |
|---|---:|
| 10 | 0.672548 |
| 25 | 0.712919 |
| 50 | 0.815496 |
| 100 | 0.895391 |
| 150 | 0.907244 |
| 200 | 0.930119 |

Q2-A:

Criterion = cross 0.90 by step 100

Result:

    BLOCKED / not satisfied

Reason:

    step 100 = 0.895391
    step 75 unavailable

Q2-B:

Gate weighting versus uniform:

    margin = +0.035002

Prerequisite:

    > +0.10

Result:

    FAIL

Q2-C:

Population versus tokenwise:

    margin = +0.437057

Result:

    PASS

Canonical theta4:

- mean cos(mu_g,b_fixed) = 0.997597
- LOO values = [0.998189, 0.997441, 0.997194, 0.997565]

Target comparison:

- GWDA target ≈ 0.984757
- causal target ≈ 0.932770

Endpoint construction is closer to the GWDA target than the causal target.

Decision:

These observations did not justify router training.

---

# 15. Q3: local inter-boundary drift

Local diagnostics:

| Transition | Norm | Mean cosine |
|---|---:|---:|
| theta3 → 10 | 3.220294 | 0.801462 |
| 10 → 25 | 1.423171 | 0.166538 |
| 25 → 50 | 1.575472 | 0.363076 |
| 50 → 100 | 1.839351 | 0.544774 |
| 100 → 150 | 1.033762 | 0.468103 |
| 150 → 200 | 0.927778 | 0.141129 |

Chained local aggregate norm:

    6.206011

Forward/reverse:

    identical

Important non-telescoping fact:

The chain does not telescope because each local boundary uses a different h_t/g_t reference.

Conclusion:

Local inter-boundary drift is not uniformly informative. The useful signal is path-level / accumulated rather than a persistent per-boundary observable.

Decision:

    Do not train.

---

# 16. Q4: reference-frame decomposition

Three frames:

- qF = fixed-function frame
- qH = current-function frame
- qHG = current-function/current-gate frame

qF telescoped exactly.

Direct theta3 → step200 qF norm:

    5.1447944641

Absolute telescoping error:

    5.058e-7

Focal LOO alignments with canonical b_fixed:

| Transition | qF | qH | qHG |
|---|---:|---:|---:|
| 50 → 100 | 0.616390 | 0.520813 | 0.544774 |
| 100 → 150 | 0.198140 | 0.485810 | 0.468103 |
| 150 → 200 | 0.087470 | 0.295217 | 0.141129 |

Interpretation:

- qF collapses late;
- qH remains larger;
- qHG generally tracks qH.

This establishes reference-distribution sensitivity.

It does **not** establish gate causality.

---

# 17. Q4B: checkpoint-anchored cumulative frame

Current cumulative shared target norms:

| Step | Norm |
|---|---:|
| 10 | 7.113515 |
| 25 | 6.723919 |
| 50 | 6.124149 |
| 100 | 6.364429 |
| 150 | 6.350389 |
| 200 | 6.300228 |

LOO qH / qHG:

| Transition | qH | qHG | qHG − qH |
|---|---:|---:|---:|
| theta3 → 10 | 0.618812 | 0.690395 | +0.071584 |
| 10 → 25 | 0.354310 | 0.305296 | −0.049014 |
| 25 → 50 | 0.199859 | 0.197437 | −0.002422 |
| 50 → 100 | 0.394425 | 0.455549 | +0.061124 |
| 100 → 150 | 0.426180 | 0.458679 | +0.032500 |
| 150 → 200 | 0.230788 | 0.099915 | −0.130873 |

Conclusion:

Re-anchoring the reference distribution to each checkpoint does not yield a persistent strong local observable.

---

# 18. Q5: exact per-image carry-plus-innovation decomposition

Identity:

    B = C + Delta

where B is the total bridge signal, C is the carried/shared component, and Delta is the local innovation.

Norms:

| Transition | ||B|| | ||Delta|| | ||C|| |
|---|---:|---:|---:|
| theta3 → 10 | 7.113515 | 3.220294 | 5.411044 |
| 10 → 25 | 6.723919 | 1.423170 | 6.437395 |
| 25 → 50 | 6.124149 | 1.575472 | 6.030265 |
| 50 → 100 | 6.364429 | 1.839351 | 5.791286 |
| 100 → 150 | 6.350389 | 1.033762 | 5.944879 |
| 150 → 200 | 6.300228 | 0.927778 | 6.281218 |

Identity reconstruction error:

    approximately 2.384e-7 to 4.768e-7

cos(Delta,B):

- 0.691756845
- 0.302867502
- 0.187762395
- 0.442072809
- 0.461135536
- 0.094089724

Pooled geometry:

| Transition | cos(C,B) | cos(Delta,B) | cos(C,Delta) | C/B | Delta/B | alpha |
|---|---:|---:|---:|---:|---:|---:|
| theta3 → 10 | 0.902941 | 0.691757 | 0.314270 | 0.760671 | 0.452701 | 0.187033 |
| 10 → 25 | 0.977552 | 0.302868 | 0.095269 | 0.957387 | 0.211658 | 0.021062 |
| 25 → 50 | 0.966514 | 0.187762 | -0.070575 | 0.984670 | 0.257256 | -0.018439 |
| 50 → 100 | 0.958561 | 0.442073 | 0.168217 | 0.909946 | 0.289005 | 0.053427 |
| 100 → 150 | 0.988024 | 0.461136 | 0.318699 | 0.936144 | 0.162787 | 0.055419 |
| 150 → 200 | 0.989129 | 0.094090 | -0.053332 | 0.996983 | 0.147261 | -0.007878 |

Notable confidence-interval sign cancellations:

- 25 → 50, T3 CI entirely negative: [-0.054758, -0.030814]
- 150 → 200, T0 CI: [-0.019394, -0.000823]
- 150 → 200, T1 CI: [-0.029622, -0.012739]

Conclusion:

After the first transition the historical cumulative direction is predominantly carried forward. Local innovations are smaller and unstable.

This is not a separately stored router-memory state and therefore does not immediately define a router update rule.

---

# 19. M1 / U4 / U5 controls

## M1: memory selection/gate desynchronization

- memory-added slots = 1.49%
- g_added = 0.2550
- displaced = 0.2493
- ratio = 1.02298

Conclusion:

The memory path was structurally inactive enough that it cannot by itself explain the failure.

## U4: archival directional control

Archival old baseline:

    19.7203

Controls:

- B_true = 21.2587 (+1.5385)
- random = +0.1399
- random-negative = −0.8392
- orthogonal = +1.8182
- cos(B_true,b_hat) = −0.529

This baseline is archival and not treated as a primary causal finding.

## U5A: corrected before-state protocol

Dataset:

    1,789 images

Paired tests:

    0 / 12 significant after Bonferroni

Smallest p-value:

    0.307456

Null conclusion:

    null not rejected

---

# 20. Historical implementation source-of-truth

Repository implementation used for the exact diagnostics:

- openmoe/models/moe.py
- openmoe/models/transformer.py
- openmoe/routers/base.py
- openmoe/routers/continual.py
- openmoe/routers/topk.py
- openmoe/data/streams.py
- configs/cifar100_milestone6_512.yaml
- scripts/train_baseline.py

Important implementation facts:

### MoE path

The sparse MoE path is hard sparse in evaluation / no-dense-surrogate evaluation.

### Transformer representation

The target L1 pre-MoE representation is:

    block.norm2(x)

The final feature is obtained after final normalization and mean pooling.

### Router API

Base routing uses:

    _route(logits, selection_logits=None)

Top-K is performed on selection logits, while gate softmax uses the original learned logits restricted to the selected experts.

### Continual router

Selection is:

    logits + memory_lambda * affinity + routing_bias

Memory observation updates are conditional on update enablement.

---

# 21. Test 5 / representation and task-ID line of inquiry

The archive includes Test 5.1A attention pooling, Test 5.3 task-supervised SupCon, representation probes, and related representation-control experiments.

Test 5.3:

    held-out task-ID = 34.1375%

The project used these studies to characterize whether task identity / representational geometry alone could explain the continual failure.

The final interpretation was that representation information and routing changes are related, but a pure task-ID router is not the intended causal mechanism.

---

# 22. Milestone 6: depth / patch / stride studies

Results index:

### Test 6.1 — depth 6

Range:

    34.11–35.20%

Interpretation:

    no depth effect

### Test 6.2 — patch 8

    32.4875%

Difference versus patch-4 seed0 reference:

    −1.55 pp

### Test 6.3 — stride 2

Seed0:

    36.4375%

Seed1:

    35.5250%

Patch4 stride4 references:

- seed0 = 34.0375%
- seed1 = 34.4125%

Two-seed patch4/stride4 mean:

    34.2250%

Two-seed patch4/stride2 mean:

    35.98125%

Seed-matched mean improvement:

    +1.75625 pp

Paired strict4 effect:

    +2.06250 pp

Fresh-random 715 kNN:

    24.41–24.55%

Chance:

    25%

Final interpretation:

Increasing token-interaction capacity is a contributor to task-ID separability.

However, stride2 changes multiple variables jointly:

- overlap
- spatial sampling
- sequence length
- positional encoding
- attention scale

Therefore the effect does not isolate attention-pair count as a sole causal variable.

---

# 23. Recovery event and scientific integrity

A power-cut recovery occurred around 2026-10-01.

Recovery manifest:

    experiments/RECOVERY_MANIFEST.md

The important integrity rule:

The remote Git repository contained the project through Test 5.3 at recovery time.

Test 6.1–6.3 binary checkpoints and local replay tensors were not available after the power cut.

The archived record explicitly states:

- lost checkpoints were not fabricated;
- recovered numerical results were preserved from the session record;
- a dirty transformer diff from the failed recovery session was preserved separately as a patch artifact.

This distinction between recovered numerical records and unavailable binaries is part of the provenance.

---

# 24. Results inventory in the repository

The archived result families include:

- attention_pool_diagnostic_seed0
- baseline_byte_matched
- baseline_byte_matched_seed1
- baseline_byte_matched_seed2
- baseline_none_seed0_final
- baseline_sample_matched
- baseline_sample_matched_seed2
- bounded_decoder
- coarse_probe_d2_512
- cross_layer_path_overlap
- crr_seed0
- crr_seed0_summary
- er_ace
- farp_cs_seed0_679
- fixed_prior_logit_adjustment_none_seed0
- milestone6
- papa_path_anchor_seed0
- probe
- rcr
- replay_A_reconstruction
- replication
- routed_support_reconciliation
- supcon_gpm_seed0_corrected
- taih_seq_corrected_balanced_q_seed0
- test_5_3_task_supcon_seed0
- diagnostics

At the final Kaggle snapshot:

    156 experiment/result files in experiments/results

    31 LFS-tracked files

The detailed path inventory is preserved in:

    docs/FINAL_KAGGLE_ARCHIVE_FILE_LIST_2026-10-05.txt

and the content hashes in:

    docs/FINAL_KAGGLE_ARCHIVE_SHA256_2026-10-05.txt

---

# 25. Important result files / artifact families

## Baseline family

- baseline_byte_matched.json
- baseline_byte_matched_seed1.json
- baseline_byte_matched_seed2.json
- baseline_none_seed0_final.json
- baseline_sample_matched.json
- baseline_sample_matched_seed2.json

## Bounded-decoder family

Representative artifacts:

- abc_B_none_exact_A_buffer_seed0.json
- abc_C_fresh_random_buffer_seed12345.pt
- abc_C_fresh_random_buffer_seed2.pt
- abc_C_none_fresh_buffer_seed0.json
- path1_multiseed_summary.json
- path1_plain_ce_seed0_manifest.json
- plain_ce_seed0_A_buffer.json
- plain_ce_seed0_C_buffer.json
- plain_ce_seed0_for_match.json
- plain_ce_seed1_A_buffer.json
- plain_ce_seed2_A_buffer.json
- plain_ce_seed2_C_buffer.json
- plain_ce_seed2_for_match.json
- probe_b_boundary3_buffer.pt
- probe_b_boundary3_buffer.pt.meta.json
- probe_b_m1_plain_ce_seed0.json
- probe_b_m1_plain_ce_seed1.json
- probe_b_m1_plain_ce_seed2.json
- probe_b_m1_replay_seed0.json
- probe_b_m1_replay_seed1.json
- probe_b_m1_replay_seed2.json
- probe_b_m3_replay_seed0.json
- probe_b_m3_replay_seed1.json
- probe_b_m3_replay_seed2.json
- probe_b_results_summary.json
- replay_715_bounded_probe_frozen_seed0.json
- replay_715_bounded_probe_seed0.json
- replay_715_seed2_buffer.pt
- replay_715_seed2_buffer_run.json
- replay_715_seed2_for_decoder.json
- associated task checkpoints

## CRR family

- expert_functional_activation_audit.json
- expert_subspace_codrift.json
- farp_pre_audit.json
- farp_pre_audit_cosine.json
- pathway_2x2_decomposition_corrected.json
- result.json
- crr_seed0_summary.md

## FARP

- realization.json
- result.json
- checkpoint/task_0.pt
- checkpoint/task_1.pt
- checkpoint/task_2.pt
- checkpoint/task_3.pt
- checkpoint/task_4.pt
- farp_anchor.pt

## RCR

- rcr_722_beta0.1_seed0.json
- rcr_722_beta1.0_seed0.json
- rcr_722_beta10.0_seed0.json
- rcr_seed0_beta_sweep_summary.json
- replay_722_seed0.json

## Representation / probe families

- probe/none_seed0.json
- probe/head_masked_seed0.json
- probe/head_masked_frozen_old_seed0.json
- probe/seed0_screening_summary.json
- supcon_gpm_seed0_corrected/representation_audit.json
- supcon_gpm_seed0_corrected/result.json
- test_5_3_task_supcon_seed0/result.json
- task-ID probe files

## Replay reconstruction

- matched_694/slda_ncm/slda_vs_ncm_seed0.json

## Replication

- arm_A_seed0_679/result.json
- arm_A_seed0_reproduced.json
- corresponding task checkpoints

## Diagnostics

Important final diagnostic files include:

- fwr_finite_update_counterfactual_confirmation.json
- fwr_first_order_sign_corrected_reference.json
- fwr_first_order_write_allocation_precheck.json
- fwr_forced_pair_first_order_28_pairs.json
- fwr_step10_replay_mu_finite_confirmation.json
- fwr_step10_replay_mu_finite_confirmation_CORRECTED.json
- fwr_theta4_corrected_regression.json
- fwr_training_frontier_persistence.json
- q1_expert1_t3_alignment_theta4_seed0_per_class.csv
- q1_expert_output_t3_alignment_theta4_seed0.json
- q2_theta0_anchored_online_estimability.json
- q3_local_inter_boundary_drift_alignment.json
- q4_local_drift_frame_attribution.json
- q4b_checkpoint_anchored_cumulative_frame_715_loo.json
- q4b_endpoint_resolution.json
- q5_exact_carry_innovation_decomposition.json
- t3_14_pair_routing_map_theta4_seed0.json

FWR Cell 2 trajectory archive:

    experiments/results/diagnostics/fwr_multistep_step10_pair34/

Expected contents:

- fwr_multistep_step10_pair34_trajectory.csv
- fwr_multistep_step10_pair34_prefix_spearman.csv
- fwr_multistep_step10_pair34.json
- fwr_multistep_step10_pair34_run.md
- EXECUTION_TRANSCRIPT_SOURCE.md

Corresponding code snapshot:

    experiments/code_snapshots/fwr_multistep_step10_pair34_cell2.py

---

# 26. FWR: Functional Write Routing

## 26.1 Definition

FWR asks:

> Does routing experts that receive current-task updates change their future functional-drift trajectory in a Pareto-meaningful way?

Define expert utility:

    U_e = eta ||grad_e L_T4||^2

Define a function-drift-related quantity:

    q_e = E[g_e(h) <E_e(h), mu_hat>]

First-order functional-write change:

    DeltaA_1st = -eta <grad q, grad L>

The key research idea is not “add a score to routing.”

Instead:

- identify the causal effect of writing into particular expert parameters;
- estimate how that write changes future functional drift;
- compare first-order predictions with finite forced interventions;
- only then ask whether the mechanism generalizes.

---

# 27. FWR Cell 1 — forced pair screen and finite validation

## 27.1 Canonical theta4 corrected forced pair

Forced pair:

    (4,6)

Replay:

    715

Tokens:

    45,760

mu norm:

    8.875405563

Corrected first-order DeltaA:

    -1.13960225e-4

Rank:

    21/28 by DeltaA
    24/28 by U

First-order Pareto frontier:

    (0,6)
    (2,6)
    (0,5)
    (0,2)
    (2,7)
    (2,3)
    (3,7)

Positive predicted DeltaA:

    (2,3) +6.39507925e-6
    (3,7) +2.08109668e-5

Existing finite validation order by DeltaA:

    (4,7) > (4,6) > (1,4)

Existing validation order by U:

    (1,4) > (4,7) > (4,6)

Spearman:

    1.0 for both orderings

## 27.2 Training-frontier persistence test

Fixed first128 T4 images.

At theta3:

    28/28 positive

Frontier:

    (0,7), (5,7)

At step10:

    6/28 positive

Candidates:

- (3,4) +2.371220959e-4
- (3,6) +1.328225801e-4
- (4,6) +1.139400835e-4
- (0,3) +8.771266352e-5
- (0,6) +5.743813542e-5
- (0,4) +1.015356149e-5

At step25:

    0/28 positive

At step50:

    0/28 positive

Criterion A:

    none

Criterion B:

    false

Classification:

    NO_A_B_CRITERION_MET

Frontier recurrence:

- (2,4) = frontier 3/4
- (3,4) = frontier 3/4

Sign patterns:

- +---
- ++--

Interpretation:

High-utility recurrence exists, but the desired stable FWR persistence criterion is not established.

---

# 28. FWR Cell 1 — corrected finite step10 confirmation

Exact step10 checkpoint SHA:

    e69fb593e6b158db1b5db16ffe26c75da0113bcc8ee9fa9d9c8abd8077823041

Replay mu norm:

    7.113514800

Selected forced pairs:

### Pair (3,4)

Loss drop:

    0.07383108139038

U:

    0.0746871950247

DeltaA first-order:

    +2.514987564491e-4

DeltaA finite:

    +2.514130012556e-4

### Pair (3,6)

Loss drop:

    0.05516052246094

U:

    0.05638946098024

DeltaA first-order:

    +1.860294239713e-4

DeltaA finite:

    +1.859455883920e-4

### Pair (4,6)

Loss drop:

    0.04963636398315

U:

    0.05054719085631

DeltaA first-order:

    +6.049882693875e-6

DeltaA finite:

    +5.988438021053e-6

### Pair (0,3)

Loss drop:

    0.09161305427551

U:

    0.09087210215253

DeltaA first-order:

    +2.491106761137e-4

DeltaA finite:

    +2.490616318854e-4

### Pair (0,6)

Loss drop:

    0.03850507736206

U:

    0.03905220772997

DeltaA first-order:

    +5.953502333388e-5

DeltaA finite:

    +5.949740981931e-5

### Pair (0,4)

Loss drop:

    0.05103230476379

U:

    0.05234172577133

DeltaA first-order:

    +5.315729547187e-6

DeltaA finite:

    +5.268800508726e-6

Validation:

- sign agreement = 6/6
- Spearman write vs finite = +1.0
- utility vs loss-drop = +1.0
- positive finite DeltaA = 6/6
- immutability = PASS

---

# 29. FWR canonical theta4 regression

Pairs:

### (4,6)

Finite:

    -1.139604857752e-4

First-order:

    -1.139602339364e-4

### (4,7)

Finite:

    -3.297458161841e-5

First-order:

    -3.296647510571e-5

### (1,4)

Finite:

    -1.780266372057e-4

First-order:

    -1.780213427611e-4

Validation:

- sign agreement = 3/3
- functional-write Spearman = +1.0
- U vs loss-drop Spearman = +1.0
- archive regression = PASS
- immutability = PASS

---

# 30. FWR Cell 2 — exact multi-step causal-predictor validation

Protocol:

Start from exact prospective task-4 step10.

Forced pair:

    (3,4)

Number of steps:

    K = 20

Step size:

    eta = 0.001

At every step k:

1. recompute natural h_k / g_k on exact 715-example replay;
2. recompute mu_k from theta0 → theta_k;
3. compute first-order DeltaA;
4. apply direct SGD intervention to the forced pair;
5. measure finite DeltaA using the same pre-update h_k / g_k / mu_hat_k.

Important technical shape correction:

L1 h:

    [B, 64, 512]

Router dense gates:

    [B*64, 8]

Therefore h is flattened to:

    [B*64, 512]

with aligned gates:

    [B*64, 8]

Total aligned tokens:

    45,760

Forced patch target:

    L1 ContinualRouter

This correction was critical.

The patch must target:

    find_l1_moe(work_model).router

rather than the whole TinyMoETransformer.

---

# 31. FWR Cell 2 — 20-step result

Result:

    Spearman rho = +1.000000
    p = 0
    sign agreement = 20/20

Aggregate:

    sum first-order = -3.712977038049e-3
    sum finite = -3.714323043823e-3
    cumulative residual = -1.346005774394e-6

Residual statistics:

    mean residual = -6.730028871971e-8
    SD = 1.950945780578e-7
    mean absolute residual = 1.574394900672e-7

Relative error:

    median = 4.113310e-4
    maximum = 7.337738e-3

Prefix rank stability:

    rho = +1.0 for n = 3 through n = 20

Sub-ranges:

    early steps 0..4 = +1.0
    first 10 = +1.0
    late 10 = +1.0

Therefore:

    no early-valid / late-collapse regime observed

---

# 32. FWR Cell 2 — full per-step trajectory

Columns conceptually are:

    k, DeltaA_1st, DeltaA_finite

| k | First-order | Finite |
|---:|---:|---:|
| 0 | +2.514987718314e-4 | +2.515316009521e-4 |
| 1 | +1.391399709973e-4 | +1.387596130371e-4 |
| 2 | +5.668270750903e-5 | +5.626678466797e-5 |
| 3 | -1.006377260637e-5 | -1.001358032227e-5 |
| 4 | -6.471338565461e-5 | -6.508827209473e-5 |
| 5 | -1.130984164774e-4 | -1.130104064941e-4 |
| 6 | -1.544553379063e-4 | -1.541376113892e-4 |
| 7 | -1.899721100926e-4 | -1.902580261230e-4 |
| 8 | -2.163069875678e-4 | -2.162456512451e-4 |
| 9 | -2.415829658275e-4 | -2.413988113403e-4 |
| 10 | -2.630269154906e-4 | -2.630949020386e-4 |
| 11 | -2.818156208377e-4 | -2.819299697876e-4 |
| 12 | -2.982773003168e-4 | -2.985000610352e-4 |
| 13 | -3.098197339568e-4 | -3.098249435425e-4 |
| 14 | -3.209745045751e-4 | -3.210306167603e-4 |
| 15 | -3.288959851488e-4 | -3.287792205811e-4 |
| 16 | -3.354343061801e-4 | -3.355741500854e-4 |
| 17 | -3.384808078408e-4 | -3.385543823242e-4 |
| 18 | -3.438499697950e-4 | -3.437995910645e-4 |
| 19 | -3.495303681120e-4 | -3.496408462524e-4 |

Largest relative linearization errors:

- k=2 = 7.337738e-3
- k=4 = 5.793028e-3
- k=3 = 4.987422e-3
- k=1 = 2.733635e-3
- k=6 = 2.057077e-3

Interpretation:

FWR first-order prediction remains rank-valid and sign-valid over the whole 20-step forced trajectory.

Local linearization error stays below 1% in every step.

This establishes **multi-step predictor validity** for forced pair (3,4).

It does **not** establish continual-learning performance improvement from a trained FWR router.

---

# 33. FWR guardrails

The final Cell 2 guardrails were:

- exact step10 checkpoint = PASS
- theta0 identity = PASS
- exact replay identity = PASS
- task counts = [179,179,179,178,0]
- token alignment = PASS
- forced patch = L1 ContinualRouter
- work/probe synchronization = PASS
- observation disabled
- no optimizer
- no training
- canonical state mutation = NO

These guardrails are important because the intervention was intentionally causal and should not alter the canonical checkpoint or introduce optimizer/history effects.

---

# 34. FWR scientific interpretation

What is now supported:

1. The FWR quantity is mathematically defined.
2. It can be estimated exactly enough on the frozen diagnostic state.
3. First-order predictions agree with finite interventions.
4. This agreement survives 20 sequential direct-SGD interventions in the forced (3,4) trajectory.
5. Rank and sign validity remain stable throughout that forced trajectory.

What is still not supported:

1. that pair (3,4) is universally optimal;
2. that the FWR signal generalizes across independent T4 batches;
3. that a router following FWR improves task-level retention;
4. that one intervention width or one checkpoint trajectory is sufficient;
5. that the mechanism will survive integration with real continual-learning optimization.

Therefore:

    predictor validity = established
    causal task-level benefit = not established
    router training = not yet justified

---

# 35. Git archive history relevant to the project

Important main/research commits include:

- 50c1f40fbd00fac88da822b5ea7905a500ea3101 — Preserve CGP Task-1 fine trajectory script
- 8ebe6dbe933da079022809fb8a783a67189d9e03 — Archive post-GELU tangent diagnostics
- 6deded147cee0e561b57c5321d03e20a93492fb4 — Archive post-GELU tangent diagnostics
- 2c7826a0c4d1801c4051686b415a9d453c70ea23 — Archive post-GELU tangent diagnostics
- 17d2c1b81eaaf141bb0d55ce5c12f4263055278a — Archive post-GELU metric recovery
- ba3cdaee0122d5c543cd92f62d88cc60a7fe79ad — Finalize FARP router implementation
- 674a621e4ebc2d32d378c65ea0fd56bc3f12eeea — Archive validated research results and checkpoints
- 38f660da9d04688a76c0b1419748224b3e58ed1a — Archive FARP pre-audits and PAPA result
- a161c5a4ff220a5e95a3bb1a22bfdedf7000be03 — Archive complete experiment results and FARP diagnostics
- b073a70b5147d07f62983aa7e6530292a58d86cf — Archive FARP implementation and routing pathology results
- cb3b3a33ff3dc9ccf7cd5efe17c9e0a460223e51 — Add CRR local-PCGrad integration and seed-0 diagnostics
- ab1e8a349bc44f0163b7e51f53a2f9e2d45804d3 — archive coarse superclass interpretation result
- 14b5a74ecd8e2a0aa431d7bc525aea98f5522574 — recover milestone 6 experiments and results
- b2e811aabc23dbfcaa385e405aa9659d8376d84e — Add Test 5.3 task-supervised SupCon results
- 93a62817438ad9d962f254d4fb05eb70ed04c1fc — Add Test 5.1A attention pool diagnostic result
- fa09188f5475881deb1b8a0a6e5a8d07f8d0d2f8 — Add curated experimental results
- b10201a6604a87fe48fca3382a6cfc58dcf781e8 — Add corrected SupCon GPM results
- a71355c4b77e28c0a2d17cecfc803a6148d512f7 — Add TAIH sequential experiment files
- 138af57fb0d57104b09d9dc81472c6fdf840b409 — Disable RACE during evaluation
- 283b087e15bcebc7f33a0ea414b5cfb20a7467de — Add Fix B RACE replay gating
- 14faca9f877a0e06a0b2e2eab5ab521cfd1e6bdf — Record RACE beta per layer in manifest
- bd229dd8def3613a54267e0aa7327494a1d69dbf — Fix RACE introduction affinity write
- 5cdc8215b980bbed7aca8fec6d9592592e399f02 — Calibrate RACE beta per layer
- 73976fe81ba077b7315e1017aa9c6f7fafa7aaac — Preserve RACE research artifacts
- b341d32f7c2d27cc4e139190e88f102a365702ff — Add verified RACE router implementation
- ed4edf71d72c88854f2bdcea735e9c0c5e8f3087 — Checkpoint verified RACE router repairs
- c1e940daebf24e36b1d5b2d7ce6b08ef30e9689d — Add replicated expert-plasticity results
- 4738ae723ac441afe781c7e30644925700057e0c — Add trainable expert override
- 02b416a5df5f8949c364038f6b0439379970716b — Add margin-aware continual router

---

# 36. FWR research branch archive

Branch:

    research/checkpoint-2026-10-05-fwr

Base:

    main

Purpose:

Freeze the validated research state before the next FWR robustness/generalization experiment.

Archived branch commits:

1. 011c300ef95467875b20d3f16a637b07cdc16fd2
   docs: archive 2026-10-05 research state

2. 9b6816a65f15ecba1b938dda8fcd2be61deab8b2
   exp: archive FWR 20-step trajectory

3. aa22755a39a1c980dee4cdcdd5828f363981b9cb
   exp: archive FWR prefix correlations

4. dcee557a560f4d1a493a7be18285daa0fb9cffef
   exp: archive FWR machine-readable result

5. c1584ec8aa12c30b430a027084d8afbd54446ea1
   exp: archive FWR run record

6. 0ba7f71c7769f69fca9e13c7ffb8890d530ffda5
   exp: snapshot corrected FWR Cell 2 protocol

7. 77a31ebedafd8c8a683d792406da692497615995
   exp: archive FWR execution transcript metadata

---

# 37. Final Kaggle archive event

Local Kaggle repository:

    /kaggle/working/openmoe-router

Local final branch:

    research/final-kaggle-archive-2026-10-05

At the last verified Kaggle snapshot:

    result files in working tree = 156
    result files tracked by Git = 156
    LFS files = 31

Local final branch HEAD:

    e6a656ee09def5a653bc92d7f1d46c927012671a

The branch contains the final local archive state.

However, the push from Kaggle failed with:

    remote: Invalid username or token. Password authentication is not supported for Git operations.
    fatal: Authentication failed for 'https://github.com/shaiksameer667ss-prog/openmoe-router.git/'

The GitHub token was successfully retrieved from Kaggle Secrets before the push attempt.

Therefore:

**the final local Kaggle branch was not yet confirmed on the GitHub remote at the time of this document.**

This is a transport/authentication problem, not a scientific archive problem.

The remote GitHub repository already has push permission for the linked account.

---

# 38. Git LFS archive state

At the final Kaggle snapshot:

    31 LFS-tracked files

Representative LFS artifacts include:

- bounded_decoder fresh/random buffers
- bounded_decoder checkpoint files
- FARP anchor
- FARP task checkpoints
- replication task checkpoints
- test_5_3 task checkpoints

FARP anchor is large and must remain LFS-backed.

This is why the project should not be archived by normal Git blobs alone.

---

# 39. Nested duplicate repository issue

The Kaggle filesystem contained duplicate nested trees such as:

    openmoe-router/openmoe-router/
    openmoe-router/openmoe-router/openmoe-router/

At one stage, Git emitted:

    warning: adding embedded git repository: openmoe-router

The intended policy is:

- preserve the real outer repository;
- do not add nested duplicate repositories as submodules;
- do not rely on them as research source-of-truth;
- use exact SHA matching for checkpoints;
- keep data/caches out of the research archive.

The nested duplicate is working-tree noise, not part of the intended project archive.

---

# 40. Important archival distinction

There are three separate layers of truth:

## Layer A — source code / configuration

Tracked in GitHub:

    openmoe/
    configs/
    scripts/
    tests/
    benchmarks/
    docs/
    notebooks/

## Layer B — experiment outputs

Tracked under:

    experiments/results/

and related experiment archive directories.

## Layer C — exact binary identity

Preserved through:

- Git LFS
- SHA256 manifests
- exact checkpoint hashes
- exact replay hashes
- code snapshots
- run records
- execution transcript metadata

All three layers are needed for serious reproducibility.

---

# 41. Exact diagnostic code-path invariants

FWR diagnostics require:

- observation disabled;
- evaluation mode;
- no optimizer state;
- direct SGD only for the forced intervention;
- canonical checkpoints not mutated;
- exact replay identity checked by SHA256;
- exact token alignment.

For FWR Cell 2:

    h aligned = [45760,512]
    gates aligned = [45760,8]

The existing helper expected for the forced-pair patch:

    install_forced_pair_forward(router, pair)

The forced router target:

    find_l1_moe(work_model).router

These details should not be changed casually in replication.

---

# 42. Known source / archive manifest files

Durable documents already in the repository:

- experiments/RESEARCH_ARCHIVE_MANIFEST.txt
- experiments/RESULTS_INDEX.md
- experiments/RECOVERY_MANIFEST.md
- docs/RESEARCH_STATE_2026-10-05.md
- docs/GIT_CHECKPOINT_MANIFEST_2026-10-05.md
- docs/FINAL_KAGGLE_ARCHIVE_FILE_LIST_2026-10-05.txt
- docs/FINAL_KAGGLE_ARCHIVE_SHA256_2026-10-05.txt
- docs/FINAL_RESEARCH_CHECKPOINT_2026-10-05.md

This master archive should be treated as the human-readable map across these files.

---

# 43. Reproducibility checklist

A future researcher should be able to recover the canonical diagnostic setup by checking:

1. Split-CIFAR100 task split 0–19 / 20–39 / 40–59 / 60–79 / 80–99.
2. Exact replay SHA256.
3. Task checkpoint SHA256s.
4. Model/config exactness.
5. Router selection/gating semantics.
6. L1 target representation = norm2 output before MoE.
7. 64 tokens/image.
8. 45,760 replay tokens.
9. FWR helper / forced pair patch location.
10. observation disabled.
11. no optimizer for frozen causal diagnostics.
12. exact checkpoint immutability.
13. source result JSON/CSV/run metadata.
14. negative mechanisms retained in the archive.

---

# 44. Scientific decision tree at snapshot time

Current state:

    Canonical failure observed
        ↓
    Raw routing drift measured
        ↓
    Representation-controlled drift measured
        ↓
    Expert identity checked
        ↓
    Expert functional drift checked
        ↓
    Generic routing / representation / optimizer mechanisms tested
        ↓
    Historical retention bridge found
        ↓
    Shared population direction established
        ↓
    Online estimability partly supported, partly failed
        ↓
    Local drift tested
        ↓
    Reference-frame decomposition tested
        ↓
    Carry / innovation decomposition tested
        ↓
    FWR proposed as mechanism
        ↓
    FWR first-order screen
        ↓
    FWR finite step10 validation
        ↓
    FWR theta4 regression
        ↓
    FWR 20-step sequential validation
        ↓
    predictor validity PASS
        ↓
    task-level causal benefit still OPEN
        ↓
    ROUTER TRAINING NOT YET JUSTIFIED

---

# 45. What must happen before router training

The next experiments must answer:

### Gate 1 — pair-level robustness

Repeat FWR forced interventions for additional expert pairs.

The current multi-step validation uses:

    (3,4)

One pair is not enough.

### Gate 2 — independent batch robustness

Repeat on an independent T4 batch rather than the same fixed first128 batch.

### Gate 3 — task-level retention causality

Test whether predicted write-routing choices actually improve old-task retention.

This is the most important missing causal link.

### Gate 4 — intervention width

Test whether the effect survives different intervention magnitudes / widths.

### Gate 5 — multiple trajectory checkpoints

Verify that FWR behavior is not an artifact of one step10 → 20-step forced path.

Only after these gates are satisfied should a trainable router intervention be considered.

---

# 46. What must NOT be concluded

Do not conclude:

- “FWR is proven to improve continual learning.”
- “The correct router is known.”
- “The historical retention direction should simply be added to logits.”
- “The shared direction is sufficient as a memory state.”
- “Task-ID routing solves the problem.”
- “Raw routing drift is the main causal failure.”
- “Expert identity migration explains the failure.”
- “A single positive forced pair demonstrates generality.”

The evidence supports a narrower conclusion:

**FWR is a promising, mechanism-grounded causal predictor whose first-order estimate has survived exact finite and 20-step sequential validation, but its generalization and task-level retention benefit remain unproven.**

---

# 47. GitHub archive pointers

## Main repository

    https://github.com/shaiksameer667ss-prog/openmoe-router

## FWR validated checkpoint branch

    research/checkpoint-2026-10-05-fwr

## Master project archive branch

    research/master-project-archive-2026-10-05

This branch contains this master narrative archive as a durable GitHub document.

## Local final Kaggle archive branch

    research/final-kaggle-archive-2026-10-05

Local branch HEAD at snapshot:

    e6a656ee09def5a653bc92d7f1d46c927012671a

Its push still needs to succeed before the Kaggle filesystem is discarded.

---

# 48. Archive integrity rules

1. Never delete a failed experiment because it is negative.
2. Never replace an exact checkpoint with a different checkpoint merely because filenames match.
3. Prefer SHA256 identity over filename identity.
4. Preserve both JSON numerical results and binary checkpoints when available.
5. Use Git LFS for large checkpoints.
6. Preserve code snapshots for corrected diagnostics.
7. Preserve run records and execution transcript metadata.
8. Keep recovered-but-unavailable binaries explicitly marked as unavailable.
9. Do not fabricate lost artifacts.
10. Record scientific decisions next to the numerical evidence.
11. Keep causal interventions separate from descriptive evaluations.
12. Do not train the router before the specified FWR gates are passed.

---

# 49. Final checkpoint summary

Date:

    2026-10-05

Canonical model:

    2-layer ViT-style sparse MoE
    hidden 512
    8 experts
    Top-2
    4 heads
    FFN 1024
    64 tokens/image

Canonical continual endpoint:

    T0 1.65%
    T1 0.90%
    T2 2.35%
    T3 3.95%
    T4 45.10%
    old mean 2.2125%

Major mechanistic conclusions:

- raw routing drift is substantial;
- representation explains much of the routing drift;
- expert identity is stable;
- expert functions drift;
- generic mechanisms tested so far are closed;
- retention has a strong shared population direction;
- local innovations are smaller and unstable after the first transition;
- FWR first-order prediction is finite-step and multi-step validated;
- FWR causal task-retention benefit remains open.

FWR Cell 2:

    forced pair = (3,4)
    eta = 0.001
    K = 20
    Spearman = +1.0
    sign = 20/20
    median relative error = 4.113310e-4
    max relative error = 7.337738e-3
    cumulative residual = -1.346006e-6

Final scientific decision:

    DO NOT TRAIN ROUTER YET.

Next gate:

    FWR robustness/generalization
    +
    task-level retention causality

---

# 50. One-line project status

**OpenMoE-Router has moved from generic routing heuristics toward a functionally grounded FWR mechanism, and the predictor is now validated enough to justify causal/generalization testing—but not yet router training.**

---

# 51. Provenance statement

This archive was assembled from:

- the OpenMoE-Router GitHub research archive;
- the repository's research manifests and results index;
- the 2026-10-05 research-state and Git checkpoint documents;
- the exact Kaggle execution records for the FWR diagnostics;
- the exact checkpoint/replay SHA identities;
- the recovered scientific record from the project conversation.

Where binaries were lost during recovery, this document preserves the verified numerical record without fabricating missing files.

Where a mechanism was closed, the closure is retained as evidence rather than removed.

