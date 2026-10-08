# RM2b-pilot — T3-Calibrated Risk-Aware Router Pilot

Protocol ID: RM2b-RISK-CONTROLLER-PILOT-v1.0
Status: locked before RM2b-pilot execution
Date locked: 2026-10-08

## Purpose

Test whether a frozen, expert-conditional counterfactual margin predictor trained on
the T3 representation regime can causally improve retention during T4 continual
training.

RM2b-pilot is a controller-operation pilot, not the five-seed primary experiment.
It is executed only once at seed 0 with four paired arms.

## Scientific question

Does acting on a learned estimate of the counterfactual change in old-vs-new
margin reduce forgetting relative to the unchanged router, while shuffled and
sign-reversed risk signals fail to reproduce the improvement?

A positive counterfactual margin delta means the alternative expert improves the
old-vs-new margin.

## Prerequisites

1. Reconstruct T3 from the canonical same-bundle training procedure used for the
   F/E-BOTH work. The reconstruction must be fresh; do not reuse a previously
   generated T3 bundle.
2. Save a complete T3 continuation bundle containing:
   - model state
   - optimizer state
   - continual-router non-gradient state (routing bias and memory prototypes/counts)
   - replay state, when replay is part of the canonical protocol
   - training seed and provenance
3. Collect a new RM1d-style expert counterfactual matrix at the T3 checkpoint.
4. Train and freeze the T3 A3 risk head from that matrix before any RM2b T4
   intervention run.
5. All four pilot arms must start from byte-identical copies of the same T3
   continuation bundle and use matched seed 0.

No T3 checkpoint, replay buffer, or optimizer state may be modified in place.

## Canonical model and stream

- Dataset: Split-CIFAR-100
- Stream: 5 tasks x 20 classes
- Model: TinyMoETransformer
- hidden_dim = 512
- ff_dim = 1024
- num_experts = 8
- top_k = 2
- num_heads = 4
- image_size = 32
- patch_size = 4
- depth = 2
- Router: continual
- Use the exact canonical router/training configuration used for the F/E same-bundle
  experiment, including optimizer, learning-rate, task-step count, replay policy,
  data loader construction, task order, and task-boundary procedure.
- T4 pilot length: exactly 200 optimization steps.

The canonical configuration file is:
configs/cifar100_milestone6_512.yaml

## T3 counterfactual collection

Use the same sampling and counterfactual operationalization as RM1d:

- 8,000 source examples total
- 2,000 examples from each source task T0, T1, T2, T3
- final MoE block only
- example-level representative route uses slot-1 expert e1 and distinct slot-2
  expert e2
- six remaining experts are candidate slot-2 replacements
- candidate is evaluated only where RM1d eligibility is satisfied
- e1 remains fixed
- only slot 2 is counterfactually replaced
- original slot-1/slot-2 gate masses are retained for the counterfactual effect
- positive delta means the replacement improves the margin

Save the raw T3 counterfactual matrix before any fitting.

## A3 target and fitting

Use the raw signed counterfactual target:

    Y_raw = Delta margin = counterfactual margin - original margin

Do not residualize by task or class.

A3 is the locked factorized expert-specific linear model:

    R_e(h) = w_e^T h + b_e

There are eight independent linear maps from the 512-dimensional pooled hidden
representation to one scalar, one per expert.

### T3 held-out QA

Before fitting the final deployment head, perform 5-fold grouped CV:

- seed = 0
- split unit = base example
- 6,400 training examples / 1,600 held-out examples per fold
- all six candidate rows from a base example inherit its fold
- no test-fold row is used for fitting
- no calibration or threshold fitting uses the held-out fold

Report aggregate OOF:
- R2
- sign accuracy
- MAE
- fold mean/std for all three metrics

This held-out R2 is an operation QA measure for the T3-deployed head.

After QA, fit the final A3 head once on all eligible T3 rows and freeze it.
Store its coefficient matrices and the training-distribution mean/std used for
risk-score standardization.

## Risk score

For each candidate expert e, compute the frozen A3 prediction R_e(h).

For the controller only, standardize using the fixed all-T3 deployment-head
prediction distribution:

    z_e = (R_e(h) - mu_T3) / sigma_T3

The intervention strength is locked:

    alpha = 0.25

No alpha tuning is allowed during the pilot or full experiment.

## RM2b routing intervention

The intervention is deliberately constrained to remain close to the RM1d
counterfactual:

1. Run the unchanged continual router to obtain baseline selection scores,
   baseline Top-2 {e1,e2}, and baseline gate masses.
2. Keep e1 fixed.
3. Form the candidate set {e2} union eligible outsider candidates.
4. For each outsider candidate c:

       S'_c = S_c + alpha * z_c

   where S_c is the router's normal selection score (including its normal
   memory affinity and routing-bias terms).
5. Keep e2's score unmodified.
6. Select:

       e2' = argmax_{c in {e2} union eligible candidates} S'_c

7. The final route is {e1,e2'}.
8. If e2'=e2, routing is unchanged.
9. If e2' differs from e2, this is a single slot-2 replacement. No slot-1 replacement
   is allowed, and no step may replace both original experts.
10. For a changed route, preserve the original two gate masses rather than
    recomputing gates from the candidate's raw logit. This isolates expert identity
    as the intervention variable and matches the RM1d counterfactual convention.
11. If no eligible outsider exists, keep the original route.

This means the controller can change at most one expert per token and cannot alter
the original slot-1 expert.

## Four pilot arms

A. BASELINE
   Normal continual router. No risk adjustment.

B. RISK_AWARE
   Use the positive A3 risk score exactly as specified above.

C. SHUFFLED_RISK
   At each routed token, independently permute the six candidate risk scores across
   candidate expert identities using the locked deterministic random generator:
   seed = 7000 + arm_seed.
   The permutation is applied before the alpha adjustment.
   All other operations are identical to RISK_AWARE.

D. SIGN_REVERSED
   Use the negative of the locked A3 standardized risk score:

       z_reversed = -z

   with the same alpha = 0.25.

No arm-specific tuning is allowed.

## Load balancing

No new balancing loss or balancing penalty is introduced.

The existing continual router's native routing-bias update remains unchanged.

Measure expert utilization during T4 for every arm:
- per-expert route share
- maximum expert route share
- dead-expert count
- routing entropy
- token-level route-change rate
- example-level route-change rate

A collapse is a result, not a reason to modify the intervention during the pilot.

## Primary endpoint

For old tasks T0-T3, define T3-to-T4 forgetting in percentage points as:

    F = mean_i [ Acc_T3(i) - Acc_T4(i) ], i in {T0,T1,T2,T3}

where Acc_T3(i) is measured immediately before T4 training and Acc_T4(i) is
measured after exactly 200 T4 optimization steps.

For each arm:

    DeltaF_arm = F_arm - F_BASELINE

Lower forgetting is better.

Because every arm begins from the same T3 state, differences are paired within
seed 0 and arise from the T4 controller intervention.

## Pilot diagnostics and gates

### Operation checks

The pilot is operationally valid only if:
- the four T3 starting bundles are byte-identical
- all T4 task batches are identical across arms except for the controller arm
- the baseline arm reproduces the saved T3->T4 baseline trajectory under the same seed
- RM2b controller produces deterministic output under repeated evaluation of the same
  T3 batch
- the A3 prediction tensor is finite

### Required pilot outputs

1. T3 A3 held-out OOF R2, sign accuracy, and MAE.
2. Example-level route-change rate for RISK_AWARE.
3. Token-level route-change rate for RISK_AWARE.
4. Per-expert route share at T4 step 200 for all arms.
5. T3->T4 forgetting F for all four arms.
6. DeltaF relative to BASELINE for all four arms.
7. Complete accuracy matrix for T3 pre-boundary and T4 final evaluation.

### Pilot progression rule

Proceed to RM2b-full only when:
- A3 held-out R2 is finite and > 0;
- RISK_AWARE has non-zero example-level route-change rate;
- all four arms complete 200 T4 steps without divergence or implementation failure.

The pilot itself does NOT apply G1-G5 as an efficacy decision because n=1 is not an
adequate estimate of the five-seed mean.

If the controller is inactive (zero route change), or A3 is degenerate, stop and
debug/revise in a separately numbered follow-up before any five-seed efficacy run.

No hyperparameter may be changed inside this pilot based on observed forgetting.

## Full-experiment gates, locked here before the pilot

These gates apply to the later five-seed RM2b-full experiment and are not tuned from
pilot results.

G1:
    mean_F(RISK_AWARE) <= mean_F(BASELINE) - 2.0 pp

G2:
    mean_F(SHUFFLED_RISK) >= mean_F(RISK_AWARE) + 1.5 pp

G3:
    mean_F(SIGN_REVERSED) >= mean_F(RISK_AWARE) + 1.5 pp

G4:
    median example-level route-change rate for RISK_AWARE is in [0.05, 0.50]

G5:
    best-expert route share for RISK_AWARE <= 0.60 at T4 step 200

Decision:
    G1 AND G2 AND G3 -> mechanism-level result; RM2c justified
    G1 but not G2/G3 -> retention improvement is not attributable specifically
                       to the correctly signed risk signal; report negative
    NOT G1 -> no retention improvement at the locked intervention strength;
              any alpha sweep must be a separately preregistered follow-up

## Prohibitions

No post-hoc changes to:
- alpha
- A3 architecture
- training target
- T3 sample count
- CV split rule
- arm definitions
- task order
- T4 step count
- forgetting definition
- progression gates
- load-balancing objective

No T4 evaluation outcome may be used to refit, recalibrate, or select the deployed
risk head.

## Provenance and artifacts

Before execution:
- record Git HEAD
- record Python, NumPy, PyTorch, CUDA, GPU
- hash the T3 reconstruction source and T3 checkpoint bundle
- hash the T3 counterfactual NPZ
- hash the final frozen A3 coefficients
- save the exact fold assignments and RNG seeds

Save:
experiments/results/RM2b_pilot/rm2b_pilot_results.json
experiments/results/RM2b_pilot/rm2b_pilot_summary.md
experiments/results/RM2b_pilot/rm2b_pilot_metadata.json
experiments/results/RM2b_pilot/rm2b_pilot_stdout.log
experiments/results/RM2b_pilot/t3_counterfactuals.npz
experiments/results/RM2b_pilot/t3_a3_state.npz
experiments/results/RM2b_pilot/route_diagnostics.npz

Do not overwrite any prior RM1d or RM2a artifact.

## Dependency record

RM1d:
- raw counterfactual SHA-256:
  c8a9a768dbe3569275cc72ce437d8e9d1a8bf8aa5adab462bb4d01b37b984408
- decision: CASE_1_PER_EXAMPLE_SIGNAL

RM2a:
- decision: RM2B_JUSTIFIED
- A3 held-out R2:
  F = 0.207272
  E-BOTH = 0.179516

These prior results justify running RM2b but are not used as RM2b outcome evidence.
