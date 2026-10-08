# RM2a — Offline Expert Risk Head Training and Evaluation

Protocol ID: FWR-RM2a-RISK-HEAD-v1.0
Status: preregistered before execution
Purpose: Train and evaluate a risk head offline on the saved RM1d expert-level counterfactual margin matrix.

Training target:
Delta margin(a | x) = m(x; a) - m0(x)
where positive values mean the counterfactual expert improves the old-vs-new margin.

No base-model training.
No checkpoint modification.
No replay modification.
No optimizer state from the base model.
No new model checkpoint writes.

## Data

Source artifact:
experiments/results/RM1d/rm1d_counterfactuals_F_E.npz

Expected SHA-256:
c8a9a768dbe3569275cc72ce437d8e9d1a8bf8aa5adab462bb4d01b37b984408

Rows:
48,000 per checkpoint = 8,000 examples × 6 alternative experts.

Checkpoints:
- F
- E-BOTH v4

Primary target:
Y_res = Delta margin(a | x), residualized by source task and source class using the same joint-group residualization convention used in RM1d.

The RM1d raw counterfactual matrix is treated as the fixed source dataset. No additional forward passes are required for RM2a.

## Inputs

For all models use the same locked expert identity representation:
expert embedding dimension d_emb = 8
representation = one-hot expert identity

A1:
h_L1(x) concatenated with one-hot(expert)
input dimension = 512 + 8 = 520
model = linear ridge/linear regression head

A2:
h_L1(x) concatenated with one-hot(expert)
input dimension = 520
model = MLP 512 -> 256 -> 1 with expert embedding concatenated to h
hidden activation = ReLU
single scalar output
No architecture tuning.

A3:
factorized per-expert linear head
eight separate linear maps from h (512 -> 1)
Equivalent output form:
R_e(h) = w_e^T h + b_e
No shared expert embedding parameters.
No architecture tuning.

All three models are trained independently for each checkpoint.

## Split

5-fold cross-validation
seed = 0
identical folds across A1, A2, and A3
all 48,000 rows for each checkpoint
no grouping changes after seeing results

The held-out fold is never used for parameter fitting, model selection, calibration fitting, or threshold selection.

Primary model comparison:
best held-out CV R2 among A1/A2/A3.

## Reported metrics

Per model and checkpoint report:

1. Held-out CV R2
2. Held-out sign accuracy:
   sign(predicted Delta margin) == sign(observed Delta margin)
3. MAE in margin units
4. Calibration:
   10 equal-frequency prediction bins using held-out predictions only;
   report mean predicted Delta margin and mean observed Delta margin per bin.

Also report:
- mean and std of held-out fold R2
- mean and std of held-out fold sign accuracy
- mean and std of held-out fold MAE
- aggregate held-out predictions for reproducibility

## Locked gates

Gate 1:
best-model held-out CV R2 >= 0.05 at F
AND
best-model held-out CV R2 >= 0.05 at E-BOTH

Gate 2:
best-model held-out sign accuracy >= 0.55 at F
AND
best-model held-out sign accuracy >= 0.55 at E-BOTH

Gate 3:
A1 does not outperform both A2 and A3 by more than 0.01 in held-out CV R2.

Interpretation:

Gate 1 AND Gate 2 pass:
=> RM2b is justified.

Gate 1 fails:
=> RM1d was an in-sample/CV signal that does not meet the preregistered held-out training threshold; do not proceed to controller training.

Gate 2 fails:
=> magnitude is learnable but directional sign is not sufficiently reliable; do not use the head as an inference-time controller.

Gate 3 is descriptive/model-selection guidance:
if A1 beats A2 and A3 by >0.01, treat the relationship as sufficiently linear and avoid unnecessary MLP/factorized complexity in RM2b.

## Sign convention

Delta margin = counterfactual margin - original margin.

Positive = counterfactual expert helps.
Negative = counterfactual expert hurts.

Do not flip the sign.

## Reproducibility requirements

Before fitting:
- verify RM1d NPZ SHA-256
- record Git HEAD
- record Python, NumPy, PyTorch versions
- record CPU/GPU environment
- record exact fold indices or a deterministic hash of them
- record model hyperparameters
- record random seeds

After fitting:
- save raw held-out predictions for every fold/model/checkpoint
- save fold metrics
- save aggregate metrics
- save calibration tables
- save a human-readable summary
- save stdout
- save provenance metadata

No post-hoc tuning of:
- expert embedding size
- hidden width
- activation
- regularization
- fold count
- seed
- gate thresholds
- decision thresholds

## Expected artifacts

experiments/results/RM2a/rm2a_results.json
experiments/results/RM2a/rm2a_summary.md
experiments/results/RM2a/rm2a_predictions.npz
experiments/results/RM2a/rm2a_metadata.json
experiments/results/RM2a/rm2a_stdout.log

No checkpoint or replay artifact may be modified.
