# OpenMoE-Router — Mechanism Archive Manifest
Date: 2026-10-03
Branch: openmoe-router-mechanism-archive
Archive base commit: 74881fb209c6d7bc99bb97db2edd50dfb79d40e8

## Purpose

Record the post-archive experiment artifacts that were generated in the Kaggle workflow and must be preserved alongside the mechanism archive.

## Post-archive diagnostics expected

### Canonical 56C/72R
- experiments/results/diagnostics/canonical_56C72R_checkpoints/task_0.pt
- experiments/results/diagnostics/canonical_56C72R_checkpoints/task_1.pt
- experiments/results/diagnostics/canonical_56C72R_checkpoints/task_2.pt
- experiments/results/diagnostics/canonical_56C72R_checkpoints/task_3.pt
- experiments/results/diagnostics/canonical_56C72R_checkpoints/task_4.pt
- experiments/results/diagnostics/canonical_56C72R_full_training.json
- experiments/results/diagnostics/canonical_56C72R_posthoc_head_refit.json

Endpoint:
T0=1.2, T1=1.15, T2=1.4, T3=6.7, T4=43.1, old mean=2.613%.
Post-hoc head refit: old mean 2.613 -> 9.188 pp (+6.575 pp), T3 6.7 -> 12.95 pp, T4 43.1 -> 35.5 pp.

### Canonical 64C/64R reproduction
- experiments/results/diagnostics/canonical_64C64R_fresh_checkpoints/task_0.pt
- experiments/results/diagnostics/canonical_64C64R_fresh_checkpoints/task_1.pt
- experiments/results/diagnostics/canonical_64C64R_fresh_checkpoints/task_2.pt
- experiments/results/diagnostics/canonical_64C64R_fresh_checkpoints/task_3.pt
- experiments/results/diagnostics/canonical_64C64R_fresh_checkpoints/task_4.pt
- experiments/results/diagnostics/canonical_64C64R_fresh.json

Boundary metrics reproduce the archived canonical seed-0 result:
T0=1.65, T1=0.90, T2=2.35, T3=3.95, T4=45.10, old mean=2.213%.

### Prospective T3->T4, 56C/72R
- experiments/results/diagnostics/prospective_t3_t4_full_model_old_head_frozen_56C72R.json
- experiments/results/diagnostics/prospective_t3_t4_full_model_old_head_trainable_56C72R.json
- experiments/results/diagnostics/prospective_t3_t4_head_only_56C72R.json
- experiments/results/diagnostics/prospective_trainable_posthoc_head_refit_56C72R.json
- experiments/results/diagnostics/prospective_trainable_posthoc_head_refit_56C72R.pt
- experiments/results/diagnostics/prospective_trainable_linear_probe.json
- experiments/results/diagnostics/prospective_trainable_head_initialization_sweep_56C72R.json

Key endpoints:
Frozen old head: old=4.275, T3=7.45, T4=39.25%.
Trainable old head: old=3.475, T3=4.65, T4=39.85%.
Post-hoc refit of prospective trainable encoder: old=2.787, T3=3.75, T4=42.55% in the initialization sweep default/prospective-head condition.
Linear probe on prospective encoder: old mean=18.763%, T3=23.45%, T4=23.15%.

### Prospective T3->T4, 64C/64R
- experiments/results/diagnostics/prospective_64C64R_checkpoint.pt
- experiments/results/diagnostics/prospective_64C64R_then_56C72R_head_refit.json

Endpoint before refit:
old=3.038%, T3=4.65%, T4=40.3%.
After refit:
old=3.000%, T3=4.85%, T4=42.65%.

## Other post-archive trajectory/diagnostic artifacts known from the Kaggle workflow

- experiments/results/diagnostics/t3_t4_checkpoint_interpolation_oracle_refined.json
- experiments/results/diagnostics/t3_t4_old_task_ce_trajectory.json
- experiments/results/diagnostics/t3_t4_reconstructed_trajectory/checkpoints/t4_intermediate/task_4_step_010.pt
- experiments/results/diagnostics/t3_t4_reconstructed_trajectory/checkpoints/t4_intermediate/task_4_step_025.pt
- experiments/results/diagnostics/t3_t4_reconstructed_trajectory/checkpoints/t4_intermediate/task_4_step_050.pt
- experiments/results/diagnostics/t3_t4_reconstructed_trajectory/checkpoints/t4_intermediate/task_4_step_075.pt
- experiments/results/diagnostics/t3_t4_reconstructed_trajectory/checkpoints/t4_intermediate/task_4_step_100.pt
- experiments/results/diagnostics/t3_t4_reconstructed_trajectory/checkpoints/t4_intermediate/task_4_step_125.pt
- experiments/results/diagnostics/t3_t4_reconstructed_trajectory/checkpoints/t4_intermediate/task_4_step_150.pt
- experiments/results/diagnostics/t3_t4_reconstructed_trajectory/checkpoints/t4_intermediate/task_4_step_175.pt
- experiments/results/diagnostics/t3_t4_reconstructed_trajectory/checkpoints/t4_intermediate/task_4_step_200.pt

## Git preservation status

As of the base archive commit, the remote branch contains the earlier experiment archive and the restored Arm-A seed-0 checkpoints. The post-archive artifacts listed above were generated in the Kaggle workflow but are not present in the GitHub tree verified before this manifest was written.

This manifest is a preservation index; it does NOT claim that binary checkpoints or raw Kaggle result files have been uploaded when their bytes were not available to the current GitHub-writing environment.

## Important scientific status

Do not reopen closed router-only hypotheses.
Do not claim mechanism improvement from diagnostic/oracle results unless the preregistered gate actually passed.
Current strongest supported thesis remains: historical computation is lost through representation drift plus historical expert-function drift, with the prospective refit 2x2 indicating training history/procedure is a major differentiator.
