# FWR Evidence Ledger — 2026-10-06

## Scope

This ledger records the validated FWR diagnostic chain already committed to main before the next expensive mechanistic-bridge run. It is a provenance/index document, not a replacement for the underlying CSV/JSON artifacts.

## 1. Corrected FWR predictor validation

### Step-10 pair audit
- Pair: experts (3,4)
- K=20
- Correct replay source
- Spearman: 1.0
- Sign agreement: 20/20
- Cumulative first-order change: -3.712977e-3
- Cumulative finite change: -3.714323e-3
- Residual: -1.346006e-6

Artifact directory:
experiments/results/diagnostics/fwr_multistep_step10_pair34/

### Router-trainable corrected audit
- Pair: experts (3,4)
- K=20
- Spearman: 0.999624
- Sign agreement: 20/20

Artifact directory:
experiments/results/diagnostics/fwr_multistep_step10_pair34_batch2_router_trainable/

## 2. Cell 4 — all-pair local validation

- 28 unordered expert pairs
- 2 exact T4 batches
- K=5
- Pair-ranking Spearman: 1.0 for every tested cell
- Cross-batch predicted/finite Spearman: 0.980296
- Sign agreement: 28/28
- Baseline positive counts at step 10: batch1 6/28; batch2 5/28

Artifact directory:
experiments/results/diagnostics/fwr_cell4_pair_ranking_28pairs_2batches_k5/

## 3. Design B — task-boundary reopening

The theta3 boundary state was positive for all 28 pairs. Along the natural theta3 trajectory the effect contracted sharply by step 10 and was fully closed by step 25.

The theta3 cross-boundary Spearman values were in the observed range 0.875205–0.978654.

Artifact directory:
experiments/results/diagnostics/fwr_design_B_task_boundary_reopening/

## 4. Corrected theta3 K20 closure

- batch1: 8/28 crossed by K20; 20/28 censored
- batch2: 7/28 crossed by K20; 21/28 censored
- Positive pair counts: k0 28/28; k1 28/28; k5 28/28; k10 28/28
- Predictor QA minimum Spearman: 0.999453
- Predictor sign agreement: 28/28

Artifact directory:
experiments/results/diagnostics/fwr_theta3_k20_closure/

## 5. Corrected theta3 K40 closure and expert-7 analysis

The corrected run used the exact replay H/G source and T4 loss contract.

- batch1: 26/28 crossed
- batch2: 24/28 crossed
- pooled censoring: 6/56 = 10.71%
- Model A contains_7: +3.6843413883, CI [2.507772595, 4.860910182], p=8.38367395e-10
- Model B cumulative pair displacement: coefficient -131.060996723, p=0.0832103376
- Model C contains_7: +3.72651877038, CI [2.566523895, 4.886513646], p=3.04551561e-10
- Model C pair displacement: -170.922590143, CI [-330.012513294, -11.832666994], p=0.035226886
- Marginal expert-7 enrichment did not pass the preregistered enrichment check in either batch.
- The invalid earlier K40 run is quarantined and excluded.

Authoritative artifact directory:
experiments/results/diagnostics/fwr_theta3_k40_closure_replay_corrected/

## 6. theta1 / theta2 K40

The corrected theta1/theta2 K40 runs were not statistically interpretable under the locked power floor.

- theta1: 0/28 crossed in both batches; 28/28 censored
- theta2: 3/28 crossed in both batches; 25/28 censored
- Interaction not estimable
- Decision: NOT_INTERPRETABLE

Artifact directories:
- experiments/results/diagnostics/fwr_theta1_k40_closure_replay_corrected/
- experiments/results/diagnostics/fwr_theta12_k40_closure_replay_corrected/
- experiments/results/diagnostics/fwr_theta2_k40_closure_replay_corrected/

## 7. Experiment A — parameter-subspace attribution

Experiment A was a complete audit at the coarse parameter-group level.

- Rows: 2240
- 28 pairs
- 2 batches
- 40 steps
- Coverage: PASS
- Completeness: PASS
- JSON reopen: PASS
- Mean |E|/|D_total|: 0.141331677
- Weighted |E|/|D_total|: 0.023977860
- Sum |E|: 1.210744107
- Sum |B|: 48.84073960
- Sum |R|: 1.491622407
- Sum |D_total|: 50.49425273
- Fixed-frame E QA: sign 2240/2240
- Global Spearman: total vs E 0.740143258; total vs B 0.997371471; total vs R 0.483864907

Interpretation: E is a small local first-order component; the coarse complement is dominated by upstream/shared parameters, but that coarse B group is too broad to identify the responsible mechanism.

## 8. Experiment B — coarse subgroup attribution

Weighted absolute attribution:
- E: 0.023977849
- U_embed: 0.408798353
- U_block0: 0.518851744
- U_block1_pre: 0.063374644
- R_L0: 0.005691324
- R_L1: 0.029149336

Dominant absolute-count:
- U_block0: 1304/2240 = 58.2%
- U_embed: 847/2240 = 37.8%
- R_L1: 74/2240
- E: 13/2240
- U_block1_pre: 2/2240

Important reporting constraint:
U_block0 + U_embed = 0.927650097 is an accumulated weighted-absolute attribution ratio relative to the accumulated |D_total| denominator. It must not be described as 92.8% of signed D_total.

## 9. Next mechanistic bridge

The locked preregistration splits the coarse upstream groups into E_patch, E_pos, B0_norm1, B0_attn, B0_norm2, B0_experts, R_L0, U_block1_pre, E, and R_L1, plus structural-zero audits Z_nonselected_L1_expert and Z_downstream.

For upstream subgroups, each counterfactual is applied only to a shadow model and followed by a full replay recomputation to obtain H'_X. The natural pre-update G_k remains the weighting measure for every subgroup. R_L1 is structural-zero for the expert-output bridge and is recorded as NOT_EVALUABLE.

The preregistered alignment threshold is: aligned cos >= +0.50; weakly aligned +0.20 <= cos < +0.50; orthogonal |cos| < 0.20; anti-aligned cos <= -0.20.

No causal claim is made from alignment alone.

Preregistration:
experiments/results/diagnostics/fwr_B_subsplit_mechanistic_bridge/fwr_B_subsplit_prereg.json

Manifest:
experiments/results/diagnostics/fwr_B_subsplit_mechanistic_bridge/fwr_B_subsplit_manifest.json

Status:
LOCKED / RESULT PENDING

## Repository provenance

All entries above refer to the locked repository line ending at commit 392940ea2f4352e47978521d7ba7169df25e5885.