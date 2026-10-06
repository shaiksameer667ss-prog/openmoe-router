# OpenMoE-Router — Results Index

| Experiment | Result |
|---|---|
| No replay | 8.083% final accuracy; 36.588 pp forgetting |
| Replay 715 | 715 examples; 8,797,360 B |
| Test 5.3 SupCon | 34.1375% held-out task-ID |
| Test 6.1 Depth 6 | 34.11–35.20%; no depth effect |
| Test 6.2 Patch 8 | 32.4875%; -1.55 pp vs patch4 seed0 |
| Test 6.3 stride2 seed0 | 36.4375% |
| Test 6.3 stride2 seed1 | 35.5250% |
| Test 6.3 patch4 seed1 | 34.4125% |
| Test 6.3 paired mean effect | +1.75625 pp reference |
| Test 6.3 paired strict4 effect | +2.06250 pp |
| Fresh-random 715 kNN | 24.41–24.55%; chance 25% |
| **FWR corrected Step-10 pair audit** | **Spearman 1.0; sign 20/20; finite-vs-first-order agreement** |
| **FWR corrected router-trainable audit** | **rho 0.999624; sign 20/20** |
| **FWR Cell 4 — 28 pairs × 2 batches × K5** | **pair-ranking rho 1.0; cross-batch rho 0.980296; sign 28/28** |
| **FWR Design B — theta1/theta2/theta3** | **theta3 boundary 28/28 positive; natural step10 6/28 (batch1), 5/28 (batch2); step25 0/28** |
| **FWR theta3 K20 corrected** | **8/28 and 7/28 closures by batch; K20 censoring 20/28 and 21/28; predictor QA passed** |
| **FWR theta3 K40 corrected** | **26/28 and 24/28 crossed; pooled censoring 6/56; strong expert-7 association; marginal expert-7 enrichment unsupported** |
| **FWR theta1/theta2 K40 corrected** | **theta1 0/28 crossed; theta2 3/28; both below power floor and NOT_INTERPRETABLE** |
| **FWR Experiment A** | **E mean local ratio 0.1413, weighted 0.0240; broad upstream components dominate** |
| **FWR Experiment B** | **U_block0 weighted 0.5189; U_embed 0.4088; combined accumulated absolute group magnitude 0.9277 relative to accumulated |D_T| denominator** |
| **FWR B subsplit mechanistic bridge** | **B0_attn weighted_abs_ratio 0.569560; signed_share +0.592406; mean_cos -0.039482; all 2240 evaluable rows orthogonal** |
| **FWR Experiment C — natural retention map** | **mean H_B0_attn 0.440389; B0_attn ranked #1 at every checkpoint; E_patch #2 at 0.333178** |
| **Experiment D — B0_attn freeze** | **31.1625 pp forgetting; Δforget -0.875 pp; T4 43.25%; INCONCLUSIVE** |
| **Experiment E — E_patch freeze** | **30.5500 pp forgetting; Δforget -1.4875 pp; T4 43.10%; INCONCLUSIVE** |

## FWR evidence chain

The FWR first-order predictor was validated before mechanistic subgroup analysis: pair ranking was stable, first-order and finite changes agreed closely, and signs were consistent across the corrected audits.

Experiment A showed that the selected L1 expert parameter group E is only a small weighted share of the accumulated first-order attribution, while the broader upstream representation stack carries most of the absolute change.

Experiment B refined that result: U_block0 and U_embed were the dominant coarse groups. Their combined accumulated absolute group magnitude was 0.927650097 relative to the accumulated |D_total| denominator. This is not a statement that they account for 92.8% of signed D_total; the quantity is the preregistered weighted absolute attribution ratio.

The mechanistic-bridge experiment then split the dominant upstream groups into patch embedding, positional embedding, block-0 normalization/attention/expert/router subgroups and tested alignment with the independently defined historical retention direction mu_g. B0_attn dominated the bridge magnitude, but mean cosine alignment with mu_g was slightly negative and all evaluable rows were classified as orthogonal.

## Authoritative corrected FWR theta3 K40 interpretation

The corrected theta3 K40 run found strong association between expert-7 involvement and earlier FWR closure after adjustment for cumulative pair displacement, while marginal expert-7 enrichment itself was not supported.

Endorsed wording:

> Expert-7 involvement is strongly associated with earlier FWR closure, and this association persists after adjustment for cumulative pair displacement. Marginal expert-7 enrichment itself is not supported.

The invalid earlier K40 replay source is explicitly quarantined under PROTOCOL_DEVIATION_HG_SOURCE_T4_BATCH and is not used as evidence.

## Experiment C — retention-map result

Protocol: FWR-C-DIRECT-NATURAL-RETENTION-MAP v1.0.

Primary endpoint:
mean across checkpoints of max(I_B0_attn, 0) / sum_Y max(I_Y, 0).

Primary result:
mean H_B0_attn = 0.44038869170896877.

Classification:
PRIMARY_RETENTION_PATHWAY.

Mean H leaders:
B0_attn 0.440389; E_patch 0.333178; U_block1_pre 0.115221; head_new_rows 0.051462; B0_experts 0.020248.

B0_attn was the leader at k=10,25,50,100,150,200.

Finite validation sign agreement:
33/36 raw rows = 91.67%.

## Experiment D — causal intervention

Intervention:
freeze B0 attention tensors:
blocks.0.attn.in_proj_weight,
blocks.0.attn.in_proj_bias,
blocks.0.attn.out_proj.weight,
blocks.0.attn.out_proj.bias.

Arm-A' forgetting:
32.0375 pp.

D forgetting:
31.1625 pp.

Delta forgetting:
-0.875 pp.

T4 accuracy:
43.25%.

Plasticity cost:
1.85 pp.

Preregistered verdict:
INCONCLUSIVE.

## Experiment E — causal intervention

Intervention:
freeze only:
patch_embed.weight,
patch_embed.bias.

The D reconstructed T3 live bundle was reused without reconstruction.

Bundle SHA256:
7f5af253a4b4f6e4f3a5587a3cb5e116bf9dc923a2611b27c1594e6b965f7d13.

Canonical theta3 SHA256:
26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc.

E forgetting:
30.5500 pp.

Arm-A' reference:
32.0375 pp.

Delta forgetting:
-1.4875 pp.

T4 accuracy:
43.10%.

Final T0..T4:
3.00%, 2.50%, 3.50%, 5.80%, 43.10%.

Plasticity cost:
2.00 pp.

Preregistered verdict:
INCONCLUSIVE.

## Overall D/E interpretation

Experiment C identified B0_attn as the primary predicted retention pathway and E_patch as the second-ranked pathway. Neither full-scope freeze intervention crossed the preregistered 2 pp causal-support threshold.

This does not establish that either group is irrelevant. It establishes that the tested intervention strength did not demonstrate the preregistered level of aggregate causal leverage.

D and E are closed. No D/E reruns are authorized.

## Recovery provenance

The 2026-10-07 recovery was performed after loss of the Kaggle runtime caused by a power interruption. The original Kaggle run artifacts are not assumed to be recoverable from the lost runtime. Numeric E/D results in this recovery record are copied from the completed run output preserved in the conversation; derived per-task forgetting values are computed from the locked historical best accuracies and the reported final accuracies.

Recovered source hashes retained:
- D reconstructed T3 live bundle: 7f5af253a4b4f6e4f3a5587a3cb5e116bf9dc923a2611b27c1594e6b965f7d13
- canonical theta3: 26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc

The large binary T3 bundle is not recreated or substituted here.
