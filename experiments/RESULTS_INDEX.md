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
| **FWR B subsplit mechanistic bridge** | **Preregistered and locked; result run pending** |

## FWR evidence chain

The FWR first-order predictor was validated before mechanistic subgroup analysis: pair ranking was stable, first-order and finite changes agreed closely, and signs were consistent across the corrected audits.

Experiment A showed that the selected L1 expert parameter group E is only a small weighted share of the accumulated first-order attribution, while the broader upstream representation stack carries most of the absolute change.

Experiment B refined that result: U_block0 and U_embed were the dominant coarse groups. Their combined accumulated absolute group magnitude was 0.927650097 relative to the accumulated |D_total| denominator. This is not a statement that they account for 92.8% of signed D_total; the quantity is the preregistered weighted absolute attribution ratio.

The next mechanistic-bridge experiment therefore splits U_embed and U_block0 into patch embedding, positional embedding, block-0 normalization/attention/expert/router subgroups, and tests whether the FWR-dominant subgroup's expert-output drift aligns with the independently defined historical retention direction mu_g.

## Authoritative corrected FWR theta3 K40 interpretation

The corrected theta3 K40 run found strong association between expert-7 involvement and earlier FWR closure after adjustment for cumulative pair displacement, while marginal expert-7 enrichment itself was not supported.

Endorsed wording:

> Expert-7 involvement is strongly associated with earlier FWR closure, and this association persists after adjustment for cumulative pair displacement. Marginal expert-7 enrichment itself is not supported.

The invalid earlier K40 replay source is explicitly quarantined under PROTOCOL_DEVIATION_HG_SOURCE_T4_BATCH and is not used as evidence.

## Current locked next experiment

experiments/results/diagnostics/fwr_B_subsplit_mechanistic_bridge/ contains the locked preregistration and manifest for the mechanistic-bridge run. No mechanistic-bridge result is claimed until that run completes with all guardrails passing.
