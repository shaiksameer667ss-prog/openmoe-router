# Experiment C — Natural Retention Map

Protocol: FWR-C-DIRECT-NATURAL-RETENTION-MAP v1.0

Primary endpoint:
mean across checkpoints of max(I_B0_attn,0) / sum_Y max(I_Y,0).

Primary result:
- mean H_B0_attn = 0.44038869170896877
- classification = PRIMARY_RETENTION_PATHWAY
- B0_attn ranked first at k=10,25,50,100,150,200
- E_patch ranked second with mean H = 0.333178
- U_block1_pre = 0.115221

Finite validation:
- 33/36 raw rows sign agreement = 91.67%
- three raw sign disagreements were recorded at k=10 B0_attn old-task2, k=200 B0_attn old-task1, and k=200 E_patch old-task2.

Interpretation:
Experiment C identifies B0_attn as the primary predicted retention pathway, with E_patch second. This is an attribution/retention-map result, not a causal intervention result.
