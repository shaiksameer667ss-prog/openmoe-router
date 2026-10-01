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

## Final Test 6.3 interpretation

Patch4/stride4 two-seed reference mean: 34.2250%.
Patch4/stride2 two-seed reference mean: 35.98125%.
Seed-matched mean improvement: +1.75625 pp.

The effect is positive in both seed-matched comparisons.
The experiment supports increased token-interaction capacity as a
contributor to task-ID separability.
It does not isolate attention-pair count as the sole causal variable.
