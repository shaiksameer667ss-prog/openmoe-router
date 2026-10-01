# CRR Seed-0 Results

Archived from the seed-0 CRR diagnostic and held-out task-ID probe.

## Endpoint coordinates

These are stored accuracy-matrix endpoint quantities:

| Arm | Old T0-T3 mean | T4 |
|---|---:|---:|
| B — whole-model PCGrad | 3.8625% | 37.7500% |
| C — current-route local expert PCGrad | 5.4125% | 17.1000% |

## Projection diagnostics

### Arm B

- CRR history rows: 800
- Projection rate: mean 0.9455; median 0.9663; p10 0.8652; p90 1.0000
- Projected-away norm: mean 8.89661; median 8.74138; p90 13.7726
- Per-step cosine mean: mean -0.3801; median -0.3896; p10 -0.5049; p90 -0.2484
- Negative cosine fraction: `<0` = 0.998; `<-0.1` = 0.993; `<-0.3` = 0.761
- Per-step cosine minimum: mean -0.8862; median -0.9129; p10 -0.9809; p90 -0.7605
- Negative minimum fraction: `<0` = 1.000; `<-0.1` = 1.000; `<-0.3` = 0.999
- Gradient norm ratio was not logged.

### Arm C

- CRR history rows: 800
- Projection rate: mean 0.9152; median 0.9375; p10 0.7969; p90 1.0000
- Projected-away norm: mean 2.40806; median 2.21772; p90 3.69355
- Cross-expert replay cosine: mean 0.1039; median 0.1089
- Local-vs-global relative update: mean 0.6694; median 0.6600
- Zero-local replay rate: mean 0.1512
- Historical coverage: mean 0.6365
- Gradient norm ratio was not logged.

## Held-out 4-way task-ID probe

Final feature protocol: all 50K training samples used to fit a LogisticRegression
decoder; held-out evaluation uses T0-T3 test samples.

| Arm | Task-ID accuracy | Chance |
|---|---:|---:|
| B | 34.9000% | 25.0000% |
| C | 33.6750% | 25.0000% |

C - B = -1.2250 pp.

## Interpretation boundary

The task-ID probe measures linear recoverability of task identity from the final
representation. It is not an old-task end-to-end decision-realization metric and
should not be conflated with realization-frontier coordinates.

The seed-0 probe therefore shows no increase in final-representation task-ID
separability from Arm C relative to Arm B.
