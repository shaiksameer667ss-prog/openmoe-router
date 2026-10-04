# GWDA L1 covariance alignment — exact 715 replay

Date: 2026-10-04  
Branch: `archive-all-experiments-final-2026-10-04`

## Purpose

Final estimator-consistent diagnostic for whether the non-trivial L1 gate–drift covariance component aligns with the independently fitted historical-retention vector.

## Locked estimator

- canonical replay: 715 examples
- task counts: T0=179, T1=179, T2=179, T3=178
- leave-one-task-out by held-out replay task
- ridge alpha = 1.0
- rank = 4
- Top-K = 2
- uncentered target
- target: `d_g(h) = sum_e g_e(h) delta_e(h)`
- fit microbatch: 4 images
- current model on CUDA; historical model on CPU for memory-safe execution

## Estimator consistency

| Fold | Rerun ||b_fixed|| | Archived |
|---|---:|---:|
| T0 | 8.221852303 | 8.221852 |
| T1 | 8.326368443 | 8.326368 |
| T2 | 8.224108014 | 8.224108 |
| T3 | 8.083627402 | 8.083628 |

All differences are below 1e-6. The exact-715 estimator is therefore reproduced.

## Decisive alignments

| Fold | cos(c_gdelta,b_fixed) | cos(mu_marg,b_fixed) | cos(mu_g,b_fixed) |
|---|---:|---:|---:|
| T0 | -0.049602 | +0.895562 | +0.930849 |
| T1 | -0.047233 | +0.909445 | +0.946151 |
| T2 | -0.023936 | +0.891417 | +0.933613 |
| T3 | -0.020704 | +0.878102 | +0.920467 |
| Mean | **-0.035369** | **+0.893632** | **+0.932770** |

The covariance norm ratio is:

`||c_gdelta|| / ||mu_g|| = 0.278019034`

The decomposition reconstructs exactly (reported reconstruction error: 0.000e+00).

## Scientific decision

The covariance component is non-trivial geometrically but is not aligned with the retention vector. The marginal gate-exposure × marginal historical-drift component is strongly aligned, as is total `mu_g`.

**Decision: covariance-aware router / GDCR branch CLOSED. No router training is justified from this hypothesis.**

The earlier ~9.00-norm bridge artifact is not the reference estimator for this final diagnostic; the exact-715 LOO estimator above is the locked target.

## Runtime

Fit pass completed in approximately 35.2 s with the historical model staged on CPU and current model on CUDA.
