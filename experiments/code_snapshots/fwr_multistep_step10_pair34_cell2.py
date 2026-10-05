"""
FWR multi-step Cell 2 reproducibility snapshot.

Protocol:
- exact prospective task-4 step10 checkpoint
- exact theta0 checkpoint
- exact 715-example replay
- forced pair (3,4)
- 20 sequential direct SGD steps, eta=1e-3
- natural h/g recomputed at every step
- first-order DeltaA = -eta <grad q, grad L>
- finite DeltaA evaluated on the same pre-update h/g/mu_hat
- router observation disabled
- eval mode
- no optimizer state

Critical shape correction:
    L1 h from TinyMoETransformer is [B,64,512].
    Router dense gates are [B*64,8].
    Flatten h to [B*64,512] before combining.

Critical patch target:
    find_l1_moe(model).router
not TinyMoETransformer itself.

Core algorithm:

for k in range(20):
    H_k, G_k = collect_natural_h_g(work_model, replay_715)
    mu_k = E_H[sum_e G_k[e] * (E_theta0,e(H_k) - E_thetak,e(H_k))]
    mu_hat_k = mu_k / ||mu_k||

    q_k = E_H[
        sum_{e in (3,4)}
        G_k[e] * <E_thetak,e(H_k), mu_hat_k>
    ]

    DeltaA_1st_k = -eta * <grad(q_k), grad(L_forced_pair_k)>

    A_before = E_H[
        sum_{e in (3,4)}
        G_k[e] * <E_thetak,e(H_k), mu_hat_k>
    ]

    theta_{k+1} = theta_k - eta * grad(L_k^(S))

    A_after = same expression, but with theta_{k+1} expert weights
        while retaining the PRE-UPDATE H_k, G_k and mu_hat_k.

    DeltaA_finite_k = A_after - A_before

Record:
    k
    loss
    mu_norm
    DeltaA_1st
    DeltaA_finite
    residual
    relative_abs_error
    sign_match
    pair utility
    h_rms
    gate entropy

Primary analysis:
    Spearman(delta_A_1st[0:20], delta_A_finite[0:20])

Secondary:
    cumulative residual
    mean per-step residual
    prefix Spearman n=3..20
    early/late rho
    largest relative linearization errors
"""

