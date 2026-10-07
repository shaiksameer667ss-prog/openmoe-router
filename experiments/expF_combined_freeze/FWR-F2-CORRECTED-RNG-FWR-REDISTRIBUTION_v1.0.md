# FWR-F2-CORRECTED-RNG-FWR-REDISTRIBUTION v1.0

**Status:** LOCKED — PREREGISTERED BEFORE EXECUTION

## 1. Purpose

This preregistration supersedes the causal interpretation of the executed Experiment F run. The executed F run is retained as a provenance record for its freeze instrumentation, but its causal comparison and antagonism classification are withdrawn because the T3→T4 stochastic trajectory was not RNG-matched.

F2 tests whether the combined B0-attention + patch-embedding freeze has an interpretable incremental effect when the exact reconstructed T3 stochastic state is restored, and whether harmful retention write mass is redistributed into alternative trainable pathways.

## 2. Scientific questions

### Primary question

When B0 attention and patch embedding are both frozen at the T4 boundary, what is the incremental effect relative to the matched-current-bundle E-BOTH v4 patch-only freeze?

### Mechanistic question

When the two leading natural harmful write pathways are frozen, does harmful retention write mass migrate into other trainable parameter groups, disappear, or increase?

## 3. Locked experiment identity

- Experiment ID: `FWR-F2-CORRECTED-RNG-FWR-REDISTRIBUTION-v1.0`
- Bundle SHA256: `60a83ce0ff0729dc44de530d24f2fcea9047606ad992f13d71a1814fbf63b91c`
- Canonical theta3 SHA256: `26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc`
- Canonical replay SHA256: `de9136168116970af6051e73e64b208ee776ac127367f52862daabcaf65da8ed`
- Execution checkout commit: `3854bd254634eafcb5ac29edf37818654fe281de`
- Source-tree baseline before documentation-only commits: `6932d6c1c9a192e0d547875cf498b586f3112d3e`
- Historical local-only F preregistration checkout `34382c0b2cc68f3d456486df096e1d278c83d263`: **UNRECOVERABLE FROM REMOTE/LOCAL OBJECT DATABASE**
- Seed identifier: `0` (T4 replay/global RNG is restored from the saved T3 state; no fresh T4 seed is permitted)

## 4. Starting-state identity

F2 MUST load the verified current C/E/F T3 live bundle and use its saved:

- model parameters;
- persistent AdamW optimizer state;
- continual-router state embedded in the model;
- replay buffer;
- Python RNG state;
- NumPy RNG state;
- PyTorch CPU RNG state;
- PyTorch CUDA RNG states;
- replay-generator RNG state.

The T3 endpoint must reproduce exactly:

```text
[0.0410, 0.0285, 0.0765, 0.4080]
```

No T4 training may begin until the endpoint and all provenance gates pass.

## 5. RNG lock

Immediately after restoration, and before construction or consumption of the real T4 loader, compute and log a combined RNG fingerprint over:

```python
pickle.dumps((
    torch.get_rng_state(),
    torch.cuda.get_rng_state_all(),
    np.random.get_state(),
    random.getstate(),
    replay_generator.get_state(),
))
```

hashed with SHA256.

The restored F2 fingerprint MUST equal the reference fingerprint derived from the exact saved T3 RNG states in the verified `60a83ce0...` bundle. Because the archived E-BOTH v4 execution restores those same saved T3 states but does not contain a durable combined fingerprint log, the reference fingerprint is to be generated directly from the verified bundle rather than inferred from an unlogged runtime value.

Log component hashes as well as the combined hash so a mismatch can be localized to Python, NumPy, CPU, CUDA, or the replay generator.

After restoration:

- do NOT call `manual_seed()` on the T4 replay generator;
- do NOT construct a replacement generator with `SEED + 100_000`;
- do NOT consume `next(iter(t4_loader))` or any other real T4 diagnostic batch;
- do NOT execute any diagnostic operation that advances the training RNG stream.

## 6. Intervention

Freeze exactly these six tensors at the T4 boundary:

```text
blocks.0.attn.in_proj_weight
blocks.0.attn.in_proj_bias
blocks.0.attn.out_proj.weight
blocks.0.attn.out_proj.bias
patch_embed.weight
patch_embed.bias
```

No other parameter may be frozen.

Expected frozen parameter count:

```text
1,075,712
```

Expected trainable tensor count:

```text
83
```

Expected persistent AdamW optimizer-state transition:

```text
89 -> 83
```

The six frozen tensors must remain bitwise identical throughout T4.

## 7. Training protocol

Use the canonical continual-learning T4 protocol:

- persistent AdamW;
- learning rate `3e-4`;
- weight decay `0.05`;
- 200 total optimizer steps;
- 10 warmup steps;
- 190 adaptation steps;
- current batch `64`;
- replay batch `64`;
- replay capacity `715`;
- canonical continual-router callbacks;
- no additional regularizer, loss, router, or optimizer intervention.

The actual training data trajectory must begin from the restored T3 RNG state and restored replay-generator state.

## 8. Checkpoint schedule

Save a complete F2 training checkpoint at exactly:

```text
k = 10, 25, 50, 100, 150, 200
```

Each checkpoint must preserve enough state to verify provenance and to reproduce the trajectory from that checkpoint, including model, optimizer, replay state, and RNG states.

The checkpoints are saved before final evaluation.

## 9. FWR measurement

At each locked checkpoint:

```text
10, 25, 50, 100, 150, 200
```

measure the same FWR / gradient-interference statistic and the same 12-group partition used by Experiment C. Do not redefine the FWR statistic, group definitions, or alignment thresholds.

Use the same:

- incoming T4 data definition;
- replay definition;
- `eta = 1e-3`;
- checkpoint positions;
- positive harmful-mass calculation;
- aggregation rules;

as the validated natural-trajectory C analysis.

### Non-mutation protocol

FWR diagnostics MUST be structurally separated from the training process. At each checkpoint:

1. Save the complete F2 training state to disk, including model, optimizer, replay state, and all relevant RNG states.
2. Continue the F2 training trajectory without running FWR diagnostics in the training process.
3. After training completes, launch a fresh process/namespace for each saved checkpoint.
4. Load the checkpoint into that fresh process and run the FWR diagnostics on the loaded copy.
5. Discard the diagnostic process/copy after measurement.

No FWR diagnostic may mutate the saved F2 checkpoint or the training trajectory. The checkpoint files are the sole interface between F2 training and FWR measurement.

If process separation is unavailable for an execution environment, the run must instead log the full RNG fingerprint immediately before and after every diagnostic phase and assert exact equality before resuming training. This fallback is explicitly weaker than the separate-process protocol and must be identified as such in the result record.

## 10. Absolute harmful-mass and redistribution quantities

The validated Experiment C quantity `H_X` is a normalized share. It MUST NOT be used directly as the numerator or denominator of the F2 redistribution index, because its normalization forces the group shares to sum to one and can create a purely arithmetic increase when groups are frozen.

F2 therefore uses the underlying **absolute positive harmful mass** before normalization. For group `X` at checkpoint `k`, define:

```text
M_X(nat, k) = sum_t max(I^nat_{t,X,k}, 0)
M_X(F2,  k) = sum_t max(I^F2_{t,X,k},  0)
```

where `I` is the same signed gradient-interference quantity used by Experiment C.

Let `FROZEN` denote the six F2 tensors. Report at every checkpoint:

```text
M_frozen(nat, k) = sum_{X in FROZEN} M_X(nat, k)
M_frozen(F2,  k) = sum_{X in FROZEN} M_X(F2,  k)
M_other(nat, k)  = sum_{X not in FROZEN} M_X(nat, k)
M_other(F2,  k)   = sum_{X not in FROZEN} M_X(F2,  k)
```

`M_frozen(nat, k)` is the natural-trajectory comparator for how much harmful mass is removed from the intervention targets. `M_frozen(F2, k)` should be zero by construction because the six target tensors are frozen; any nonzero value indicates an implementation or measurement failure and must trigger an audit.

The redistribution index is:

```text
redistribution_index(k)
    = M_other(F2, k) / M_other(nat, k)
```

The pooled index is:

```text
redistribution_index_pooled
    = sum_k M_other(F2, k) / sum_k M_other(nat, k)
```

If a natural denominator is exactly zero, that checkpoint index is `UNDEFINED` and MUST be reported as such, not imputed. The pooled index is likewise `UNDEFINED` if its pooled denominator is zero.

The normalized C-style shares `H_X` may still be reported as a secondary descriptive quantity, but they are not used to classify redistribution.

## 11. Locked redistribution classifications

Classification precedence is:

```text
index > 1.2
    -> AMPLIFICATION

0.8 <= index <= 1.2
    -> MASS_CONSERVED

0.4 <= index < 0.8
    -> PARTIAL_REDUCTION

index < 0.4
    -> MASS_DISAPPEARS
```

`MASS_CONSERVED` means that absolute non-frozen harmful mass in F2 is approximately equal to that in the natural trajectory, consistent with harmful mass having migrated rather than disappeared. It does not by itself identify which specific groups receive the redistributed mass.

These bands are locked before execution and may not be tuned after results are observed.

## 12. Primary endpoint A — incremental intervention effect

Using the matched current bundle:

```text
Delta_forget_F2
    = forgetting(F2)
      - forgetting(E-BOTH v4)
```

E-BOTH v4 is the current-bundle patch-only freeze reference on bundle `60a83ce0...`.

This quantity is a pairwise incremental comparison, not a four-cell factorial interaction and not a new causal-support gate.

No antagonism/synergy/causal classification band is applied to endpoint A.

Report endpoint A as a point estimate together with the RNG fingerprint match and all provenance/invariant checks.

## 13. Primary endpoint B — redistribution

Report:

- `redistribution_index(k)` at all six checkpoints;
- pooled redistribution index;
- per-group absolute `M_X(F2, k)`;
- per-checkpoint `M_frozen(nat, k)` and `M_frozen(F2, k)`;
- `M_other(F2, k)` and `M_other(nat, k)`;
- normalized `H_X` shares as secondary descriptive quantities;
- comparison with the natural C trajectory.

The mechanistic interpretation is conditional on these measurements and must not be inferred from endpoint A alone.

## 14. Secondary trajectory comparison

For each parameter group `X`, compare F2 and natural C across the six checkpoints using the same per-group trajectory summaries as the validated C analysis, including Spearman correlation where defined.

The main question is whether harmful mass removed from the frozen groups is accompanied by increased absolute harmful mass in:

```text
U_block1_pre
B0_experts
R_L0
R_L1
classifier/head groups
other downstream groups
```


### E-BOTH v4 FWR status

FWR measurement on the archived E-BOTH v4 trajectory is **deferred** under this preregistration. The available E-BOTH v4 artifacts contain the trained pre-evaluation and final checkpoints, but not the six locked intermediate checkpoints required for the FWR trajectory comparison. Therefore no six-state E-BOTH FWR column is inferred or reconstructed from unsupported intermediate states. A separate E-BOTH-FWR study may be preregistered later if needed.

No one group may be declared compensatory solely from rank order; the conclusion must follow the locked trajectory measurements.

## 15. Required validity gates

F2 is interpretable only if all of the following pass:

1. repository HEAD equals the locked HEAD;
2. active T3 bundle SHA equals `60a83ce0...`;
3. theta3 SHA equals `26f0c00c...`;
4. canonical replay SHA equals `de913616...`;
5. T3 endpoint is exact;
6. restored combined RNG fingerprint equals the bundle-derived reference fingerprint;
7. the real T4 loader is not diagnostically consumed before training;
8. exactly six F tensors are frozen;
9. no other tensor is frozen;
10. optimizer state transitions `89 -> 83`;
11. all six frozen tensors remain bitwise identical;
12. six checkpoints are produced at `10,25,50,100,150,200`;
13. FWR diagnostics do not mutate the training trajectory.

Failure of any gate invalidates the F2 causal/redistribution analysis and requires audit before any rerun.

## 16. Multiple seeds

F2 remains a single-seed run. No empirical uncertainty interval is claimed from F2 alone.

If repeated seeds are later authorized, they must be a separately preregistered replication study and must not be retroactively pooled into F2.

## 17. Prohibited post-result changes

After T4 training begins, do not change:

- RNG restoration protocol;
- replay-generator initialization/restoration;
- checkpoint positions;
- FWR definition;
- group definitions;
- harmful-mass aggregation;
- redistribution thresholds;
- primary endpoint definition;
- intervention tensor set;
- optimizer protocol;
- training step counts.

Invalid, failed, or quarantined runs must remain in provenance records and must not be silently promoted into the primary F2 result.

## 18. Interpretation lock

F2 will not be described as proving antagonism merely because endpoint A is positive.

The preferred interpretation hierarchy is:

```text
RNG mismatch / protocol failure
    -> uninterpretable

valid F2 + redistribution_index evidence
    -> compensatory redistribution / amplification, if supported

valid F2 + low non-frozen harmful mass
    -> disappearance of harmful write pressure

valid F2 + no clear mass shift
    -> unresolved null / alternative mechanism
```

Any stronger causal or mechanistic claim requires evidence beyond the point estimate of endpoint A.

## 19. Required artifacts

At completion, preserve:

- F2 pre-evaluation checkpoint;
- F2 final checkpoint;
- all six intermediate checkpoints;
- F2 result JSON;
- RNG fingerprint JSON with component hashes;
- FWR per-checkpoint results;
- pooled redistribution summary;
- provenance manifest with SHA256 values;
- invalid/quarantined attempt records, if any.

## 20. Lock statement

This preregistration is locked before F2 execution. No threshold tuning, intervention changes, RNG substitutions, diagnostic loader consumption, or post hoc endpoint redefinition is permitted after training begins.

## 21. Provenance amendment — pre-execution

The originally recorded F execution checkout `34382c0b2cc68f3d456486df096e1d278c83d263` cannot be recovered from the current Kaggle Git object database or the remote GitHub repository.

Before F2 execution, the remote history was audited. The verifiable F2 branch currently ends at `3854bd254634eafcb5ac29edf37818654fe281de`. A GitHub commit comparison from `6932d6c1c9a192e0d547875cf498b586f3112d3e` to `3854bd254634eafcb5ac29edf37818654fe281de` shows only documentation-file changes and no model/training source changes. Therefore F2 will execute from the verifiable `3854bd...` checkout, while preserving the unrecoverable `34382c...` identifier as historical provenance only.

This amendment is recorded before any F2 training step. No scientific endpoint, intervention, RNG rule, checkpoint schedule, FWR definition, harmful-mass aggregation, or classification band is changed by this provenance amendment.
