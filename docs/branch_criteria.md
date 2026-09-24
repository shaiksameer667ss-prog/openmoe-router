# Diagnostic Branch Criteria

Status: preregistered
Purpose: define the decision rule for selecting the next OpenMoE-Router research milestone before inspecting new probe or decomposition results.

## 1. Research constitution

The project follows:

> **DIAGNOSE → DECIDE → BUILD**

No new method milestone is selected because it is interesting or because a result looks promising. A method is selected only when the preregistered diagnostic criteria identify a dominant failure mechanism.

The diagnostic phase must also allow a valid negative outcome:

> **No dominant MoE-specific mechanism at this scale**

in which case the project pivots to the systems/framework contribution rather than inventing a post-hoc mechanism.

---

## 2. Experimental scope

The primary diagnostic conditions are:

- `none`
- `head_masked`
- `head_masked_frozen_old`

The probe matrix compares, for each training condition:

- learned classifier head
- NCM with refit prototypes
- linear probe

across the five task boundaries.

`head_masked_ncm` is not treated as a separate training condition. It is an evaluation protocol applied to the `head_masked` checkpoint.

Primary diagnostic scale:

- CIFAR-100
- 5 tasks
- 20 classes per task
- 2-layer, 8-expert MoE
- existing OpenMoE-Router Phase 1/Milestone 6 protocol
- held-out CIFAR-100 test evaluation

The first diagnostic pass is seed 0.

Additional seeds are used only for confirmation when a candidate branch passes the seed-0 screening threshold.

---

## 3. Seed-0 screening and confirmation

The diagnostic phase uses two stages.

### Stage A: seed-0 screening

Run the probe and four-cell decomposition on seed 0.

A branch becomes a candidate only if its preregistered magnitude criterion is satisfied.

### Stage B: confirmation

Only candidate branches are replicated on seeds 1 and 2.

A candidate branch is confirmed when:

- at least 2 of 3 seeds show the same directional effect, and
- the effect is not driven by a single task boundary.

If no branch passes seed-0 screening, or no candidate survives confirmation, the final outcome is:

> **No dominant MoE-specific mechanism at this scale.**

No new mechanism is then introduced for the purpose of forcing a positive result.

---

## 4. Decoder-dominant branch

The decoder branch is selected when both magnitude and consistency criteria hold.

### Magnitude

For old-task accuracy at the final boundary:

1. Linear-probe accuracy is at least 1.5× the learned-head accuracy on the same checkpoint.
2. Linear-probe accuracy remains within 10 percentage points of its value at the boundary where the corresponding old tasks were introduced.

### Consistency

The magnitude criteria must hold for at least 2 of the 3 training conditions:

- `none`
- `head_masked`
- `head_masked_frozen_old`

For confirmation, the same directional relationship must be observed in at least 2 of 3 seeds.

### Interpretation

A decoder-dominant result means that substantial old-class information remains linearly accessible in the current feature representation while the learned classifier performs substantially worse.

This supports a decoder/misalignment interpretation within this experimental regime.

It does not establish that the decoder is the universal source of continual-learning forgetting.

---

## 5. Trunk-dominant branch

The trunk branch is selected when both magnitude and consistency criteria hold.

### Magnitude

For old-task linear-probe accuracy:

- final-boundary probe accuracy is more than 20 percentage points below the probe accuracy at the boundary where those tasks were introduced.

### Consistency

The magnitude criterion must hold for at least 2 of the 3 training conditions:

- `none`
- `head_masked`
- `head_masked_frozen_old`

For confirmation, the same directional effect must be present in at least 2 of 3 seeds.

### Interpretation

A trunk-dominant result means that discriminative information has substantially degraded in the feature representation itself.

This supports a representation/trunk-drift interpretation within this experimental regime.

It does not establish that the trunk is the universal source of continual-learning forgetting.

---

## 6. Router/expert-dominant branch

The router/expert branch is selected only when the local counterfactual decomposition identifies a sufficiently large routing contribution.

For a MoE layer, define the four counterfactual cells:

- `y00`: old routing + old experts
- `y10`: new routing + old experts
- `y01`: old routing + new experts
- `y11`: new routing + new experts

All four cells use the same pre-MoE hidden activation.

### Main effects

Routing main effect:

\[
R_{\mathrm{main}} = y10 - y00
\]

Expert main effect:

\[
X_{\mathrm{main}} = y01 - y00
\]

Interaction:

\[
I = y11 - y10 - y01 + y00
\]

Total observed change:

\[
\Delta y = y11 - y00
\]

### Magnitude

The routing main-effect share is:

\[
S_R =
\frac{\|y10-y00\|}
{\|y11-y00\|}
\]

A routing-dominant candidate requires:

- \(S_R > 0.40\)
- at least one MoE layer satisfies this criterion
- the criterion holds at boundaries 2, 3, and 4
- the denominator is above the preregistered numerical floor for relative-change measurements

Also record the absolute normalized routing effect:

\[
A_R =
\frac{\|y10-y00\|}
{\|y00\|}
\]

to avoid interpreting unstable ratios produced by negligible total output change.

### Consistency

The routing-dominant criterion must have the same directional interpretation across at least 2 of 3 seeds.

The effect must not be attributable to a single boundary.

### Interpretation

The four-cell decomposition attributes local output change under controlled counterfactual substitutions.

A large routing main effect does **not** by itself prove that routing is the sole cause of forgetting. Routing and expert changes can interact, and upstream representation drift can alter the pre-MoE activation.

Therefore local and full-network results are reported separately.

---

## 7. Local and full-network decomposition modes

Both modes must always be run.

### Local counterfactual mode

The same pre-MoE activation is fed through the four combinations of old/new router and old/new experts.

Purpose:

> Measure the contribution of the layer's own routing and expert parameter changes while holding the layer input fixed.

### Full-network observed mode

The actual checkpoint-specific activations are used.

Purpose:

> Measure what the layer actually contributes in the trained network after upstream changes have propagated into its input.

These modes are stored as separate result sections:

```text
local_counterfactual
full_network_observed
