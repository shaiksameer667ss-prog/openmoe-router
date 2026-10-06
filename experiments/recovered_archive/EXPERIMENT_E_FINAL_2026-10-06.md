# Experiment E — E_patch Freeze

Protocol:
- Experiment C ranked E_patch second.
- D reconstructed T3 live bundle was reused directly; no second T0->T3 reconstruction.
- Intervention froze only patch_embed.weight and patch_embed.bias.
- Persistent AdamW retained all 89 parameter tensors.
- Both continual routers remained enabled.
- T4 used 64 current + 64 replay, 200 steps total (10 warmup + 190 adaptation).

Source integrity:
- D T3 bundle SHA256 = 7f5af253a4b4f6e4f3a5587a3cb5e116bf9dc923a2611b27c1594e6b965f7d13
- canonical theta3 SHA256 = 26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc

Intervention checks:
- frozen parameter count = 2
- frozen parameters = patch_embed.bias, patch_embed.weight
- E_patch remained in persistent AdamW parameter groups
- 87 parameter tensors remained trainable
- non-E_patch learning detected
- both continual routers remained enabled

Result:
- Arm-A' old-task forgetting reference = 32.0375 pp
- E old-task forgetting = 30.5500 pp
- Δforget_E = -1.4875 pp
- T4 accuracy = 43.10%
- plasticity cost = 2.00 pp
- preregistered verdict = INCONCLUSIVE

Final T0..T4 accuracies:
T0 3.00%, T1 2.50%, T2 3.50%, T3 5.80%, T4 43.10%.

Derived forgetting per task from the locked historical best accuracies:
T0 28.30 pp, T1 25.00 pp, T2 33.90 pp, T3 35.00 pp.

E is CLOSED. Do not rerun.
