# OpenMoE-Router — Power-Loss Recovery Note

Date: 2026-10-07

The Kaggle runtime containing the working tree and completed Experiment D/E reporting artifacts was lost following a power interruption before the intended Git push completed.

Remote audit found:
- repository: shaiksameer667ss-prog/openmoe-router
- default branch: main
- remote HEAD: 33a6aceebe571ff978be530aaf1790b4b5b49e95
- A/B artifacts are present on GitHub at that commit.
- D/E protocol/result commits were not present on GitHub.

This recovery commit restores the textual scientific record from the completed run outputs preserved in the conversation. It does not recreate missing binary Kaggle artifacts and does not rerun any experiment.

Authoritative recovered D result:
- Arm-A' forgetting: 32.0375 pp
- D forgetting: 31.1625 pp
- Δforget: -0.875 pp
- T4: 43.25%
- decision: INCONCLUSIVE

Authoritative recovered E result:
- Arm-A' forgetting reference: 32.0375 pp
- E forgetting: 30.5500 pp
- Δforget: -1.4875 pp
- T4: 43.10%
- final T0..T4: 3.00%, 2.50%, 3.50%, 5.80%, 43.10%
- decision: INCONCLUSIVE

Source integrity retained:
- D T3 bundle SHA256: 7f5af253a4b4f6e4f3a5587a3cb5e116bf9dc923a2611b27c1594e6b965f7d13
- canonical theta3 SHA256: 26f0c00cbe53d20d535cc701c2e53c7d488af5eca51a8117c9b2da21d85307fc

Important:
- D is CLOSED.
- E is CLOSED.
- No D/E rerun is implied by this recovery.
- Any future binary checkpoint recovery must be separately verified; this text record is not a substitute for the original file bytes.
