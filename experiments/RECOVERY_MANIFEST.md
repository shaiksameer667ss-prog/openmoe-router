# OpenMoE-Router — Power-Cut Recovery Manifest

Recovery timestamp: 2026-10-01T02:14:02
Remote base: b2e811a
Branch: main
Remote: https://github.com/shaiksameer667ss-prog/openmoe-router.git

## Status

The Kaggle working filesystem was lost after a power cut.
The GitHub remote contained the project through Test 5.3.
Test 6.1–6.3 binaries were not pushed before the loss.
Their verified numerical results are recovered here from the session record.
No lost checkpoint or replay tensor is fabricated.

## Test 6.3

Patch4/stride4 seed0 reference: 34.0375%.
Patch4/stride4 seed1 reference: 34.4125%.
Patch4/stride2 seed0 reference: 36.4375%.
Patch4/stride2 seed1 reference: 35.5250%.
Two-seed stride4 mean: 34.2250%.
Two-seed stride2 mean: 35.98125%.
Two-seed mean effect: +1.75625 pp.

Interpretation: small, directionally replicated improvement in task-ID
separability from increased token-interaction capacity.

The stride2 manipulation changes overlap, spatial sampling, sequence
length, positional encoding, and attention scale jointly.

## Binary artifact status

Original Test 6.1–6.3 checkpoints and local replay tensors are lost.
They were not recovered from GitHub and are not represented as recovered.

The exact dirty transformer diff from the failed recovery session is
preserved separately as a .patch artifact.
