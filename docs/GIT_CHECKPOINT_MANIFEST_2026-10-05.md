# Git Checkpoint Manifest — OpenMoE-Router — 2026-10-05

## Branch

research/checkpoint-2026-10-05-fwr

Base:
main

Purpose:
Freeze the validated research state before the next FWR robustness/generalization experiment. Prior main history is preserved; this branch adds archival documentation and experiment records.

## New archival commits on this branch

1. 011c300ef95467875b20d3f16a637b07cdc16fd2
   docs: archive 2026-10-05 research state

2. 9b6816a65f15ecba1b938dda8fcd2be61deab8b2
   exp: archive FWR 20-step trajectory

3. aa22755a39a1c980dee4cdcdd5828f363981b9cb
   exp: archive FWR prefix correlations

4. dcee557a560f4d1a493a7be18285daa0fb9cffef
   exp: archive FWR machine-readable result

5. c1584ec8aa12c30b430a027084d8afbd54446ea1
   exp: archive FWR run record

6. 0ba7f71c7769f69fca9e13c7ffb8890d530ffda5
   exp: snapshot corrected FWR Cell 2 protocol

7. 77a31ebedafd8c8a683d792406da692497615995
   exp: archive FWR execution transcript metadata

8. This manifest is the next archival commit.

## Archived files

- docs/RESEARCH_STATE_2026-10-05.md
- experiments/results/diagnostics/fwr_multistep_step10_pair34/fwr_multistep_step10_pair34_trajectory.csv
- experiments/results/diagnostics/fwr_multistep_step10_pair34/fwr_multistep_step10_pair34_prefix_spearman.csv
- experiments/results/diagnostics/fwr_multistep_step10_pair34/fwr_multistep_step10_pair34.json
- experiments/results/diagnostics/fwr_multistep_step10_pair34/fwr_multistep_step10_pair34_run.md
- experiments/results/diagnostics/fwr_multistep_step10_pair34/EXECUTION_TRANSCRIPT_SOURCE.md
- experiments/code_snapshots/fwr_multistep_step10_pair34_cell2.py

## Exact artifact identities

Step-10 checkpoint:
e69fb593e6b158db1b5db16ffe26c75da0113bcc8ee9fa9d9c8abd8077823041

Theta0:
66dc3ac61695a96da3f23352d7c63c2bdd3e24f02056b94ed71c39a495c51ef7

Replay file:
de9136168116970af6051e73e64b208ee776ac127367f52862daabcaf65da8ed

Replay payload:
ac54b7af81500046ef1472d2236eb6eb7aa3e738aa17903e6d4ed7976bdc4963

## Scientific checkpoint

FWR Cell 2 multi-step:
- forced pair (3,4)
- eta=0.001
- 20 sequential direct-SGD steps
- exact replay = 715 examples
- exact token alignment = 45,760
- Spearman rho = 1.000000
- sign agreement = 20/20
- median relative error = 4.113310e-4
- maximum relative error = 7.337738e-3
- cumulative residual = -1.346006e-6
- no early-valid/late-collapse regime observed

Decision:
DO NOT TRAIN ROUTER YET.

Next gate:
FWR robustness/generalization, especially additional forced pairs and independent T4-batch validation, followed by task-level retention causality before any router intervention.
