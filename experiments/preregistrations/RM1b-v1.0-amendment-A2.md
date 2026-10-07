# RM1b-v1.0 Amendment A2 — Correct T3 checkpoint indexing

Date: 2026-10-08
Status: locked before corrected rerun

The prior A1 amendment correctly changed F and E-BOTH to checkpoint-relative indexing but retained an incorrect T3 split.

The corrected checkpoint-relative margin is:

T3:
max(logits[0:40]) - max(logits[40:60])

F:
max(logits[0:80]) - max(logits[80:100])

E-BOTH v4:
max(logits[0:80]) - max(logits[80:100])

This is the final RM1b-v1.0 amendment. No decision threshold, permutation procedure, CV procedure, seed, or sample selection is changed.

All earlier RM1b runs remain invalid for scientific decision-making and are preserved as provenance artifacts.

The corrected run is the only run eligible for the RM1b decision.
