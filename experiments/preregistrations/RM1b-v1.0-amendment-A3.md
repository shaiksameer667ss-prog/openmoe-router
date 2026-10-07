# RM1b-v1.0 Amendment A3 — Final checkpoint-relative indexing

Date: 2026-10-08
Status: FINAL / locked before rerun

The final checkpoint-relative class ranges are:

- T3 checkpoint (after task 3): old classes 0–59; current/new classes 60–79.
- F final checkpoint (after task 4): old classes 0–79; current/new classes 80–99.
- E-BOTH v4 final checkpoint (after task 4): old classes 0–79; current/new classes 80–99.

Therefore:

T3: max(logits[0:60]) - max(logits[60:80])
F: max(logits[0:80]) - max(logits[80:100])
E-BOTH: max(logits[0:80]) - max(logits[80:100])

The uploaded A1/A2 execution that used T3 old 0:40 vs new 40:60 is invalid and preserved as provenance only. The final corrected runner is scripts/rm1b_corrected.py.

No decision threshold, sample set, routing signature, CV, permutation count, ridge alpha, or seed is changed.

The corrected rerun is the only RM1b run eligible for scientific interpretation.
