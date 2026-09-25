# RCR Preregistration Amendment v2

**Amendment date:** 2026-09-26  
**Status:** FROZEN  
**Applies to:** Routing-Consistent Replay (RCR) preregistration for the CIFAR-100 Phase 1 / Milestone 6 experiment

## Purpose

This amendment records a design identity discovered before any RCR experiment was run.

## E/C identity

The preregistered condition **C (`RCR_722`)** and condition **E (`replay_722 + routing_consistency`)** are identical under the RCR definition.

RCR is defined as replay augmented with the routing-consistency objective against compressed per-class historical routing references. The compressed routing reference is an implementation detail required by the memory-budget design; it does not constitute a separate mechanism.

Therefore:

- **C** = replay_722 + routing consistency using the RCR historical routing references.
- **E** = replay_722 + routing consistency using the same RCR historical routing references.

No experimental distinction exists between them.

## Amendment

Condition E is no longer treated as an independent comparator.

E is retained only as a **labeled replication/audit condition** if executed. It must not be reported as an independent method or as independent evidence confirming C.

The previously stated intent of E — isolating the routing-consistency term from the memory-compression component — is dropped. That isolation is not achievable under the frozen RCR definition without introducing a new condition after preregistration.

## Primary comparison

The primary RCR comparison remains:

**A (`replay_722`) vs C (`RCR_722`)**

This comparison tests the effect of adding the RCR routing-consistency package at the same replay capacity.

Any observed effect must be attributed to the **routing-consistency plus compressed-routing-reference package as a whole**. The experiment does not isolate the routing-consistency objective from the compressed routing-reference representation.

Accordingly, the result must not be described as demonstrating that the routing-consistency term alone caused an observed improvement.

## Freed experimental slot

The experimental slot previously occupied by independent condition E is reassigned to the preregistered **ß sweep on seed 0**:

**ß ? {0.1, 1.0, 10.0}**

The sweep is used to select ß once on seed 0, with **ß = 1.0** remaining the preregistered primary value unless the existing selection rule specifies otherwise.

The previously preregistered outcome thresholds are unchanged.

## Optional determinism audit

A second execution of C at seed 0 may be performed solely as a determinism audit.

If performed, it is a duplicate execution of the same condition, not an independent scientific replication. Its purpose is to check reproducibility of the RCR execution path.

A duplicate C run must not be counted as an additional seed, additional independent observation, or separate comparator.

If the determinism audit is not executed, no experimental inference is affected.

## Timing and knowledge state

This amendment was identified and recorded **before any RCR training run**.

The identity of C and E was discovered from the preregistered definitions themselves, not from inspection of RCR experimental outcomes.

No RCR result has been used to motivate, select, or modify this amendment.

## Scope preservation

All other preregistered RCR protocol elements, including replay capacities, memory accounting, ß candidates, evaluation boundaries, seed-selection rules, and outcome thresholds, remain unchanged unless explicitly amended in a later dated document.

---

**Amendment conclusion:** E is removed as an independent comparator; the freed slot is assigned to the preregistered seed-0 ß sweep; no RCR experimental result informed this amendment.
