# RCR Preregistration Amendment v3

**Amendment date:** 2026-09-26  
**Status:** FROZEN  
**Applies to:** Routing-Consistent Replay (RCR) preregistration for the CIFAR-100 Phase 1 / Milestone 6 experiment

## Purpose

This amendment corrects the replay-capacity specification in the frozen RCR preregistration before any RCR experimental run.

## Capacity correction

The strict memory-matched RCR capacity is **722 examples, not 723**.

The audited byte accounting is:

| Replay capacity | Replay bytes | RCR references + router buffers | Total method-state bytes | Relation to 8,911,776-byte target |
|---:|---:|---:|---:|---|
| 722 | 8,883,488 | 22,912 | 8,906,400 | 5,376 bytes under |
| 723 | 8,895,792 | 22,912 | 8,918,704 | 6,928 bytes over |

Therefore 723 exceeds the stability method-state target and cannot be described as the strict memory-matched capacity.

## Corrected experimental conditions

The conditions are corrected as follows:

- **A = `replay_722`**
- **C = `RCR_722` at ß = 1.0**
- **D = `RCR_723`**, retained as the secondary condition at the slight over-budget capacity.

The primary comparison remains:

**A (`replay_722`) vs C (`RCR_722`)**

## Relation to previous preregistration

The earlier preregistration specified 723 as the strict capacity. That specification was incorrect because the complete method-state accounting includes the replay memory, 6,400-byte RCR class-reference state, and 16,512-byte continual-router buffers.

The correction is made from the pre-run byte audit and is recorded before any RCR experimental result is observed.

## ß sweep

The seed-0 ß sweep remains:

**ß ? {0.1, 1.0, 10.0}**

The primary RCR condition is **RCR_722 at ß = 1.0**.

The ß sweep informs assessment of the preregistered primary value; it does not change the primary comparison or thresholds after observing results.

## Seed policy

No confirmation seeds are run at this stage.

Seeds 1–2 are reserved for the preregistered confirmation procedure after the seed-0 result establishes a signal requiring confirmation.

The secondary `RCR_723` condition is not run unless the seed-0 results provide a preregistered reason to examine the slight over-budget condition.

## Thresholds

The previously frozen outcome thresholds remain unchanged:

- Positive: `RCR_722 old-task mean = replay_722 old-task mean + 0.010`
- Weak positive: `+0.005 = ? < +0.010`
- Null: `|?| < 0.005`
- Negative: `? < -0.005`

The primary metric remains the boundary-4 old-task mean.

## Timing and knowledge state

This amendment is recorded **before the first RCR training run**.

The capacity correction was derived from audited memory accounting and not from any observed RCR outcome.

No RCR experimental result informed this amendment.

## Scope preservation

All other preregistered RCR protocol elements remain unchanged unless explicitly amended in a later dated document.

---

**Amendment conclusion:** the strict memory-matched capacity is corrected from 723 to 722; A is `replay_722`, C is `RCR_722` at ß = 1.0, and D is `RCR_723` as the secondary over-budget condition.
