# Bounded Decoder Ablation Design

## Purpose

Determine whether memory-bounded NCM recovery is attributable to:

1. training-time replay preserving the representation,
2. decoder refitting itself,
3. or the particular retained buffer contents.

The comparison is evaluated at boundary 4 using the same held-out CIFAR-100 test split and the same 715-example NCM decoder budget.

## Conditions

| Condition | Training | Decoder-fit buffer |
|---|---|---|
| A | Replay-trained | Exact 715 images retained by the replay-trained run |
| B | `none`, no training-time replay | The exact same 715 images as A |
| C | `none`, no training-time replay | A fresh, independently drawn 715-image buffer |

Condition B holds decoder-fit images fixed while changing whether the model was trained with replay.

Condition C holds the `none` training regime fixed while changing the decoder-fit buffer contents.

The existing archived bounded replay result is capacity 715. Its retained images were not serialized, so an exact A/B comparison requires a replay-trained run with buffer serialization before conditions B and C are evaluated.

## Interpretation patterns

| Pattern | Interpretation |
|---|---|
| B ˜ A and C ˜ A | Decoder refitting is sufficient; training-time replay is not required for the recovery under this protocol. |
| B << A and C << A | Training-time replay preserves the representation that the decoder refit exploits. |
| B < A and C < B | Both training-time replay and the particular retained buffer contribute. |
| B ˜ A and C << A | The particular buffer contents matter despite little dependence on training-time exposure; investigate further. |

No conclusion is drawn until the three rows are observed together.

## Measurements

For each condition at boundary 4, record:

- frozen NCM old-task mean, where introduction-time references are available
- replay-refit NCM old-task mean
- learned-head old-task mean
- per-task NCM refit accuracies
- decoder state bytes
- replay-buffer size and exact byte count
- number of classes represented

The primary comparison is the old-task mean of the replay-refit NCM.

## Memory accounting

The decoder state is 102,400 B for 100 class means at 256 dimensions in FP32.

The decoder memory claim applies to the decoder state and is not a claim that the three training conditions have identical training-time method-state footprints.

## Causal wording

Condition B must not be described as a buffer on which the `none` model was trained.

The exact question is:

> Given the same images available for decoder refit, does replay training change how much current-feature class information can be recovered?

Condition C asks whether the result depends materially on the specific retained images.

## Relationship to prior results

The previously archived bounded replay-trained seed-0 result found:

- learned head old-task mean: 0.046375
- replay-refit NCM old-task mean: 0.092875
- frozen introduction-time NCM old-task mean: 0.012375

The frozen-to-refit old-task gap was 0.0805.

Those values remain archived raw results and are not changed by this ablation design.

## Order of operations

1. Obtain a replay-trained run with serialization of the exact 715 retained images.
2. Train a fresh `none` model with the same training protocol and save the boundary-4 checkpoint.
3. Evaluate condition B on that checkpoint using the exact A buffer.
4. Evaluate condition C on the same `none` checkpoint using an independent 715-image buffer.
5. Report A, B, and C together before drawing an interpretation.
6. Replicate the bounded decoder at seeds 1 and 2 only after this ablation is complete.

## Preregistration discipline

This ablation does not modify the frozen branch criteria or retrospectively redefine earlier criteria. It is a control experiment for interpreting the already-observed seed-0 bounded decoder result.
