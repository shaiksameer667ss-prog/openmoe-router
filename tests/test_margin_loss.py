from __future__ import annotations

import pytest
import torch

from openmoe.training.engine import relative_old_class_margin_loss


def test_relative_old_class_margin_matches_definition() -> None:
    logits = torch.tensor(
        [
            [1.0, 2.0, 0.5, 3.0],
            [4.0, 1.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([2, 3], dtype=torch.long)

    actual = relative_old_class_margin_loss(
        logits=logits,
        labels=labels,
        current_batch_size=2,
        current_class_start=2,
    )

    old_max = torch.tensor([2.0, 4.0])
    true_logits = torch.tensor([0.5, 3.0])
    expected = torch.nn.functional.softplus(
        old_max - true_logits
    ).mean()

    assert torch.allclose(actual, expected, atol=1e-7)


def test_relative_old_class_margin_is_zero_for_task_zero() -> None:
    logits = torch.randn(4, 10)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    actual = relative_old_class_margin_loss(
        logits=logits,
        labels=labels,
        current_batch_size=4,
        current_class_start=0,
    )

    assert actual.item() == 0.0


def test_relative_old_class_margin_rejects_invalid_batch_size() -> None:
    logits = torch.randn(4, 10)
    labels = torch.tensor([2, 3, 4, 5], dtype=torch.long)

    with pytest.raises(ValueError):
        relative_old_class_margin_loss(
            logits=logits,
            labels=labels,
            current_batch_size=0,
            current_class_start=2,
        )

    with pytest.raises(ValueError):
        relative_old_class_margin_loss(
            logits=logits,
            labels=labels,
            current_batch_size=5,
            current_class_start=2,
        )
