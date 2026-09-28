
from __future__ import annotations

import torch
import torch.nn.functional as F

from openmoe.training.engine import er_ace_loss


def test_er_ace_is_sum_of_current_and_replay_ce():
    torch.manual_seed(0)

    logits = torch.randn(
        4,
        8,
        requires_grad=True,
    )

    labels = torch.tensor(
        [2, 3, 0, 1],
        dtype=torch.long,
    )

    actual = er_ace_loss(
        logits=logits,
        labels=labels,
        current_batch_size=2,
        current_class_start=2,
        classes_per_task=2,
    )

    expected = (
        F.cross_entropy(
            logits[:2, 2:4],
            labels[:2] - 2,
        )
        + F.cross_entropy(
            logits[2:, :4],
            labels[2:],
        )
    )

    assert torch.allclose(
        actual,
        expected,
        atol=1e-7,
    )


def test_er_ace_unseen_logits_have_zero_gradient():
    torch.manual_seed(1)

    logits = torch.randn(
        4,
        8,
        requires_grad=True,
    )

    labels = torch.tensor(
        [2, 3, 0, 1],
        dtype=torch.long,
    )

    loss = er_ace_loss(
        logits=logits,
        labels=labels,
        current_batch_size=2,
        current_class_start=2,
        classes_per_task=2,
    )

    loss.backward()

    assert torch.equal(
        logits.grad[:, 4:],
        torch.zeros_like(
            logits.grad[:, 4:]
        ),
    )


def test_er_ace_without_replay_is_exact_standard_ce():
    torch.manual_seed(2)

    logits = torch.randn(
        6,
        10,
        requires_grad=True,
    )

    labels = torch.tensor(
        [0, 1, 2, 3, 4, 5],
        dtype=torch.long,
    )

    actual = er_ace_loss(
        logits=logits,
        labels=labels,
        current_batch_size=6,
        current_class_start=0,
        classes_per_task=6,
    )

    expected = F.cross_entropy(
        logits,
        labels,
    )

    assert torch.equal(
        actual,
        expected,
    )
