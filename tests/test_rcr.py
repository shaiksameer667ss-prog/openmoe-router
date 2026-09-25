from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from openmoe.continual.rcr import RCRState


class ToyRoutingModel(nn.Module):
    """Tiny differentiable router used only for the RCR unit test."""

    def __init__(self) -> None:
        super().__init__()

        self.scale = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(
        self,
        images: torch.Tensor,
    ):
        class_ids = images[:, 0].long()

        class0_logits = torch.tensor(
            [2.0, 0.0, -1.0],
            device=images.device,
        )

        class1_logits = torch.tensor(
            [0.0, 2.0, -1.0],
            device=images.device,
        )

        base = torch.where(
            class_ids[:, None, None] == 1,
            class1_logits[None, None, :],
            class0_logits[None, None, :],
        )

        logits = (
            base
            * self.scale
        )

        logits = logits.repeat(
            1,
            2,
            1,
        ).reshape(
            images.shape[0] * 2,
            3,
        )

        return SimpleNamespace(
            telemetry=[
                {
                    "routing": SimpleNamespace(
                        logits=logits
                    )
                }
            ]
        )


def make_loader() -> DataLoader:
    images = torch.tensor(
        [
            [0.0],
            [0.0],
            [1.0],
            [1.0],
        ]
    )

    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    task_ids = torch.zeros(
        4,
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(
            images,
            labels,
            task_ids,
        ),
        batch_size=2,
        shuffle=False,
    )


def test_capture_class_references() -> None:
    model = ToyRoutingModel()
    loader = make_loader()
    state = RCRState(
        num_classes=2,
        num_layers=1,
        num_experts=3,
    )

    state.capture_class_references(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert sorted(
        state.class_references
    ) == [0, 1]

    assert len(
        state.class_references[0]
    ) == 1

    assert state.class_references[
        0
    ][0].shape == (3,)

    assert state.class_references[
        1
    ][0].shape == (3,)

    assert torch.isclose(
        state.class_references[0][0].sum(),
        torch.tensor(1.0),
    )

    assert torch.isclose(
        state.class_references[1][0].sum(),
        torch.tensor(1.0),
    )

    assert (
        state.reference_bytes
        == 2 * 3 * 4
    )


def test_routing_consistency_loss_backpropagates() -> None:
    model = ToyRoutingModel()
    loader = make_loader()
    state = RCRState(
        num_classes=2,
        num_layers=1,
        num_experts=3,
    )

    state.capture_class_references(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    images, labels, _ = next(
        iter(loader)
    )

    with torch.no_grad():
        model.scale.fill_(0.5)

    loss = state.routing_consistency_loss(
        model=model,
        images=images,
        labels=labels,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(
        loss
    )

    assert loss.item() > 0.0

    loss.backward()

    assert model.scale.grad is not None

    assert (
        model.scale.grad.abs().item()
        > 1e-8
    )


def test_historical_references_are_detached() -> None:
    model = ToyRoutingModel()
    loader = make_loader()
    state = RCRState(
        num_classes=2,
        num_layers=1,
        num_experts=3,
    )

    state.capture_class_references(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    for references in (
        state.class_references.values()
    ):
        for reference in references:
            assert reference.requires_grad is False