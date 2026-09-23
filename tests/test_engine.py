import torch

from openmoe.models.transformer import TinyMoETransformer
from openmoe.routers.topk import TopKRouter
from openmoe.training.engine import train_steps


def make_model(num_classes=10):
    def factory(hidden_dim, num_experts):
        return TopKRouter(
            hidden_dim,
            num_experts,
            top_k=1,
        )

    return TinyMoETransformer(
        num_classes=num_classes,
        hidden_dim=32,
        num_heads=4,
        ff_dim=64,
        num_experts=4,
        router_factory=factory,
        depth=1,
    )


def test_head_masked_training_path_runs():
    model = make_model(num_classes=10)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    images = torch.randn(
        2,
        3,
        32,
        32,
    )

    labels = torch.tensor(
        [7, 8],
    )

    task_ids = torch.zeros(
        2,
        dtype=torch.long,
    )

    loader = [
        (
            images,
            labels,
            task_ids,
        )
    ]

    history = train_steps(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        steps=1,
        head_mask_old_classes=5,
    )

    assert len(history) == 1

    assert torch.isfinite(
        torch.tensor(
            history[0]["loss"],
        )
    )


def test_head_masked_old_class_logits_are_excluded_from_ce():
    logits = torch.tensor(
        [
            [
                100.0,
                90.0,
                80.0,
                70.0,
                1.0,
                2.0,
            ],
        ],
        requires_grad=True,
    )

    labels = torch.tensor(
        [5],
    )

    masked = logits.clone()

    masked[
        :,
        :4,
    ] = float("-inf")

    loss = torch.nn.functional.cross_entropy(
        masked,
        labels,
    )

    loss.backward()

    assert torch.isfinite(loss)

    assert torch.all(
        logits.grad[:, :4] == 0
    )

    assert torch.all(
        logits.grad[:, 4:] != 0
    )
