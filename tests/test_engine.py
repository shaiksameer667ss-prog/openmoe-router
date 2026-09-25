import torch

from openmoe.models.transformer import TinyMoETransformer
from openmoe.routers.topk import TopKRouter
from openmoe.training.engine import evaluate_ncm, train_steps


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

def test_evaluate_ncm_uses_class_means():
    class FeatureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def extract_features(self, images):
            return images[:, :2]

    model = FeatureModel()

    images = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.2, 10.0],
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    task_ids = torch.zeros(
        4,
        dtype=torch.long,
    )

    loader = [
        (
            images,
            labels,
            task_ids,
        )
    ]

    accuracy = evaluate_ncm(
        model=model,
        prototype_loaders=[loader],
        evaluation_loader=loader,
        device=torch.device("cpu"),
    )

    assert accuracy == 1.0

def test_train_steps_rcr_path_runs():
    from openmoe.continual.rcr import RCRState

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

    state = RCRState(
        num_classes=10,
        num_layers=1,
        num_experts=4,
    )

    state.capture_class_references(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    history = train_steps(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        steps=1,
        rcr_state=state,
        rcr_beta=1.0,
        rcr_replay_batch_size=1,
    )

    assert len(history) == 1
    assert "rcr" in history[0]
    assert torch.isfinite(
        torch.tensor(history[0]["rcr"])
    )
