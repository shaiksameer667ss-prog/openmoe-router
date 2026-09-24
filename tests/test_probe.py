import torch
from torch import nn

from openmoe.continual.probe import collect_features


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(2.0)
        )
        self.grad_enabled_seen: list[bool] = []

    def extract_features(self, images):
        self.grad_enabled_seen.append(
            torch.is_grad_enabled()
        )
        return images[:, :2] * self.scale


def make_loader():
    images = torch.tensor(
        [
            [1.0, 2.0, 9.0],
            [3.0, 4.0, 8.0],
            [5.0, 6.0, 7.0],
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [0, 1, 1],
        dtype=torch.long,
    )

    task_ids = torch.zeros(
        3,
        dtype=torch.long,
    )

    return [
        (
            images,
            labels,
            task_ids,
        )
    ]


def test_collect_features_returns_features_and_labels_without_grad():
    model = FeatureModel()

    features, collected_labels = collect_features(
        model=model,
        loader=make_loader(),
        device=torch.device("cpu"),
    )

    expected_features = torch.tensor(
        [
            [2.0, 4.0],
            [6.0, 8.0],
            [10.0, 12.0],
        ]
    )

    expected_labels = torch.tensor(
        [0, 1, 1],
        dtype=torch.long,
    )

    assert torch.equal(
        features,
        expected_features,
    )

    assert torch.equal(
        collected_labels,
        expected_labels,
    )

    assert all(
        flag is False
        for flag in model.grad_enabled_seen
    )


def test_collect_features_rejects_empty_loader():
    model = FeatureModel()

    try:
        collect_features(
            model=model,
            loader=[],
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "empty loader" in str(exc)
    else:
        raise AssertionError(
            "collect_features should reject an empty loader"
        )


def test_collect_features_restores_training_mode():
    model = FeatureModel()
    model.train()

    assert model.training is True

    collect_features(
        model=model,
        loader=make_loader(),
        device=torch.device("cpu"),
    )

    assert model.training is True


def test_collect_features_preserves_eval_mode():
    model = FeatureModel()
    model.eval()

    assert model.training is False

    collect_features(
        model=model,
        loader=make_loader(),
        device=torch.device("cpu"),
    )

    assert model.training is False


def test_collect_features_concatenates_multiple_batches_in_order():
    model = FeatureModel()

    loader = [
        (
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [4],
                dtype=torch.long,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        ),
        (
            torch.tensor(
                [
                    [5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [5, 6],
                dtype=torch.long,
            ),
            torch.tensor(
                [0, 0],
                dtype=torch.long,
            ),
        ),
    ]

    features, labels = collect_features(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        features,
        torch.tensor(
            [
                [2.0, 4.0],
                [10.0, 12.0],
                [16.0, 18.0],
            ]
        ),
    )

    assert torch.equal(
        labels,
        torch.tensor(
            [4, 5, 6],
            dtype=torch.long,
        ),
    )


def test_collect_features_rejects_malformed_batch():
    model = FeatureModel()

    loader = [
        (
            torch.tensor(
                [[1.0, 2.0, 3.0]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )
    ]

    try:
        collect_features(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "(images, labels, task_id)" in str(exc)
    else:
        raise AssertionError(
            "collect_features should reject malformed batches"
        )


def test_collect_features_rejects_wrong_feature_shape():
    class BadFeatureModel(nn.Module):
        def extract_features(self, images):
            return images.unsqueeze(1)

    model = BadFeatureModel()

    loader = [
        (
            torch.tensor(
                [[1.0, 2.0, 3.0]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )
    ]

    try:
        collect_features(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "[B, D]" in str(exc)
    else:
        raise AssertionError(
            "collect_features should reject non-[B, D] features"
        )